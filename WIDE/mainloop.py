import os
import math
import time
import tqdm
import pickle
import swanlab
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed.nn as dist_nn
import regex as re

from lightning import LightningDataModule
from lightning.fabric import Fabric
from lightning.fabric.strategies.model_parallel import ModelParallelStrategy
from lightning.fabric.strategies.single_device import SingleDeviceStrategy
from datasets import Dataset, load_dataset
from torch.optim.lr_scheduler import LRScheduler
from transformers import PreTrainedModel, PretrainedConfig, PreTrainedTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from torchmetrics import MeanMetric, SumMetric, Metric
from argparse import Namespace
from functools import cache, partial
from typing import Optional, Dict, List, Any
from itertools import chain
from peft import LoraConfig, get_peft_model

from transformer_fsdp_hook import fsdp_hook

### Preprocessing func ###
def tokenize_sample(dataset, tokenizer: PreTrainedTokenizer, args: Namespace):
    return dataset.map(
        lambda x: tokenizer(
            x["text"],
            padding="max_length",
            max_length=args.max_length,
            truncation=True,
        ),
        batched=True,
        batch_size=16,
        num_proc=32,
        remove_columns=dataset.column_names,
        keep_in_memory=True,
    )

def tokenize_concat(dataset, tokenizer: PreTrainedTokenizer, args: Namespace):
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer.eos_token_id is None, concat mode needs an EOS token.")

    def tokenize_with_eos(examples):
        tokenized = tokenizer(
            examples["text"],
            add_special_tokens=False,
        )

        tokenized["input_ids"] = [
            input_ids + [eos_id]
            for input_ids in tokenized["input_ids"]
        ]

        tokenized["attention_mask"] = [
            attention_mask + [1]
            for attention_mask in tokenized["attention_mask"]
        ]

        return tokenized

    tokenized = dataset.map(
        tokenize_with_eos,
        batched=True,
        batch_size=16,
        num_proc=32,
        remove_columns=dataset.column_names,
        keep_in_memory=True,
    )

    def group_texts(examples):
        concatenated = {
            k: list(chain.from_iterable(examples[k]))
            for k in examples.keys()
        }

        total_length = len(concatenated["input_ids"])
        total_length = (total_length // args.max_length) * args.max_length

        return {
            k: [
                t[i:i + args.max_length]
                for i in range(0, total_length, args.max_length)
            ]
            for k, t in concatenated.items()
        }

    return tokenized.map(
        group_texts,
        batched=True,
        batch_size=512,
        num_proc=32,
        keep_in_memory=True,
    )

@cache
def is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1

@cache
def is_main_process():
    return dist.is_initialized() and dist.get_rank() == 0

def get_router_names(model: PreTrainedModel):
    for name, module in model.named_modules():
        if hasattr(module, '_router_names'):
            return module._router_names

class CosineLRSchedule(LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup: Optional[int]=10000,
        max_lr: Optional[float]=1e-4,
        min_lr: Optional[float]=1e-6,
        max_steps: Optional[float]=-1,
        **kwargs
    ):
        assert max_steps > 0, "max_steps must be greater than 0"
        self.warmup = warmup
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.max_steps = max_steps

        super(CosineLRSchedule, self).__init__(optimizer)
    
    def get_lr(self) -> List[float]:
        step = max(1, self._step_count)
        if step <= self.warmup:
            scale = step / self.warmup
            return [min(lr * scale, self.max_lr) for lr in self.base_lrs]
        else:
            scale = (self.min_lr + 0.5 * (self.max_lr - self.min_lr) * \
                    (1.0 + math.cos(((step - self.warmup) / (max(self.max_steps, step) - self.warmup)) * math.pi))) / self.max_lr
            if scale * self.max_lr < self.min_lr:
                scale = self.min_lr / self.max_lr
            return [min(lr * scale, self.max_lr) for lr in self.base_lrs]


class HFDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data: Dict[str, np.ndarray], # tokenized files
        config: PretrainedConfig,
        args: Namespace,
        **kwargs,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.data = data
        self.config = config
        self.args = args
    
    def __getitem__(self, index: int):
        input_ids = torch.tensor(self.data['input_ids'][index])
        attention_mask = torch.tensor(self.data['attention_mask'][index])
        labels = input_ids.clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }
    
    def __len__(self):
        return len(self.data['input_ids'])


def custom_collate_fn(batch: List[Dict[str, torch.Tensor]]):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }


class LightiningDataWrapper(LightningDataModule):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        train_dataset_path: str,
        eval_dataset_path: str,
        config: PretrainedConfig,
        args: Namespace,
        **kwargs,
    ):
        super().__init__()

        self.config = config
        self.args = args

        with open(train_dataset_path, 'rb') as f:
            train_dataset = pickle.load(f)
        with open(eval_dataset_path, 'rb') as f:
            eval_dataset = pickle.load(f)

        # 2. convert to torch dataset
        self.train_dataset = HFDataset(tokenizer, train_dataset, config, args)
        self.eval_dataset = HFDataset(tokenizer, eval_dataset, config, args)
        print(f'Finish loading cached datasets: {args.train_dataset} and {args.eval_dataset}')
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            num_workers=vars(self.args).get('num_workers', 8),
            collate_fn=partial(custom_collate_fn),
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            num_workers=vars(self.args).get('num_workers', 8),
            collate_fn=partial(custom_collate_fn),
        )


#########################################################
#                  --- trainer ---
#########################################################
class Trainer:
    def __init__(
        self,
        fabric: Fabric,
        model: PreTrainedModel,
        datamodule: LightningDataModule,
        config: PretrainedConfig,
        args: Namespace,
        **kwargs,
    ):
        self.config = config
        self.args = args

        # setup distribution
        self.model = fabric.setup_module(model)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            self.args.max_lr,
            weight_decay=self.args.weight_decay,
        )
        schedule = CosineLRSchedule(
            optimizer,
            warmup=int(self.args.max_steps * self.args.warmup_ratio),
            max_lr=self.args.max_lr,
            min_lr=vars(self.args).get('min_lr', 0),
            max_steps=self.args.max_steps,
        )

        self.optimizer = fabric.setup_optimizers(optimizer)
        self.schedule = schedule
        self.model: PreTrainedModel = self.model.train()

        self.train_dataloader, self.eval_dataloader = fabric.setup_dataloaders(datamodule.train_dataloader(), datamodule.val_dataloader())

        self.fabric = fabric

        # prepare training
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        self.max_epochs = getattr(self.args, 'max_epochs', -1)
        self.max_steps = args.max_steps
        self.eval_steps = args.eval_steps
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.log_steps = 10

        self.router_names = get_router_names(self.model)
        print(f"Router names: {self.router_names}")
        self.train_metrics = {
            'loss|lm': MeanMetric().to(self.fabric.strategy.root_device),
            'loss|sparsity': MeanMetric().to(self.fabric.strategy.root_device),
            'sparsity|total': MeanMetric().to(self.fabric.strategy.root_device),
        }
        self.eval_metrics = {
            'loss|lm': MeanMetric().to(self.fabric.strategy.root_device),
            'sparsity|total': MeanMetric().to(self.fabric.strategy.root_device),
        }
        for router_name in self.router_names:
            self.train_metrics[f'sparsity|{router_name}'] = MeanMetric().to(self.fabric.strategy.root_device)
            self.eval_metrics[f'sparsity|{router_name}'] = MeanMetric().to(self.fabric.strategy.root_device)

        if self.fabric.is_global_zero:
            # setup swanlab logger
            swanlab.init(
                project=self.args.project_name,
                experiment_name=f"{self.args.model_name.replace('/', '_')}-{self.args.wrapper_type}-{self.args.sparse_target}-{self.args.sparsity}-A{args.attn_group_size}-F{args.ffn_group_size}-tag:{self.args.tag}",
                logdir=self.save_dir,
                config=vars(self.args),
            )

            print(f"------------- Architecture -------------")
            print(self.model)
            print(f"Total Param: {sum([p.numel() for p in self.model.parameters()])}")
            print(f"Trainable Param: {sum([p.numel() for p in self.model.parameters() if p.requires_grad])}")
    
    def save(self):
        # https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html
        sharded_sd = self.model.state_dict()
        state_dict = {}
        for name, param in sharded_sd.items():
            full_param = param.full_tensor() if torch.cuda.device_count() > 1 else param
            if self.fabric.is_global_zero:
                state_dict[name] = full_param.cpu()
            else:
                del full_param

        # filter out router and lora
        if self.fabric.is_global_zero:
            router_dict = {}
            lora_dict = {}

            for k in state_dict.keys():
                if 'router' in k:
                    new_k = re.findall(r'[\w\W]*?(layers.[\d]+.[\w\W]+)', k)[-1]
                    new_k = f"model.{new_k}"
                    router_dict[new_k] = state_dict[k]
                    print(f"|---> Sucessful catch router {new_k}, size{state_dict[k].shape}")
                elif 'lora_A' in k or 'lora_B' in k:
                    new_k = re.findall(r'[\w\W]*?(layers.[\d]+.[\w\W]+)', k)[-1]
                    new_k = re.sub(r'.block', '', new_k)
                    new_k = f"model.{new_k}"
                    lora_dict[new_k] = state_dict[k]
                    print(f"|---> Sucessful catch lora {new_k}, size{state_dict[k].shape}")
            
            output_path = f"{self.save_dir}/router_stage1"
            if self.args.lora: output_path = f"{self.save_dir}/router_stage2"
            os.makedirs(output_path, exist_ok=True)
            
            if len(router_dict) > 0: torch.save(router_dict, os.path.join(output_path, f"router.pth"))
            if len(lora_dict) > 0: torch.save(lora_dict, os.path.join(output_path, f"lora.pth"))
            print(f"Sucessful save router and lora to {output_path}")
    
    def evaluate(self, global_steps: int):
        self.fabric.print(f"Start evaluate at step {global_steps}...")
        self.model.eval()

        if self.fabric.is_global_zero:
            progress_bar = tqdm.tqdm(
                total=len(self.eval_dataloader),
                leave=False,
            )

        for name, module in self.model.named_modules():
            if "WrapperModel" in module.__class__.__name__: tgt_model = module

        torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.eval_dataloader):
                outputs: CausalLMOutputWithPast = self.model(**batch)
                lm_loss = outputs.loss
                all_sparsitys: Dict[str, torch.Tensor] = tgt_model.compute_sparsity()

                self.eval_metrics['loss|lm'].update(lm_loss)

                for router_name in self.router_names:
                    if router_name in all_sparsitys: self.eval_metrics[f'sparsity|{router_name}'].update(all_sparsitys[router_name])

                self.eval_metrics['sparsity|total'].update(all_sparsitys['total'])

                if self.fabric.is_global_zero:
                    progress_bar.update()
                    progress_bar.set_description(f"Evaluation step {batch_idx}/{len(self.eval_dataloader)}")
        
        torch.cuda.synchronize()
        eval_duration = time.perf_counter() - start_time
        if self.fabric.is_global_zero: progress_bar.close()

        # logging metrics
        log_metrics = {'eval/duration': eval_duration}
        for k, v in self.eval_metrics.items():
            if isinstance(v, Metric): log_metrics[f'eval/{k}'] = v.compute().item()
        
        if self.fabric.is_global_zero:
            swanlab.log(log_metrics, step=global_steps)
        
        for k, v in self.eval_metrics.items():
            if isinstance(v, Metric): v.reset()
        
        self.model.train()
    
    def epilogue(self):
        self.save()
        self.fabric.print("------------ End Training ------------")
        if self.fabric.is_global_zero:
            swanlab.finish()

            if self.profiler is not None:
                self.profiler.stop()
                print(self.profiler.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs: CausalLMOutputWithPast = self.model(**batch)
        lm_loss = outputs.loss

        # 1. compute target sparsity loss
        tgt_model = None
        for name, module in self.model.named_modules():
            if "WrapperModel" in module.__class__.__name__: tgt_model = module

        all_sparsitys: Dict[str, torch.Tensor] = tgt_model.compute_sparsity()
        router_zero_prob: Dict[str, torch.Tensor] = tgt_model.router_zero_prob
        active_router_rates = {
            name: router_zero_prob[name]
            for name in ('attention', 'ffn')
            if name in router_zero_prob
        }
        if not active_router_rates:
            raise RuntimeError("No active router sparsity was produced by the model")

        total_router_loss = torch.stack(list(active_router_rates.values())).mean()
        target_sparsities = {
            'attention': self.args.sparsity_attention,
            'ffn': self.args.sparsity_ffn,
        }
        configured_targets = {
            name: target_sparsities[name]
            for name in active_router_rates
            if target_sparsities[name] >= 0
        }

        if configured_targets:
            if configured_targets.keys() != active_router_rates.keys():
                raise ValueError(
                    "Set a per-router sparsity target for every active router, "
                    "or leave all per-router targets unset"
                )
            target_losses = [
                abs(configured_targets[name] - active_router_rates[name])
                for name in active_router_rates
            ]
            sparsity_loss = 20 * torch.stack(target_losses).mean()
        else:
            sparsity_loss = 20 * abs(self.args.sparsity - total_router_loss)
        
        loss = lm_loss + sparsity_loss

        # 2. logging metrics
        self.train_metrics['loss|lm'].update(lm_loss.detach())
        self.train_metrics['loss|sparsity'].update(sparsity_loss.detach())
        for router_name in self.router_names:
            if router_name in all_sparsitys: self.train_metrics[f'sparsity|{router_name}'].update(all_sparsitys[router_name])

        self.train_metrics['sparsity|total'].update(all_sparsitys['total'])

        return loss

    def fit(self):
        local_steps = 0 # num dataloader steps
        global_steps = 0 # num optimizer steps
        global_epochs = 0 # num epochs have been executed
        terminate = False

        self.profiler = None
        if self.fabric.is_global_zero:
            # setup progress bar
            progress_bar = tqdm.tqdm(
                total=self.max_steps,
                desc=f"Epoch {global_epochs}/{self.max_epochs} | Step {global_steps}/{self.max_steps}",
                leave=False,
            )

            if self.args.profiling:
                self.profiler = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    schedule=torch.profiler.schedule(wait=1, warmup=5, active=4, repeat=1),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=False,
                )
                self.profiler.start()
        
        while not terminate:
            torch.cuda.empty_cache()

            for batch_idx, batch in enumerate(self.train_dataloader):
                local_steps += 1
                is_accumulating = (local_steps % self.gradient_accumulation_steps) != 0

                with self.fabric.no_backward_sync(self.model, enabled=is_accumulating):
                    loss = self.train_step(batch)
                    self.fabric.backward(loss / self.gradient_accumulation_steps)
                
                if self.profiler is not None:
                    self.profiler.step()
                
                if is_accumulating: continue

                # optimizer update step
                global_steps += 1
                grad_norm = None
                lr = None

                if self.args.max_grad_norm > 0:
                    grad_norm = self.fabric.clip_gradients(
                        module=self.model,
                        optimizer=self.optimizer,
                        max_norm=self.args.max_grad_norm,
                        error_if_nonfinite=False,
                    )
                
                if self.optimizer is not None:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                
                if self.schedule is not None:
                    self.schedule.step()
                    lr = self.optimizer.param_groups[0]["lr"]

                # logging metrics
                if self.fabric.is_global_zero:
                    progress_bar.update()
                    progress_bar.set_description(f"Epoch {global_epochs}/{self.max_epochs} | Step {global_steps}/{self.max_steps}")
                
                if global_steps % self.log_steps == 0:
                    log_metrics = {}
                    if grad_norm is not None: log_metrics['train/grad_norm'] = grad_norm.item()
                    if lr is not None: log_metrics['train/lr'] = lr

                    for k, v in self.train_metrics.items():
                        if isinstance(v, Metric): log_metrics[f'train/{k}'] = v.compute().item()

                    if self.fabric.is_global_zero: swanlab.log(log_metrics, step=global_steps)

                    for k, v in self.train_metrics.items():
                        if isinstance(v, Metric): v.reset()
                
                # eval
                if global_steps % self.eval_steps == 0:
                    self.evaluate(global_steps)
                
                # update terminate
                terminate = (global_steps >= self.max_steps) or (self.max_epochs > 0 and (global_epochs >= self.max_epochs))
                if terminate: break

            global_epochs += 1
            terminate = (global_steps >= self.max_steps) or (self.max_epochs > 0 and (global_epochs >= self.max_epochs))
            if terminate: break
        
        if self.fabric.is_global_zero: progress_bar.close()
        self.epilogue()


def mainloop_func(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: PretrainedConfig,
    args: Namespace,
):
    # apply lora wrapper
    if args.lora:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
            lora_dropout=0.1,
            bias="none",
        )

        model = get_peft_model(model, lora_config)
        for name, param in model.named_parameters():
            if 'router' in name:
                param.requires_grad = True if not args.freeze_router else False
    
    else:
        for name, param in model.named_parameters():
            if 'router' not in name:
                param.requires_grad = False
            else: param.requires_grad = True
    
    model = model.to(torch.bfloat16)

    # init fabric
    if torch.cuda.device_count() > 1:
        torch.distributed.init_process_group(backend='cuda:nccl,cpu:gloo' if args.cpu_offload else 'nccl')
        strategy = ModelParallelStrategy(
            parallelize_fn=partial(fsdp_hook, cpu_offload=args.cpu_offload),
            data_parallel_size=torch.cuda.device_count(),
            tensor_parallel_size=1,
            save_distributed_checkpoint=False,
        )
    else:
        strategy = SingleDeviceStrategy(device='cuda:0')

    fabric = Fabric(
        accelerator='auto',
        strategy=strategy,
        precision='bf16-mixed',
    )
    fabric.launch()
    fabric.seed_everything(args.seed)

    # preprocess dataset
    base_path = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_path, 'datasets')
    os.makedirs(dataset_path, exist_ok=True)

    train_dataset_path = os.path.join(dataset_path, f"{args.train_dataset.replace('/', '_')}_{args.max_length}_{args.train_data_mode}_train.pkl")
    eval_dataset_path = os.path.join(dataset_path, f"{args.eval_dataset.replace('/', '_')}_{args.max_length}_eval.pkl")

    if not (os.path.exists(train_dataset_path) and os.path.exists(eval_dataset_path)):
        if fabric.is_global_zero:
            train_dataset = load_dataset(args.train_dataset)
            if args.train_dataset == args.eval_dataset:
                eval_dataset = train_dataset
                if 'test' not in train_dataset:
                    train_valid_split = train_dataset['train'].train_test_split(test_size=args.eval_dataset_ratio)
                    train_dataset = train_valid_split
                    eval_dataset = train_valid_split
            else:
                eval_dataset = load_dataset(args.eval_dataset)

            if args.train_data_mode == 'sample':
                train_dataset = tokenize_sample(train_dataset['train'], tokenizer, args)
            elif args.train_data_mode == 'concat':
                train_dataset = tokenize_concat(train_dataset['train'], tokenizer, args)

            eval_dataset = tokenize_sample(eval_dataset['test'], tokenizer, args)
            
            print(f'Finish mapping datasets: {args.train_dataset} and {args.eval_dataset}')

            train_dataset = {k: train_dataset[k] for k in train_dataset.column_names}
            eval_dataset = {k: eval_dataset[k] for k in eval_dataset.column_names}

            print(f'Finish converting datasets: {args.train_dataset} and {args.eval_dataset}')
            
            with open(train_dataset_path, 'wb') as f:
                pickle.dump(train_dataset, f)
            with open(eval_dataset_path, 'wb') as f:
                pickle.dump(eval_dataset, f)
            
            print(f'Finish caching datasets: {args.train_dataset} and {args.eval_dataset}')

    fabric.barrier()

    # init dataloader
    with fabric.rank_zero_first():
        data_wrapper = LightiningDataWrapper(
            tokenizer=tokenizer,
            train_dataset_path=train_dataset_path,
            eval_dataset_path=eval_dataset_path,
            config=config,
            args=args,
        )
    
    trainer = Trainer(
        fabric=fabric,
        model=model,
        datamodule=data_wrapper,
        config=config,
        args=args,
    )
    trainer.fit()
    if torch.cuda.device_count() > 1: torch.distributed.destroy_process_group()
