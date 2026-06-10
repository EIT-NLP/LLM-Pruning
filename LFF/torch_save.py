from transformers import TrainerCallback
import os
import torch
class SaveCheckpointCallback(TrainerCallback):
    def __init__(self, save_steps, model_name, method, sparsity, lora):
        self.save_steps = save_steps
        self.save_dir = f"{model_name}_{method}_{sparsity}_lora{lora}"
        os.makedirs(self.save_dir, exist_ok=True)
        
    def on_step_end(self, args, state, control, **kwargs):

        # 检查是否达到保存步数
        if state.global_step % self.save_steps == 0:
            # 获取模型
            model = kwargs.get('model')
            if model is not None:
                # 构建保存路径
                checkpoint_dir = os.path.join(self.save_dir, f"checkpoint-{state.global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                file_path = os.path.join(checkpoint_dir, "model.pth")
                
                # 保存模型
                torch.save(model.state_dict(), file_path)
                print(f"Model saved at step {state.global_step}")

class SaveCheckpointCallback_train_lw(TrainerCallback):
    def __init__(self, save_steps, model_name, method, sparsity, lora):
        self.save_steps = save_steps
        self.save_dir = f"checkpoint_lw"
        os.makedirs(self.save_dir, exist_ok=True)
        
    def on_step_end(self, args, state, control, **kwargs):

        # 检查是否达到保存步数
        if state.global_step % self.save_steps == 0:
            # 获取模型
            model = kwargs.get('model')
            if model is not None:
                # 构建保存路径
                checkpoint_dir = os.path.join(self.save_dir, f"checkpoint-{state.global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                file_path = os.path.join(checkpoint_dir, "model.pth")
                # 保存模型
                model_dict = model.state_dict()
                # 只保存需要的参数
                model_dict = {k: v for k, v in model_dict.items() if 'lw'in k}
                torch.save(model_dict, file_path)
                print(f"Model saved at step {state.global_step}")

                model_dict = model.state_dict()
                # 只保存需要的参数
                model_dict = {k: v for k, v in model_dict.items() if 'router'in k}
                file_path = os.path.join(checkpoint_dir, "init_router_model.pth")
                torch.save(model_dict, file_path)
                print(f"init router Model saved at step {state.global_step}")


class SaveCheckpointCallback_retrain_lw(TrainerCallback):
    def __init__(self, save_steps, model_name, method, sparsity, lora):
        self.save_steps = save_steps
        self.save_dir = f"checkpoint_retrain_lw"
        os.makedirs(self.save_dir, exist_ok=True)
        
    def on_step_end(self, args, state, control, **kwargs):

        # 检查是否达到保存步数
        if state.global_step % self.save_steps == 0:
            # 获取模型
            model = kwargs.get('model')
            if model is not None:
                # 构建保存路径
                checkpoint_dir = os.path.join(self.save_dir, f"checkpoint-{state.global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                file_path = os.path.join(checkpoint_dir, "model.pth")
                # 保存模型
                model_dict = model.state_dict()
                # 只保存需要的参数
                model_dict = {k: v for k, v in model_dict.items() if 'lw'in k}
                torch.save(model_dict, file_path)
                print(f"Model saved at step {state.global_step}")

                torch.save(model, "saved_models/router_retrain_lw.pt")

class SaveCheckpointCallback_train_router(TrainerCallback):
    def __init__(self, save_steps, model_name, method, sparsity, lora):
        self.save_steps = save_steps
        self.save_dir = f"checkpoint_router"
        os.makedirs(self.save_dir, exist_ok=True)
        
    def on_step_end(self, args, state, control, **kwargs):

        # 检查是否达到保存步数
        if state.global_step % self.save_steps == 0:
            # 获取模型
            model = kwargs.get('model')
            if model is not None:
                # 构建保存路径
                checkpoint_dir = os.path.join(self.save_dir, f"checkpoint-{state.global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                file_path = os.path.join(checkpoint_dir, "model.pth")
                # 保存模型
                model_dict = model.state_dict()
                # 只保存需要的参数
                model_dict = {k: v for k, v in model_dict.items() if 'router'in k}
                torch.save(model_dict, file_path)
                print(f"Model saved at step {state.global_step}")

                torch.save(model, "saved_models/llama3_8b_router.pt")

class SaveCheckpointCallback_retrain_router(TrainerCallback):
    def __init__(self, save_steps, model_name, method, sparsity, lora):
        self.save_steps = save_steps
        self.save_dir = f"checkpoint_retrain_router"
        os.makedirs(self.save_dir, exist_ok=True)
        
    def on_step_end(self, args, state, control, **kwargs):

        # 检查是否达到保存步数
        if state.global_step % self.save_steps == 0:
            # 获取模型
            model = kwargs.get('model')
            if model is not None:
                # 构建保存路径
                checkpoint_dir = os.path.join(self.save_dir, f"checkpoint-{state.global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                file_path = os.path.join(checkpoint_dir, "model.pth")
                # 保存模型
                model_dict = model.state_dict()
                # 只保存需要的参数
                model_dict = {k: v for k, v in model_dict.items() if 'router'in k}
                torch.save(model_dict, file_path)
                print(f"Model saved at step {state.global_step}")

                torch.save(model, "saved_models/retrain_router.pt")
                
class SaveCheckpointCallback_train_lw_and_router(TrainerCallback):
    def __init__(self, save_steps, model_name, method, sparsity, lora):
        self.save_steps = save_steps
        self.save_dir = f"checkpoint_lw_and_router"
        os.makedirs(self.save_dir, exist_ok=True)
        
    def on_step_end(self, args, state, control, **kwargs):
        # 检查是否达到保存步数
        if state.global_step % self.save_steps == 0:
            # 获取模型
            model = kwargs.get('model')
            if model is not None:
                # 构建保存路径
                checkpoint_dir = os.path.join(self.save_dir, f"checkpoint-{state.global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                file_path = os.path.join(checkpoint_dir, "model.pth")
                # 保存模型
                model_dict = model.state_dict()
                # 只保存需要的参数
                model_dict = {k: v for k, v in model_dict.items() if ('router'in k or 'lw' in k)}
                torch.save(model_dict, file_path)
                print(f"Model saved at step {state.global_step}")

class SaveCheckpointCallback_train_lora(TrainerCallback):
    def __init__(self, save_steps, model_name, method, sparsity, lora):
        self.save_steps = save_steps
        self.save_dir = f"checkpoint_lora"
        os.makedirs(self.save_dir, exist_ok=True)
        
    def on_step_end(self, args, state, control, **kwargs):
        # 检查是否达到保存步数
        if state.global_step % self.save_steps == 0:
            # 获取模型
            model = kwargs.get('model')
            if model is not None:
                # 构建保存路径
                checkpoint_dir = os.path.join(self.save_dir, f"checkpoint-lora-{state.global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                file_path = os.path.join(checkpoint_dir, "model.pth")
                # 保存模型
                model_dict = model.state_dict()
                # 只保存需要的参数
                model_dict = {k: v for k, v in model_dict.items() if 'lora'in k}
                torch.save(model_dict, file_path)
                print(f"Model saved at step {state.global_step}")

                torch.save(model, "saved_models/llama3_8b_router_lora.pt")

class SaveCheckpointCallback_post_training(TrainerCallback):
    def __init__(self, save_steps, model_name, method, sparsity):
        self.save_steps = save_steps
        self.save_dir = f"{model_name}_{method}_{sparsity}_post_training"
        os.makedirs(self.save_dir, exist_ok=True)
        
    def on_step_end(self, args, state, control, **kwargs):
        # 检查是否达到保存步数
        if state.global_step % self.save_steps == 0:
            # 获取模型
            model = kwargs.get('model')
            if model is not None:
                # 构建保存路径
                checkpoint_dir = os.path.join(self.save_dir, f"checkpoint-{state.global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                file_path = os.path.join(checkpoint_dir, "model.bin")
                
                # 保存模型
                torch.save(model.state_dict(), file_path)
                print(f"Model saved at step {state.global_step}")

class SaveCheckpointCallback_original(TrainerCallback):
    def __init__(self, save_steps, model_name):
        self.save_steps = save_steps
        self.save_dir = f"{model_name}_original_lora"
        os.makedirs(self.save_dir, exist_ok=True)
        
    def on_step_end(self, args, state, control, **kwargs):
        # 检查是否达到保存步数
        if state.global_step % self.save_steps == 0:
            # 获取模型
            model = kwargs.get('model')
            if model is not None:
                # 构建保存路径
                checkpoint_dir = os.path.join(self.save_dir, f"checkpoint-{state.global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                file_path = os.path.join(checkpoint_dir, "model.bin")
                
                # 保存模型
                torch.save(model.state_dict(), file_path)
                print(f"Model saved at step {state.global_step}")

class CustomEvalCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # 每隔 200 个 global_step 进行评估
        if state.global_step % 100 == 0 and state.global_step > 0:
            trainer = kwargs['trainer']
            eval_results = trainer.evaluate()
            print(f"Evaluation at step {state.global_step}: {eval_results}")