import torch
import torch.nn as nn

from transformers import PreTrainedModel
from torch.distributed.fsdp import MixedPrecisionPolicy, CPUOffloadPolicy, OffloadPolicy, fully_shard
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from typing import Optional

def fsdp_hook(
    model: PreTrainedModel,
    device_mesh: DeviceMesh,
    cpu_offload: Optional[bool]=False,
):
    tp_mesh = device_mesh["tensor_parallel"]
    dp_mesh = device_mesh["data_parallel"]

    tgt_model = model
    for name, module in model.named_modules():
        if hasattr(module, 'model') and hasattr(module.model, 'layers'):
            tgt_model = module
            break

    if tp_mesh.size() > 1:
        # skip ForCausalLM due to Liger Kernel fuse CE
        tgt_model = parallelize_module(tgt_model, tp_mesh, {})

        plan = {
            "layers.0": PrepareModuleInput(
                input_layouts=(Replicate(), None),
                desired_input_layouts=(Shard(1), None),
                use_local_output=True,
            ),
        }
        parallelize_module(tgt_model.model, tp_mesh, plan)

        for layer_id, block in enumerate(tgt_model.model.layers):
            plan = {
                'self_attn.q_proj': ColwiseParallel(),
                'self_attn.k_proj': ColwiseParallel(),
                'self_attn.v_proj': ColwiseParallel(),
                'self_attn.o_proj': RowwiseParallel(),
                'mlp.gate_proj': ColwiseParallel(),
                'mlp.up_proj': ColwiseParallel(),
                'mlp.down_proj': RowwiseParallel(),
            }
            parallelize_module(block, tp_mesh, plan)
    
    if dp_mesh.size() > 1:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
        
        for layer_id, block in enumerate(tgt_model.model.layers):
            reshard_after_forward = int(layer_id) < len(tgt_model.model.layers) - 1
            fully_shard(
                block,
                mesh=dp_mesh,
                reshard_after_forward=reshard_after_forward,
                mp_policy=mp_policy,
                offload_policy=CPUOffloadPolicy() if cpu_offload else OffloadPolicy(),
            )
        
        fully_shard(tgt_model.model, mesh=dp_mesh, mp_policy=mp_policy)
        fully_shard(tgt_model, mesh=dp_mesh, mp_policy=mp_policy)
        
    return model