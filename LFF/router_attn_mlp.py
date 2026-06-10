# inspired by  https://github.com/kyegomez/Mixture-of-Depths
import logging
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Any

from transformers import PreTrainedModel, DynamicCache, Cache
# class TokenRouter(nn.Module):
#     def __init__(self, embed_dim):
#         super().__init__()
#         # 直接从输入维度到输出权重预测
#         self.weight_predictor = nn.Linear(embed_dim, 2)
        
#         # 使用 He Kaiming 初始化
#         nn.init.kaiming_uniform_(self.weight_predictor.weight, nonlinearity='linear')
        
#         # 初始化 bias 为 0
#         if self.weight_predictor.bias is not None:
#             nn.init.zeros_(self.weight_predictor.bias)

#     def forward(self, x):
#         # 保存输入的原始数据类型
#         original_type = x.dtype
        
#         # 计算权重预测
#         weights = self.weight_predictor(x.to(self.weight_predictor.weight.dtype))
        
#         return weights.to(original_type)

# 添加第三种类别的预测：可以使用轻量级网络恢复精度的token
class TokenRouter(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        # 中间层的维度是 embed_dim 的四分之一
        intermediate_dim = embed_dim //4 - 200
        # 增加一个中间层
        self.hidden_layer = nn.Linear(embed_dim, intermediate_dim)
        self.relu = nn.ReLU() 
        self.weight_predictor = nn.Linear(intermediate_dim, 2)
        # self.weight_predictor = nn.Linear(intermediate_dim, 3)
        
        # 使用 He Kaiming 初始化
        nn.init.kaiming_uniform_(self.hidden_layer.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.weight_predictor.weight, nonlinearity='linear')

        # 初始化 bias 为 0
        if self.hidden_layer.bias is not None:
            nn.init.zeros_(self.hidden_layer.bias)
        if self.weight_predictor.bias is not None:
            nn.init.zeros_(self.weight_predictor.bias)

    def forward(self, x):
        original_type = x.dtype
        
        # 先通过中间层并激活，再传递到 weight_predictor
        x = self.hidden_layer(x.to(self.hidden_layer.weight.dtype))
        x = self.relu(x)  # 使用 ReLU 激活函数
        
        # 计算最终的权重
        weights = self.weight_predictor(x.to(self.weight_predictor.weight.dtype))
    
        return weights.to(original_type)

# 下采样+上采样的轻量级网络
class LightweightNetwork(nn.Module):
    def __init__(self, hidden_size, lw_net_rank=100):
        super().__init__()
        lw_net_rank=100
        self.linear1 = nn.Linear(hidden_size, lw_net_rank)
        # self.act = nn.ReLU()
        self.linear2 = nn.Linear(lw_net_rank, hidden_size)

        # 使用 He Kaiming 初始化
        # nn.init.kaiming_uniform_(self.linear1.weight, nonlinearity='relu')
        # nn.init.kaiming_uniform_(self.linear2.weight, nonlinearity='linear')
        nn.init.normal_(self.linear1.weight, mean=0, std=1e-3)
        nn.init.normal_(self.linear2.weight, mean=0, std=1e-3)

        # 初始化 bias 为 0
        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear1.bias)
        if self.linear2.bias is not None:
            nn.init.zeros_(self.linear2.bias)

    def forward(self, x):
        if x.dtype != self.linear1.weight.dtype:
            x = x.to(self.linear1.weight.dtype)
        x = self.linear1(x)   
        x = nn.functional.silu(x)  
        # x = self.act(x)
        x = self.linear2(x)
        return x

# hadamard的轻量级网络
# class LightweightNetwork(nn.Module):
#     def __init__(self, hidden_size, lw_net_rank=100):
#         super().__init__()
#         lw_net_rank=100
#         self.weight = nn.Parameter(torch.randn(hidden_size))
#         nn.init.normal_(self.weight, mean=0, std=1e-3)

#     def forward(self, x):
#         x = x * self.weight.unsqueeze(0).unsqueeze(0)  # 扩展维度以匹配输入形状
#         return x

# class LightweightNetwork(nn.Module):
#     def __init__(self, hidden_size, lw_net_rank=1):
#         super().__init__()
#         self.intermediate_size = int(hidden_size * lw_net_rank)  # 低秩网络的中间层大小
#         self.hidden_size = hidden_size
#         mlp_bias = False  # 是否使用 bias
#         self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
#         self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
#         self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=mlp_bias)
#         self.act_fn = F.silu

#     def forward(self, x):
#         x = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
#         return x

class LightweightNetwork_attn(nn.Module):
    def __init__(self, hidden_size, lw_net_rank=1):
        super().__init__()

    def forward(self, x):
        return torch.zeros_like(x)  # 返回一个与输入形状相同的零张量，模拟轻量级网络的输出

def compute_cos(tensor1, tensor2):
    cos_sim = F.cosine_similarity(tensor1, tensor2, dim=-1)
    return cos_sim.mean().item()  # 返回平均余弦相似度

class CosineSimilarityLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CosineSimilarityLoss, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        # 计算余弦相似度
        cos_sim = nn.functional.cosine_similarity(x, y, dim=-1)
        
        # 因为要最大化余弦相似度，所以取负值来最小化损失
        loss = 1-cos_sim
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class router_attn_mlp_llama(nn.Module):
    def __init__(self, block, hidden_size, args):
        super().__init__()
        self.router_attention = TokenRouter(hidden_size)
        self.router_mlp = TokenRouter(hidden_size)
        self.block = block
        self.training_step = 0
        self.args= args

        # initialize the total tokens and skipped tokens
        self.total_tokens = 0
        self.skipped_attn_tokens = 0
        self.skipped_mlp_tokens = 0
        self.lw_attn_tokens = 0
        self.lw_mlp_tokens = 0
        

        # record the sparsity of the routers
        self.attn_router_zero_prob = 0.0  
        self.mlp_router_zero_prob = 0.0   

        # 初始化存储 token 路由信息的字典
        self.routing_matrix = {
            "attention": None,
            "mlp": None
        }

        # 初始化轻量级网络
        # self.lw_attn = LightweightNetwork_attn(hidden_size, args.lw_net_rank)
        self.lw_attn = LightweightNetwork(hidden_size, args.lw_net_rank)
        self.lw_mlp = LightweightNetwork(hidden_size, args.lw_net_rank)

        # freeze the parameters of the block
        for param in self.block.parameters():
            param.requires_grad = False
        
        self.train_lw_only = args.train_lw_only
        self.lw_attn_loss = None
        self.lw_mlp_loss = None
        self.lw_loss = nn.SmoothL1Loss(reduction='none')  # 使用SmoothL1Loss作为轻量级网络的损失函数
        # self.lw_loss = CosineSimilarityLoss(reduction='none')
        # self.lw_loss = nn.MSELoss(reduction='none')
        self.cos_mlp_ori = None
        self.cos_mlp_lw = None
        self.cos_attn_ori = None
        self.cos_attn_lw = None

        self.retrain_lw = args.retrain_lw
        self.init_router = args.init_router
        self.post_norm_router = args.post_norm_router  # 是否在后置层归一化后计算路由器
        self.skipgpt = args.skipgpt  # 是否使用skipgpt

        self.ce_loss = nn.CrossEntropyLoss()
        self.log_file = "sparsity.txt"

        self.regu_loss = 0
    
    def return_lw_loss(self):
        return self.lw_attn_loss, self.lw_mlp_loss

    def reset_lw_loss(self):
        self.lw_attn_loss = None
        self.lw_mlp_loss = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        **kwargs,
    ):
        
        b, s, feature_dim = hidden_states.shape
        TRAIN_FOLLOW_ROUTER = self.retrain_lw
        # TRAIN_FOLLOW_ROUTER = True
        if self.train_lw_only or TRAIN_FOLLOW_ROUTER:
            hidden_states_copy = hidden_states.detach()
            with torch.no_grad():
                # 防止attention mask为None
                if attention_mask is None:
                    attention_mask = torch.ones((b, s), device=hidden_states.device)
                # 训练过程中temperature逐渐降低
                if self.training and any(param.requires_grad for param in self.router_attention.parameters()):
                    if self.training_step <  self.args.gradient_accumulation_steps * self.args.max_steps_stage:
                        self.training_step += 1
                    temperature = self.args.initial_temperature - (self.args.initial_temperature - self.args.final_temperature) * ((self.training_step-1) // self.args.gradient_accumulation_steps )/ ( self.args.max_steps_stage)
                else:
                    temperature = self.args.final_temperature
                # llama attn
                residual = hidden_states_copy
                
                if TRAIN_FOLLOW_ROUTER:
                    weights = self.router_attention(hidden_states_copy)
                    gumbel_weights = F.gumbel_softmax(weights, tau=temperature, hard=True, dim=-1)
                    # lw_attn_mask = gumbel_weights[:, :, 2] * attention_mask
                    lw_attn_mask = gumbel_weights[:, :, 1] * attention_mask
                hidden_states = self.block.input_layernorm(hidden_states)
                attn_input = hidden_states
                attn_output, self_attn_weights, present_key_value = self.block.self_attn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
                hidden_states = residual + attn_output
                attn_label = hidden_states
                # llama mlp
                residual = hidden_states
                if TRAIN_FOLLOW_ROUTER:
                    weights_mlp = self.router_mlp(hidden_states)
                    gumbel_weights_mlp = F.gumbel_softmax(weights_mlp, tau=temperature, hard=True, dim=-1)
                    lw_mlp_mask = gumbel_weights_mlp[:, :, 1] * attention_mask
                hidden_states = self.block.post_attention_layernorm(hidden_states)
                mlp_input = hidden_states
                mlp_output = self.block.mlp(hidden_states)
                hidden_states = residual + mlp_output
            # attn loss
            lw_attn = self.lw_attn(attn_input.detach())
            lw_out_attn = hidden_states_copy + lw_attn
            self.lw_attn_loss = self.lw_loss(lw_attn.float(), attn_output.float())

            # router init
            attn_input_reshape = attn_input.detach().squeeze(0) # 【1， 128， 4096】 → 【128， 4096】
            sum_lw_attn_loss = self.lw_attn_loss.sum(dim=-1).squeeze(0)
            _, indices = torch.sort(sum_lw_attn_loss) # 对重构误差进行排序
            labels = torch.zeros(s, dtype=torch.long) # 此处sequence length 为128
            labels[indices[:s//4]] = 1  

            attn_router_output = self.router_attention(attn_input_reshape)
            attn_router_init_loss = self.ce_loss(attn_router_output, labels.to('cuda'))


            # mlp loss
            lw_mlp = self.lw_mlp(mlp_input.detach())
            self.lw_mlp_loss = self.lw_loss(lw_mlp.float(), mlp_output.float())
            lw_out = residual + lw_mlp

            mlp_input_reshape = mlp_input.detach().squeeze(0) # 【1， 128， 4096】 → 【128， 4096】
            sum_lw_mlp_loss = self.lw_mlp_loss.sum(dim=-1).squeeze(0)
            _, indices = torch.sort(sum_lw_mlp_loss) # 对重构误差进行排序
            labels = torch.zeros(s, dtype=torch.long) # 此处sequence length 为128
            labels[indices[:s//4]] = 1 

            mlp_router_output = self.router_mlp(mlp_input_reshape)
            mlp_router_init_loss = self.ce_loss(mlp_router_output, labels.to('cuda'))


            # 只计算router内的loss
            if TRAIN_FOLLOW_ROUTER:
                self.lw_attn_loss = self.lw_attn_loss * lw_attn_mask.unsqueeze(-1)
                self.lw_mlp_loss = self.lw_mlp_loss * lw_mlp_mask.unsqueeze(-1)
                self.lw_attn_loss = self.lw_attn_loss.sum() / lw_attn_mask.sum() / feature_dim if lw_attn_mask.sum() > 0 else self.lw_attn_loss.sum()
                self.lw_mlp_loss = self.lw_mlp_loss.sum() / lw_mlp_mask.sum() / feature_dim if lw_mlp_mask.sum() > 0 else self.lw_mlp_loss.sum()
            else:
                self.lw_attn_loss = self.lw_attn_loss.mean()
                self.lw_mlp_loss = self.lw_mlp_loss.mean()

            # 计算router init loss
            if self.init_router:
                self.lw_attn_loss = attn_router_init_loss
                self.lw_mlp_loss = mlp_router_init_loss
            
            self.cos_mlp_ori = compute_cos(residual, hidden_states)
            self.cos_mlp_lw = compute_cos(lw_out, hidden_states)
            self.cos_attn_ori = compute_cos(attn_label, hidden_states_copy)
            self.cos_attn_lw = compute_cos(lw_out_attn, attn_label)

            # 更新过lw损失后进入下一层
            outputs = (hidden_states,)
            if output_attentions:
                outputs += (self_attn_weights,)
            if use_cache:
                outputs += (present_key_value,)
            return outputs

        # check for NaN in the input tokens
        if torch.isnan(hidden_states).any():
            warnings.warn(
                "NaN detected in input tokens, this is not intended to happen, please check your model.")
            hidden_state = torch.where(torch.isnan(hidden_states), torch.zeros_like(hidden_states), hidden_states)
        # 防止attention mask为None
        if attention_mask is None:
            attention_mask = torch.ones((b, s), device=hidden_states.device)

        # 更新 self.total_tokens，只统计 attention_mask 为1的 token
        self.total_tokens += attention_mask.sum().item()

        # 训练过程中temperature逐渐降低
        if self.training and any(param.requires_grad for param in self.router_attention.parameters()):
            if self.training_step <  self.args.gradient_accumulation_steps * self.args.max_steps_stage:
                self.training_step += 1
            temperature = self.args.initial_temperature - (self.args.initial_temperature - self.args.final_temperature) * ((self.training_step-1) // self.args.gradient_accumulation_steps )/ ( self.args.max_steps_stage)
        else:
            temperature = self.args.final_temperature
        # temperature = self.args.initial_temperature
        # temperature = 3

       

        # perform attention
        residual = hidden_states
        hidden_states = self.block.input_layernorm(hidden_states)

        # 轻量级网络输出（模拟attention）
        lw_attn = self.lw_attn(hidden_states)

         # 计算gumbel softmax之前的权重 router 在 rmsnorm 之后计算
        if self.post_norm_router:
            weights = self.router_attention(hidden_states)
        else: 
            weights = self.router_attention(residual)  

        # 计算gumbel softmax
        gumbel_weights = F.gumbel_softmax(weights, tau=temperature, hard=True, dim=-1)

        # # gumbel weights的最后一个维度是长度为2的one-hot vectors，第一个代表是否执行，第二个代表是否跳过，我们取出第一个维度代表selected_mask
        # selected_mask = gumbel_weights[:, :, 1] * attention_mask 
        # gumbel_weights_gate = gumbel_weights[:, :, 0]

        # 此时 gumble weights为3维
        selected_mask = gumbel_weights[:, :, 1] * attention_mask 
        gumbel_weights_gate = gumbel_weights[:, :, 0]
        # lw_attn_mask = gumbel_weights[:, :, 2] * attention_mask
        lw_attn_mask = selected_mask

        # 统计跳过 Attention 的次数
        self.skipped_attn_tokens += selected_mask.sum().item()
        self.lw_attn_tokens += lw_attn_mask.sum().item()
        # 记录router_attention的0类概率
        self.attn_router_zero_prob = gumbel_weights_gate.mean()
        # self.attn_router_zero_prob = 1-(selected_mask.mean() + 0.75*lw_attn_mask.mean())  # 计算0类概率

        # attn计算中mask掉跳过token
        # print(attention_mask.shape, attention_mask.sum())
        # print(gumbel_weights_gate.shape, gumbel_weights_gate.sum())
        # exit()
        # attention_mask = attention_mask*gumbel_weights_gate.detach()

        


        hidden_states, self_attn_weights, present_key_value = self.block.self_attn(
            hidden_states=hidden_states.bfloat16(),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        # 计算被router选中的token变换前后余弦相似度以及相应的损失
        cos_attn = nn.functional.cosine_similarity(residual, residual+hidden_states, dim=-1)
        cos_loss_attn = ((1-cos_attn)*selected_mask).sum()/(selected_mask.sum()+1)

         # 将attention的结果乘以gumbel weights

        # hidden_states = hidden_states + residual

        # 需要处理三部分计算结果：1. residue（对应完全跳过）2. hidden states （对应原始输出）3. lw （对应轻量级网络输出）
        # 注意轻量级网络输出拟合两层间的残差，因此residue都要加
        if self.skipgpt:
            hidden_states = residual + hidden_states * gumbel_weights_gate.unsqueeze(-1)
        else:
            hidden_states = residual + hidden_states * gumbel_weights_gate.unsqueeze(-1) + lw_attn * lw_attn_mask.unsqueeze(-1)
        

        # #新公式
        # hidden_states = (hidden_states + residual)*gumbel_weights_gate.unsqueeze(-1) + residual*selected_mask.unsqueeze(-1)
        

        # Fully Connected
        residual = hidden_states
        hidden_states = self.block.post_attention_layernorm(hidden_states)

        # 计算mlp gumbel softmax之前的权重  在rmsnorm之后计算router
        if self.post_norm_router:
            weights_mlp = self.router_mlp(hidden_states)
        else: 
            weights_mlp = self.router_mlp(residual)
        # 计算mlp的gumbel softmax
        gumbel_weights_mlp = F.gumbel_softmax(weights_mlp, tau=temperature, hard=True, dim=-1)

        # 计算gate
        selected_mask_mlp = gumbel_weights_mlp[:, :, 1] * attention_mask  
        gumbel_weights_gate_mlp = gumbel_weights_mlp[:, :, 0]
        # lw_mlp_mask = gumbel_weights_mlp[:, :, 2] * attention_mask
        lw_mlp_mask = selected_mask_mlp

        # 记录router_mlp的0类概率
        # self.mlp_router_zero_prob = 1-(selected_mask_mlp.mean() + 0.75*lw_mlp_mask.mean())
        self.mlp_router_zero_prob = gumbel_weights_gate_mlp.mean()
        # 统计跳过 MLP 的次数
        self.skipped_mlp_tokens += selected_mask_mlp.sum().item()
        self.lw_mlp_tokens += lw_mlp_mask.sum().item()
        lw_mlp = self.lw_mlp(hidden_states)
        hidden_states = self.block.mlp(hidden_states.bfloat16())

        # 计算被router选中的token变换前后余弦相似度以及相应的损失
        cos_mlp = nn.functional.cosine_similarity(residual, residual+hidden_states, dim=-1)
        cos_loss_mlp = ((1-cos_mlp)*selected_mask_mlp).sum()/(selected_mask_mlp.sum()+1)

        self.regu_loss = cos_loss_attn + cos_loss_mlp

        # 将mlp的结果乘以gumbel weights

        # hidden_states = hidden_states * gumbel_weights_gate_mlp.unsqueeze(-1) + residual
        # 需要处理三部分计算结果：1. residue（对应完全跳过）2. hidden states （对应原始输出）3. lw （对应轻量级网络输出）
        # hidden_states = residual + hidden_states * gumbel_weights_gate_mlp.unsqueeze(-1) + lw_mlp * lw_mlp_mask.unsqueeze(-1)
        if self.skipgpt:
            hidden_states = residual + hidden_states * gumbel_weights_gate_mlp.unsqueeze(-1)
        else:
            hidden_states = residual + hidden_states * gumbel_weights_gate_mlp.unsqueeze(-1) + lw_mlp * lw_mlp_mask.unsqueeze(-1)

        # 记录最新的路由信息
        self.routing_matrix["attention"] = selected_mask.to(torch.float32).detach().cpu().numpy()
        self.routing_matrix["mlp"] = selected_mask_mlp.to(torch.float32).detach().cpu().numpy()
        
        outputs = (hidden_states.bfloat16(),)
        

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)


        # # 记录sparsity到本地文件
        # sparsity = self.compute_sparsity()
        # self.log_sparsity(sparsity)

        return outputs

    def log_sparsity(self, sparsity_values):
        """将sparsity数值写入文件，每行记录一个样本的四个数值"""
        # 确保是列表或可迭代对象
        if not isinstance(sparsity_values, (list, tuple)):
            sparsity_values = [sparsity_values]
        
        # 转换为字符串，用逗号分隔
        line = ",".join(map(str, sparsity_values)) + "\n"
        
        # 追加写入文件（a模式确保每个样本都被记录）
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(line)

    def compute_sparsity(self):
        attn_sparsity = self.skipped_attn_tokens  / self.total_tokens if self.total_tokens > 0 else 0
        attn_lw_sparsity = self.lw_attn_tokens / self.total_tokens if self.total_tokens > 0 else 0
        mlp_sparsity = self.skipped_mlp_tokens / self.total_tokens if self.total_tokens > 0 else 0
        mlp_lw_sparsity = self.lw_mlp_tokens / self.total_tokens if self.total_tokens > 0 else 0
        return attn_sparsity, mlp_sparsity, attn_lw_sparsity, mlp_lw_sparsity
    
    def return_cos_mlp(self):
        return self.cos_mlp_ori, self.cos_mlp_lw, self.cos_attn_ori, self.cos_attn_lw

    def reset_sparsity_counts(self):
        self.total_tokens = 0
        self.skipped_attn_tokens = 0
        self.skipped_mlp_tokens = 0
        self.lw_attn_tokens = 0
        self.lw_mlp_tokens = 0
        self.regu_loss = 0



class router_attn_mlp_gemma (nn.Module):
    def __init__(self, block, hidden_size, args):
        super().__init__()
        self.router_attention = TokenRouter(hidden_size)
        self.router_mlp = TokenRouter(hidden_size)
        self.block = block
        self.training_step = 0
        self.args= args

        # initialize the total tokens and skipped tokens
        self.total_tokens = 0
        self.skipped_attn_tokens = 0
        self.skipped_mlp_tokens = 0

        # record the sparsity of the routers
        self.attn_router_zero_prob = 0.0  
        self.mlp_router_zero_prob = 0.0   

        # 初始化存储 token 路由信息的字典
        self.routing_matrix = {
            "attention": None,
            "mlp": None
        }

        # freeze the parameters of the block
        for param in self.block.parameters():
            param.requires_grad = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        # gemma特定代码
        if self.block.is_sliding and attention_mask is not None:  # efficient SDPA and no padding
            # Flash-attn is a 2D tensor
            if self.block.config._attn_implementation == "flash_attention_2":
                if past_key_value is not None:  # when decoding
                    attention_mask = attention_mask[:, -self.block.sliding_window :]
            else:
                min_dtype = torch.finfo(hidden_states.dtype).min
                sliding_window_mask = torch.tril(
                    torch.ones_like(attention_mask, dtype=torch.bool), diagonal=-self.block.sliding_window
                )
                attention_mask = torch.where(sliding_window_mask, min_dtype, attention_mask)
                if attention_mask.shape[-1] <= 1:  # when decoding
                    attention_mask = attention_mask[:, :, :, -self.block.sliding_window :]
        b, s, _ = hidden_states.shape

        self.total_tokens += b * s

        # check for NaN in the input tokens
        if torch.isnan(hidden_states).any():
            warnings.warn(
                "NaN detected in input tokens, this is not intended to happen, please check your model.")

        # 防止attention mask为None
        if attention_mask is None:
            attention_mask = torch.ones((b, s), device=hidden_states.device)

        # 训练过程中temperature逐渐降低
        if self.router_attention.training:
            if self.training_step <  self.args.gradient_accumulation_steps * self.args.max_steps_stage:
                self.training_step += 1
            temperature = self.args.initial_temperature - (self.args.initial_temperature - self.args.final_temperature) * ((self.training_step-1) // self.args.gradient_accumulation_steps )/ ( self.args.max_steps_stage)
        else:
            temperature = self.args.final_temperature

        # 计算gumbel softmax之前的权重
        weights = self.router_attention(hidden_states)

        # 计算gumbel softmax
        gumbel_weights = F.gumbel_softmax(weights, tau=temperature, hard=True, dim=-1)

        # gumbel weights的最后一个维度是长度为2的one-hot vectors，第一个代表是否执行，第二个代表是否跳过，我们取出第一个维度代表selected_mask
        selected_mask = gumbel_weights[:, :, 1]
        gumbel_weights_gate = gumbel_weights[:, :, 0]

        # 统计跳过 Attention 的次数
        self.skipped_attn_tokens += selected_mask.sum().item()
        # 记录router_attention的0类概率
        self.attn_router_zero_prob = gumbel_weights_gate.mean()

        
        # perform attention
        residual = hidden_states
        hidden_states = self.block.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.block.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = self.block.post_attention_layernorm(hidden_states)

         # 将attention的结果乘以gumbel weights
        hidden_states = hidden_states * gumbel_weights_gate.unsqueeze(-1) + residual
        
        # 计算mlp gumbel softmax之前的权重
        weights_mlp = self.router_mlp(residual)

        # 计算mlp的gumbel softmax
        gumbel_weights_mlp = F.gumbel_softmax(weights_mlp, tau=temperature, hard=True, dim=-1)

        # 计算gate
        selected_mask_mlp = gumbel_weights_mlp[:, :, 1]
        gumbel_weights_gate_mlp = gumbel_weights_mlp[:, :, 0]

        # 记录router_mlp的0类概率
        self.mlp_router_zero_prob = gumbel_weights_gate_mlp.mean()
        # 统计跳过 MLP 的次数
        self.skipped_mlp_tokens += selected_mask_mlp.sum().item()

        # Fully Connected
        residual = hidden_states
        hidden_states = self.block.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.block.mlp(hidden_states)
        hidden_states = self.block.post_feedforward_layernorm(hidden_states)

        # # 将mlp的结果乘以gumbel weights
        hidden_states = hidden_states * gumbel_weights_gate_mlp.unsqueeze(-1) + residual
        # hidden_states = hidden_states  + residual

        # 记录最新的路由信息
        self.routing_matrix["attention"] = selected_mask.to(torch.float32).detach().cpu().numpy()
        self.routing_matrix["mlp"] = selected_mask_mlp.to(torch.float32).detach().cpu().numpy()
        
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
    
    def compute_sparsity(self):
        attn_sparsity = self.skipped_attn_tokens / self.total_tokens if self.total_tokens > 0 else 0
        mlp_sparsity = self.skipped_mlp_tokens / self.total_tokens if self.total_tokens > 0 else 0
        return attn_sparsity, mlp_sparsity

    def reset_sparsity_counts(self):
        self.total_tokens = 0
        self.skipped_attn_tokens = 0
        self.skipped_mlp_tokens = 0




def apply_router_attn_mlp(model: PreTrainedModel, args) -> PreTrainedModel:
    hidden_size = model.config.hidden_size
    new_layers = nn.ModuleList()
    idx = 0
    if model.__class__.__name__ == "LlamaForCausalLM":
        for i, layer in enumerate(model.model.layers):
            new_layer = router_attn_mlp_llama(layer, hidden_size, args)
            new_layers.append(new_layer)
            # if idx > 1 and idx < 30: # 部分替换 前面3层和后面2层不替换    
            #     new_layer = router_attn_mlp_llama(layer, hidden_size, args)
            #     new_layers.append(new_layer)
            # else: new_layers.append(layer)
            # idx +=1
    
    # elif model.__class__.__name__ == "Gemma2ForCausalLM":
    #     for i, layer in enumerate(model.model.layers):
    #         new_layer = router_attn_mlp_gemma(layer, hidden_size, args)
    #         new_layers.append(new_layer)

    model.model.layers = new_layers    
    class_name = model.__class__.__name__

    # Insert MoD before the For
    if 'For' in class_name:
        parts = class_name.split('For', 1)
        modified_class_name = parts[0] + 'MoDFor' + parts[1]
    else:
        modified_class_name = 'MoD' + class_name  # If it doesn't find any i prepends MoD

    model.__class__.__name__ = modified_class_name

    return model
