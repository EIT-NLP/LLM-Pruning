#!/bin/bash

base_path=...
num_gpus=4
per_device_train_batch_size=1
per_device_eval_batch_size=2
gradient_accumulation_steps=4
train_steps=10000
eval_steps=20000
model_name=Llama-3.1-8B
sparse_target=all
route_rank_size=32

: "${SPARSITY:=0.0}"
: "${ATTN_G:=1}"
: "${FFN_G:=1}"
sparsity=$SPARSITY
attn_group_size=$ATTN_G
ffn_group_size=$FFN_G

echo "sparsity=$sparsity"
echo "attn_group_size=$attn_group_size"
echo "ffn_group_size=$ffn_group_size"

export SWANLAB_SAVE_DIR=${base_path}/swanlab_cache
export SWANLAB_API_KEY=...

export HF_HOME=...
export HF_TOKEN=...

export MODELSCOPE_CACHE=...

mkdir -p "${base_path}/WIDE/${model_name}"

for wrapper_type in attn_group_ffn_block
do
    save_dir=${base_path}/WIDE/${model_name}/${model_name}_${wrapper_type}_${sparse_target}_${sparsity}_A${attn_group_size}_F${ffn_group_size}
    router_dir=${base_path}/WIDE/${model_name}/${model_name}_${wrapper_type}_${sparse_target}_${sparsity}_A${attn_group_size}_F${ffn_group_size}
    tag="None"

    torchrun --nproc_per_node=${num_gpus} ${base_path}/WIDE/main.py \
        --model_name ${model_name} \
        --model_path ${base_path}/modelscope_cache/models/LLM-Research/${model_name} \
        --project_name WIDE-Train \
        --save_dir ${save_dir} \
        --wrapper_type ${wrapper_type} \
        --train_dataset ${base_path}/huggingface_cache/RedPajama-Data-1T-Sample-subset850000 \
        --eval_dataset ${base_path}/huggingface_cache/wikitext-2-raw-v1 \
        --max_length 4096 \
        --per_device_train_batch_size ${per_device_train_batch_size} \
        --per_device_eval_batch_size ${per_device_eval_batch_size} \
        --gradient_accumulation_steps ${gradient_accumulation_steps} \
        --eval_steps ${eval_steps} \
        --initial_temperature 5 \
        --final_temperature 0.5 \
        --sparsity ${sparsity} \
        --warmup_ratio 0.1 \
        --max_steps ${train_steps} \
        --max_lr 2e-3 \
        --sparse_target ${sparse_target} \
        --attn_group_size ${attn_group_size} \
        --ffn_group_size ${ffn_group_size} \
        --route_rank_size ${route_rank_size} \
        --tag ${tag}

    python ${base_path}/WIDE/main.py \
        --benchmark \
        --benchmark_tasks wikitext arc_easy arc_challenge boolq winogrande piqa openbookqa hellaswag \
        --benchmark_batch_size auto \
        --benchmark_max_batch_size 16 \
        --benchmark_max_length 4096 \
        --router_ckpt_path ${save_dir}/router_stage1/router.pth \
        --model_name ${model_name} \
        --model_path ${base_path}/modelscope_cache/models/LLM-Research/${model_name} \
        --project_name WIDE-Bench \
        --save_dir ${save_dir} \
        --wrapper_type ${wrapper_type} \
        --sparsity ${sparsity} \
        --sparse_target ${sparse_target} \
        --attn_group_size ${attn_group_size} \
        --ffn_group_size ${ffn_group_size} \
        --route_rank_size ${route_rank_size} \
        --tag ${tag}
    
    torchrun --nproc_per_node=${num_gpus} ${base_path}/WIDE/main.py \
        --lora \
        --model_name ${model_name} \
        --model_path ${base_path}/modelscope_cache/models/LLM-Research/${model_name} \
        --project_name WIDE-Train \
        --save_dir ${save_dir} \
        --wrapper_type ${wrapper_type} \
        --train_dataset ${base_path}/huggingface_cache/RedPajama-Data-1T-Sample-subset850000 \
        --eval_dataset ${base_path}/huggingface_cache/wikitext-2-raw-v1 \
        --max_length 4096 \
        --router_ckpt_path ${save_dir}/router_stage1/router.pth \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size ${per_device_eval_batch_size} \
        --gradient_accumulation_steps 4 \
        --eval_steps ${eval_steps} \
        --initial_temperature 5 \
        --final_temperature 0.5 \
        --sparsity ${sparsity} \
        --warmup_ratio 0.1 \
        --max_steps ${train_steps} \
        --max_lr 2e-4 \
        --sparse_target ${sparse_target} \
        --attn_group_size ${attn_group_size} \
        --ffn_group_size ${ffn_group_size} \
        --route_rank_size ${route_rank_size} \
        --cpu_offload \
        --tag ${tag}_lora

    python ${base_path}/WIDE/main.py \
        --benchmark \
        --benchmark_tasks wikitext arc_easy arc_challenge boolq winogrande piqa openbookqa hellaswag \
        --benchmark_batch_size auto \
        --benchmark_max_batch_size 16 \
        --benchmark_max_length 4096 \
        --router_ckpt_path ${save_dir}/router_stage2/router.pth \
        --lora_ckpt_path ${save_dir}/router_stage2/lora.pth \
        --model_name ${model_name} \
        --model_path ${base_path}/modelscope_cache/models/LLM-Research/${model_name} \
        --project_name WIDE-Bench \
        --save_dir ${save_dir} \
        --wrapper_type ${wrapper_type} \
        --sparsity ${sparsity} \
        --sparse_target ${sparse_target} \
        --attn_group_size ${attn_group_size} \
        --ffn_group_size ${ffn_group_size} \
        --route_rank_size ${route_rank_size} \
        --tag ${tag}_lora
done