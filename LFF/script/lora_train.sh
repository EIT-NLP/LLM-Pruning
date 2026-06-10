
CUDA_VISIBLE_DEVICES=1 nohup python main.py \
    --lora_hc \
    --regu_weight 0 \
    --sparsity 0.25 \
    --evaluation_strategy "steps" \
    --eval_steps 100 \
    --max_steps_stage 2000 \
    --learning_rate 2e-4 \
    --lr_scheduler_type 'cosine' \
    --max_length 2048 \
    --warmup_ratio 0.1 \
    --post_norm_router \
    --gradient_accumulation_steps 16 \
    --skipgpt > "hc_log/regularized_lora.txt" 2>&1 &
    

