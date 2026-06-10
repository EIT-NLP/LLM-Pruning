python main.py \
    --train_lw_and_router \
    --evaluation_strategy "steps" \
    --eval_steps 100 \
    --max_steps_stage 1000 \
    --learning_rate 2e-3 \
    --lr_scheduler_type "cosine" \
    --max_length 1024 \
    --logging_steps 1 \
    --warmup_ratio 0.1 \
    # --gradient_accumulation_steps 8