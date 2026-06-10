CUDA_VISIBLE_DEVICES=1 python main.py \
    --retrain_lw \
    --lw_net_rank 0.65 \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine