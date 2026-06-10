CUDA_VISIBLE_DEVICES=1 python main.py \
    --train_lw_only \
    --init_router \
    --lw_net_rank 0.65 \
    --learning_rate 1e-3 \
    --lr_scheduler_type cosine