print_autotune=1
nsys_profile=0
model_name=llama3.1-8b
model_path=...

benchmark_metric=ttft
num_repeat=50

batch_size=(1 8 16 32)
seq_len=(1024)
sparsity=(0.5 0.5 0.5)

RUN_DENSE=1
RUN_STATIC_M=1
RUN_STATIC_K=1
RUN_STATIC_K_LOWRANK=1
RUN_STATIC_NK=1
RUN_STATIC_NK_CROSS=1
RUN_DYNAMIC_M=1
RUN_DYNAMIC_NK=1

if [[ $RUN_DENSE == 1 ]]; then
    dynamic=static
    style=dense
    config_name=dense

    if [[ $nsys_profile == 0 ]]; then
        python main.py \
            --model_name $model_name \
            --model_path $model_path \
            --dynamic $dynamic \
            --style $style \
            --config_name $config_name \
            --benchmark_metric $benchmark_metric \
            --num_warmup 1 \
            --num_repeat $num_repeat \
            --sparsity ${sparsity[@]} \
            --batch_size ${batch_size[@]} \
            --seq_len ${seq_len[@]} \
            --liger_kernel \
            --inplace_update_kvcache \
            --cuda_graph

    else
        nsys_prefix="${benchmark_metric}_${batch_size[0]}_${seq_len[0]}_${dynamic}_${style}_${config_name}_${sparsity[0]}"
        nsys profile --trace=cuda,nvtx -o ${nsys_prefix}.nsys-rep --force-overwrite true python main.py \
            --model_name $model_name \
            --model_path $model_path \
            --dynamic $dynamic \
            --style $style \
            --config_name $config_name \
            --benchmark_metric $benchmark_metric \
            --num_warmup 1 \
            --num_repeat $num_repeat \
            --sparsity $sparsity \
            --batch_size ${batch_size[@]} \
            --seq_len ${seq_len[@]} \
            --inplace_update_kvcache \
            --liger_kernel
    fi
fi

if [[ $RUN_STATIC_M == 1 ]]; then
    dynamic=static
    style=propagate
    config_name=static_m

    python main.py \
        --model_name $model_name \
        --model_path $model_path \
        --dynamic $dynamic \
        --style $style \
        --config_name $config_name \
        --benchmark_metric $benchmark_metric \
        --num_warmup 1 \
        --num_repeat $num_repeat \
        --sparsity ${sparsity[@]} \
        --batch_size ${batch_size[@]} \
        --seq_len ${seq_len[@]} \
        --liger_kernel \
        --inplace_update_kvcache \
        --cuda_graph
fi

if [[ $RUN_STATIC_K == 1 ]]; then
    dynamic=static
    style=unstructured
    config_name=static_k

    python main.py \
        --model_name $model_name \
        --model_path $model_path \
        --dynamic $dynamic \
        --style $style \
        --config_name $config_name \
        --benchmark_metric $benchmark_metric \
        --num_warmup 1 \
        --num_repeat $num_repeat \
        --sparsity ${sparsity[@]} \
        --batch_size ${batch_size[@]} \
        --seq_len ${seq_len[@]} \
        --liger_kernel \
        --inplace_update_kvcache \
        --cuda_graph
fi

if [[ $RUN_STATIC_K_LOWRANK == 1 ]]; then
    dynamic=static
    style=lowrank
    config_name=static_k_lowrank

    python main.py \
        --model_name $model_name \
        --model_path $model_path \
        --dynamic $dynamic \
        --style $style \
        --config_name $config_name \
        --benchmark_metric $benchmark_metric \
        --num_warmup 1 \
        --num_repeat $num_repeat \
        --sparsity ${sparsity[@]} \
        --batch_size ${batch_size[@]} \
        --seq_len ${seq_len[@]} \
        --liger_kernel \
        --inplace_update_kvcache \
        --cuda_graph
fi

if [[ $RUN_STATIC_NK == 1 ]]; then
    dynamic=static
    style=propagate
    config_name=static_nk

    python main.py \
        --model_name $model_name \
        --model_path $model_path \
        --dynamic $dynamic \
        --style $style \
        --config_name $config_name \
        --benchmark_metric $benchmark_metric \
        --num_warmup 1 \
        --num_repeat $num_repeat \
        --sparsity ${sparsity[@]} \
        --batch_size ${batch_size[@]} \
        --seq_len ${seq_len[@]} \
        --liger_kernel \
        --inplace_update_kvcache \
        --cuda_graph
fi

if [[ $RUN_STATIC_NK_CROSS == 1 ]]; then
    dynamic=static
    style=propagate
    config_name=static_nk_cross

    python main.py \
        --model_name $model_name \
        --model_path $model_path \
        --dynamic $dynamic \
        --style $style \
        --config_name $config_name \
        --benchmark_metric $benchmark_metric \
        --num_warmup 1 \
        --num_repeat $num_repeat \
        --sparsity ${sparsity[@]} \
        --batch_size ${batch_size[@]} \
        --seq_len ${seq_len[@]} \
        --liger_kernel \
        --inplace_update_kvcache \
        --cuda_graph
fi

if [[ $RUN_DYNAMIC_M == 1 ]]; then
    dynamic=dynamic
    style=propagate
    config_name=dynamic_m

    python main.py \
        --model_name $model_name \
        --model_path $model_path \
        --dynamic $dynamic \
        --style $style \
        --config_name $config_name \
        --benchmark_metric $benchmark_metric \
        --num_warmup 1 \
        --num_repeat $num_repeat \
        --sparsity ${sparsity[@]} \
        --batch_size ${batch_size[@]} \
        --seq_len ${seq_len[@]} \
        --liger_kernel \
        --inplace_update_kvcache \
        --cuda_graph
fi

if [[ $RUN_DYNAMIC_NK == 1 ]]; then
    dynamic=dynamic
    style=sparse_attention
    config_name=dynamic_nk

    python main.py \
        --model_name $model_name \
        --model_path $model_path \
        --dynamic $dynamic \
        --style $style \
        --config_name $config_name \
        --benchmark_metric $benchmark_metric \
        --num_warmup 1 \
        --num_repeat $num_repeat \
        --sparsity ${sparsity[@]} \
        --batch_size ${batch_size[@]} \
        --seq_len ${seq_len[@]} \
        --liger_kernel \
        --inplace_update_kvcache \
        --cuda_graph
fi