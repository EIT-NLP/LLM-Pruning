# lm_eval --model hf \
#     --model_args pretrained=/data/chao_han/model/Meta-Llama-3_1-8B,trust_remote_code=true \
#     --tasks arc_challenge \
#     --num_fewshot 25 \
#     --device cuda:0 \
#     --batch_size auto:1 \
CUDA_VISIBLE_DEVICES=0 nohup python eval.py > eval_result.txt 2>&1 &
