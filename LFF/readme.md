# Bring Future Vision: Dynamic Computation Allocation Guided by Lightweight Feature Forecaster

> This repository is built based on the SkipGPT code developed by EIT-NLP, with the original code repository available at: https://github.com/EIT-NLP/SkipGPT.

## Training

* Prepare the local path to the Llama-3.1-8B model and the RedPajama dataset, then specify the corresponding parameters in main.py (or the Shell script).
* Run script\lw_train.sh for LFF Training, this process takes less than 5 minutes on a single A6000 GPU.
* Run script\router_train.sh for router Training, around 7 hours.
* Run script\router_train.sh for Lora Finetuning, around 7 hours.

## Evaluation

* Run script\eval.sh