## Quick Start

### Installation
This project is implemented based torch, Huggingface, FlashAttention, DeepSpeed, and vLLM libraries. To obtain the dependencies, we provide the following three ways:

**1. Using pip**
```bash
# Make sure torch 2.1.2 and cuda 12.1 is installed
pip install -r requirements.txt
```
### Create Experiment Script

We first specify the configuration file for the experiment, and then, we explain how to run the training and evaluation using a configuration file.

**VinePPO Experiments**
- `configs/polIter_rho1bSft2_vineppo_MATH.jsonnet`
- `configs/polIter_rho1bSft2_vineppo_GSM8K.jsonnet`
- `configs/polIter_deepseekSft2_vineppo_MATH.jsonnet`
- `configs/polIter_deepseekSft2_vineppo_GSM8K.jsonnet`

**PPO Experiments**
- `configs/polIter_rho1bSft2_ppo_MATH.jsonnet`
- `configs/polIter_rho1bSft2_ppo_GSM8K.jsonnet`
- `configs/polIter_deepseekSft2_ppo_MATH.jsonnet`
- `configs/polIter_deepseekSft2_ppo_GSM8K.jsonnet`

**DPO Experiments**
- `configs/polIter_rho1bSft2_dpo_positive_MATH.jsonnet`
- `configs/polIter_rho1bSft2_dpo_positive_GSM8K.jsonnet`
- `configs/polIter_deepseekSft2_dpo_positive_MATH.jsonnet`
- `configs/polIter_deepseekSft2_dpo_positive_GSM8K.jsonnet`

**RestEM Experiments**
- `configs/polIter_rho1bSft2_restem_MATH.jsonnet`
- `configs/polIter_rho1bSft2_restem_GSM8K.jsonnet`
- `configs/polIter_deepseekSft2_restem_MATH.jsonnet`
- `configs/polIter_deepseekSft2_restem_GSM8K.jsonnet`

Once you have selected the configuration file, you can run the training and evaluation using the following script:
```bash

CONFIGSTR="configs/<config_file>.jsonnet"
APP_DIRECTORY="experiments/<path_to_output_dir>"

export APP_SEED="42"
export WANDB_RUN_ID="<unique_wandb_run_id>" # Optional

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# Run the training
deepspeed --no_local_rank --num_gpus=$NUM_GPUS  \
         src/treetune/main.py --configs "$CONFIGSTR" \
            run_iteration_loop

# Run the evaluation
deepspeed --no_local_rank --num_gpus=$NUM_GPUS   \
         src/treetune/main.py --configs "$CONFIGSTR" \
            run_evaluation

```
### Running the experiments
To run the experiments, you can use the following script:
1. Normal local run
```bash
chmod +x run.sh
./run.sh
```
