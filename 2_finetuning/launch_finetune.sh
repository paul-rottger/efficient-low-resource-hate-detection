#!/bin/sh

#SBATCH --job-name=test-finetune
#SBATCH --clusters=htc
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --partition=short 
#SBATCH --output=outputs.out
#SBATCH --error=errors.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=paul.rottger@oii.ox.ac.uk

# reset modules
module purge

# load python module
module load Anaconda3/2020.11

# activate the right conda environment
source activate $DATA/conda-envs/lrh-env

# Display GPU status
nvidia-smi

for trainpath in $DATA/low-resource-hate/0_data/main/1_clean/dynabench2021_english/train/train_20_rs1.csv; do
    python finetune.py \
        --model_name_or_path $DATA/low-resource-hate/default-models/twitter-xlm-roberta-base \
        --train_file $trainpath \
        --validation_file $trainpath \
        --test_file $trainpath \
        --dataset_cache_dir $DATA/low-resource-hate/z_cache/datasets \
        --do_train \
        --per_device_train_batch_size 4 \
        --num_train_epochs 3 \
        --max_seq_length 128 \
        --save_strategy "no" \
        --do_eval \
        --evaluation_strategy "epoch" \
        --output_dir $DATA/low-resource-hate/finetuned-models/xlmt_dynabench2021_english \
        --overwrite_output_dir
done