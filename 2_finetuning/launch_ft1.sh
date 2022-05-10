#!/bin/sh

#SBATCH --job-name=ft1
#SBATCH --clusters=arc
#SBATCH --ntasks-per-node=16
#SBATCH --time=10:00:00
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

for split in 10 20 30 40 50 100 200 300 400 500 1000 2000 3000 4000 5000 10000 20000; do
    python finetune_and_test.py \
        --model_name_or_path $DATA/low-resource-hate/default-models/twitter-xlm-roberta-base \
        --train_file $DATA/low-resource-hate/0_data/main/1_clean/dyn21_en/train/train_${split}_rs1.csv \
        --validation_file $DATA/low-resource-hate/0_data/main/1_clean/dyn21_en/dev_500.csv \
        --dataset_cache_dir $DATA/low-resource-hate/z_cache/datasets \
        --do_train \
        --per_device_train_batch_size 16 \
        --num_train_epochs 3 \
        --max_seq_length 128 \
        --save_strategy "no" \
        --do_eval \
        --evaluation_strategy "epoch" \
        --output_dir $DATA/low-resource-hate/english-base-models/xlmt_dyn21_en_${split}_rs1 \
        --overwrite_output_dir
done