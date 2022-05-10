#!/bin/sh

#SBATCH --job-name=rs1
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

# Display GPU status
nvidia-smi

# Pick base model to then finetune
basemodel="xlmt_dynabench2021_english_20k"

for dataset in basile2019_spanish fortuna2019_portuguese ousidhoum2019_french ousidhoum2019_arabic sanguinetti2020_italian; do
    for split in 10_rs1 20_rs1 50_rs1 100_rs1 200_rs1 500_rs1 1000_rs1 2000_rs1; do
        python finetune_and_test.py \
            --model_name_or_path $DATA/low-resource-hate/english-base-models/${basemodel} \
            --train_file $DATA/low-resource-hate/0_data/main/1_clean/${dataset}/train/train_${split}.csv \
            --validation_file $DATA/low-resource-hate/0_data/main/1_clean/${dataset}/test_2500.csv \
            --dataset_cache_dir $DATA/low-resource-hate/z_cache/datasets \
            --do_train \
            --per_device_train_batch_size 16 \
            --num_train_epochs 3 \
            --max_seq_length 128 \
            --save_strategy "no" \
            --do_eval \
            --evaluation_strategy "epoch" \
            --output_dir $DATA/low-resource-hate/finetuned-models/${basemodel}_${dataset}_${split} \
            --overwrite_output_dir
    done
done