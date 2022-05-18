#!/bin/sh

#SBATCH --job-name=ft2-max
#SBATCH --clusters=arc
#SBATCH --ntasks-per-node=16
#SBATCH --time=11:59:00
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

# Pick base model to then continue finetuning (-->FT2)
basemodel="xlmt_dyn21_en_20000_rs1"

for dataset in bas19_es for19_pt ous19_ar san20_it; do
    python finetune_and_test.py \
        --model_name_or_path $DATA/low-resource-hate/english-base-models/${basemodel}/ \
        --train_file $DATA/low-resource-hate/0_data/main/1_clean/${dataset}/train_*.csv \
        --validation_file $DATA/low-resource-hate/0_data/main/1_clean/${dataset}/dev_*.csv \
        --dataset_cache_dir $DATA/low-resource-hate/z_cache/datasets \
        --do_train \
        --per_device_train_batch_size 16 \
        --num_train_epochs 5 \
        --max_seq_length 128 \
        --do_eval \
        --per_device_eval_batch_size 32 \
        --evaluation_strategy "epoch" \
        --save_strategy "epoch" \
        --save_total_limit 1 \
        --load_best_model_at_end \
        --metric_for_best_model "macro_f1" \
        --output_dir $DATA/low-resource-hate/finetuned-models/full-sample/multilingual-models/${basemodel}_${dataset}_X_full \
        --overwrite_output_dir
done