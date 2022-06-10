#!/bin/sh

#SBATCH --job-name=xd0-hi-mhc
#SBATCH --clusters=arc
#SBATCH --ntasks-per-node=16
#SBATCH --time=11:59:00
#SBATCH --partition=short
#SBATCH --output=outputs.out
#SBATCH --error=errors.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=paul.rottger@oii.ox.ac.uk

# reset modules and load Python
module purge
module load Anaconda3/2020.11

# activate the right conda environment
source activate $DATA/conda-envs/lrh-env

for dataset in has21_hi; do
    for modelpath in $DATA/low-resource-hate/finetuned-models/random-sample/multilingual-models/xlmt_dyn21_en_0*${dataset}*/; do
        for testpath in $DATA/low-resource-hate/0_data/hatecheck/*_$(basename $dataset | cut -c7-9).csv; do
            python finetune_and_test.py \
                --model_name_or_path ${modelpath} \
                --dataset_cache_dir $DATA/low-resource-hate/z_cache/datasets \
                --do_predict \
                --test_file ${testpath} \
                --test_results_dir $DATA/low-resource-hate/results/hatecheck_$(basename $dataset | cut -c7-9) \
                --test_results_name $(basename $modelpath).csv \
                --per_device_eval_batch_size 64 \
                --max_seq_length 128 \
                --output_dir .
        done
    done
done