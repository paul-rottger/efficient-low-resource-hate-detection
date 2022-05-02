#!/bin/sh

python finetune_and_test.py \
    --model_name_or_path "cardiffnlp/twitter-xlm-roberta-base" \
    --test_file ../0_data/main/1_clean/dynabench2021_english/train/train_20_rs1.csv \
    --dataset_cache_dir ../z_cache \
    --do_predict \
    --test_results_dir ../z_results \
    --test_results_name firsttest.csv \
    --output_dir . \
    --max_seq_length 128 \
