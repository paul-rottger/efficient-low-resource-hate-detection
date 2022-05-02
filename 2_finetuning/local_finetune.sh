#!/bin/sh

python finetune_and_test.py \
    --model_name_or_path "cardiffnlp/twitter-xlm-roberta-base" \
    --train_file ../0_data/main/1_clean/dynabench2021_english/train/train_20_rs1.csv \
    --validation_file ../0_data/main/1_clean/dynabench2021_english/train/train_20_rs1.csv \
    --dataset_cache_dir ../z_cache \
    --do_train \
    --per_device_train_batch_size 4 \
    --num_train_epochs 3 \
    --max_seq_length 128 \
    --save_strategy "no" \
    --do_eval \
    --evaluation_strategy "epoch" \
    --output_dir ../xlmt_dynabench2021_english \
    --overwrite_output_dir