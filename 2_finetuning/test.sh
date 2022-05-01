#!/bin/sh

basemodel="xlmt_dynabench2021_english_20k"

for dataset in basile2019_spanish fortuna2019_portuguese ousidhoum2019_french ousidhoum2019_arabic sanguinetti2020_italian; do
    for split in 500_rs1 1000_rs1 2000_rs1; do 
        echo $DATA/low-resource-hate/0_data/main/1_clean/${dataset}/train/train_${split}.csv
        echo $DATA/low-resource-hate/finetuned-models/${basemodel}_${dataset}_${split}
    done
done