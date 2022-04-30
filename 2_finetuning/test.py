#!/usr/bin/env python

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

df = pd.read_csv("../0_data/main/1_clean/dynabench2021_english/train/train_20_rs1.csv")

tokenizer = AutoTokenizer.from_pretrained("HannahRoseKirk/Hatemoji")
model = AutoModelForSequenceClassification.from_pretrained("HannahRoseKirk/Hatemoji")

inputs = tokenizer(list(df.text), return_tensors="pt", padding=True)

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_ids = [l.argmax().item() for l in logits]

for i in zip(df.text, predicted_class_ids):
    print(i[0])
    print(i[1])
    print()