import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("HannahRoseKirk/Hatemoji")

model = AutoModelForSequenceClassification.from_pretrained("HannahRoseKirk/Hatemoji")

print("Test classification")
print(model.predict("This is a test phrase"))