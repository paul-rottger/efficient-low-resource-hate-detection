import pandas as pd

df = pd.read_csv("./0_data/main/1_clean/basile2019_spanish/test2500.csv")

print(df.text[1])