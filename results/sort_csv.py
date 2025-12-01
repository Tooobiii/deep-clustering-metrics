import pandas as pd

df = pd.read_csv("results.csv")

df_sorted = df.sort_values(by=["model", "dataset", "size", "architecture"])

df_sorted.to_csv("results_sorted.csv", index=False)