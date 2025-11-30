import pandas as pd

df = pd.read_excel("results.xlsx")

df_sorted = df.sort_values(by=["model", "dataset", "size", "architecture"])
#df_sorted = df.sort_values(by=["seed"])

df_sorted.to_excel("results_sorted.xlsx", index=False)