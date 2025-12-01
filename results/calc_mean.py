import pandas as pd

df = pd.read_csv("results_sorted.csv")

group_cols = ["dataset", "size", "model", "architecture"]

exclude = group_cols + ["seed", "n_clusters", "embedding_size", "clustering_epochs", "pretrain_epochs"]
exclude += [c for c in df.columns if c.endswith(".sd")]

numeric_cols = df.select_dtypes(include="number").columns
metric_cols = [c for c in numeric_cols if c not in exclude]

df_mean = df.groupby(group_cols)[metric_cols].mean().reset_index()

df_std = df.groupby(group_cols)[metric_cols].std().reset_index()

new_cols = group_cols.copy()
for col in metric_cols:
    new_cols.append(col)
    new_cols.append(col + ".sd")
    df_mean[col + ".sd"] = df_std[col]

df_mean = df_mean[new_cols]

df_mean.to_csv("results_mean.csv", index=False)