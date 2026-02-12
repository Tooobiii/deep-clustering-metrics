import pandas as pd
from pathlib import Path

from config import DATASET_POLICY

### CONFIG
ft_type = "cvi"  # or ft

INPUT_CSV = f"results_prepped.csv"
OUTPUT_DIR = Path(f"{ft_type}_results")

TARGET = "UCA"
META_COLS = ["dataset", "size", "model", "architecture", "n_clusters"]
EXTERNAL_COLS = ["UCA", "NMI", "ARI"]

OUTPUT_DIR.mkdir(exist_ok=True)


def compute_correlations(df, target, cols):
    df_sub = df[[target] + cols]

    # Spearman
    corr_s = df_sub.corr(method="spearman")
    rho = corr_s[target].drop(target)

    # Kendall
    corr_k = df_sub.corr(method="kendall")
    tau = corr_k[target].drop(target)

    # Valid runs
    n_valid = {
        col: df_sub[[target, col]].dropna().shape[0]
        for col in cols
    }

    result = pd.DataFrame({
        "metric": rho.index,
        "spearman_rho": rho.values,
        "kendall_tau": [tau[m] for m in rho.index],
        "n_valid": [n_valid[m] for m in rho.index],
    })

    # Sort by Spearman
    result = result.sort_values("spearman_rho", ascending=False)

    return result

### LOAD DATA
df = pd.read_csv(INPUT_CSV)

cols = [
    c for c in df.columns
    if c not in META_COLS + EXTERNAL_COLS
]

### GLOBAL CORRELATIONS
global_results = compute_correlations(df, TARGET, cols)
global_results.to_csv(OUTPUT_DIR / f"correlation_{ft_type}_global_aggregated.csv", index=False)

### DATASET CORRELATIONS
for dataset, df_ds in df.groupby("dataset"):
    if df_ds[TARGET].notna().sum() < 3:
        continue  # too few points to correlate

    res = compute_correlations(df_ds, TARGET, cols)
    res.to_csv(OUTPUT_DIR / f"correlation_{ft_type}_data_{dataset}_aggregated.csv", index=False)

### MODEL CORRELATIONS
for model, df_m in df.groupby("model"):
    if df_m[TARGET].notna().sum() < 3:
        continue

    res = compute_correlations(df_m, TARGET, cols)
    res.to_csv(OUTPUT_DIR / f"correlation_{ft_type}_model_{model}_aggregated.csv", index=False)

### ARCHITECTURE CORRELATIONS
for arch, df_a in df.groupby("architecture"):
    if df_a[TARGET].notna().sum() < 3:
        continue

    res = compute_correlations(df_a, TARGET, cols)
    res.to_csv(OUTPUT_DIR / f"correlation_{ft_type}_arch_{arch}_aggregated.csv", index=False)

### GROUPED ANALYSIS
def add_dataset_groups(df, policy):
    for key in ["dim_group", "size_group", "k_group", "data_type"]:
        df[key] = df["dataset"].map(
            lambda d: policy.get(d, {}).get(key)
        )
    return df

def run_grouped_analysis(df, group_col, prefix):
    for group, df_g in df.groupby(group_col):
        if df_g[TARGET].notna().sum() < 3:
            continue

        res = compute_correlations(df_g, TARGET, cols)
        res.to_csv(
            OUTPUT_DIR / f"correlation_{ft_type}_{prefix}_{group}_aggregated.csv",
            index=False
        )

df = add_dataset_groups(df, DATASET_POLICY)

### DIMENSIONALITY CORRELATIONS
run_grouped_analysis(df, "dim_group", "dim")

### DATASET SIZE CORRELATIONS
run_grouped_analysis(df, "size_group", "size")

### NUMBER OF CLUSTERS CORRELATIONS
run_grouped_analysis(df, "k_group", "k")

### DATA TYPE CORRELATIONS
run_grouped_analysis(df, "data_type", "type")

df = add_dataset_groups(df, DATASET_POLICY)