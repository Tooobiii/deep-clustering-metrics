import argparse

import pandas as pd
from pathlib import Path

from config_datasets import DATASET_POLICY

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", type=str, default=f"results_prepped.csv")
    parser.add_argument("-ft", "--type", type=str, default="cvi", choices=["cvi", "meta"])
    parser.add_argument("-t", "--target", type=str, required=True, choices=["UCA", "NMI", "ARI"])
    parser.add_argument("-c", "--calc-sd", type=bool, default=False)

    return parser.parse_args()


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

### PREPARATIONS
META_COLS = ["dataset", "size", "model", "architecture", "n_clusters"]
EXTERNAL_COLS = ["UCA", "NMI", "ARI"]

args = parse_arguments()

input_csv = args.input
output_dir = Path(f"{args.type}_results")

output_dir.mkdir(exist_ok=True)

### LOAD DATA
df = pd.read_csv(input_csv)

cols = [
    c for c in df.columns
    if c not in META_COLS + EXTERNAL_COLS
]

### GLOBAL CORRELATIONS
global_results = compute_correlations(df, args.target, cols)
global_results.to_csv(output_dir / f"correlation_{args.type}_global_aggregated.csv", index=False)

### DATASET CORRELATIONS
for dataset, df_ds in df.groupby("dataset"):
    if df_ds[args.target].notna().sum() < 3:
        continue  # too few points to correlate

    res = compute_correlations(df_ds, args.target, cols)
    res.to_csv(output_dir / f"correlation_{args.type}_data_{dataset}_aggregated.csv", index=False)

### MODEL CORRELATIONS
for model, df_m in df.groupby("model"):
    if df_m[args.target].notna().sum() < 3:
        continue

    res = compute_correlations(df_m, args.target, cols)
    res.to_csv(output_dir / f"correlation_{args.type}_model_{model}_aggregated.csv", index=False)

### ARCHITECTURE CORRELATIONS
for arch, df_a in df.groupby("architecture"):
    if df_a[args.target].notna().sum() < 3:
        continue

    res = compute_correlations(df_a, args.target, cols)
    res.to_csv(output_dir / f"correlation_{args.type}_arch_{arch}_aggregated.csv", index=False)

### GROUPED ANALYSIS
def add_dataset_groups(df, policy):
    for key in ["dim_group", "size_group", "k_group", "data_type"]:
        df[key] = df["dataset"].map(
            lambda d: policy.get(d, {}).get(key)
        )
    return df

def run_grouped_analysis(df, group_col, prefix):
    for group, df_g in df.groupby(group_col):
        if df_g[args.target].notna().sum() < 3:
            continue

        res = compute_correlations(df_g, args.target, cols)
        res.to_csv(
            output_dir / f"correlation_{args.type}_{prefix}_{group}_aggregated.csv",
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