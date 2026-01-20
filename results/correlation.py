import pandas as pd
from pathlib import Path

### CONFIG
ft_type = "cvi"  # or ft

INPUT_CSV = f"results_prepped.csv"
OUTPUT_DIR = Path(f"{ft_type}_results")

TARGET = "UCA"
META_COLS = ["dataset", "size", "model", "architecture"]
EXTERNAL_COLS = ["UCA", "NMI", "ARI"]

OUTPUT_DIR.mkdir(exist_ok=True)


def compute_correlations(df, target, cols):
    df_sub = df[[target] + cols]

    corr = df_sub.corr(method="spearman")
    rho = corr[target].drop(target)

    n_valid = {
        col: df_sub[[target, col]].dropna().shape[0]
        for col in cols
    }

    result = pd.DataFrame({
        "metric": rho.index,
        "spearman_rho": rho.values,
        "n_valid": [n_valid[m] for m in rho.index],
    })

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
global_results.to_csv(OUTPUT_DIR / f"correlation_{ft_type}_global.csv", index=False)

### DATASET CORRELATIONS
for dataset, df_ds in df.groupby("dataset"):
    if df_ds[TARGET].notna().sum() < 3:
        continue  # too few points to correlate

    res = compute_correlations(df_ds, TARGET, cols)
    res.to_csv(OUTPUT_DIR / f"correlation_{ft_type}_data_{dataset}.csv", index=False)

### MODEL CORRELATIONS
for model, df_m in df.groupby("model"):
    if df_m[TARGET].notna().sum() < 3:
        continue

    res = compute_correlations(df_m, TARGET, cols)
    res.to_csv(OUTPUT_DIR / f"correlation_{ft_type}_model_{model}.csv", index=False)

### ARCHITECTURE CORRELATIONS
for arch, df_a in df.groupby("architecture"):
    if df_a[TARGET].notna().sum() < 3:
        continue

    res = compute_correlations(df_a, TARGET, cols)
    res.to_csv(OUTPUT_DIR / f"correlation_{ft_type}_arch_{arch}.csv", index=False)