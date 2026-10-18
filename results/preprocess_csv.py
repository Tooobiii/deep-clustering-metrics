import argparse

import numpy as np
import pandas as pd
from pathlib import Path

from config_results import CVI, INVERT_CVIS, META_COLS, EXT_COLS

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", type=str, default=f"results/results.csv")
    parser.add_argument("-o", "--output", type=str, default=f"results/")
    parser.add_argument("-a", "--aggregate", action="store_false")
    parser.add_argument("-sd", "--calc-sd", action="store_true")

    return parser.parse_args()

def drop_cols(df):
    drop_cols = ["seed", "embedding_size", "clustering_epochs", "pretrain_epochs",
                 "g_mean.mean", "h_mean.mean", "sd_ratio"]

    drop_cols += [c for c in df.columns if c.endswith(".sd")]
    df = df.drop(columns=drop_cols, errors="ignore")
    return df

def normalize_datasets(df):
    df.loc[(df["dataset"] == "reuters") & (df["size"] <= 10000), "dataset"] = "reuters10k"
    df.loc[(df["dataset"] == "reuters") & (df["size"] > 10000), "dataset"] = "reuters50k"
    return df

def replace_architecture_labels(df):
    arch = df["architecture"].astype("string")

    df["architecture"] = np.select(
        [
            arch.str.contains("1000->1000", regex=False, na=False),
            arch.str.contains("500->500", regex=False, na=False),
            arch.str.contains("250->250", regex=False, na=False),
        ],
        ["broad", "normal", "slim"],
        default=arch
    )
    return df

def invert_metrics(df):
    existing = [c for c in INVERT_CVIS if c in df.columns]

    if existing:
        df[existing] = df[existing].astype(float) * -1

    return df

def sort_results(df):
    return df.sort_values(
        by=["dataset", "model", "architecture"]
    )

def aggregate_results(df, calc_sd):
    group_cols = ["dataset", "model", "architecture", "size", "n_clusters"]

    numeric_cols = df.select_dtypes(include="number").columns
    metric_cols = [c for c in numeric_cols if c not in group_cols]

    mean_df = df.groupby(group_cols)[metric_cols].mean().reset_index()

    if not calc_sd:
        return mean_df[group_cols + metric_cols]

    std_df = df.groupby(group_cols)[metric_cols].std().reset_index()
    sd_df = std_df[metric_cols].rename(columns={c: f"{c}.sd" for c in metric_cols})

    out = pd.concat([mean_df, sd_df], axis=1)

    ordered_cols = []
    for c in metric_cols:
        ordered_cols.extend([c, f"{c}.sd"])

    return out[group_cols + ordered_cols]

def split_results(df, output_dir):
    shared_cols = [c for c in (META_COLS + EXT_COLS) if c in df.columns]

    cvi_cols = [c for c in CVI if c in df.columns]
    cvi_df = df[shared_cols + cvi_cols]

    feature_cols = [c for c in df.columns if c not in shared_cols and c not in cvi_cols]
    feature_df = df[shared_cols + feature_cols]

    df.to_csv(output_dir / "results_combined.csv", index=False)
    cvi_df.to_csv(output_dir / "results_cvi.csv", index=False)
    feature_df.to_csv(output_dir / "results_feature.csv", index=False)

def main():
    args = parse_arguments()
    df = pd.read_csv(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    df = drop_cols(df)
    df = normalize_datasets(df)
    df = replace_architecture_labels(df)
    df = invert_metrics(df)
    df = sort_results(df)

    if args.aggregate:
        df = aggregate_results(df, args.calc_sd)
    split_results(df, output_dir)

if __name__ == "__main__":
    main()