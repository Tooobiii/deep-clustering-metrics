import numpy as np
import pandas as pd

INFILE = "results.csv"
OUTFILE = "results_prepped.csv"

CALC_SD = False

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
    invert_metrics = [
        "Davies-Bouldin",
        "Davies-Bouldin starred",
        "Ray-Turi",
        "Kwon",
        "Tang",
        "SD1",
        "SD2",
        "S_Dbw",
        "Vsv1",
        "Vsv2",
        "C Index",
        "McClain-Rao",
        "Cluster entropy",
        "Sym-Davies-Bouldin",
        "Negentropy Increment",
    ]

    existing = [c for c in invert_metrics if c in df.columns]

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


def main():
    df = pd.read_csv(INFILE)

    df = drop_cols(df)
    df = normalize_datasets(df)
    df = replace_architecture_labels(df)
    df = invert_metrics(df)
    df = sort_results(df)

    df = aggregate_results(df, CALC_SD)
    df.to_csv(OUTFILE, index=False)

if __name__ == "__main__":
    main()