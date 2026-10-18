import os

import numpy as np
import pandas as pd

def save_results(row, embedding, labels, output):
    if not os.path.exists(output):
        csv = pd.DataFrame(columns=row.keys())
        csv.to_csv(output, index=False)

    csv = pd.read_csv(output)

    # Create new columns
    for col in row.keys():
        if col not in csv.columns:
            csv[col] = None

    # Skip unused columns
    for col in csv.columns:
        if col not in row:
            row[col] = None

    csv = pd.concat([csv, pd.DataFrame([row])], ignore_index=True)
    csv.to_csv(output, index=False)

    base_dir = os.path.dirname(output)
    emb_dir = os.path.join(base_dir, "embeddings")
    os.makedirs(emb_dir, exist_ok=True)

    run_name = row["run_name"]
    emb_path = os.path.join(emb_dir, f"{run_name}.npz")
    np.savez_compressed(
        emb_path,
        embedding=np.ascontiguousarray(embedding),
        pseudo_labels=np.asarray(labels),
        dataset = row["dataset"],
        seed = row["seed"],
        model=row["model"],
        architecture = row["architecture"],
    )