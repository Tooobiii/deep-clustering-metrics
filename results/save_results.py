import os

import pandas as pd

def save_results(row, output):
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