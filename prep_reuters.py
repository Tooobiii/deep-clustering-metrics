import argparse
import numpy as np
from clustpy.data import load_reuters
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser(description="Prepare random Reuters subset and save indices.")
    parser.add_argument("--subset_size", type=int, required=True,
                        help="Number of documents in the subset.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--stratify", action="store_true",
                        help="Whether to stratify the sample according to labels.")
    parser.add_argument("--output", type=str, default="reuters_subset_indices.npy",
                        help="File to save the indices.")
    args = parser.parse_args()

    # Laden des Datensatzes
    data, labels = load_reuters(return_X_y=True)

    # Stratify-Option
    stratify_labels = labels if args.stratify else None

    subset_idx, _ = train_test_split(
        np.arange(len(labels)),
        train_size=args.subset_size,
        random_state=args.seed,
        stratify=stratify_labels
    )

    np.save(f"{args.subset_size}_{args.output}", subset_idx)
    print(f"Saved {len(subset_idx)} indices to {args.output}")

if __name__ == "__main__":
    main()