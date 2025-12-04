import argparse
import numpy as np
from clustpy.data import load_reuters
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset_size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--stratify", action="store_true")
    parser.add_argument("--output", type=str, default="../subset_incides/reuters_subset_indices.npy")
    args = parser.parse_args()

    # Load data
    data, labels = load_reuters(return_X_y=True)

    # Keep cluster ratio from original
    stratify_labels = labels if args.stratify else None

    subset_idx, _ = train_test_split(
        np.arange(len(labels)),
        train_size=args.subset_size,
        random_state=args.seed,
        stratify=stratify_labels
    )

    np.save(f"{args.subset_size}_{args.output}", subset_idx)

if __name__ == "__main__":
    main()