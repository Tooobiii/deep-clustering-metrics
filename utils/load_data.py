import numpy as np

from clustpy.data.preprocessing import z_normalization

from config import DATASETS


def load_normalized_data(dataset, subset_amount=None):
    load_data = DATASETS[dataset]
    data, labels = load_data(return_X_y=True)

    # Subsetting for reuters
    if dataset == "reuters" and subset_amount is not None:
        subset_indices = np.load(f"subset_indices/{subset_amount}_reuters_subset_indices.npy")

        data = data[subset_indices]
        labels = labels[subset_indices]

    data = z_normalization(data)
    n_clusters = len(np.unique(labels))

    return data, labels, n_clusters