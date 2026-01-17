from clustpy.data import (load_iris, load_mnist, load_fmnist, load_reuters, load_newsgroups, load_cifar10,
                          load_cifar100, load_coil20, load_stl10, load_usps, load_pendigits)
from clustpy.deep import DEC, DCN, IDEC, DKM

DATASETS = {
    "iris": load_iris, # just for testing
    "mnist": load_mnist, # image
    "fmnist": load_fmnist, # image
    "cifar10": load_cifar10,  # image
    "cifar100": load_cifar100, # image
    "coil20": load_coil20, # image
    "stl10": load_stl10, # image
    "usps": load_usps, # image
    "reuters": load_reuters, # text
    "newsgroups": load_newsgroups, # text
    "pendigits": load_pendigits # other
}

TEXT_DATASETS = {"reuters", "newsgroups"}

# Model args
MODELS = {
    "DEC": DEC, # just for testing
    "IDEC": IDEC,
    "DCN": DCN,
    "DKM": DKM,
}

def dataset_policy(dataset, n):
    is_text = dataset in TEXT_DATASETS

    compute_distance = n < 20000
    compute_complexity = compute_distance
    compute_dbcv = (not is_text) and (n < 20000)

    return {
        "distance": compute_distance,
        "complexity": compute_complexity,
        "dbcv": compute_dbcv,
    }