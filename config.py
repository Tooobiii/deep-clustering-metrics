from clustpy.data import (load_iris, load_mnist, load_fmnist, load_reuters, load_newsgroups, load_cifar10,
                          load_cifar100, load_coil20, load_stl10, load_usps)
from clustpy.deep import DEC, DCN, IDEC, DKM

DATASETS = {
    "iris": load_iris, # just for testing
    "mnist": load_mnist,
    "fmnist": load_fmnist,
    "cifar10": load_cifar10,
    "cifar100": load_cifar100,
    "coil20": load_coil20,
    "stl10": load_stl10,
    "usps": load_usps,
    "reuters": load_reuters,
    "newsgroups": load_newsgroups,
}

# Model args
MODELS = {
    "DEC": DEC, # just for testing
    "IDEC": IDEC,
    "DCN": DCN,
    "DKM": DKM,
}