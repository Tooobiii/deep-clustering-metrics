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

# Model args
MODELS = {
    "DEC": DEC, # just for testing
    "IDEC": IDEC,
    "DCN": DCN,
    "DKM": DKM,
}