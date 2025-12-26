import argparse

import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset", type=str, required=True, choices=["iris", "mnist", "fmnist",
                        "cifar10", "cifar100", "coil20", "stl10", "usps", "reuters", "newsgroups", "pendigits"])
    parser.add_argument("-n", "--subset_amount", type=int, choices=[10000, 20000, 50000], default=None)
    parser.add_argument("-s", "--seed", type=int, default=np.random.randint(1, 999999))

    parser.add_argument("-m", "--model", type=str, required=True, choices=["DEC", "IDEC", "DCN", "DKM"])
    parser.add_argument("-hl", "--hidden_layers", type=int, nargs="+", default=[500, 500, 2000])
    parser.add_argument("-e", "--embedding_size", type=int, default=10)
    parser.add_argument("-p", "--pretrain_epochs", type=int, default=None)
    parser.add_argument("-c", "--clustering_epochs", type=int, default=None)

    parser.add_argument("-l", "--clustering_loss", type=float, default=None)
    parser.add_argument("-pl", "--pretrain_learning_rate", type=float, default=None)
    parser.add_argument("-cl", "--clustering_learning_rate", type=float, default=None)

    parser.add_argument("-o", "--output", type=str, default=f"results/results.csv")

    return parser.parse_args()