import argparse
import numpy as np
import torch
import pandas as pd
import inspect
import random
from clustpy.data import (load_iris, load_mnist, load_fmnist, load_reuters, load_newsgroups, load_cifar10,
                          load_cifar100, load_coil20, load_stl10, load_usps, load_pendigits)
from clustpy.deep import DEC, DCN, IDEC, DKM
from clustpy.data.preprocessing import z_normalization
from clustpy.deep.neural_networks import FeedforwardAutoencoder
from clustpy.metrics import unsupervised_clustering_accuracy
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from extract_features import Extractor

def build_autoencoder(input_dim, embedding_dim, hidden_layers, activations=torch.nn.LeakyReLU):
    ae_layers = [input_dim] + hidden_layers + [embedding_dim]
    return (FeedforwardAutoencoder, {
        "layers": ae_layers,
        "activation_fn": activations,
    })

EXCEL_PATH = "results/results.xlsx"

# General args
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


# Parse arguments
parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", type=str, required=True, choices=DATASETS.keys())
parser.add_argument("-n", "--subset_amount", type=int, choices=[10000,20000,50000], default=None)
parser.add_argument("-s", "--seed", type=int, default=np.random.randint(1,999999))

parser.add_argument("-m", "--model", type=str, required=True, choices=MODELS.keys())
parser.add_argument("-hl", "--hidden_layers", type=int, nargs="+", default=[500,500,2000])
parser.add_argument("-e", "--embedding_size", type=int, default=None)
parser.add_argument("-p", "--pretrain_epochs", type=int, default=None)
parser.add_argument("-c", "--clustering_epochs", type=int, default=None)

parser.add_argument("-l", "--clustering_loss", type=float, default=None)
parser.add_argument("-pl", "--pretrain_learning_rate", type=float, default=None)
parser.add_argument("-cl", "--clustering_learning_rate", type=float, default=None)

args = parser.parse_args()

# Seeding
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

# Load data set
load_data = DATASETS[args.dataset]
data, labels = load_data(return_X_y=True)

# Subsetting for reuters
if args.dataset == "reuters" and args.subset_amount is not None:
    subset_indices = np.load(f"subset_indices/{args.subset_amount}_reuters_subset_indices.npy")

    data = data[subset_indices]
    labels = labels[subset_indices]

data = z_normalization(data)
n_clusters = len(np.unique(labels))


# Initialize model
ModelClass = MODELS[args.model]
sig = inspect.signature(ModelClass)
DEFAULTS = {
    k: v.default
    for k, v in sig.parameters.items()
    if v.default is not inspect.Parameter.empty
}

embedding_size = args.embedding_size if args.embedding_size is not None else DEFAULTS["embedding_size"]
pretrain_epochs = args.pretrain_epochs if args.pretrain_epochs is not None else DEFAULTS["pretrain_epochs"]
clustering_epochs = args.clustering_epochs if args.clustering_epochs is not None else DEFAULTS["clustering_epochs"]
architecture = build_autoencoder(input_dim=data.shape[1], embedding_dim=embedding_size, hidden_layers=args.hidden_layers)
clustering_loss = args.clustering_loss if args.clustering_loss is not None else DEFAULTS["clustering_loss_weight"]
pretrain_optimizer = {"lr": args.pretrain_learning_rate} if args.pretrain_learning_rate is not None else DEFAULTS["pretrain_optimizer_params"]
clustering_optimizer = {"lr": args.clustering_learning_rate} if args.clustering_learning_rate is not None else DEFAULTS["clustering_optimizer_params"]

model = ModelClass(n_clusters=n_clusters,
                   neural_network=architecture,
                   random_state=args.seed,
                   pretrain_epochs=pretrain_epochs,
                   clustering_epochs=clustering_epochs,
                   clustering_loss_weight=clustering_loss,
                   pretrain_optimizer_params=pretrain_optimizer,
                   clustering_optimizer_params=clustering_optimizer,
                   device="cuda"
                   )


# Train model
model.fit(data)
model.neural_network.eval()

# Extract architecture
layers = []
for name, module in model.neural_network.named_modules():
    if isinstance(module, torch.nn.Linear):
        layers.append(f"{module.in_features}->{module.out_features}")
    elif type(module).__name__ != "Sequential":
        layers.append(type(module).__name__)

arch_str = " | ".join(layers)

# Get final embeddings
with torch.no_grad():
    data_tensor = torch.tensor(data, dtype=torch.float32, device="cuda")
    embeddings = model.neural_network.encoder(data_tensor).cpu().numpy()

pseudo_labels = model.labels_


# Calculate UCA, NMI, ARI
UCA = unsupervised_clustering_accuracy(labels, pseudo_labels)
NMI = normalized_mutual_info_score(labels, pseudo_labels)
ARI = adjusted_rand_score(labels, pseudo_labels)

# Extract features
extractor = Extractor(embeddings, pseudo_labels)
feature_dict = extractor.extract_all()

# Build results
row = {
    "dataset": args.dataset,
    "size": len(data),
    "seed": args.seed,
    "model": args.model,
    "architecture": arch_str,
    "n_clusters": n_clusters,
    "embedding_size": embedding_size,
    "pretrain_epochs": pretrain_epochs,
    "clustering_epochs": clustering_epochs,
    "UCA": UCA,
    "NMI": NMI,
    "ARI": ARI
}

for feat, val in feature_dict.items():
    if val is None or (isinstance(val, float) and np.isnan(val)):
        row[feat] = None
    else:
        row[feat] = float(val)


# Save results
excel = pd.read_excel(EXCEL_PATH)

# Create new columns
for col in row.keys():
    if col not in excel.columns:
        excel[col] = None

# Skip unused columns
for col in excel.columns:
    if col not in row:
        row[col] = None

excel = pd.concat([excel, pd.DataFrame([row])], ignore_index=True)
excel.to_excel(EXCEL_PATH, index=False)