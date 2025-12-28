import random

import numpy as np
import torch

from clustpy.metrics import unsupervised_clustering_accuracy
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from args import parse_arguments
from results.save_results import save_results
from utils.extract_metrics import Dispersion
from utils.load_data import load_normalized_data
from utils.model import initialize_model, extract_architecture

# Parse Arguments
args = parse_arguments()

# Seeding
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

# Load data
data, labels, n_clusters = load_normalized_data(args.dataset, args.subset_amount)

# Initialize model
model = initialize_model(args, data, n_clusters)

# Train model
model.fit(data)
model.neural_network.eval()

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
feature_dict = Dispersion(embeddings, pseudo_labels).extract_all()

# Build results
row = {
    "dataset": args.dataset,
    "size": len(data),
    "seed": args.seed,
    "model": args.model,
    "architecture": extract_architecture(model),
    "n_clusters": n_clusters,
    "embedding_size": args.embedding_size,
    "pretrain_epochs": model.pretrain_epochs,
    "clustering_epochs": model.clustering_epochs,
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
save_results(row, args.output)