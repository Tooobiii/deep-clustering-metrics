import numpy as np
import torch
from sklearn.metrics import pairwise_distances
from utils.load_data import load_normalized_data
from utils.model import initialize_model
from args import parse_arguments
from results.save_results import save_results


class Validator:
    def __init__(self, embeddings, pseudo_labels, n_clusters):
        self.embeddings = embeddings
        self.pseudo_labels = pseudo_labels
        self.n_clusters = n_clusters
        self.unique_labels = np.unique(pseudo_labels)
        self.dist_matrix = pairwise_distances(self.embeddings)

    def generate_subset(self):
        subset_idx = np.random.choice(len(self.pseudo_labels), size=int(len(self.pseudo_labels) * 0.2), replace=False)
        return self.embeddings[subset_idx], self.pseudo_labels[subset_idx]

    def generate_subset_matrix(self):
        subset_embedding, subset_pseudo_label = self.generate_subset()
        return pairwise_distances(subset_embedding), subset_pseudo_label

    def intra_cluster_distance_full(self):
        return np.mean([
            np.mean(self.dist_matrix[self.pseudo_labels == c][:, self.pseudo_labels == c])
            for c in self.unique_labels
        ])

    def intra_cluster_distance_subset(self):
        subset_matrix, subset_pseudo_label = self.generate_subset_matrix()
        return np.mean([
            np.mean(subset_matrix[subset_pseudo_label == c][:, subset_pseudo_label == c])
            for c in np.unique(subset_pseudo_label)
        ])


args = parse_arguments()

data, labels, n_clusters = load_normalized_data(args.dataset, args.subset_amount)

model = initialize_model(args, data, n_clusters)
model.fit(data)
model.neural_network.eval()

with torch.no_grad():
    data_tensor = torch.tensor(data, dtype=torch.float32, device="cuda")
    embeddings = model.neural_network.encoder(data_tensor).cpu().numpy()

validator = Validator(embeddings, model.labels_, n_clusters)

row = {
    "dataset": args.dataset,
    "size": len(data),
    "seed": args.seed,
    "model": args.model,
    "intra_cluster_full": validator.intra_cluster_distance_full(),
    "intra_cluster_subset": validator.intra_cluster_distance_subset()
}

save_results(row, f"../results/results_val.csv")