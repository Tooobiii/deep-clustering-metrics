import numpy as np
from pycvi import cvi_func as cv
from sklearn.metrics import pairwise_distances, silhouette_score, calinski_harabasz_score, davies_bouldin_score

class Extractor:
    def __init__(self, embeddings, pseudo_labels):
        self.embeddings = embeddings
        self.pseudo_labels = pseudo_labels

    def dunn(self):
        return cv.dunn(self.embeddings, self.pseudo_labels)

    def davies_bouldin_index(self):
        return davies_bouldin_score(self.embeddings, self.pseudo_labels)

    def calinski_harabasz_index(self):
        return calinski_harabasz_score(self.embeddings, self.pseudo_labels)

    def intra_cluster_distance(self):
        distances = pairwise_distances(self.embeddings)
        return np.mean([
            np.mean(distances[self.pseudo_labels == c][:, self.pseudo_labels == c])
            for c in np.unique(self.pseudo_labels)
        ])

    def inter_cluster_distance(self):
        centroids = np.vstack([
            self.embeddings[self.pseudo_labels == c].mean(axis=0)
            for c in np.unique(self.pseudo_labels)
        ])
        distances = pairwise_distances(centroids)
        return np.mean(distances[np.triu_indices_from(distances, k=1)])

    def silhouette_score(self):
        return silhouette_score(self.embeddings, self.pseudo_labels)

    def extract_all(self):
        return {
            "dunn": self.dunn(),
        #    "davies_bouldin": self.davies_bouldin_index(),
            "calinski_harabasz": self.calinski_harabasz_index(),
            "inter_cluster_distance": self.inter_cluster_distance(),
            "intra_cluster_distance": self.intra_cluster_distance(),
            "silhouette_score": self.silhouette_score(),
        }

    # TODO More features