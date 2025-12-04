import numpy as np
from pycvi import cvi_func as cv
from sklearn.metrics import pairwise_distances, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from pymfe.mfe import MFE

import time


class Extractor:
    def __init__(self, embeddings, pseudo_labels, n_clusters):
        self.embeddings = embeddings
        self.pseudo_labels = pseudo_labels
        self.n_clusters = n_clusters
        self.unique_labels = np.unique(pseudo_labels)
        self.now = time.time()
        #self.dist_matrix = pairwise_distances(self.embeddings)
        #self.centroids = self.compute_centroids()

### HELPING FUNCTIONS ###
    def compute_centroids(self):
        return np.vstack([
            self.embeddings[self.pseudo_labels == c].mean(axis=0)
            for c in self.unique_labels
            ])


### COHESION METRICS (DISTANCE WITHIN A CLUSTER) ###
    def intra_cluster_distance(self):
        print("intra started after:", time.time() - self.now)
        distances = pairwise_distances(self.embeddings)
        return np.mean([
            np.mean(distances[self.pseudo_labels == c][:, self.pseudo_labels == c])
            for c in self.unique_labels
        ])

    def silhouette_score(self):
        print("silhouette_score started after:", time.time() - self.now)
        return silhouette_score(self.embeddings, self.pseudo_labels)


### SEPARATION METRICS (DISTANCE BETWEEN CLUSTERS) ###
    def inter_cluster_distance(self):
        print("inter started after:", time.time() - self.now)
        centroids = self.compute_centroids()
        distances = pairwise_distances(centroids)
        return np.mean(distances[np.triu_indices_from(distances, k=1)])

    def dunn(self):
        print("dunn started after:", time.time() - self.now)
        mfe = MFE(groups=["clustering"], features=["vdu"])
        mfe.fit(self.embeddings, self.pseudo_labels)
        _, values = mfe.extract()
        return values[0]

    def intercluster_seperation_index(self):
        print("intercluster_seperation_index started after:", time.time() - self.now)
        mfe = MFE(groups=["clustering"], features=["int"])
        mfe.fit(self.embeddings, self.pseudo_labels)
        _, values = mfe.extract()
        return values[0]


### DISPERSION METRICS (COMBINE COHESION AND SEPARATION)
    def calinski_harabasz_index(self):
        print("calinski_harabasz_index started after:", time.time() - self.now)
        return calinski_harabasz_score(self.embeddings, self.pseudo_labels)

    def davies_bouldin_index(self):
        print("davies_bouldin_index started after:", time.time() - self.now)
        return davies_bouldin_score(self.embeddings, self.pseudo_labels)

    def score_function(self):
        print("score_function started after:", time.time() - self.now)
        return cv.score_function(self.embeddings, transform_multilabel(self.pseudo_labels))


### INFORMATION-BASED METRICS
    def normalized_relative_entropy(self):
        print("normalized_relative_entropy started after:", time.time() - self.now)
        mfe = MFE(groups=["clustering"], features=["nre"])
        mfe.fit(self.embeddings, self.pseudo_labels)
        _, values = mfe.extract()
        return values[0]

    def extract_all(self):
        result = {
            "intra_cluster_distance": self.intra_cluster_distance(),
            "dunn": self.dunn(),
            "davies_bouldin": self.davies_bouldin_index(),
            "calinski_harabasz": self.calinski_harabasz_index(),
            "inter_cluster_distance": self.inter_cluster_distance(),
            "silhouette_score": self.silhouette_score(),
            "score_function": self.score_function(),
            "intercluster_seperation_index": self.intercluster_seperation_index(),
            "normalized_relative_entropy": self.normalized_relative_entropy()
        }

        print("all done after:", time.time() - self.now)

        return result

def transform_multilabel(labels):
    return [[int(label)] for label in labels]