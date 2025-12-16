import numpy as np
from permetrics import ClusteringMetric
from pycvi import cvi_func as cv
from sklearn.metrics import pairwise_distances, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from pymfe.mfe import MFE

import time


class Extractor:
    def __init__(self, embedding, pseudo_labels, n_clusters):
        self.embedding = embedding
        self.pseudo_labels = pseudo_labels
        self.n_clusters = n_clusters
        self.unique_labels = np.unique(pseudo_labels)
        self.centroids = self.compute_centroids()
        self.evaluator = ClusteringMetric()

        self.now = time.time()


    def compute_centroids(self):
        return np.vstack([
            self.embedding[self.pseudo_labels == c].mean(axis=0)
            for c in self.unique_labels
            ])


### DISTANCE-BASED METRICS ###
    def dunn(self):
        print("Dunn index started after:", time.time() - self.now)
        return self.evaluator.dunn_index(self.embedding, self.pseudo_labels, use_modified=False)

    def intercluster_separation_index(self):
        print("Intercluster separation index started after:", time.time() - self.now)
        mfe = MFE(groups=["clustering"], features=["int"])
        mfe.fit(self.embedding, self.pseudo_labels)
        _, values = mfe.extract()
        return values[0]

    def silhouette_score(self):
        print("Silhouette score started after:", time.time() - self.now)
        return silhouette_score(self.embedding, self.pseudo_labels)


### DISPERSION BASED METRICS (COMBINE COHESION AND SEPARATION)
    def ball_hall_index(self):
        print("Ball Hall index started after:", time.time() - self.now)
        return self.evaluator.ball_hall_index(self.embedding, self.pseudo_labels)

    def calinski_harabasz_index(self):
        print("Calinski-Harabasz index started after:", time.time() - self.now)
        return calinski_harabasz_score(self.embedding, self.pseudo_labels)

    def davies_bouldin_index(self):
        print("Davies-Bouldin index started after:", time.time() - self.now)
        return davies_bouldin_score(self.embedding, self.pseudo_labels)

    def hartigan_index(self):
        print("Hartigan index started after:", time.time() - self.now)
        return self.evaluator.hartigan_index(self.embedding, self.pseudo_labels)

    def pbm_index(self):
        print("PBM index started after:", time.time() - self.now)
        center = self.embedding.mean(axis=0)
        e1 = np.sum(np.linalg.norm(self.embedding - center, axis=1))
        ek = sum(
            np.sum(np.linalg.norm(self.embedding[self.pseudo_labels == i] - self.centroids[i], axis=1))
            for i in range(self.n_clusters))
        dk = np.max(pairwise_distances(self.centroids))
        return ((e1 / ek) * dk) ** 2 if ek > 0 else np.nan

    def score_function(self):
        print("ScoreFunction started after:", time.time() - self.now)
        return cv.score_function(self.embedding, transform_multilabel(self.pseudo_labels))

    def sdbw_index(self):
        print("S_Dbw index started after:", time.time() - self.now)
        scat = np.mean([
            np.linalg.norm(self.embedding[self.pseudo_labels == i] - self.centroids[i])
            for i in range(self.n_clusters)
        ])
        density = 0.0
        for i in range(self.n_clusters):
            for j in range(i + 1, self.n_clusters):
                mid = self.centroids[i] + self.centroids[j] / 2
                density += np.sum(np.linalg.norm(self.embedding - mid, axis=1) < scat)
        return scat + density

    def xie_beni_index(self):
        print("Xie-Beni index started after:", time.time() - self.now)
        return self.evaluator.xie_beni_index(self.embedding, self.pseudo_labels)

### INFORMATION-BASED METRICS
    def normalized_relative_entropy(self):
        print("Normalized relative entropy started after:", time.time() - self.now)
        mfe = MFE(groups=["clustering"], features=["nre"])
        mfe.fit(self.embedding, self.pseudo_labels)
        _, values = mfe.extract()
        return values[0]

    def extract_all(self):
        result = {
            "Dunn_Index": self.dunn(),
            "Intercluster_Separation_Index": self.intercluster_separation_index(),
            "Silhouette_Score": self.silhouette_score(),

            "Ball-Hall_Index": self.ball_hall_index(),
            "Calinski_Harabasz_Index": self.calinski_harabasz_index(),
            "Davies-Bouldin_Index": self.davies_bouldin_index(),
            "Hartigan_Index": self.hartigan_index(),
            "Normalized relative entropy": self.normalized_relative_entropy(),
            "PBM-Index": self.pbm_index(),
            "Score_Function": self.score_function(),
            "S_Dbw": self.sdbw_index(),
            "Xie-Beni_Index": self.xie_beni_index()
        }

        print("All done after:", time.time() - self.now)

        return result

def transform_multilabel(labels):
    return [[int(label)] for label in labels]