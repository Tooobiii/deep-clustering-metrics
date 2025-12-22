import numpy as np
from permetrics import ClusteringMetric
from pycvi import cvi_func as cv
from sklearn.metrics import pairwise_distances, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from pymfe.mfe import MFE

import time


class Extractor:
    def __init__(self, embedding, pseudo_labels, n_clusters):
        self.X = embedding
        self.y = pseudo_labels
        self.K = n_clusters
        self.unique_labels = np.unique(pseudo_labels)
        self.n, self.p = self.X.shape

        self.centroids = self._centroids()
        self.barycenter = self.X.mean(axis=0)
        self.cluster_sizes = self._cluster_sizes()

        self._precompute_distances()

        self.evaluator = ClusteringMetric()
        self.now = time.time()


    def _centroids(self):
        return np.vstack([
            self.X[self.y == c].mean(axis=0)
            for c in self.unique_labels
            ])

    def _cluster_sizes(self):
        return np.array([
            np.sum(self.y == c)
            for c in self.unique_labels
        ])

    def _precompute_distances(self):
        """Save frequently used distances"""
        # x_i - c_k
        self.within_centroid_dist = {
            c : self._dist(self.X[self.y == c], ck)
            for c, ck in zip(self.unique_labels, self.centroids)
        }

        # ||c_k - c_k'||
        self.between_centroid_dist = pairwise_distances(self.centroids)
        np.fill_diagonal(self.between_centroid_dist, np.inf) # Avoid zeros on the diagonal


    def _dist(self, a, b):
        """Calculate Euclidean distance"""
        # ||a - b||^2
        return np.linalg.norm(a - b, axis=1)

    def cluster_variances(self):
        """
        Calculate variance for each cluster
        Returns array of shape (K,)
        """
        # (1 / n_k) * \sum ||x_i - ck||^2
        return np.array([
            np.sum(self.within_centroid_dist[c] ** 2) / nk
            for c, nk in zip(self.unique_labels, self.cluster_sizes)
        ])

    def min_centroid_distance(self):
        """Shortest distance between centroids"""
        return np.min(self.between_centroid_dist)

    def max_centroid_distance(self):
        """Highest distance between centroids"""
        return np.max(self.between_centroid_dist)

    def mean_centroid_distance(self):
        """Average distance between centroids"""
        return np.mean(self.between_centroid_dist[np.isfinite(self.between_centroid_dist)])

    def cluster_radii(self):
        """
        Average distance between cluster points and centroid
        Return array of shape (K,)
        """
        # (1 / n_k) * \sum ||x_i - c_k||
        return np.array([
            np.mean(self.within_centroid_dist[c])
            for c in self.unique_labels
        ])

    def max_cluster_radius(self):
        """Highest cluster radius"""
        return np.max(self.cluster_radii())

    def tss(self):
        """Calculate total sum of squares"""
        # \sum ||x_i-b||^2
        return np.sum(self._dist(self.X, self.barycenter) ** 2)

    def wss(self):
        """Calculate within-group sum of squares"""
        # \sum \sum ||x_i - c_k||^2
        return sum(np.sum(d ** 2)
        for d in self.within_centroid_dist.values()
        )

    def bss(self):
        """Calculate between-group sum of squares"""
        # \sum \sum n_k * ||c_k - b||^2
        return sum(nk * np.linalg.norm(ck - self.barycenter) ** 2
        for nk, ck in zip(self.cluster_sizes, self.centroids)
    )



### DISTANCE-BASED METRICS ###
    def dunn(self):
        print("Dunn index started after:", time.time() - self.now)
        return self.evaluator.dunn_index(self.X, self.y, use_modified=False)

    def intercluster_separation(self):
        print("Intercluster separation index started after:", time.time() - self.now)
        mfe = MFE(groups=["clustering"], features=["int"])
        mfe.fit(self.X, self.y)
        _, values = mfe.extract()
        return values[0]

    def silhouette(self):
        print("Silhouette score started after:", time.time() - self.now)
        return silhouette_score(self.X, self.y)


### DISPERSION BASED METRICS ###
    def ball_hall(self):
        print("Ball Hall index started after:", time.time() - self.now)
        return self.evaluator.ball_hall_index(self.X, self.y)

    def banfield_raftery(self):
        pass #TODO

    def calinski_harabasz(self):
        print("Calinski-Harabasz index started after:", time.time() - self.now)
        return calinski_harabasz_score(self.X, self.y)

    def davies_bouldin(self):
        print("Davies-Bouldin index started after:", time.time() - self.now)
        return davies_bouldin_score(self.X, self.y)

    def davies_bouldin2(self):
        pass # TODO

    def dis(self):
        pass # TODO

    def dis_star(self):
        pass # TODO

    def fukuyuma_sugeno(self):
        pass # TODO

    def halkidi(self):
        pass # TODO

    def hartigan(self):
        print("Hartigan index started after:", time.time() - self.now)
        return self.evaluator.hartigan_index(self.X, self.y)

    def hartigan2(self):
        print("Hartigan index started after:", time.time() - self.now)
        return np.log(self.bss() / self.wss())

    def kwon(self):
        pass # TODO

    def partition_separation(self):
        pass # TODO

    def pbm(self):
        pass # TODO

    def ratkowsky_lance(self):
        pass # TODO

    def r_squared(self):
        return self.bss() / self.tss()

    def score_function(self):
        print("ScoreFunction started after:", time.time() - self.now)
        return cv.score_function(self.X, transform_multilabel(self.y))

    def scat(self):
        pass # TODO

    def scat_star(self):
        pass

    def sdbw(self):
        pass # TODO

    def sd1(self):
        pass # TODO

    def sd2(self):
        pass # TODO

    def tang(self):
        pass

    def trace_w(self):
        print("Trace W started after:", time.time() - self.now)
        return self.wss()

    def vsv1(self):
        pass # TODO

    def vsv2(self):
        pass # TODO

    def wch(self):
        pass # TODO

    def wemmert_gancarski(self):
        pass

    def wemmert_gancarski2(self):
        pass

    def xie_beni(self):
        print("Xie-Beni index started after:", time.time() - self.now)
        return self.evaluator.xie_beni_index(self.X, self.y)

    def xie_beni2(self):
        pass # TODO check implementation of xie_beni

    def zalik_sv(self):
        pass # TODO


### INFORMATION-BASED METRICS
    def normalized_relative_entropy(self):
        print("Normalized relative entropy started after:", time.time() - self.now)
        mfe = MFE(groups=["clustering"], features=["nre"])
        mfe.fit(self.X, self.y)
        _, values = mfe.extract()
        return values[0]


### EXTRACTION
    def extract_all(self):
        result = {
            "Dunn Index": self.dunn(),
            "Silhouette Score": self.silhouette(),

            "Ball-Hall Index": self.ball_hall(),
            "Calinski-Harabasz Index": self.calinski_harabasz(),
            "Davies-Bouldin Index": self.davies_bouldin(),
            "PBM-Index": self.pbm(),
            "S_Dbw": self.sdbw(),

            "Normalized relative entropy": self.normalized_relative_entropy(),


            "Intercluster_Separation_Index": self.intercluster_separation(),
            "Hartigan Index": self.hartigan(),
            "Score_Function": self.score_function(),
            "Xie-Beni_Index": self.xie_beni(),
        }

        print("All done after:", time.time() - self.now)

        return result

def transform_multilabel(labels):
    return [[int(label)] for label in labels]