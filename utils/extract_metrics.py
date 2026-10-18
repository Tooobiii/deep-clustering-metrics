import time
from functools import wraps

import numpy as np
from permetrics import ClusteringMetric # Dunn
from pymfe.mfe import MFE
from scipy.special import logsumexp
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.neighbors import NearestNeighbors

def metric(name=None):
    def decorator(func):
        func._is_metric = True
        func._metric_name = name or func.__name__
        return func
    return decorator

def timed(label=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            name = label or func.__name__
            print(f"{name} took {time.time() - start:.4f}s")
            return result
        return wrapper
    return decorator

class Metrics:
    def __init__(self, embedding, pseudo_labels):
        self.X = embedding
        self.y = pseudo_labels
        self.K = len(np.unique(self.y))
        self.labels = np.unique(self.y)
        self.n, self.p = self.X.shape
        self.centroids = self._centroids()
        self.cluster_sizes = self._cluster_sizes()
        self.barycenter = self.X.mean(axis=0)
        self._precompute_distances()

    def _precompute_distances(self):
        """
        Precompute frequently used distances

        Computes and stores:
        - Distances of points assigned to their assigned cluster centroids
        - Pairwise distances between all cluster centroids

        within_centroid_dist : dict[int, ndarray]
            For each cluster index k, an array of shape (n_k,) containing
            distances ||x_i -c_k|| for all points i assigned to cluster k.

        between_centroid_dist : ndarray of shape (K, K)
            Matrix of centroid distances ||c_k - c_k'||.
        """
        self.within_centroid_dist = {
            k : self._dist(self.X[self.y == label], self.centroids[k])
            for k, label in enumerate(self.labels)
        }

        self.between_centroid_dist = pairwise_distances(self.centroids)
        np.fill_diagonal(self.between_centroid_dist, np.inf)

    def _centroids(self):
        """
        Compute cluster centroids

        :return:
        centroids : ndarray of shape (K, p)
            Centroid c_k of each cluster k, defined as the mean of all points
            assigned to that cluster.
        """
        return np.vstack([
            self.X[self.y == c].mean(axis=0)
            for c in self.labels
            ])

    def _dist(self, a, b):
        """
        Compute Euclidean distances between multiple vectors and a single vector.

        :parameter
        a : ndarray of shape (m, p)
            Matrix containing m vectors in p-dimensional space.
        b : ndarray of shape (p,)
            Reference vector.

        :return
        distances : array of shape (m,)
            Euclidean distances ||a_i - b|| for i = 1, ..., m
        """
        return np.linalg.norm(a - b, axis=1)

    def _cluster_sizes(self):
        """
        Compute cluster sizes
        :return:
        cluster_sizes : ndarray of shape (K,)
            Number of points n_k assigned to each cluster k.
        """
        return np.array([
            np.sum(self.y == c)
            for c in self.labels
        ])

    def min_centroid_distance(self):
        """
        Compute the minimum distance between distinct centroids.

        :return
        distance : float
            Minimum centroid-to-centroid distance
            min ||c_k - c_k'||
        """
        return np.min(self.between_centroid_dist)

    def max_centroid_distance(self):
        """
        Compute the maximum distance between centroids.

        :return
        distance : float
            Maximum centroid-to-centroid distance
            max ||c_k - c_k'||
        """
        return np.max(self.between_centroid_dist[np.isfinite(self.between_centroid_dist)])

    def extract_all(self):
        results = {}

        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and getattr(attr, "_is_metric", False):
                results[attr._metric_name] = attr()

        return results


class Separation(Metrics):
    def __init__(self, embedding, pseudo_labels):
        super().__init__(embedding, pseudo_labels)

    ### HELPING FUNCTIONS ###
    def cluster_variances(self):
        """
        Compute within-cluster variances.

        :return
        variances : ndarray of shape (K,)
            Cluster variances, defined as
            (1 / n_k) * \\sum ||x_i - c_k||^2
        """
        return np.array([
            np.sum(self.within_centroid_dist[c] ** 2) / nk
            for c, nk in zip(self.labels, self.cluster_sizes)
        ])

    def mean_centroid_distance(self):
        """
        Compute the mean distance between distinct centroids.

        :return
        distance : float
            Mean centroid-to-centroid distance
        """
        return np.mean(self.between_centroid_dist[np.isfinite(self.between_centroid_dist)])

    def cluster_radii(self):
        """
        Compute cluster raddii.

        :return
        radii : ndarray of shape (K,)
            Mean distance of points in each cluster to their centroid
            (1 / n_k) * \\sum ||x_i - c_k||
        """
        return np.array([
            np.mean(self.within_centroid_dist[c])
            for c in self.labels
        ])

    def max_cluster_radius(self):
        """
        Compute the maximum cluster radius

        :return
        radius : float
            Maximum cluster radius max((1 / n_k) * \\sum ||x_i - c_k||)
        """
        return np.max(self.cluster_radii())

    def tss(self):
        """
        Compute total sum of squares

        :return
        tss : float
            TSS = \\sum ||x_i - b||^2
        """
        return np.sum(self._dist(self.X, self.barycenter) ** 2)

    def wss(self):
        """
        Compute within-group sum of squares

        :return
        wss : float
            WSS = \\sum \\sum ||x_i - c_k||^2
        """
        return sum(np.sum(d ** 2)
        for d in self.within_centroid_dist.values()
        )

    def bss(self):
        """
        Compute between-group sum of squares

        :return
        bss : float
            BSS = \\sum ||c_k - b||^2
        """
        return sum(nk * np.linalg.norm(ck - self.barycenter) ** 2
        for nk, ck in zip(self.cluster_sizes, self.centroids)
    )

    ### METRICS ###
    @metric("Ball-Hall")
    @timed("Ball-Hall")
    def ball_hall(self):
        return np.mean(self.cluster_variances())

    @metric("Calinski-Harabasz")
    @timed("Calinski-Harabasz")
    def calinski_harabasz(self):
        return (self.bss() / (self.K - 1)) / (self.wss() / (self.n - self.K))

    @metric("Davies-Bouldin")
    @timed("Davies-Bouldin")
    def davies_bouldin(self):
        radii = self.cluster_radii()
        db = []

        for k in range(self.K):
            ratios = []
            for kp in range(self.K):
                if k != kp:
                    ratios.append(
                        (radii[k] + radii[kp]) / self.between_centroid_dist[k, kp]
                    )
            db.append(np.max(ratios))

        return np.mean(db)

    @metric("Davies-Bouldin starred")
    @timed("Davies-Bouldin starred")
    def davies_bouldin_star(self):
        radii = self.cluster_radii()
        vals = []

        for k in range(self.K):
            max_num = max(
                (radii[k] + radii[kp].item())
                for kp in range(self.K) if kp != k
            )
            min_den = min(
                self.between_centroid_dist[k, kp]
                for kp in range(self.K) if kp != k
            )
            vals.append(max_num / min_den)

        return np.mean(vals)

    @metric("Hartigan")
    @timed("Hartigan")
    def hartigan(self):
        return np.log(self.bss() / self.wss())

    @metric("Kwon")
    @timed("Kwon")
    def kwon(self):
        num = self.wss() + np.sum(
            np.linalg.norm(self.centroids - self.barycenter, axis=1) ** 2
        ) / self.K
        den = self.min_centroid_distance() ** 2
        return num / den

    @metric("Partition Separation")
    @timed("Partition Separation")
    def partition_separation(self):
        max_nk = np.max(self.cluster_sizes)
        c_bar = self.centroids.mean(axis=0)
        den = np.mean(np.linalg.norm(self.centroids - c_bar, axis=1) ** 2)

        total = 0.0
        for k in range(self.K):
            min_d = min(self.between_centroid_dist[k, kp] ** 2
                        for kp in range(self.K) if kp != k
                        )
            total += (self.cluster_sizes[k] / max_nk - np.exp(-min_d / den))

        return total

    @metric("PBM")
    @timed("PBM")
    def pbm(self):
        e1 = np.sum(self._dist(self.X, self.barycenter))
        ek = sum([np.sum(d) for d in self.within_centroid_dist.values()])
        dk = self.max_centroid_distance()
        return ((1 / self.K) * (e1 / ek) * dk) ** 2

    @metric("Ratkowsky-Lance")
    @timed("Ratkowsky-Lance")
    def ratkowsky_lance(self):
        num = np.zeros(self.p)
        den = np.zeros(self.p)

        for j in range(self.p):
            xj = self.X[:, j]
            bj = xj.mean()

            den[j] = np.sum((xj - bj) ** 2)

            for nk, ck in zip(self.cluster_sizes, self.centroids[:, j]):
                num[j] += nk * (ck - bj) ** 2

        return np.sqrt((1 / (self.K * self.p)) * np.sum(num / den))

    @metric("Ray-Turi")
    @timed("Ray-Turi")
    # Crisp Xie-Beni
    def ray_turi(self):
        return (self.wss() / self.n) / (self.min_centroid_distance() ** 2)

    @metric("R-Squared")
    @timed("R-Squared")
    def r_squared(self):
        return self.bss() / self.tss()

    @metric("Score Function")
    @timed("Score Function")
    def score_function(self):
        term1 = sum(
            nk * np.linalg.norm(ck - self.centroids.mean(axis=0))
            for nk, ck in zip(self.cluster_sizes, self.centroids)
        )

        term2 = sum((1 / nk) *
            np.sum(d)
            for nk, d in zip(self.cluster_sizes, self.within_centroid_dist.values())
        )

        return 1.0 - (np.exp(np.exp((1 / (self.n * self.K)) * (term1 - term2)))) ** -1

    @metric("Tang")
    @timed("Tang")
    def tang(self):
        pair_sum = 0.0
        for k in range(self.K - 1):
            for kp in range(k + 1, self.K):
                pair_sum += self.between_centroid_dist[k, kp] ** 2

        num = self.wss() + (2.0 / (self.K * (self.K - 1))) * pair_sum
        den = (self.min_centroid_distance() ** 2) + (1.0 / self.K)
        return num / den

    @metric("Trace_W")
    @timed("Trace_W")
    def trace_w(self):
        return self.wss()

    @metric("Wemmert-Gancarski")
    @timed("Wemmert-Gancarski")
    def wemmert_gancarski(self):
        total = 0.0

        for k in range(self.K):
            sum_ratios = 0.0
            for xi in self.X[self.y == self.labels[k]]:
                num = np.linalg.norm(xi - self.centroids[k])
                den = min(np.linalg.norm(xi - self.centroids[kp])
                            for kp in range(self.K) if kp != k
                            )
                sum_ratios += num / den

            total += max(0.0, self.cluster_sizes[k] - sum_ratios)

        return total / self.n

    @metric("Wemmert-Gancarski starred")
    @timed("Wemmert-Gancarski starred")
    def wemmert_gancarski_star(self):
        total = 0.0

        for k in range(self.K):
            cluster_sum = 0.0
            for xi in self.X[self.y == self.labels[k]]:
                d_ck = np.linalg.norm(xi - self.centroids[k])
                d_min = min(np.linalg.norm(xi - self.centroids[kp])
                            for kp in range(self.K) if kp != k
                            )
                cluster_sum += 1.0 - (d_ck / d_min)

            total += cluster_sum / self.cluster_sizes[k]

        return total / self.K

    @metric("Zalik SV")
    @timed("Zalik SV")
    def zalik_sv(self):
        num = sum(min(self.between_centroid_dist[k,kp]
                      for kp in range(self.K) if kp != k)
                  for k in range (self.K)
                  )

        den = sum(np.max(self.within_centroid_dist[k])
                  for k in range (self.K)
                  )

        return num / den


class Density(Metrics):
    def __init__(self, embedding, pseudo_labels):
        super().__init__(embedding, pseudo_labels)

    ### HELPING FUNCTIONS ###
    def dis(self):
        dists = self.between_centroid_dist[np.isfinite(self.between_centroid_dist)]
        ratio = np.max(dists) / np.min(dists)

        inv_sum = 0.0
        for k in range(self.K):
            inv_sum += 1.0 / sum(
                self.between_centroid_dist[k, kp]
                for kp in range(self.K) if kp != k
            )

        return ratio * inv_sum

    def dis_star(self):
        sigma = np.sqrt(
            np.mean([
                np.sum(self.within_centroid_dist[k] ** 2) / self.cluster_sizes[k]
                for k in range(self.K)
            ])
        )

        nck = []
        for k in range(self.K):
            nck.append(np.sum(
                self._dist(self.X, self.centroids[k]) <= sigma)
            )

        dis = 0.0
        for k in range(self.K):
            for kp in range(self.K):
                if k != kp:
                    mid = 0.5 * (self.centroids[k] + self.centroids[kp])
                    n_mid = np.sum(self._dist(self.X, mid) <= sigma)
                    dis += n_mid / max(nck[k], nck[kp])

        dis /= (self.K * (self.K - 1))

        return dis

    def scat(self):
        num = np.mean([
            np.sum(self.within_centroid_dist[k] ** 2 / self.cluster_sizes[k])
            for k in range(self.K)
        ])

        den = np.sum(self._dist(self.X, self.barycenter) ** 2) / self.n

        return num / den

    def scat_star(self):
        return max(np.sum(
            self.within_centroid_dist[k] ** 2 / self.cluster_sizes[k]) /
                   (np.sum(self._dist(self.X, self.barycenter) ** 2) / self.n)
            for k in range(self.K)
        )

    ### METRICS ###
    @metric("DBCV")
    @timed("DBCV")
    def dbcv(self):
        return ClusteringMetric(X=self.X, y_pred=self.y).density_based_clustering_validation_index()

    @metric("S_Dbw")
    @timed("S_Dbw")
    def sdbw(self):
        return self.scat() + self.dis_star()

    @metric("SD1")
    @timed("SD1")
    def sd1(self):
        return self.scat() + self.dis()

    @metric("SD2")
    @timed("SD2")
    def sd2(self):
        return self.scat_star() + self.dis()

    @metric("Vsv1")
    @timed("Vsv1")
    def vsv1(self):
        vu = np.mean([np.sum(
            self.within_centroid_dist[k]) / self.cluster_sizes[k]
                      for k in range(self.K)
                      ])

        vo = self.K / self.min_centroid_distance()

        return vu + vo

    @metric("Vsv2")
    @timed("Vsv2")
    def vsv2(self):
        vu_star = max(np.sum(
            self.within_centroid_dist[k]) / self.cluster_sizes[k]
                      for k in range(self.K)
        )

        vo = self.K / self.min_centroid_distance()

        return vu_star + vo


class Distance(Metrics):
    @timed("Precomputing distances")
    def __init__(self, embedding, pseudo_labels):
        super().__init__(embedding, pseudo_labels)

        self.D = pairwise_distances(self.X)
        self._iu = np.triu_indices(self.n, k=1)
        self.distances = self.D[self._iu]

        self._precompute()

        self.evaluator = ClusteringMetric()

    ### HELPING FUNCTIONS ###
    def _precompute(self):
        self._same_cluster = (
            self.y[self._iu[0]] == self.y[self._iu[1]]
        )
        self._different_cluster = ~self._same_cluster

        self.nT = self.n * (self.n - 1) // 2
        self.nW = np.sum(self._same_cluster)
        self.nB = np.sum(self._different_cluster)
        self.SW = np.sum(self.distances[self._same_cluster])
        self.SB = np.sum(self.distances[self._different_cluster])

        self._sorted_distances = np.sort(self.distances)
        self._sd_distances = np.std(self.distances, ddof=0)

    def s_min(self):
        return np.sum(self._sorted_distances[: self.nW])

    def s_max(self):
        return np.sum(self._sorted_distances[-self.nW :])

    def sd(self):
        return self._sd_distances

    ### METRICS ###
    @metric("C Index")
    @timed("C Index")
    def c_index(self):
        return (self.SW - self.s_min()) / (self.s_max() - self.s_min())

    @metric("Dunn")
    @timed("Dunn")
    def dunn(self):
        return self.evaluator.dunn_index(self.X, self.y, use_modified=False)

    @metric("McClain-Rao")
    @timed("McClain-Rao")
    def mcclain_rao(self):
        return (self.SW / self.nW) / (self.SB / self.nB)

    @metric("Point Biserial")
    @timed("Point Biserial")
    def point_biserial(self):
        meanB = np.float64(self.SB) / np.float64(self.nB)
        meanW = np.float64(self.SW) / np.float64(self.nW)
        scale = np.sqrt(np.float64(self.nB) * np.float64(self.nW)) / np.float64(self.nT)
        return (meanB - meanW) * scale / np.float64(self.sd())

    @metric("Silhouette")
    @timed("Silhouette")
    def silhouette(self):
        return silhouette_score(self.X, self.y)

class Information(Metrics):
    def __init__(self, embedding, pseudo_labels):
        super().__init__(embedding, pseudo_labels)

    ### HELPING FUNCTIONS
    def _gaussian_mixture_log_probs(self):
        sq_dist = np.column_stack([np.sum((self.X - self.centroids[k]) ** 2, axis=1)
            for k in range(self.K)
        ])

        sse = sum(np.sum(sq_dist[self.y == label, k])
            for k, label in enumerate(self.labels)
        )

        sigma2 = sse / (self.n * self.p)

        weights = self.cluster_sizes / self.n

        log_probs = (np.log(weights)[None, :]- 0.5 * self.p * np.log(2.0 * np.pi * sigma2)- sq_dist / (2.0 * sigma2))
        return log_probs, sigma2

    ### METRICS
    @metric("Shannon Entropy")
    @timed("Shannon Entropy")
    def shannon_entropy(self):
        p = self.cluster_sizes / self.n
        return -np.sum(p * np.log(p + 1e-6))

    @metric("Bayesian Information Criterion")
    @timed("Bayesian Information Criterion")
    def bic(self):
        log_probs, _ = self._gaussian_mixture_log_probs()
        log_likelihood = np.sum(logsumexp(log_probs, axis=1))
        n_parameters = self.K * self.p + self.K

        return float(-2.0 * log_likelihood+ n_parameters * np.log(self.n))

    @metric("Integrated Completed Likelihood")
    @timed("Integrated Completed Likelihood")
    def icl(self):
        log_probs, _ = self._gaussian_mixture_log_probs()

        log_norm = logsumexp(log_probs, axis=1, keepdims=True)
        log_tau = log_probs - log_norm
        tau = np.exp(log_tau)

        entropy = -np.sum(tau * log_tau)

        return float(self.bic() + 2.0 * entropy)


class Symmetry(Metrics):
    @timed("Precomputing point-symmetry distances")
    def __init__(self, embedding, pseudo_labels):
        super().__init__(embedding, pseudo_labels)

        self._dps_by_k = None
        self._sum_dps_by_k = None
        self._max_dps_by_k = None
        self._total_dps = None

        self._precompute_point_symmetry()

        self._min_intercluster_dist = None

    ### HELPING FUNCTIONS
    def _precompute_point_symmetry(self):
        dps_by_k = {}
        sum_dps_by_k = np.zeros(self.K, dtype=np.float64)
        max_dps_by_k = np.zeros(self.K, dtype=np.float64)

        for k, label in enumerate(self.labels):
            Xk = self.X[self.y == label]
            nk = Xk.shape[0]

            if nk < 2:
                dps = np.full(nk, np.nan, dtype=np.float64)
                dps_by_k[k] = dps
                sum_dps_by_k[k] = np.nan
                max_dps_by_k[k] = np.nan
                continue

            ck = self.centroids[k]
            Xk_star = 2.0 * ck - Xk

            nn = NearestNeighbors(n_neighbors=2, algorithm="auto")
            nn.fit(Xk)
            dists, _ = nn.kneighbors(Xk_star, return_distance=True)

            dsym = dists.mean(axis=1)  # Eq. (11)
            r = self.within_centroid_dist[k].astype(np.float64)
            dps = dsym * r

            dps_by_k[k] = dps
            sum_dps_by_k[k] = np.sum(dps)
            max_dps_by_k[k] = np.max(dps)

        self._dps_by_k = dps_by_k
        self._sum_dps_by_k = sum_dps_by_k
        self._max_dps_by_k = max_dps_by_k

        self._total_dps = np.nansum(sum_dps_by_k) if np.any(np.isnan(sum_dps_by_k)) else float(np.sum(sum_dps_by_k))

    def _centroid_dist_matrix(self):
        Dc = pairwise_distances(self.centroids)
        np.fill_diagonal(Dc, np.inf)
        return Dc

    def _min_intercluster_distance(self):
        if self._min_intercluster_dist is not None:
            return self._min_intercluster_dist

        D = pairwise_distances(self.X)
        same = (self.y[:, None] == self.y[None, :])
        D[same] = np.inf
        np.fill_diagonal(D, np.inf)

        self._min_intercluster_dist = float(np.min(D))
        return self._min_intercluster_dist

    ### METRICS
    @metric("Sym")
    @timed("Sym")
    def sym(self):
        Dc = self._centroid_dist_matrix()
        max_centroid_dist = float(np.max(Dc[np.isfinite(Dc)]))

        denom = self._total_dps
        if denom == 0 or np.isnan(denom):
            return np.nan

        return (max_centroid_dist / denom) / float(self.K)

    @metric("Sym-Davies-Bouldin")
    @timed("Sym-Davies-Bouldin")
    def sym_davies_bouldin(self):
        Dc = self._centroid_dist_matrix()

        avg_ps = np.empty(self.K, dtype=np.float64)
        for k, label in enumerate(self.labels):
            nk = np.sum(self.y == label)
            if nk == 0:
                avg_ps[k] = np.nan
            else:
                avg_ps[k] = self._sum_dps_by_k[k] / float(nk)

        if np.any(np.isnan(avg_ps)):
            return np.nan

        R = (avg_ps[:, None] + avg_ps[None, :]) / Dc
        np.fill_diagonal(R, -np.inf)

        return float(np.mean(np.max(R, axis=1)))

    @metric("Sym-Dunn")
    @timed("Sym-Dunn")
    def sym_dunn(self):
        numerator = self._min_intercluster_distance()

        denom = float(np.nanmax(self._max_dps_by_k))
        if denom == 0 or np.isnan(denom):
            return np.nan

        return numerator / denom


class MetaFeatures:
    def __init__(self, embedding, pseudo_labels, mfe):
        self.embedding = embedding
        self.pseudo_labels = pseudo_labels
        self.mfe = mfe

    @timed("Extracting")
    def extract(self):
        self.mfe.fit(self.embedding, self.pseudo_labels)
        features, values = self.mfe.extract()
        return dict(zip(features, values))

class Statistical(MetaFeatures):
    def __init__(self, embedding, pseudo_labels):
        super().__init__(embedding, pseudo_labels, MFE(groups=["info-theory", "statistical"]))

class Complexity(MetaFeatures):
    def __init__(self, embedding, pseudo_labels):
        super().__init__(embedding, pseudo_labels, MFE(groups=["complexity"]))