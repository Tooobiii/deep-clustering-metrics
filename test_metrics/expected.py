import numpy as np

from utils.extract_metrics import Separation, Density, Distance, Information, Symmetry

EXPECTED = [
    (Separation, {
        "ball_hall": 0.25,
        "calinski_harabasz": 18,
        "davies_bouldin": 1/3,
        "davies_bouldin_star": 1/3,
        "hartigan": np.log(9),
        "kwon": 13 / 36,
        "partition_separation": 2 * (1 - np.exp(-4)),
        "pbm": 22.5,
        "ratkowsky_lance": 1 / 2,
        "ray_turi": 1 / 36,
        "score_function": 0.845605,
        "tang": 10 / 9.5,
        "wemmert_gancarski": 1 - 0.5 / np.sqrt(9.25),
        "wemmert_gancarski_star": 1 - 0.5 / np.sqrt(9.25),
        "zalik_sv": 6,
    }),

    (Density, {
        "dbcv": 7 / 12,
        "sd1": 23 / 30,
        "sd2": 23 / 30,
        "sdbw": 1 / 10,
        "vsv1": 7 / 6,
        "vsv2": 7 / 6,
    }),

    (Distance, {
        "c_index": 0,
        "dunn": 3,
        "mcclain_rao": 2 * (np.sqrt(10) - 3),
        "point_biserial": 0.997727,
        "silhouette":1 - (2 / (3 + np.sqrt(10))),
    }),

    (Information, {
        "shannon_entropy": np.log(2),
        "bic": 19.930428,
        "icl": 19.930428,
    }),

    (Symmetry, {
        "sym": 1.5,
        "sym_davies_bouldin": 1 / 6,
        "sym_dunn": 12.0,
    })
]