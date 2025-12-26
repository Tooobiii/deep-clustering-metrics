import numpy as np

EXPECTED_SYMMETRIC = {
    "dunn": 3,
    #"intercluster_separation": None, TODO
    "silhouette": 0.675440,
    "ball_hall": 0.25,
    "banfield_raftery": -np.log(256),
    "calinski_harabasz": 18,
    "davies_bouldin": 1/3,
    "davies_bouldin_star": 1/3,
    "fukuyama_sugeno": -3.5,
    "hartigan": np.log(9),
    "kwon": 13/36,
    "partition_separation": 1.963368,
    "pbm": 22.5,
    "ratkowsky_lance": 1/2,
    "ray_turi": 1/36,
    "r_squared": 9/10,
    "score_function": 0.845605,
    "sdbw": 1/10,
    "sd1": 23/30,
    "sd2": 23/30,
    "tang": 10 / 9.5,
    "trace_w": 1,
    "vsv1": 7/6,
    "vsv2": 7/6,
    "wch": 18,
    "wemmert_gancarski": 0.835601,
    "wemmert_gancarski_star": 0.835601,
    "xie_beni_star": 1/36,
    "zalik_sv": 6,
}

EXPECTED_ASYMMETRIC = {
    None : None # TODO
}