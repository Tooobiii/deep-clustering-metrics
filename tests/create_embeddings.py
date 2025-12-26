import numpy as np

def symmetric_embedding():
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [3.0, 0.0],
        [3.0, 1.0],
    ])
    labels = np.array([0, 0, 1, 1])
    return X, labels

def asymmetric_embedding():
    X = np.array([
        [1.0, 0.0],
        [0.0, 0.5],
        [2.5, 0.0],
        [3.0, 1.0],
        [2.0, 2.0]
    ])
    labels = np.array([0, 0, 1, 1, 1])
    return X, labels