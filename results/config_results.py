DATASET_POLICY = {
    "mnist": {
        "dim_group": "low",
        "size_group": "large",
        "k_group": "medium",
        "data_type": "image",
    },
    "fmnist": {
        "dim_group": "low",
        "size_group": "large",
        "k_group": "medium",
        "data_type": "image",
    },
    "usps": {
        "dim_group": "low",
        "size_group": "small",
        "k_group": "medium",
        "data_type": "image",
    },
    "cifar10": {
        "dim_group": "medium",
        "size_group": "large",
        "k_group": "medium",
        "data_type": "image",
    },
    "cifar100": {
        "dim_group": "medium",
        "size_group": "large",
        "k_group": "many",
        "data_type": "image",
    },
    "stl10": {
        "dim_group": "high",
        "size_group": "small",
        "k_group": "medium",
        "data_type": "image",
    },
    "coil20": {
        "dim_group": "high",
        "size_group": "small",
        "k_group": "many",
        "data_type": "image",
    },
    "reuters10k": {
        "dim_group": "medium",
        "size_group": "small",
        "k_group": "few",
        "data_type": "text",
    },
    "reuters50k": {
        "dim_group": "medium",
        "size_group": "large",
        "k_group": "few",
        "data_type": "text",
    },
    "newsgroups": {
        "dim_group": "medium",
        "size_group": "small",
        "k_group": "many",
        "data_type": "text",
    },
}

INVERT_CVIS = [
    "Davies-Bouldin",
    "Davies-Bouldin starred",
    "Ray-Turi",
    "Kwon",
    "Tang",

    "DBCV",
    "SD1",
    "SD2",
    "S_Dbw",
    "Vsv1",
    "Vsv2",

    "C Index",
    "McClain-Rao",

    "BIC",
    "Shannon entropy",
    "ICL",

    "Sym-Davies-Bouldin",
    ]

CVI = [
    "Ball-Hall",
    "Calinski-Harabasz",
    "Davies-Bouldin",
    "Davies-Bouldin starred",
    "Hartigan",
    "Kwon",
    "Partition Separation",
    "PBM",
    "R-Squared",
    "Ratkowsky-Lance",
    "Ray-Turi",
    "Score Function",
    "Tang",
    "Trace_W",
    "Wemmert-Gancarski",
    "Wemmert-Gancarski starred",
    "Zalik SV",

    "DBCV",
    "SD1",
    "SD2",
    "S_Dbw",
    "Vsv1",
    "Vsv2",

    "C Index",
    "Dunn",
    "McClain-Rao",
    "Point Biserial",
    "Silhouette",
    "Shannon Entropy",
    "BIC",
    "ICL",

    "Sym",
    "Sym-Davies-Bouldin",
    "Sym-Dunn",
]

META_COLS = ["dataset", "size", "model", "architecture", "n_clusters"]
EXT_COLS = ["UCA", "NMI", "ARI"]