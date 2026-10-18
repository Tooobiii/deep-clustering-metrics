import numpy as np
import pytest

from create_embedding import symmetric_embedding
from expected import EXPECTED


@pytest.mark.parametrize(
    "metric_class, expected_dict",
    EXPECTED
)

def test_metric(metric_class, expected_dict):
    embedding, label = symmetric_embedding()
    metrics = metric_class(embedding, label)

    print("Klasse:", type(metrics))
    print("Datei:", __import__("inspect").getfile(type(metrics)))
    print("Methode vorhanden:", hasattr(metrics, "shannon_entropy"))
    print("Methoden:", [m for m in dir(metrics) if "shannon" in m])

    for metric_name, expected_value in expected_dict.items():
        value = getattr(metrics, metric_name)()
        assert np.isclose(value, expected_value, atol=1e-6)