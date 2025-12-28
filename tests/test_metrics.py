import numpy as np
import pytest

from create_embeddings import symmetric_embedding, asymmetric_embedding
from expected import EXPECTED_SYMMETRIC, EXPECTED_ASYMMETRIC
from utils.extract_metrics import Dispersion



@pytest.mark.parametrize(
    "metric_symmetric, expected_symmetric",
    EXPECTED_SYMMETRIC.items()
)

def test_symmetric(metric_symmetric, expected_symmetric):
    embedding, label = symmetric_embedding()
    dispersion = Dispersion(embedding, label)
    value = getattr(dispersion, metric_symmetric)()
    assert np.isclose(value, expected_symmetric, atol=1e-6)


@pytest.mark.parametrize(
    "metric_asymmetric, expected_asymmetric",
    EXPECTED_ASYMMETRIC.items()
)

def test_asymmetric(metric_asymmetric, expected_asymmetric):
    embedding, label = asymmetric_embedding()
    dispersion = Dispersion(embedding, label)
    value = getattr(dispersion, metric_asymmetric)()
    assert np.isclose(value, expected_asymmetric, atol=1e-6)