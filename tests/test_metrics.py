import numpy as np

from neureps_arc_nlp.metrics import compute_label_metrics


def test_perfect_label_anchor_alignment_has_unit_purity_and_mi() -> None:
    labels = np.array([0, 0, 1, 1])
    coefficients = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float)
    metrics = compute_label_metrics(coefficients, labels)
    assert metrics["usage_weighted_purity"] == 1.0
    assert np.isclose(metrics["normalized_label_coefficient_mi"], 1.0)
    assert metrics["global_top_label_probability"] == 0.5


def test_uniform_coefficients_have_zero_directional_mi() -> None:
    labels = np.array([0, 0, 1, 1])
    coefficients = np.full((4, 3), 1 / 3)
    metrics = compute_label_metrics(coefficients, labels)
    assert np.isclose(metrics["usage_weighted_purity"], 0.5)
    assert np.isclose(metrics["normalized_label_coefficient_mi"], 0.0, atol=1e-15)
