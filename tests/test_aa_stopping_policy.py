"""Independent tests for the new policy; no historical kernel is modified."""

from __future__ import annotations

import numpy as np
import pytest

from neureps_arc_nlp.aa_stopping_policy import PolicyConfig, resolve_policy, summarize_simplex_block


def check(weights, gradient, projected=None, *, scale=1.0, policy=None):
    return summarize_simplex_block(
        np.asarray(weights, dtype=np.float64),
        np.asarray(gradient, dtype=np.float64),
        np.asarray(weights if projected is None else projected, dtype=np.float64),
        gradient_scale=scale,
        policy=policy or PolicyConfig(),
    )


@pytest.mark.parametrize(
    "params",
    [
        {"stationarity_gap_tol": True}, {"stationarity_gap_tol": 0},
        {"stationarity_gap_tol": float("inf")}, {"stationarity_step_tol": "1e-6"},
        {"stationarity_check_interval": 1.5}, {"stationarity_check_interval": False},
        {"max_refinements": -1}, {"max_refinements": 5}, {"refinement_multiplier": 1},
        {"refinement_multiplier": 5}, {"perturbation_restarts": -1},
        {"perturbation_strength": 0}, {"perturbation_strength": 1.01},
        {"perturbation_strength": float("nan")},
    ],
)
def test_invalid_settings_fail_closed(params):
    with pytest.raises((TypeError, ValueError)):
        resolve_policy(params)


def test_resolved_policy_has_canonical_scalar_types():
    policy = resolve_policy({"max_refinements": np.int64(1), "stationarity_gap_tol": np.float64(1e-5)})
    assert type(policy.max_refinements) is int
    assert type(policy.stationarity_gap_tol) is float
    assert set(policy.as_dict()) == set(PolicyConfig().as_dict())


def test_boundary_stationarity_and_common_gradient_offset():
    for offset in (0, 1e8, -1e8):
        result = check([[1, 0, 0]], [[offset, offset + 2, offset + 5]])
        assert result["passed"]
        assert result["frank_wolfe_gap_max"] == 0


def test_large_inactive_gradient_cannot_hide_active_violation():
    result = check([[0, 1, 0]], [[-1, 0, 1e9]])
    assert result["frank_wolfe_gap_max"] == 1
    assert not result["passed"]


def test_tiny_step_roundoff_does_not_certify_stationarity():
    # Even if an unusably tiny step produced projected == weights, FW rejects.
    result = check([[0.5, 0.5]], [[0, 1]], [[0.5, 0.5]])
    assert result["projected_step_l1_max"] == 0
    assert not result["passed"]


def test_projected_step_is_a_separate_gate():
    result = check([[0.5, 0.5]], [[0, 0]], [[1, 0]])
    assert result["frank_wolfe_gap_max"] == 0
    assert not result["passed"]


def test_single_slow_row_is_not_hidden_by_aggregate_mean():
    weights = np.tile([1.0, 0.0], (10000, 1))
    gradient = np.zeros_like(weights)
    gradient[-1] = [1, 0]
    result = check(weights, gradient)
    assert result["violating_rows"] == 1
    assert not result["passed"]


def test_feasibility_is_independent_of_gap():
    for row in ([0.5, 0.6], [-1e-15, 1 + 1e-15]):
        assert not check([row], [[0, 0]])["passed"]
    assert not check([[0.5, 0.5]], [[0, 0]], [[0.5, 0.6]])["passed"]


def test_fixed_scale_invariance_and_zero_scale_rejection():
    for multiplier in (1e-12, 1.0, 1e12):
        result = check([[0.25, 0.75]], [[0, 2 * multiplier]], scale=multiplier)
        assert result["frank_wolfe_gap_max"] == 1.5
    for scale in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            check([[1, 0]], [[0, 0]], scale=scale)


def test_sample_duplication_gradient_and_weight_scaling():
    weights = np.array([[0.3, 0.7]], dtype=np.float64)
    grad = np.array([[1.0, 4.0]])
    original = check(weights, grad, scale=2.0)
    duplicated = check(np.repeat(weights / 2, 2, axis=1), np.repeat(2 * grad, 2, axis=1), scale=4.0)
    assert duplicated["frank_wolfe_gap_max"] == pytest.approx(original["frank_wolfe_gap_max"])


def test_stationary_collapsed_aa_is_not_a_minimum():
    x = np.array([[-1.0], [1.0]])
    a = b = np.full((2, 2), 0.5)
    z = b @ x
    residual = a @ z - x
    assert check(a, residual @ z.T)["passed"]
    assert check(b, (a.T @ residual) @ x.T)["passed"]
    perturbed = np.array([[0.6, 0.4], [0.4, 0.6]])
    assert 0.5 * np.sum((perturbed @ perturbed @ x - x) ** 2) < 0.5 * np.sum(residual ** 2)


def test_nonfinite_and_unsupported_arrays_are_rejected():
    with pytest.raises(FloatingPointError):
        check([[1, 0]], [[np.nan, 0]])
    with pytest.raises(TypeError):
        summarize_simplex_block(
            np.ones((1, 1), dtype=np.float32), np.zeros((1, 1)), np.ones((1, 1)),
            gradient_scale=1, policy=PolicyConfig(),
        )
    with pytest.raises(ValueError):
        check([[1, 0]], [[1]])


def test_torch_cpu_scalar_summary_matches_numpy():
    torch = pytest.importorskip("torch")
    weights = np.array([[0.2, 0.8], [1.0, 0.0]])
    gradient = np.array([[3.0, -2.0], [0.0, 1e4]])
    projected = np.array([[0.1, 0.9], [1.0, 0.0]])
    expected = check(weights, gradient, projected, scale=2.5)
    actual = summarize_simplex_block(
        *[torch.as_tensor(value) for value in (weights, gradient, projected)],
        gradient_scale=2.5, policy=PolicyConfig(), torch=torch,
    )
    assert actual == expected
