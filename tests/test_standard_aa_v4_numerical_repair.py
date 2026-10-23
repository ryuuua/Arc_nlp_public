from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from neureps_arc_nlp import standard_aa_v4 as solver
from neureps_arc_nlp.aa_stopping_policy import PolicyConfig


def _backend(torch_backend):
    return pytest.importorskip("torch") if torch_backend else None


def _array(values, torch):
    values = np.asarray(values, dtype=np.float64)
    return values if torch is None else torch.as_tensor(values, dtype=torch.float64)


def _numpy(values, torch):
    return np.asarray(values) if torch is None else values.detach().cpu().numpy()


def _update(block, X, A, B, *, torch, tol=0.0, initial_step=1.0, max_iter=1, max_backtracks=20):
    Z = B @ X
    residual = A @ Z - X
    objective = 0.5 * float(np.sum(_numpy(residual, torch) ** 2))
    kwargs = dict(
        residual=residual,
        objective=objective,
        max_iter=max_iter,
        tol=tol,
        initial_step=initial_step,
        backtracking_factor=0.5,
        max_backtracks=max_backtracks,
    )
    args = (X, Z, A) if block == "a" else (X, A, B)
    if block == "b":
        kwargs["archetypes"] = Z
    function = getattr(solver, f"_update_{block}_{'numpy' if torch is None else 'torch'}")
    return function(*args, **kwargs) if torch is None else function(torch, *args, **kwargs)


def _exact_b_data():
    v = [1.0392900584680278, -1.8506655954583149]
    X = np.vstack([np.tile(v, (6, 1)), [-1.0, 0.7]])
    A = np.zeros((7, 2))
    A[:6, 0] = 1.0
    A[-1, 1] = 1.0
    B = np.zeros((2, 7))
    B[0, :6] = [
        0.2145380617481609,
        0.11897090767846667,
        0.05977354231439358,
        0.1223813424239779,
        0.1495520605608664,
        0.3347840852741345,
    ]
    B[1, -1] = 1.0
    return X, A, B


@pytest.mark.parametrize("torch_backend", [False, True])
@pytest.mark.parametrize("block", ["a", "b"])
def test_exact_feasible_optimum_retains_original_at_zero_algorithmic_tolerance(torch_backend, block):
    torch = _backend(torch_backend)
    if block == "a":
        X = np.vstack([np.eye(3), [0.1, 0.2, 0.7]])
        A = X.copy()
        B = np.c_[np.eye(3), np.zeros(3)]
    else:
        X, A, B = _exact_b_data()
    X, A, B = (_array(v, torch) for v in (X, A, B))
    initial = A if block == "a" else B
    result = _update(block, X, A, B, torch=torch, initial_step=1e-3)
    assert result.available
    assert result.value is initial
    assert result.iterations == result.backtracks == 0
    assert result.objective == 0.0
    np.testing.assert_array_equal(_numpy(result.value, torch), _numpy(initial, torch))
    assert result.numerical_diagnostics["status"] == "numerical_no_op"
    assert result.numerical_diagnostics["algorithmic_tol"] == 0.0
    assert result.numerical_diagnostics["numerical_no_op_tol"] > 0.0


@pytest.mark.parametrize("torch_backend", [False, True])
def test_positive_tolerance_accepts_stable_descent_despite_full_objective_roundoff_increase(torch_backend):
    torch = _backend(torch_backend)
    X = np.array([[0.0, 0.0], [1e-8, 1e-8], [1.0, -1.0 + 1e-9]])
    A = np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2] if torch_backend else [0.7, 0.3]])
    B = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    X, A, B = (_array(v, torch) for v in (X, A, B))
    # This objective has a large irreducible orthogonal floor. The tiny
    # representable coefficient improvement must not disappear by subtraction.
    result = _update("a", X, A, B, torch=torch, tol=1e-6, initial_step=5e15)
    assert result.available and result.iterations == 1 and result.backtracks == 0
    evidence = result.numerical_diagnostics["candidate_evidence"]
    assert evidence["stable_delta"] < -evidence["stable_delta_error_bound"]
    assert evidence["direct_delta"] > 0.0
    assert evidence["displacement"] > 0.01
    # Report the actual recomputed total objective, never a clamped history.
    residual = result.value @ (B @ X) - X
    actual = 0.5 * float(np.sum(_numpy(residual, torch) ** 2))
    assert result.objective == actual


@pytest.mark.parametrize("torch_backend", [False, True])
@pytest.mark.parametrize("block", ["a", "b"])
def test_stable_quadratic_ascent_hidden_by_rounded_objective_is_rejected(torch_backend, block):
    torch = _backend(torch_backend)
    X = _array([[0.0, 0.0], [1e-8, 0.0], [0.0, 1e8]], torch)
    A = _array([[1.0, 0.0], [0.5, 0.5], [0.5, 0.5]], torch)
    B = _array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], torch)
    Z = B @ X
    current = A if block == "a" else B
    candidate = _numpy(current, torch).copy()
    if block == "a":
        candidate[2] = [0.25, 0.75]
    else:
        candidate[0] = [0.5, 0.5, 0.0]
    candidate = _array(candidate, torch)
    residual = A @ Z - X
    record = {}
    result = solver._backtracked_candidate(
        X,
        Z if block == "a" else A,
        current,
        current - candidate,
        block=block,
        preview=candidate,
        preview_delta=0.1,
        preview_raw_finite=True,
        current_z=Z,
        current_residual=residual,
        current_objective=0.5 * solver._squared_norm(residual, torch),
        initial_step=1.0,
        backtracking_factor=0.5,
        max_backtracks=0,
        no_op_tol=solver._numerical_no_op_tolerance(current),
        torch=torch,
        diagnostic_record=record,
    )
    assert result is None
    assert record["status"] == "candidate_exhaustion"
    assert record["backtracks"] == 1
    assert record["candidate_evidence"]["stable_delta"] > 0.0
    assert record["candidate_evidence"]["direct_delta"] == 0.0
    assert not record["candidate_evidence"]["confident_nonincrease"]


@pytest.mark.parametrize("torch_backend", [False, True])
@pytest.mark.parametrize("block", ["a", "b"])
@pytest.mark.parametrize("invalid", ["negative", "nonfinite", "row_sum"])
def test_invalid_current_is_not_silently_projected_or_nooped(torch_backend, block, invalid):
    torch = _backend(torch_backend)
    X = np.eye(2)
    A = B = np.eye(2)
    bad = np.eye(2)
    if invalid == "negative":
        bad[0] = [1.0 + 1e-12, -1e-12]
    elif invalid == "nonfinite":
        bad[0, 0] = np.nan
    else:
        bad[0, 0] = 1.0 + 2e-8
    if block == "a":
        A = bad
    else:
        B = bad
    X, A, B = (_array(v, torch) for v in (X, A, B))
    result = _update(block, X, A, B, torch=torch)
    assert not result.available and result.iterations == 0
    assert result.numerical_diagnostics["status"] == "invalid_current"


@pytest.mark.parametrize("torch_backend", [False, True])
@pytest.mark.parametrize("block", ["a", "b"])
@pytest.mark.parametrize("nonfinite", [False, True])
def test_invalid_candidate_cannot_trigger_a_numerical_noop(torch_backend, block, nonfinite):
    torch = _backend(torch_backend)
    current = _array(np.eye(2), torch)
    candidate = _array([[1.0, np.nan if nonfinite else -1e-17], [0.0, 1.0]], torch)
    record = {}
    result = solver._backtracked_candidate(
        current,
        current,
        current,
        _array(np.zeros((2, 2)), torch),
        block=block,
        preview=candidate,
        preview_delta=0.0,
        preview_raw_finite=True,
        current_z=current,
        current_residual=_array(np.zeros((2, 2)), torch),
        current_objective=0.0,
        initial_step=1.0,
        backtracking_factor=0.5,
        max_backtracks=0,
        no_op_tol=solver._numerical_no_op_tolerance(current),
        torch=torch,
        diagnostic_record=record,
    )
    assert result is None
    assert not record["candidate_evidence"]["simplex_feasible"]


@pytest.mark.parametrize("torch_backend", [False, True])
@pytest.mark.parametrize("block", ["a", "b"])
def test_nonfinite_archetype_cache_is_rejected_before_noop(torch_backend, block):
    torch = _backend(torch_backend)
    X = current = _array(np.eye(2), torch)
    Z = _array(np.full((2, 2), np.nan), torch)
    kwargs = dict(
        residual=_array(np.zeros((2, 2)), torch),
        objective=0.0,
        max_iter=1,
        tol=0.0,
        initial_step=1.0,
        backtracking_factor=0.5,
        max_backtracks=20,
    )
    function = getattr(solver, f"_update_{block}_{'numpy' if torch is None else 'torch'}")
    args = (X, Z, current) if block == "a" else (X, current, current)
    if block == "b":
        kwargs["archetypes"] = Z
    result = function(*args, **kwargs) if torch is None else function(torch, *args, **kwargs)
    assert not result.available
    assert result.numerical_diagnostics["status"] == "invalid_current"


@pytest.mark.parametrize("torch_backend", [False, True])
def test_numerical_noop_does_not_claim_stationarity(torch_backend):
    torch = _backend(torch_backend)
    X = _array(np.eye(2), torch)
    A = _array(np.full((2, 2), 0.5), torch)
    B = _array(np.eye(2), torch)
    result = _update("a", X, A, B, torch=torch, initial_step=1e-30)
    assert result.available and result.value is A
    assert result.numerical_diagnostics["status"] == "numerical_no_op"
    kwargs = dict(Z=B @ X, residual=A @ (B @ X) - X, x_lipschitz=1.0, policy=PolicyConfig())
    check = (
        solver._stationarity_check_numpy(X, A, B, **kwargs)
        if torch is None
        else solver._stationarity_check_torch(torch, X, A, B, **kwargs)
    )
    assert not check["a"]["passed"] and not check["passed"]


@pytest.mark.parametrize("torch_backend", [False, True])
def test_large_orthogonal_irreducible_floor_does_not_prevent_descent(torch_backend):
    torch = _backend(torch_backend)
    X = _array([[0.0, 1e14], [0.0, -1e14], [1.0, 1e14], [1.0, -1e14]], torch)
    A = _array(np.full((4, 2), 0.5), torch)
    B = _array([[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]], torch)
    result = _update("a", X, A, B, torch=torch, initial_step=1.0, max_backtracks=20)
    assert result.available and result.iterations == 1 and result.backtracks == 0
    evidence = result.numerical_diagnostics["candidate_evidence"]
    assert evidence["stable_delta"] == pytest.approx(-0.375)
    assert evidence["stable_delta_error_bound"] < 1e-13
    assert evidence["direct_delta"] == 0.0
    assert evidence["confident_nonincrease"]


@pytest.mark.parametrize("torch_backend", [False, True])
def test_ambiguous_negative_quadratic_delta_is_not_confident_descent(torch_backend):
    torch = _backend(torch_backend)
    residual = _array([[-0.5 - 1e-15]], torch)
    displacement = _array([[1.0]], torch)
    change = solver._quadratic_loss_change(residual, displacement, torch=torch)
    assert -change["stable_delta_error_bound"] < change["stable_delta"] < 0.0
    assert not change["confident_nonincrease"]


@pytest.mark.parametrize("torch_backend", [False, True])
@pytest.mark.parametrize("block_stage", ["initial_a", "b", "polish_a"])
def test_failure_captures_latest_safe_inner_factors_and_replay_metadata(monkeypatch, torch_backend, block_stage):
    torch = _backend(torch_backend)
    X = np.random.default_rng(19).normal(size=(9, 3))
    backend = "numpy" if torch is None else "torch"
    target = f"_update_{'b' if block_stage == 'b' else 'a'}_{backend}"
    original = getattr(solver, target)
    calls = 0
    expected = {}

    def fail_after_progress(*args, **kwargs):
        nonlocal calls
        calls += 1
        update = original(*args, **kwargs)
        if block_stage == "polish_a" and calls == 1:
            return update
        expected["value"] = _numpy(update.value, torch).copy()
        expected["Z"] = _numpy(update.archetypes, torch).copy()
        return replace(update, available=False)

    monkeypatch.setattr(solver, target, fail_after_progress)
    args = dict(
        k=3, seed=0, n_init=2, max_iter=3, tol=1e-5, X_val=None, method_params={"a_max_iter": 3, "b_max_iter": 3}
    )
    with pytest.raises(solver.AAOptimizationFailure, match="backtracking exhausted") as captured:
        if torch is None:
            solver.fit_standard_aa_v4_numpy(X, **args, dtype="float64")
        else:
            solver.fit_standard_aa_v4_torch(torch, X, **args, dtype=torch.float64, device=torch.device("cpu"))
    failure = captured.value
    diag = failure.diagnostics
    json.dumps(diag, allow_nan=False)
    assert set(failure.snapshot) == {"A", "B", "Z"}
    assert all(isinstance(value, np.ndarray) for value in failure.snapshot.values())
    assert diag["snapshot_safe"] and diag["block_stage"] == block_stage
    assert diag["outer_iteration"] == 1
    assert diag["attempt"]["index"] == 0 and diag["attempt"]["type"] == "independent"
    assert len(diag["accepted_history"]) == 1
    assert not diag["stationarity_certified"]
    np.testing.assert_array_equal(failure.snapshot["B" if block_stage == "b" else "A"], expected["value"])
    np.testing.assert_array_equal(failure.snapshot["Z"], expected["Z"])
    norm = diag["normalization"]
    working = (X / norm["raw_scale"] - norm["center_unit"]) / norm["centered_scale"]
    np.testing.assert_allclose(failure.snapshot["B"] @ working, failure.snapshot["Z"], atol=1e-14)
    residual = failure.snapshot["A"] @ failure.snapshot["Z"] - working
    assert diag["snapshot_normalized_objective"] == pytest.approx(0.5 * np.sum(residual**2))
    assert failure.metric_whitener is None


def test_programming_error_is_not_reclassified_as_numerical_failure(monkeypatch):
    def bug(*args, **kwargs):
        raise TypeError("programming sentinel")

    monkeypatch.setattr(solver, "_update_a_numpy", bug)
    with pytest.raises(TypeError, match="programming sentinel"):
        solver.fit_standard_aa_v4_numpy(
            np.eye(3),
            k=2,
            seed=0,
            n_init=1,
            max_iter=1,
            tol=1e-5,
            X_val=None,
            method_params={},
            dtype="float64",
        )


def test_failure_snapshot_capture_error_preserves_original_exception(monkeypatch):
    def copy_error(*args):
        raise MemoryError("copy sentinel")

    monkeypatch.setattr(solver, "_to_numpy", copy_error)
    with pytest.raises(solver.AAOptimizationFailure, match="original sentinel") as captured:
        solver._raise_block_failure(
            "original sentinel",
            solver._BlockUpdate(np.eye(2), 0.0, 0, 1, 0.0, False),
            A=np.eye(2),
            B=np.eye(2),
            Z=np.eye(2),
            block_stage="initial_a",
            iteration=1,
            a_stage=0,
            b_stage=0,
            config=solver.StandardAAV4Config(),
            max_iter=1,
            history=[],
            torch=None,
        )
    assert captured.value.snapshot == {}
    assert captured.value.diagnostics["snapshot_capture_error"] == "MemoryError"
    assert "MemoryError" in captured.value.__notes__[0]


def test_failure_notes_are_implemented_without_python311_dependency():
    failure = solver.AAOptimizationFailure("example", diagnostics={}, snapshot={})
    assert "add_note" in solver.AAOptimizationFailure.__dict__
    failure.add_note("first")
    failure.add_note("second")
    assert failure.__notes__ == ["first", "second"]


@pytest.mark.parametrize("invalid", ["negative_objective", "nonfinite_objective", "nonfinite_residual"])
def test_invalid_cached_state_is_not_saved_as_safe_failure_snapshot(invalid):
    objective = -1.0 if invalid == "negative_objective" else np.nan if invalid == "nonfinite_objective" else 0.0
    residual = np.full((2, 1), np.nan) if invalid == "nonfinite_residual" else np.zeros((2, 1))
    with pytest.raises(solver.AAOptimizationFailure) as captured:
        solver._raise_block_failure(
            "invalid state",
            solver._BlockUpdate(
                np.full((2, 2), 0.5),
                objective,
                0,
                0,
                0.0,
                False,
                np.zeros((2, 1)),
                residual,
                {"status": "invalid_current"},
            ),
            A=np.full((2, 2), 0.5),
            B=np.full((2, 2), 0.5),
            Z=np.zeros((2, 1)),
            block_stage="initial_a",
            iteration=1,
            a_stage=0,
            b_stage=0,
            config=solver.StandardAAV4Config(),
            max_iter=1,
            history=[],
            torch=None,
        )
    assert not captured.value.diagnostics["snapshot_safe"]
    assert captured.value.diagnostics["snapshot_state"] == "invalid_state_not_saved"
    assert captured.value.snapshot == {}
    json.dumps(captured.value.diagnostics, allow_nan=False)
