from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest

from neureps_arc_nlp import standard_aa_v3 as old_solver
from neureps_arc_nlp import standard_aa_v4 as solver
from neureps_arc_nlp.aa_stopping_policy import PolicyConfig


def _fit(X=None, *, torch_backend=False, **overrides):
    values = np.random.default_rng(19).normal(size=(14, 4)) if X is None else X
    options = dict(k=3, seed=0, n_init=1, max_iter=8, tol=1e-5, X_val=None)
    params = dict(a_max_iter=4, b_max_iter=4, a_tol=1e-6, b_tol=1e-6)
    params.update(overrides.pop("method_params", {}))
    options.update(overrides)
    if torch_backend:
        torch = pytest.importorskip("torch")
        return solver.fit_standard_aa_v4_torch(
            torch,
            values,
            **options,
            method_params=params,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
    return solver.fit_standard_aa_v4_numpy(values, **options, method_params=params, dtype="float64")


def _diag(payload):
    return payload["extras"]["solver_diagnostics"]


def test_real_result_is_checked_and_preserves_public_factor_contract():
    X = np.random.default_rng(19).normal(size=(14, 4))
    payload = _fit(X)
    A, B, Z = (payload[key] for key in ("coefficients_A_train", "archetype_weights_B", "archetypes_Z"))
    np.testing.assert_allclose(Z, B @ X, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(A.sum(1), 1.0, atol=1e-12)
    np.testing.assert_allclose(B.sum(1), 1.0, atol=1e-12)
    assert payload["train_mse"] == pytest.approx(np.mean(np.sum((X - A @ Z) ** 2, axis=1)))
    diag = solver.read_standard_aa_diagnostics(payload["extras"])
    assert diag["termination_reason"] in {"max_iterations", "first_order_stationary", "stalled_nonstationary"}
    assert diag["final_stationarity_check"]["passed"] is diag["stationarity_check_passed"]
    assert diag["convergence_criterion"] == "projected_gradient_and_frank_wolfe_v1"
    assert not any(diag[key] for key in ("stationarity_certified", "kkt_certified", "global_optimality_certified"))
    assert diag["gradient_mapping_used_for_stopping"] is True
    assert diag["total_outer_iterations"] == payload["n_iter"]
    assert diag["total_diagnostic_checks"] == sum(r["diagnostic_checks"] for r in diag["restart_summaries"]) + 1


def test_refinement_is_bounded_and_only_failed_blocks_refine():
    config = solver.StandardAAV4Config(
        objective_patience=1,
        min_outer_iterations=1,
        policy=PolicyConfig(max_refinements=2),
    )
    check = {"passed": False, "a": {"passed": False}, "b": {"passed": True}}
    first = solver._advance_policy(
        check, iteration=1, max_iter=20, plateau_count=1, a_stage=0, b_stage=0, config=config
    )
    assert first == (None, 1, 0, 0)
    second = solver._advance_policy(
        check, iteration=2, max_iter=20, plateau_count=1, a_stage=1, b_stage=0, config=config
    )
    assert second == (None, 2, 0, 0)
    third = solver._advance_policy(
        check, iteration=3, max_iter=20, plateau_count=1, a_stage=2, b_stage=0, config=config
    )
    assert third[0] == "stalled_nonstationary"
    assert (
        solver._advance_policy(
            check,
            iteration=20,
            max_iter=20,
            plateau_count=1,
            a_stage=0,
            b_stage=0,
            config=config,
        )[0]
        is None
    )


def test_false_plateau_refines_without_claiming_stationarity(monkeypatch):
    original_check = solver._stationarity_check_numpy
    original_a = solver._update_a_numpy
    observed = []

    def failed_check(*args, **kwargs):
        result = original_check(*args, **kwargs)
        result["passed"] = False
        result["a"]["passed"] = False
        result["b"]["passed"] = True
        return result

    def counted_a(*args, **kwargs):
        observed.append((kwargs["max_iter"], kwargs["tol"]))
        return original_a(*args, **kwargs)

    monkeypatch.setattr(solver, "_stationarity_check_numpy", failed_check)
    monkeypatch.setattr(solver, "_update_a_numpy", counted_a)
    payload = _fit(
        max_iter=20,
        tol=1.0,
        method_params={
            "objective_patience": 1,
            "min_outer_iterations": 1,
            "max_refinements": 2,
            "refinement_multiplier": 2,
        },
    )
    diag = _diag(payload)
    assert diag["termination_reason"] == "stalled_nonstationary"
    assert diag["stationarity_check_passed"] is False
    assert diag["selected_a_refinement_stage"] == 2
    assert diag["selected_b_refinement_stage"] == 0
    assert payload["n_iter"] <= 20
    assert (8, 0.0) in observed and (16, 0.0) in observed
    assert diag["total_refinements"] == 2


def test_force_budget_bypasses_early_stop_and_refinement():
    payload = _fit(
        max_iter=6,
        tol=1.0,
        method_params={
            "force_max_iterations": True,
            "objective_patience": 1,
            "min_outer_iterations": 1,
            "stationarity_gap_tol": 1e6,
            "stationarity_step_tol": 1e6,
        },
    )
    diag = _diag(payload)
    assert payload["n_iter"] == 6
    assert diag["termination_reason"] == "max_iterations"
    assert diag["stationarity_check_passed"] is True
    assert diag["gradient_mapping_used_for_stopping"] is False
    assert diag["converged"] is False
    assert diag["stopping_criterion_satisfied"] is False
    assert diag["total_refinements"] == 0


def test_numerical_pass_before_minimum_does_not_claim_stopping_satisfied():
    result = _fit(
        max_iter=2,
        method_params={
            "min_outer_iterations": 5,
            "stationarity_gap_tol": 1e6,
            "stationarity_step_tol": 1e6,
        },
    )
    diag = _diag(result)
    assert diag["termination_reason"] == "max_iterations"
    assert diag["stationarity_check_passed"] is True
    assert diag["converged"] is False
    assert diag["stopping_criterion_satisfied"] is False


@pytest.mark.parametrize("torch_backend", [False, True])
def test_force_mode_matches_frozen_v3_numerics(torch_backend):
    X = np.random.default_rng(29).normal(size=(12, 4))
    kwargs = dict(k=3, seed=2, n_init=1, max_iter=5, tol=1e-5, X_val=None)
    params = dict(a_max_iter=5, b_max_iter=5, force_max_iterations=True)
    new = _fit(X, torch_backend=torch_backend, **kwargs, method_params=params)
    if torch_backend:
        torch = pytest.importorskip("torch")
        old = old_solver.fit_standard_aa_v3_torch(
            torch,
            X,
            **kwargs,
            method_params=params,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
    else:
        old = old_solver.fit_standard_aa_v3_numpy(X, **kwargs, method_params=params, dtype="float64")
    for key in ("coefficients_A_train", "archetype_weights_B", "archetypes_Z"):
        np.testing.assert_array_equal(new[key], old[key])
    np.testing.assert_array_equal(
        [row["objective"] for row in new["extras"]["optimization_history"]],
        [row["objective"] for row in old["extras"]["optimization_history"]],
    )


def test_all_block_check_uses_same_current_iterate(monkeypatch):
    X = np.random.default_rng(1).normal(size=(7, 3))
    A = np.full((7, 2), 0.5)
    B = np.full((2, 7), 1 / 7)
    B[0] = np.eye(7)[2]
    Z = B @ X
    residual = A @ Z - X
    observed = []
    original = solver.summarize_simplex_block

    def capture(weights, gradient, projected, **kwargs):
        observed.append((weights.copy(), gradient.copy(), kwargs["gradient_scale"]))
        return original(weights, gradient, projected, **kwargs)

    monkeypatch.setattr(solver, "summarize_simplex_block", capture)
    solver._stationarity_check_numpy(
        X,
        A,
        B,
        Z=Z,
        residual=residual,
        x_lipschitz=solver._data_lipschitz_numpy(X)[0],
        policy=PolicyConfig(),
    )
    np.testing.assert_array_equal(observed[0][0], A)
    np.testing.assert_array_equal(observed[1][0], B)
    np.testing.assert_array_equal(observed[0][1], residual @ Z.T)
    np.testing.assert_array_equal(observed[1][1], (A.T @ residual) @ X.T)
    assert observed[0][2] == pytest.approx(np.mean(X**2))
    assert observed[1][2] == pytest.approx(len(X) * np.mean(X**2))


def test_final_payload_rechecks_canonicalized_returned_factors(monkeypatch):
    original_canonicalize = solver._canonicalize_final_b
    original_check = solver._stationarity_check_numpy
    calls = []

    def canonicalize(B):
        result, identity = original_canonicalize(B)
        calls.append(result.copy())
        return result, identity

    def checked(X, A, B, **kwargs):
        result = original_check(X, A, B, **kwargs)
        if calls:
            np.testing.assert_array_equal(B, calls[-1])
        return result

    monkeypatch.setattr(solver, "_canonicalize_final_b", canonicalize)
    monkeypatch.setattr(solver, "_stationarity_check_numpy", checked)
    payload = _fit(max_iter=2)
    assert len(calls) == 2
    assert _diag(payload)["returned_state_postchecks"] == 1


def test_stale_stationary_stop_is_removed_after_returned_state_check():
    config = solver.StandardAAV4Config(min_outer_iterations=1)
    failed = {"passed": False}
    assert (
        solver._checked_reason("first_order_stationary", failed, n_iter=2, max_iter=9, config=config)
        == "stalled_nonstationary"
    )
    assert (
        solver._checked_reason("first_order_stationary", failed, n_iter=9, max_iter=9, config=config)
        == "max_iterations"
    )


def test_postcanonicalization_failure_overrides_optimizer_stationary_stop(monkeypatch):
    original = solver._stationarity_check_numpy
    check_count = 0

    def passed_once(*args, **kwargs):
        nonlocal check_count
        check_count += 1
        check = original(*args, **kwargs)
        passed = check_count == 1
        check["a"]["passed"] = check["b"]["passed"] = check["passed"] = passed
        return check

    monkeypatch.setattr(solver, "_stationarity_check_numpy", passed_once)
    result = _fit(max_iter=8, method_params={"min_outer_iterations": 1, "stationarity_check_interval": 1})
    assert result["n_iter"] == 1
    assert _diag(result)["termination_reason"] == "stalled_nonstationary"
    assert _diag(result)["stationarity_check_passed"] is False
    assert check_count == 3  # optimizer, candidate after canonicalization, returned payload


def test_exact_solution_passes_but_stationary_saddle_is_not_claimed_optimal():
    X = np.asarray([[-1.0], [1.0]])
    exact_A = np.eye(2)
    exact_B = np.eye(2)
    exact = solver._stationarity_check_numpy(
        X,
        exact_A,
        exact_B,
        Z=X,
        residual=np.zeros_like(X),
        x_lipschitz=2.0,
        policy=PolicyConfig(),
    )
    assert exact["passed"] is True
    # Uniform A/B gives zero first-order gradients but positive loss. A joint
    # feasible perturbation reduces it: stationarity is not a local-minimum certificate.
    A = B = np.full((2, 2), 0.5)
    Z = B @ X
    saddle = solver._stationarity_check_numpy(
        X,
        A,
        B,
        Z=Z,
        residual=A @ Z - X,
        x_lipschitz=2.0,
        policy=PolicyConfig(),
    )
    assert saddle["passed"] is True
    shifted = (1 - 0.1) * A + 0.1 * np.eye(2)
    assert np.sum((X - shifted @ shifted @ X) ** 2) < np.sum((X - A @ B @ X) ** 2)


@pytest.mark.parametrize("torch_backend", [False, True])
def test_exact_fit_stops_only_after_minimum_with_true_joint_check(torch_backend):
    payload = _fit(np.asarray([[-1.0], [1.0]]), k=2, torch_backend=torch_backend, max_iter=8)
    diag = _diag(payload)
    assert payload["train_mse"] == pytest.approx(0.0, abs=1e-25)
    assert payload["n_iter"] == 5
    assert diag["termination_reason"] == "first_order_stationary"
    assert diag["converged"] is True
    assert diag["stationarity_check_passed"] is True
    assert diag["final_stationarity_check"]["a"]["passed"] is True
    assert diag["final_stationarity_check"]["b"]["passed"] is True


def test_constant_short_circuit_records_no_executed_attempts():
    diag = _diag(_fit(np.ones((8, 3)), n_init=2, method_params={"perturbation_restarts": 3}))
    assert diag["termination_reason"] == "constant_exact"
    assert diag["stationarity_check_passed"] is True
    assert diag["attempts_requested"] == 5
    assert diag["attempts_completed"] == 0
    assert diag["search_termination_reason"] == "constant_exact_short_circuit"
    assert diag["gradient_mapping_diagnostic_computed"] is False
    for key in (
        "total_outer_iterations",
        "total_a_inner_iterations",
        "total_b_inner_iterations",
        "total_diagnostic_checks",
        "total_refinements",
    ):
        assert diag[key] == 0


@pytest.mark.parametrize("torch_backend", [False, True])
def test_large_constant_shortcut_really_reconstructs_exactly(torch_backend):
    X = np.full((100, 2), 1e100)
    result = _fit(X, k=3, n_init=2, torch_backend=torch_backend, method_params={"perturbation_restarts": 2})
    A, B, Z = (result[key] for key in ("coefficients_A_train", "archetype_weights_B", "archetypes_Z"))
    np.testing.assert_array_equal(B @ X, Z)
    np.testing.assert_array_equal(A @ Z, X)
    assert np.mean(np.sum((X - A @ Z) ** 2, axis=1)) == result["train_mse"] == 0.0
    diag = solver.read_standard_aa_diagnostics(result["extras"])
    assert diag["training_shape"] == [100, 2]
    assert diag["inner_iteration_budgets"] == {"a_max_iter": 4, "b_max_iter": 4}
    assert diag["attempts_requested"] == 4 and diag["attempts_completed"] == 0
    assert diag["gradient_mapping_diagnostic_computed"] is False


def test_normalization_underflow_cannot_be_mislabeled_constant():
    X = np.asarray([[1e308, 1e-308], [1e308, 2e-308]])
    assert not np.all(X == X[0])
    with pytest.raises(FloatingPointError, match="Nonconstant.*collapsed"):
        solver.affine_normalize_numpy(X)
    with pytest.raises(FloatingPointError, match="Nonconstant.*collapsed"):
        solver.preflight_fit_inputs(
            X,
            k=2,
            seed=0,
            n_init=1,
            max_iter=5,
            tol=1e-5,
            X_val=None,
            method_params={},
            dtype="float64",
        )
    with pytest.raises(FloatingPointError, match="Nonconstant.*collapsed"):
        _fit(X, k=2)


@pytest.mark.parametrize("error", [1e-7, 5e-6])
def test_old_feasibility_interval_is_repaired_for_b_but_rejected_for_a(error):
    A = np.full((4, 2), 0.5)
    B = np.full((2, 4), 0.25)
    B[0, 0] += error
    with pytest.raises(FloatingPointError, match="infeasible B"):
        solver._validate_returned_factors(A, B)
    repaired_B, receipt = solver._canonicalize_final_b(B)
    assert receipt["applied"] is True
    assert receipt["trigger_row_sum_error"] == 1e-8
    assert receipt["max_row_sum_error_after"] <= 1e-8
    solver._validate_returned_factors(A, repaired_B)
    invalid_A = A.copy()
    invalid_A[0, 0] += error
    with pytest.raises(FloatingPointError, match="infeasible A"):
        solver._validate_returned_factors(invalid_A, repaired_B)


def test_b_canonicalization_rejects_outside_bounded_repair_window():
    B = np.full((2, 4), 0.25)
    B[0, 0] += 2e-4
    with pytest.raises(FloatingPointError, match="ceiling"):
        solver._canonicalize_final_b(B)


def test_restart_and_perturbation_search_is_deterministic_and_accounted():
    params = {"perturbation_restarts": 2, "force_max_iterations": True}
    first = _fit(n_init=2, max_iter=3, method_params=params)
    again = _fit(n_init=2, max_iter=3, method_params=params)
    assert _diag(first)["restart_summaries"] == _diag(again)["restart_summaries"]
    for key in ("coefficients_A_train", "archetype_weights_B"):
        np.testing.assert_array_equal(first[key], again[key])
    diag = _diag(first)
    summaries = diag["restart_summaries"]
    assert [r["attempt_type"] for r in summaries] == ["independent", "independent", "perturbation", "perturbation"]
    assert diag["total_outer_iterations"] == 12
    assert diag["attempts_completed"] == 4
    selected = summaries[diag["selected_restart_index"]]
    assert selected["returned_normalized_objective"] == min(r["returned_normalized_objective"] for r in summaries)
    assert diag["search_termination_reason"] == "attempt_budget_exhausted"


def test_appending_attempts_preserves_seed_prefix_and_best_loss():
    independent1 = _fit(n_init=1, max_iter=3)
    independent2 = _fit(n_init=2, max_iter=3)
    assert _diag(independent1)["restart_summaries"] == _diag(independent2)["restart_summaries"][:1]
    assert independent2["train_mse"] <= independent1["train_mse"] + 1e-13
    perturbed1 = _fit(n_init=2, max_iter=3, method_params={"perturbation_restarts": 1})
    perturbed2 = _fit(n_init=2, max_iter=3, method_params={"perturbation_restarts": 2})
    assert _diag(perturbed1)["restart_summaries"] == _diag(perturbed2)["restart_summaries"][:3]
    assert perturbed2["train_mse"] <= perturbed1["train_mse"] + 1e-13
    other_n_init = _fit(n_init=1, max_iter=3, method_params={"perturbation_restarts": 1})
    assert (
        _diag(other_n_init)["restart_summaries"][-1]["init_seed"]
        == _diag(perturbed1)["restart_summaries"][-1]["init_seed"]
    )


def test_restart_selection_uses_checked_objective_not_optimizer_score(monkeypatch):
    original = solver._fit_one_numpy
    count = 0

    def misleading_score(*args, **kwargs):
        nonlocal count
        count += 1
        result = original(*args, **kwargs)
        result["objective"] = -1e20 * count
        return result

    monkeypatch.setattr(solver, "_fit_one_numpy", misleading_score)
    result = _fit(n_init=3, max_iter=3)
    summaries = _diag(result)["restart_summaries"]
    expected = min(range(3), key=lambda idx: summaries[idx]["returned_normalized_objective"])
    assert _diag(result)["selected_restart_index"] == expected
    assert all(row["returned_normalized_objective"] >= 0 for row in summaries)


def test_equal_checked_scores_keep_earliest_attempt(monkeypatch):
    original = solver._fit_one_numpy
    first = None

    def same_result(*args, **kwargs):
        nonlocal first
        if first is None:
            first = original(*args, **kwargs)
        return copy.deepcopy(first)

    monkeypatch.setattr(solver, "_fit_one_numpy", same_result)
    result = _fit(n_init=3, max_iter=2)
    assert _diag(result)["selected_restart_index"] == 0
    assert _diag(result)["attempts_completed"] == 3


def test_numpy_torch_cpu_parity_with_search():
    options = dict(n_init=2, max_iter=4, method_params={"perturbation_restarts": 1})
    numpy_result = _fit(**options)
    torch_result = _fit(torch_backend=True, **options)
    for key in ("coefficients_A_train", "archetype_weights_B", "archetypes_Z"):
        np.testing.assert_allclose(numpy_result[key], torch_result[key], rtol=1e-9, atol=1e-11)
    assert _diag(numpy_result)["termination_reason"] == _diag(torch_result)["termination_reason"]
    assert _diag(numpy_result)["selected_restart_index"] == _diag(torch_result)["selected_restart_index"]


@pytest.mark.parametrize("bad", [np.nan, np.inf, complex(1, 2)])
def test_nonfinite_or_complex_input_rejected(bad):
    X = np.asarray([[1.0, bad], [2.0, 3.0]])
    with pytest.raises((ValueError, TypeError)):
        _fit(X, k=2)


def test_float32_and_unknown_parameters_rejected_before_compute(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("numeric preparation was entered")

    monkeypatch.setattr(solver, "affine_normalize_numpy", forbidden)
    with pytest.raises(ValueError, match="float64"):
        solver.fit_standard_aa_v4_numpy(
            np.ones((3, 2)),
            k=2,
            seed=0,
            n_init=1,
            max_iter=1,
            tol=1e-5,
            X_val=None,
            method_params={},
            dtype="float32",
        )
    with pytest.raises(ValueError, match="Unsupported"):
        _fit(method_params={"warm_start": True})


def test_numeric_attempt_failure_is_not_swallowed(monkeypatch):
    original = solver._fit_one_numpy
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise FloatingPointError("deliberate numeric failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(solver, "_fit_one_numpy", fail_second)
    with pytest.raises(FloatingPointError, match="deliberate"):
        _fit(n_init=2, max_iter=2)


@pytest.mark.parametrize(
    "key,value",
    [
        ("stationarity_certified", True),
        ("global_optimality_certified", True),
        ("termination_reason", "objective_plateau"),
        ("solver_dependency_identities", {}),
    ],
)
def test_new_reader_rejects_false_claims_and_identity_drift(key, value):
    extras = copy.deepcopy(_fit(max_iter=2)["extras"])
    extras["solver_diagnostics"][key] = value
    with pytest.raises(ValueError):
        solver.read_standard_aa_diagnostics(extras)


def test_force_policy_controller_never_refines():
    config = replace(solver.StandardAAV4Config(), force_max_iterations=True)
    assert solver._advance_policy(
        {"passed": False},
        iteration=10,
        max_iter=20,
        plateau_count=9,
        a_stage=0,
        b_stage=0,
        config=config,
    ) == (None, 0, 0, 9)
