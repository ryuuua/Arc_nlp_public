from __future__ import annotations

import gc
import hashlib
import weakref
from pathlib import Path

import numpy as np
import pytest

import neureps_arc_nlp.solver as v2_solver
import neureps_arc_nlp.standard_aa_v3 as solver
from neureps_arc_nlp.solver import fit_standard_aa_v2_numpy
from neureps_arc_nlp.standard_aa_v3 import (
    DIAGNOSTICS_SCHEMA_VERSION,
    SOLVER_CONTRACT_VERSION,
    SOLVER_ID,
    SOLVER_IMPLEMENTATION_DIGEST,
    SOLVER_IMPLEMENTATION_VERSION,
    fit_standard_aa_v3_numpy,
    fit_standard_aa_v3_torch,
    read_standard_aa_diagnostics,
)


def _matrix() -> np.ndarray:
    return np.random.default_rng(20260811).normal(size=(14, 5))


def _params(version: int = 3, **overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "b_update": f"aa_projected_gradient_v{version}",
        "a_max_iter": 30,
        "b_max_iter": 30,
        "a_tol": 1e-8,
        "b_tol": 1e-8,
    }
    values.update(overrides)
    return values


def _fit_numpy(X: np.ndarray, *, n_init: int = 1, **param_overrides: object) -> dict[str, object]:
    return fit_standard_aa_v3_numpy(
        X,
        k=3,
        seed=7,
        n_init=n_init,
        max_iter=12,
        tol=1e-5,
        X_val=None,
        method_params=_params(**param_overrides),
        dtype="float64",
    )


def test_numpy_v3_is_dense_feasible_monotone_and_versioned() -> None:
    X = _matrix()
    payload = _fit_numpy(X)
    A = np.asarray(payload["coefficients_A_train"])
    B = np.asarray(payload["archetype_weights_B"])
    Z = np.asarray(payload["archetypes_Z"])
    extras = payload["extras"]
    diagnostics = extras["solver_diagnostics"]

    assert extras["b_update"] == SOLVER_ID
    assert extras["solver_contract_version"] == SOLVER_CONTRACT_VERSION
    assert extras["solver_implementation_version"] == SOLVER_IMPLEMENTATION_VERSION
    assert extras["solver_implementation_digest"] == SOLVER_IMPLEMENTATION_DIGEST
    assert diagnostics["schema_version"] == DIAGNOSTICS_SCHEMA_VERSION
    assert diagnostics["dense_factorization"] is True
    assert diagnostics["sparsity_assumption"] == "none"
    assert diagnostics["cache_strategy"] == "accepted_dense_z_residual_cache_v1"
    assert diagnostics["persistent_reconstruction_cache"] is False
    assert diagnostics["preview_reused_as_backtrack_zero"] is True
    assert diagnostics["returned_state_objective_coherent"] is True
    assert diagnostics["objective_history_monotone_with_numeric_allowance"] is True
    assert diagnostics["stationarity_certified"] is False
    assert np.all(A >= -1e-12)
    assert np.all(B >= -1e-12)
    np.testing.assert_allclose(A.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(B.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(B @ X, Z, atol=1e-11, rtol=1e-11)

    raw_objective = 0.5 * float(np.sum((A @ Z - X) ** 2, dtype=np.float64))
    history = diagnostics["objective_history"]
    assert history[-1]["objective"] == pytest.approx(raw_objective, rel=0, abs=1e-14)
    assert payload["train_mse"] == pytest.approx(2.0 * raw_objective / X.shape[0])
    objectives = [float(row["normalized_objective"]) for row in history]
    assert all(current <= previous for previous, current in zip(objectives, objectives[1:]))
    assert read_standard_aa_diagnostics(extras) == diagnostics


def test_implementation_digest_tracks_exact_v3_source_artifact() -> None:
    source_bytes = Path(solver.__file__).read_bytes()

    assert SOLVER_IMPLEMENTATION_DIGEST == hashlib.sha256(source_bytes).hexdigest()
    assert SOLVER_IMPLEMENTATION_DIGEST != hashlib.sha256(source_bytes + b"\n").hexdigest()


@pytest.mark.parametrize(
    ("outer_key", "invalid_value", "message"),
    [
        ("solver_contract_version", "aa_b_update_v2", "solver contract versions"),
        (
            "solver_diagnostics_schema_version",
            "arc_nlp_standard_aa_diagnostics_v2",
            "diagnostics schema versions",
        ),
        (
            "solver_implementation_version",
            "standard_aa_v3_corrupt",
            "solver implementation versions",
        ),
        ("solver_implementation_digest", "0" * 64, "solver implementation digests"),
    ],
)
def test_v3_reader_rejects_outer_nested_identity_disagreement(
    outer_key: str,
    invalid_value: str,
    message: str,
) -> None:
    extras = dict(_fit_numpy(_matrix())["extras"])
    extras["solver_diagnostics"] = dict(extras["solver_diagnostics"])
    extras[outer_key] = invalid_value

    with pytest.raises(ValueError, match=message):
        read_standard_aa_diagnostics(extras)


def test_v3_cache_only_kernel_matches_frozen_v2_float64_solution() -> None:
    X = _matrix()
    common = {
        "k": 3,
        "seed": 7,
        "n_init": 1,
        "max_iter": 12,
        "tol": 1e-5,
        "X_val": None,
        "dtype": "float64",
    }
    v2 = fit_standard_aa_v2_numpy(X, method_params=_params(version=2), **common)
    v3 = fit_standard_aa_v3_numpy(X, method_params=_params(version=3), **common)

    for key in ("coefficients_A_train", "archetype_weights_B", "archetypes_Z"):
        np.testing.assert_array_equal(v3[key], v2[key])
    v2_history = [row["normalized_objective"] for row in v2["extras"]["optimization_history"]]
    v3_history = [row["normalized_objective"] for row in v3["extras"]["optimization_history"]]
    np.testing.assert_array_equal(v3_history, v2_history)


@pytest.mark.parametrize(
    ("dtype_name", "atol", "rtol"),
    [("float64", 1e-10, 1e-10), ("float32", 1e-4, 1e-4)],
)
def test_numpy_and_torch_cpu_v3_parity(dtype_name: str, atol: float, rtol: float) -> None:
    torch = pytest.importorskip("torch")
    X = _matrix().astype(dtype_name)
    common = {
        "k": 3,
        "seed": 7,
        "n_init": 1,
        "max_iter": 5,
        "tol": 1e-5,
        "X_val": None,
        "method_params": _params(a_max_iter=8, b_max_iter=8),
    }
    numpy_payload = fit_standard_aa_v3_numpy(X, dtype=dtype_name, **common)
    torch_payload = fit_standard_aa_v3_torch(
        torch,
        X,
        device=torch.device("cpu"),
        dtype=torch.float64 if dtype_name == "float64" else torch.float32,
        **common,
    )

    for key in ("coefficients_A_train", "archetype_weights_B", "archetypes_Z"):
        np.testing.assert_allclose(torch_payload[key], numpy_payload[key], atol=atol, rtol=rtol)
    for payload in (numpy_payload, torch_payload):
        A = np.asarray(payload["coefficients_A_train"])
        B = np.asarray(payload["archetype_weights_B"])
        np.testing.assert_allclose(A.sum(axis=1), 1.0, atol=atol, rtol=0)
        np.testing.assert_allclose(B.sum(axis=1), 1.0, atol=atol, rtol=0)
        history = payload["extras"]["optimization_history"]
        objectives = [float(row["normalized_objective"]) for row in history]
        assert all(current <= previous + atol for previous, current in zip(objectives, objectives[1:]))


def test_backtrack_zero_consumes_preprojected_preview_without_reprojection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    Z = np.asarray([[0.0, 0.0], [1.0, 1.0]])
    current = np.full((4, 2), 0.5)
    residual = current @ Z
    residual -= X
    objective = solver._objective_from_residual_numpy(residual)
    gradient = residual @ Z.T
    step = 0.1
    raw_preview = current - step * gradient
    preview = solver._project_rows_to_simplex_numpy(raw_preview)
    preview_delta = solver._relative_iterate_change_numpy(preview, current)

    projection_calls = 0
    original = solver._project_rows_to_simplex_numpy

    def counted_projection(values: np.ndarray) -> np.ndarray:
        nonlocal projection_calls
        projection_calls += 1
        return original(values)

    monkeypatch.setattr(solver, "_project_rows_to_simplex_numpy", counted_projection)
    update = solver._backtracked_a_candidate_numpy(
        X,
        Z,
        current,
        gradient,
        preview=preview,
        preview_delta=preview_delta,
        preview_raw_finite=True,
        current_residual=residual,
        current_objective=objective,
        initial_step=step,
        backtracking_factor=0.5,
        max_backtracks=2,
        no_op_tol=1e-12,
    )

    assert update is not None
    assert update[3] == 0
    assert projection_calls == 0
    assert update[-1] == pytest.approx(preview_delta)


def test_outer_loop_reuses_accepted_dense_state_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X, _normalization = solver.affine_normalize_numpy(_matrix())
    B = solver._initial_archetype_weights_numpy(X, k=3, seed=7)
    A = np.full((X.shape[0], 3), 1.0 / 3.0)
    x_lipschitz, _method = solver._data_lipschitz_numpy(X)
    original_a = solver._update_a_numpy
    original_b = solver._update_b_numpy
    a_inputs: list[tuple[object, object]] = []
    a_outputs: list[solver._BlockUpdate] = []
    b_inputs: list[tuple[object, object]] = []
    b_outputs: list[solver._BlockUpdate] = []

    def tracked_a(
        X_arg: np.ndarray,
        Z_arg: np.ndarray,
        A_arg: np.ndarray,
        **kwargs: object,
    ) -> solver._BlockUpdate:
        a_inputs.append((Z_arg, kwargs["residual"]))
        result = original_a(X_arg, Z_arg, A_arg, **kwargs)
        a_outputs.append(result)
        return result

    def tracked_b(
        X_arg: np.ndarray,
        A_arg: np.ndarray,
        B_arg: np.ndarray,
        **kwargs: object,
    ) -> solver._BlockUpdate:
        b_inputs.append((kwargs["archetypes"], kwargs["residual"]))
        result = original_b(X_arg, A_arg, B_arg, **kwargs)
        b_outputs.append(result)
        return result

    monkeypatch.setattr(solver, "_update_a_numpy", tracked_a)
    monkeypatch.setattr(solver, "_update_b_numpy", tracked_b)
    solver._fit_one_numpy(
        X,
        A,
        B,
        max_iter=2,
        objective_rtol=1e-12,
        config=solver.StandardAAV3Config(
            a_max_iter=1,
            b_max_iter=1,
            a_tol=1e-12,
            b_tol=1e-12,
            min_outer_iterations=3,
        ),
        x_lipschitz=x_lipschitz,
    )

    # First B's accepted state is passed directly to the following A polish.
    assert a_inputs[1][0] is b_outputs[0].archetypes
    assert a_inputs[1][1] is b_outputs[0].residual
    # That A polish's accepted residual and unchanged Z feed B2.
    assert b_inputs[1][0] is a_outputs[1].archetypes
    assert b_inputs[1][1] is a_outputs[1].residual


@pytest.mark.parametrize("width", (4_096, 4_099, 5_555))
def test_long_dense_float32_numpy_projection_preserves_v2_numerics(width: int) -> None:
    values = np.full((1, width), 1.0 / float(width), dtype=np.float32)

    projected = solver._project_rows_to_simplex_numpy(values)
    expected = v2_solver._project_rows_to_simplex_numpy(values)

    assert projected.dtype == np.float32
    assert np.count_nonzero(projected) == width
    np.testing.assert_array_equal(projected, expected)


@pytest.mark.parametrize("width", (4_096, 4_099, 5_555))
def test_long_dense_float32_torch_cpu_projection_preserves_v2_numerics(
    width: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    values = torch.full((1, width), 1.0 / float(width), dtype=torch.float32)
    cumsum_dtypes: list[object] = []
    original_cumsum = torch.cumsum

    def tracked_cumsum(input_tensor: object, dim: int, **kwargs: object) -> object:
        cumsum_dtypes.append(kwargs.get("dtype"))
        return original_cumsum(input_tensor, dim, **kwargs)

    monkeypatch.setattr(torch, "cumsum", tracked_cumsum)

    projected = solver._project_rows_to_simplex_torch(torch, values)
    expected = v2_solver._project_rows_to_simplex_torch(torch, values)

    assert projected.dtype == torch.float32
    assert int(torch.count_nonzero(projected)) == width
    assert bool(torch.equal(projected, expected))
    assert cumsum_dtypes == [None, None]


def test_n_init_releases_nonselected_full_numpy_restart_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factor_refs: list[
        tuple[
            weakref.ReferenceType[np.ndarray],
            weakref.ReferenceType[np.ndarray],
            weakref.ReferenceType[np.ndarray],
            weakref.ReferenceType[np.ndarray],
        ]
    ] = []
    optimizer_objectives = iter((3.0, 2.0, 1.0))

    def fake_fit(
        X: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        **_kwargs: object,
    ) -> dict[str, object]:
        fake_A = np.full_like(A, 1.0 / float(A.shape[1]))
        fake_B = np.full_like(B, 1.0 / float(B.shape[1]))
        fake_Z = fake_B @ X
        fake_residual = fake_A @ fake_Z
        fake_residual -= X
        objective = next(optimizer_objectives)
        factor_refs.append(
            (
                weakref.ref(fake_A),
                weakref.ref(fake_B),
                weakref.ref(fake_Z),
                weakref.ref(fake_residual),
            )
        )
        return {
            "A": fake_A,
            "B": fake_B,
            "Z": fake_Z,
            "residual": fake_residual,
            "objective": objective,
            "n_iter": 1,
            "termination_reason": "max_iterations",
            "history": [
                solver._history_row(
                    iteration=1,
                    objective=objective,
                    relative=0.0,
                    plateau_count=0,
                )
            ],
        }

    monkeypatch.setattr(solver, "_fit_one_numpy", fake_fit)
    payload = _fit_numpy(_matrix(), n_init=3)
    gc.collect()

    assert factor_refs[0][0]() is None
    assert factor_refs[0][1]() is None
    assert factor_refs[1][0]() is None
    assert factor_refs[1][1]() is None
    assert all(restart_refs[2]() is None for restart_refs in factor_refs)
    assert all(restart_refs[3]() is None for restart_refs in factor_refs)
    assert payload["extras"]["solver_diagnostics"]["restarts_completed"] == 3


def test_n_init_restart_selection_matches_v2_optimizer_objective() -> None:
    payload = _fit_numpy(_matrix(), n_init=3)
    diagnostics = payload["extras"]["solver_diagnostics"]
    summaries = diagnostics["restart_summaries"]

    assert diagnostics["restart_diagnostics_schema_version"] == ("arc_nlp.standard_aa_v3.restarts.v1")
    assert diagnostics["n_init_requested"] == 3
    assert diagnostics["restarts_completed"] == 3
    assert diagnostics["restart_selection_metric"] == ("optimizer_final_normalized_objective")
    assert [row["restart_index"] for row in summaries] == [0, 1, 2]
    selected = diagnostics["selected_restart_index"]
    expected = min(summaries, key=lambda row: row["optimizer_final_normalized_objective"])["restart_index"]
    assert selected == expected
    assert diagnostics["selected_init_seed"] == summaries[selected]["init_seed"]


def test_n_init_three_preserves_v2_selected_solution() -> None:
    X = _matrix().astype(np.float32)
    v2 = fit_standard_aa_v2_numpy(
        X,
        k=3,
        seed=7,
        n_init=3,
        max_iter=12,
        tol=1e-5,
        X_val=None,
        method_params=_params(version=2),
        dtype="float32",
    )
    v3 = fit_standard_aa_v3_numpy(
        X,
        k=3,
        seed=7,
        n_init=3,
        max_iter=12,
        tol=1e-5,
        X_val=None,
        method_params=_params(version=3),
        dtype="float32",
    )

    for key in ("coefficients_A_train", "archetype_weights_B", "archetypes_Z"):
        np.testing.assert_allclose(v3[key], v2[key], atol=2e-5, rtol=2e-5)


def test_payload_reconciles_canonicalized_b_with_terminal_objective() -> None:
    X_raw = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    X_normalized, normalization = solver.affine_normalize_numpy(X_raw)
    A = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.25, 0.75]],
        dtype=np.float32,
    )
    B = np.asarray(
        [[1.0 + 5e-5, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    optimizer_objective, _canonicalization = solver._returned_state_normalized_objective(X_normalized, A, B)
    data_lipschitz, data_method = solver._data_lipschitz_numpy(X_normalized)
    history = [
        solver._history_row(
            iteration=1,
            objective=optimizer_objective + 1e-8,
            relative=0.0,
            plateau_count=1,
        )
    ]
    result = {
        "history": history,
        "termination_reason": "max_iterations",
        "n_iter": 1,
        "data_lipschitz": data_lipschitz,
        "data_lipschitz_method": data_method,
        "restart_summaries": [
            {
                "restart_index": 0,
                "init_seed": 7,
                "optimizer_final_normalized_objective": optimizer_objective + 1e-8,
                "returned_state_normalized_objective": optimizer_objective,
                "outer_iterations": 1,
                "termination_reason": "max_iterations",
                "final_b_canonicalization_applied": True,
            }
        ],
        "restarts_requested": 1,
        "restart_index": 0,
        "init_seed": 7,
    }
    payload = solver._build_payload(
        X_raw=X_raw,
        seed=7,
        X_normalized=X_normalized,
        A=A,
        B=B,
        normalization=normalization,
        max_iter=1,
        objective_rtol=1e-5,
        config=solver.StandardAAV3Config(force_max_iterations=True),
        result=result,
        device_used="cpu",
        dtype_used="float32",
    )

    returned_A = np.asarray(payload["coefficients_A_train"])
    returned_B = np.asarray(payload["archetype_weights_B"])
    returned_Z = np.asarray(payload["archetypes_Z"])
    raw_objective = 0.5 * float(np.sum((returned_A @ returned_Z - X_raw.astype(np.float64)) ** 2))
    diagnostics = payload["extras"]["solver_diagnostics"]
    terminal = payload["extras"]["optimization_history"][-1]

    assert diagnostics["final_b_canonicalization"]["applied"] is True
    np.testing.assert_allclose(returned_B.sum(axis=1), 1.0, atol=1e-12, rtol=0)
    assert diagnostics["returned_state_objective_coherent"] is True
    assert terminal["objective"] == pytest.approx(raw_objective, rel=0, abs=1e-15)
    assert diagnostics["returned_state_public_objective"] == pytest.approx(raw_objective)
    assert terminal["normalized_objective"] == pytest.approx(diagnostics["returned_state_normalized_objective"])


def test_constant_solution_reports_restart_short_circuit_without_fake_runs() -> None:
    X = np.full((8, 3), 4.0)
    payload = fit_standard_aa_v3_numpy(
        X,
        k=2,
        seed=5,
        n_init=4,
        max_iter=10,
        tol=1e-5,
        X_val=None,
        method_params={"b_update": SOLVER_ID},
        dtype="float64",
    )
    diagnostics = payload["extras"]["solver_diagnostics"]

    assert diagnostics["termination_reason"] == "constant_exact"
    assert diagnostics["n_init_requested"] == 4
    assert diagnostics["restarts_completed"] == 0
    assert diagnostics["selected_restart_index"] is None
    assert diagnostics["restart_summaries"] == []
    assert payload["train_mse"] == 0.0
