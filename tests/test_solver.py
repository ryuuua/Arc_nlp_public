import hashlib
from pathlib import Path

import numpy as np

from neureps_arc_nlp.solver import fit_standard_aa_v2_numpy, infer_standard_aa_v2_numpy

ROOT = Path(__file__).parents[1]
EXPECTED_HASH = "fc7ddb1ce607a3b090e07d2643efba594dc6eb30485b38c2f02bb552c31ae1c2"


def _params() -> dict[str, object]:
    return {
        "a_max_iter": 30,
        "a_tol": 1e-6,
        "b_max_iter": 30,
        "b_tol": 1e-6,
        "a_step_scale": 1.0,
        "b_step_scale": 1.0,
        "backtracking_factor": 0.5,
        "max_backtracks": 20,
        "objective_patience": 2,
        "min_outer_iterations": 2,
        "record_history": True,
        "b_update": "aa_projected_gradient_v2",
    }


def test_solver_is_exact_paper_pinned_file() -> None:
    payload = (ROOT / "src/neureps_arc_nlp/solver.py").read_bytes()
    assert hashlib.sha256(payload).hexdigest() == EXPECTED_HASH


def test_fit_and_validation_inference_are_simplex_feasible() -> None:
    rng = np.random.default_rng(3)
    train = rng.normal(size=(12, 4))
    valid = rng.normal(size=(5, 4))
    result = fit_standard_aa_v2_numpy(
        train,
        k=3,
        seed=0,
        n_init=1,
        max_iter=5,
        tol=1e-5,
        X_val=valid,
        method_params=_params(),
        dtype="float64",
    )
    normalized = (valid / result["normalization_raw_scale"] - result["normalization_center_unit"]) / result[
        "normalization_centered_scale"
    ]
    inferred = infer_standard_aa_v2_numpy(
        normalized,
        result["normalization_archetypes"],
        max_iter=result["inference_max_iter"],
        tol=result["inference_tol"],
        step_scale=result["inference_step_scale"],
        dtype="float64",
    )
    assert inferred.shape == (5, 3)
    assert np.all(inferred >= -1e-9)
    assert np.allclose(inferred.sum(axis=1), 1.0)


def test_constant_matrix_uses_exact_short_circuit() -> None:
    result = fit_standard_aa_v2_numpy(
        np.ones((8, 3)),
        k=2,
        seed=0,
        n_init=1,
        max_iter=5,
        tol=1e-5,
        X_val=None,
        method_params=_params(),
        dtype="float64",
    )
    assert result["train_mse"] == 0.0
    assert result["extras"]["solver_diagnostics"]["termination_reason"] == "constant_exact"
