import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from neureps_arc_nlp.app import run_fit


def test_minimal_numpy_fit_writes_metrics_diagnostics_history_and_arrays(tmp_path: Path) -> None:
    rng = np.random.default_rng(8)
    centers = np.asarray([[-2.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    train_labels = np.repeat(np.arange(3), 5)
    valid_labels = np.repeat(np.arange(3), 2)
    X_train = centers[train_labels] + rng.normal(scale=0.1, size=(len(train_labels), 2))
    X_valid = centers[valid_labels] + rng.normal(scale=0.1, size=(len(valid_labels), 2))
    input_path = tmp_path / "input.npz"
    np.savez(input_path, X_train=X_train, X_valid=X_valid, labels_valid=valid_labels)

    config = OmegaConf.create(
        {
            "condition": {"dataset": "toy", "model": "toy"},
            "k": 3,
            "seed": 0,
            "input": {
                "path": str(input_path),
                "keys": {"train": "X_train", "valid": "X_valid", "labels_valid": "labels_valid"},
            },
            "output": {"dir": "result", "overwrite": False},
            "solver": {
                "id": "aa_projected_gradient_v2",
                "backend": "numpy",
                "device": "cpu",
                "dtype": "float64",
                "n_init": 1,
                "max_iter": 8,
                "tol": 1e-5,
                "method_params": {
                    "b_update": "aa_projected_gradient_v2",
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
                },
            },
        }
    )
    output = run_fit(config, base_dir=tmp_path)

    assert {item.name for item in output.iterdir()} == {
        "fit_arrays.npz",
        "metrics.json",
        "optimization_history.jsonl",
        "solver_diagnostics.json",
    }
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["solver"]["id"] == "aa_projected_gradient_v2"
    assert 0.0 <= metrics["metrics"]["label_alignment_valid"]["normalized_label_coefficient_mi"] <= 1.0
    diagnostics = json.loads((output / "solver_diagnostics.json").read_text())
    assert diagnostics["stationarity_certified"] is False
    assert diagnostics["kkt_certified"] is False
    assert (output / "optimization_history.jsonl").read_text().strip()
    with np.load(output / "fit_arrays.npz", allow_pickle=False) as arrays:
        assert arrays["coefficients_A_valid"].shape == (len(valid_labels), 3)
