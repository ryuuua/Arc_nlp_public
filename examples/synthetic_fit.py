"""Fit and replay a small synthetic AA example using the portable package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_module

from neureps_arc_nlp import load_fit
from neureps_arc_nlp.app import run_fit
from neureps_arc_nlp.metrics import compute_reconstruction_metrics


def run_example(profile: str, output: Path) -> dict[str, object]:
    """Write a fresh fit and verify saved-coordinate inference on Validation."""
    output = output.resolve()
    input_path = output.with_name(output.name + "-input.npz")
    if output.exists() or input_path.exists():
        raise FileExistsError("Choose a fresh output path; existing example files are preserved")
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    vertices = np.array([[-1.0, -0.5], [1.0, -0.5], [0.0, 1.0]])
    train_weights = rng.dirichlet(np.ones(3), size=90)
    valid_weights = np.vstack([np.eye(3), rng.dirichlet(np.ones(3), size=27)])
    X_train = train_weights @ vertices
    X_valid = valid_weights @ vertices
    labels_valid = valid_weights.argmax(axis=1)
    np.savez(input_path, X_train=X_train, X_valid=X_valid, labels_valid=labels_valid)
    overrides = [
        f"solver={profile}",
        "k=3",
        "seed=7",
        "solver.backend=numpy",
        "solver.device=cpu",
        "solver.dtype=float64",
        "solver.max_iter=40",
        "solver.method_params.a_max_iter=80",
        "solver.method_params.b_max_iter=80",
    ]
    if profile == "standard_v4":
        overrides += [
            "solver.method_params.max_refinements=0",
            "solver.method_params.perturbation_restarts=0",
        ]
    with initialize_config_module(config_module="neureps_arc_nlp.configs", version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)
    cfg.condition.dataset = "synthetic_triangle"
    cfg.condition.model = "known_coordinates"
    cfg.input.path = str(input_path)
    cfg.output.dir = str(output)
    cfg.output.overwrite = False
    run_fit(cfg, base_dir=output.parent)

    fitted = load_fit(output)
    A_valid = fitted.transform(X_valid)
    assert A_valid.shape == (30, 3)
    assert np.isfinite(A_valid).all()
    assert A_valid.min() >= -1e-8
    np.testing.assert_allclose(A_valid.sum(axis=1), 1.0, atol=1e-6)
    with np.load(output / "fit_arrays.npz", allow_pickle=False) as arrays:
        np.testing.assert_allclose(A_valid, arrays["coefficients_A_valid"], atol=1e-8, rtol=1e-7)
        np.testing.assert_allclose(fitted.archetypes_Z, arrays["archetype_weights_B"] @ X_train, atol=1e-6, rtol=1e-6)
    metrics = compute_reconstruction_metrics(X_valid, A_valid, fitted.archetypes_Z)
    return {"solver_id": fitted.solver_id, "output": str(output), "validation_reconstruction": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=("paper_v2", "standard_v3", "standard_v4"), default="paper_v2")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path("artifacts/examples") / f"synthetic_{args.solver}"
    print(json.dumps(run_example(args.solver, output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
