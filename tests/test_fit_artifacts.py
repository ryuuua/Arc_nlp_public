import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from neureps_arc_nlp import app, load_fit, transform
from neureps_arc_nlp.dispatch import resolve_solver


def _config(tmp_path, profile):
    rng = np.random.default_rng(19)
    X = rng.normal(size=(14, 3)) + np.asarray([3, -5, 10])
    V = rng.normal(size=(5, 3)) + np.asarray([3, -5, 10])
    source = tmp_path / "input.npz"
    np.savez(source, X_train=X, X_valid=V, labels_valid=np.arange(5) % 2)
    with initialize_config_module(version_base="1.3", config_module="neureps_arc_nlp.configs"):
        cfg = compose(config_name="config", overrides=[
            f"solver={profile}", "k=3", "solver.backend=numpy", "solver.device=cpu",
            "solver.dtype=float64", "solver.max_iter=3", "solver.method_params.a_max_iter=10",
            "solver.method_params.b_max_iter=10", "solver.method_params.min_outer_iterations=2",
        ])
    if profile == "standard_v4":
        cfg.solver.method_params.max_refinements = 0
    cfg.input.path = str(source)
    cfg.output.dir = str(tmp_path / "fit")
    return cfg, X, V


@pytest.mark.parametrize("profile", ["paper_v2", "standard_v3", "standard_v4"])
def test_saved_state_replays_validation_and_new_rows(profile, tmp_path) -> None:
    cfg, X, V = _config(tmp_path, profile)
    root = app.run_fit(cfg)
    fitted = load_fit(root)
    with np.load(root / "fit_arrays.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(fitted.transform(V), archive["coefficients_A_valid"])
    coefficients = transform(fitted, X[:2])
    assert coefficients.shape == (2, 3)
    np.testing.assert_allclose(coefficients.sum(axis=1), 1)
    assert not fitted.archetypes_Z.flags.writeable
    with pytest.raises(ValueError, match="feature count"):
        fitted.transform(np.ones((2, 4)))
    with pytest.raises(ValueError, match="finite real"):
        fitted.transform(np.full((2, 3), np.nan))
    payload = json.loads((root / "metrics.json").read_text())
    diagnostics = json.loads((root / "solver_diagnostics.json").read_text())
    if profile == "paper_v2":
        spec = resolve_solver(cfg.solver.id)
        expected = spec.fit("numpy")(
            X, k=3, seed=0, n_init=1, max_iter=3, tol=cfg.solver.tol, X_val=V,
            method_params=OmegaConf.to_container(cfg.solver.method_params), dtype="float64",
        )
        np.testing.assert_array_equal(fitted.archetypes_Z, expected["archetypes_Z"])
        assert payload["schema_version"] == "neureps_aa_fit.v1"
    if profile == "standard_v4":
        assert diagnostics["stopping_policy"]["max_refinements"] == 0
        assert "restart_summaries" in diagnostics and "final_stationarity_check" in diagnostics
        assert "aa_stopping_policy" in diagnostics["solver_dependency_identities"]
        assert "aa_stopping_policy.py" in payload["provenance"]["solver_sources_sha256"]


def _replace_diagnostics(root, update):
    path = root / "solver_diagnostics.json"
    diagnostics = json.loads(path.read_text())
    update(diagnostics)
    path.write_text(json.dumps(diagnostics))
    metrics_path = root / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["artifacts"]["solver_diagnostics_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    metrics_path.write_text(json.dumps(metrics))


@pytest.mark.parametrize("mutation,match", [
    (lambda d: d.pop("inference_state"), "saved inference state"),
    (lambda d: d["inference_state"].update(max_iter=True), "max_iter"),
    (lambda d: d["inference_state"].update(tol=0), "tol"),
    (lambda d: d["inference_state"].update(source_sha256={}), "source or contract"),
    (lambda d: d["inference_state"].update(solver_id="aa_projected_gradient_v3"), "identities disagree"),
])
def test_loader_rejects_invalid_replay_contract(tmp_path, mutation, match) -> None:
    cfg, _, _ = _config(tmp_path, "paper_v2")
    root = app.run_fit(cfg)
    _replace_diagnostics(root, mutation)
    with pytest.raises(ValueError, match=match):
        load_fit(root)


@pytest.mark.parametrize("filename", ["fit_arrays.npz", "solver_diagnostics.json"])
def test_loader_rejects_modified_artifact_bytes(tmp_path, filename) -> None:
    cfg, _, _ = _config(tmp_path, "paper_v2")
    root = app.run_fit(cfg)
    with (root / filename).open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_fit(root)


@pytest.mark.parametrize("name,replacement,match", [
    ("normalization_raw_scale", np.asarray(0.0), "normalization_raw_scale"),
    ("normalization_center_unit", np.zeros(4), "center shape"),
    ("archetype_weights_B", np.ones((2, 14)), "factor shapes"),
    ("normalization_archetypes", np.full((3, 3), np.nan), "finite real"),
    ("normalization_archetypes", np.full((3, 3), 100.0), "raw and normalized archetypes disagree"),
])
def test_loader_validates_arrays_even_when_hashes_are_updated(tmp_path, name, replacement, match) -> None:
    cfg, _, _ = _config(tmp_path, "paper_v2")
    root = app.run_fit(cfg)
    path = root / "fit_arrays.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays[name] = replacement
    np.savez_compressed(path, **arrays)
    metrics_path = root / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["artifacts"]["arrays_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    metrics_path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match=match):
        load_fit(root)


@pytest.mark.parametrize("profile", ["standard_v3", "standard_v4"])
def test_torch_cpu_uses_strict_runtime_and_reload_can_infer_on_numpy(tmp_path, profile) -> None:
    torch = pytest.importorskip("torch")
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    warn_only_before = torch.is_deterministic_algorithms_warn_only_enabled()
    cfg, _, V = _config(tmp_path, profile)
    cfg.solver.backend = "torch"
    try:
        root = app.run_fit(cfg)
        diagnostics = json.loads((root / "solver_diagnostics.json").read_text())
        assert diagnostics["runtime"]["runtime_contract"] == "aa_torch_runtime_v2"
        assert diagnostics["runtime"]["device_used"] == "cpu"
        assert diagnostics["runtime"]["deterministic_algorithms_enabled"] is True
        fitted = load_fit(root)
        with np.load(root / "fit_arrays.npz", allow_pickle=False) as arrays:
            np.testing.assert_allclose(fitted.transform(V), arrays["coefficients_A_valid"], atol=1e-10, rtol=1e-10)
    finally:
        torch.use_deterministic_algorithms(deterministic_before, warn_only=warn_only_before)


@pytest.mark.parametrize("profile", ["paper_v2", "standard_v3"])
def test_float32_affine_roundoff_does_not_reject_valid_saved_factors(tmp_path, profile, monkeypatch) -> None:
    cfg, _, _ = _config(tmp_path, profile)
    cfg.k = 2
    cfg.solver.dtype = "float32"
    X = np.asarray([[1e8], [3], [-99999992]], dtype=np.float32)
    V = X[:2]
    np.savez(cfg.input.path, X_train=X, X_valid=V, labels_valid=np.asarray([0, 1]))
    spec = resolve_solver(cfg.solver.id)
    original_fit = spec.fit("numpy")

    def fit_with_cancelling_convex_combination(*args, **kwargs):
        result = original_fit(*args, **kwargs)
        # Admissible k=2 factors with a near-zero raw archetype. This reproduces
        # the kernels' independent raw/working-space products without requiring
        # a particular optimizer trajectory to select this convex combination.
        B = np.asarray([[1 / 3, 1 / 3, 1 / 3], [1, 0, 0]], dtype=np.float64)
        X_normalized, _ = spec.module.affine_normalize_numpy(X)
        result["archetype_weights_B"] = B
        result["archetypes_Z"] = B @ np.asarray(X, dtype=np.float64)
        result["normalization_archetypes"] = B @ np.asarray(X_normalized, dtype=np.float64)
        return result

    monkeypatch.setattr(spec.module, original_fit.__name__, fit_with_cancelling_convex_combination)
    root = app.run_fit(cfg)
    with np.load(root / "fit_arrays.npz", allow_pickle=False) as arrays:
        restored_raw = (
            arrays["normalization_archetypes"] * arrays["normalization_centered_scale"]
            + arrays["normalization_center_unit"]
        ) * arrays["normalization_raw_scale"]
        assert not np.allclose(restored_raw, arrays["archetypes_Z"], rtol=1e-5, atol=1e-7)
        np.testing.assert_array_equal(load_fit(root).transform(V), arrays["coefficients_A_valid"])


def test_v4_failure_publishes_only_atomic_evidence_and_preserves_exception(tmp_path, monkeypatch) -> None:
    cfg, X, _ = _config(tmp_path, "standard_v4")
    module = resolve_solver(cfg.solver.id).module
    failure = module.AAOptimizationFailure(
        "test numerical failure", diagnostics={"snapshot_safe": True},
        snapshot={"A": np.ones((14, 3)) / 3, "B": np.ones((3, 14)) / 14, "Z": np.zeros((3, 3))},
    )

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(module, "fit_standard_aa_v4_numpy", fail)
    with pytest.raises(module.AAOptimizationFailure) as caught:
        app.run_fit(cfg)
    assert caught.value is failure
    assert not Path(cfg.output.dir).exists()
    evidence, = (tmp_path / "_failed_fits").iterdir()
    assert not evidence.name.startswith(".pending")
    assert {p.name for p in evidence.iterdir()} == {"manifest.json", "snapshot.npz"}
    manifest = json.loads((evidence / "manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["completed"] is False
    with np.load(evidence / "snapshot.npz", allow_pickle=False) as arrays:
        _, expected = module.affine_normalize_numpy(X)
        np.testing.assert_array_equal(arrays["normalization_center_unit"], expected.center_unit)
        np.testing.assert_array_equal(arrays["A"], failure.snapshot["A"])


def test_failure_writer_error_does_not_mask_original_solver_error(tmp_path, monkeypatch) -> None:
    cfg, _, _ = _config(tmp_path, "standard_v4")
    module = resolve_solver(cfg.solver.id).module
    failure = module.AAOptimizationFailure("original", diagnostics={}, snapshot={})

    def fail(*args, **kwargs):
        raise failure

    def disk_failure(*args, **kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(module, "fit_standard_aa_v4_numpy", fail)
    monkeypatch.setattr(app, "write_failure_evidence", disk_failure)
    with pytest.raises(module.AAOptimizationFailure) as caught:
        app.run_fit(cfg)
    assert caught.value is failure
    assert any("disk unavailable" in note for note in failure.__notes__)
    assert not Path(cfg.output.dir).exists()
