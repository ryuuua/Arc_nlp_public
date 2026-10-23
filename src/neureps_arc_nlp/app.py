from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import hydra
import numpy as np
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from . import __version__
from .artifacts import (
    _canonical_json,
    _sha256_file,
    _write_json,
    inference_state,
    save_fit_arrays,
    write_failure_evidence,
)
from .dispatch import SolverSpec, preflight_solver, resolve_solver
from .metrics import compute_all_metrics


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(_canonical_json(dict(row)) + b"\n")


def _resolve_path(value: Any, base_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _load_input(path: Path, keys: DictConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"input embedding archive does not exist: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = [str(keys.train), str(keys.valid), str(keys.labels_valid)]
        missing = [name for name in required if name not in archive]
        if missing:
            raise ValueError(f"input archive is missing arrays: {missing}")
        X_train = np.asarray(archive[required[0]])
        X_valid = np.asarray(archive[required[1]])
        labels_valid = np.asarray(archive[required[2]])
    if X_train.ndim != 2 or X_valid.ndim != 2:
        raise ValueError("X_train and X_valid must be two-dimensional")
    if X_train.shape[1] != X_valid.shape[1]:
        raise ValueError("X_train and X_valid feature counts differ")
    if labels_valid.ndim not in {1, 2} or labels_valid.shape[0] != X_valid.shape[0]:
        raise ValueError("labels_valid must align with X_valid rows")
    if not np.isfinite(X_train).all() or not np.isfinite(X_valid).all():
        raise ValueError("embedding matrices must be finite")
    return X_train, X_valid, labels_valid


def _normalized_validation(X_valid: np.ndarray, result: Mapping[str, Any]) -> np.ndarray:
    return (
        X_valid / float(result["normalization_raw_scale"]) - np.asarray(result["normalization_center_unit"])
    ) / float(result["normalization_centered_scale"])


def _fit_numpy(
    cfg: DictConfig, X_train: np.ndarray, X_valid: np.ndarray, spec: SolverSpec,
) -> tuple[dict[str, Any], np.ndarray]:
    method_params = OmegaConf.to_container(cfg.solver.method_params, resolve=True)
    result = spec.fit("numpy")(
        X_train,
        k=int(cfg.k),
        seed=int(cfg.seed),
        n_init=int(cfg.solver.n_init),
        max_iter=int(cfg.solver.max_iter),
        tol=float(cfg.solver.tol),
        X_val=X_valid,
        method_params=method_params,
        dtype=str(cfg.solver.dtype),
    )
    A_valid = spec.infer("numpy")(
        _normalized_validation(X_valid, result),
        result["normalization_archetypes"],
        max_iter=result["inference_max_iter"],
        tol=result["inference_tol"],
        step_scale=result["inference_step_scale"],
        dtype=str(cfg.solver.dtype),
    )
    return result, A_valid


def _fit_torch(
    cfg: DictConfig, X_train: np.ndarray, X_valid: np.ndarray, spec: SolverSpec,
) -> tuple[dict[str, Any], np.ndarray]:
    runtime_metadata = None
    if spec.revision != "v2":
        from .aa_runtime_v2 import configure_torch_runtime_v2

        runtime = configure_torch_runtime_v2(
            seed=int(cfg.seed), device=cfg.solver.device,
            deterministic=cfg.solver.get("deterministic", True),
        )
        runtime_metadata = runtime.metadata
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Torch backend requires: python -m pip install -e '.[gpu]'") from exc

    requested_device = str(cfg.solver.device)
    if spec.revision == "v2":
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("solver.device=cuda was requested but CUDA is unavailable")
        device = torch.device(requested_device)
        torch.manual_seed(int(cfg.seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(cfg.seed))
    else:
        device = runtime.device
    dtype = torch.float32 if str(cfg.solver.dtype) == "float32" else torch.float64

    method_params = OmegaConf.to_container(cfg.solver.method_params, resolve=True)
    result = spec.fit("torch")(
        torch,
        X_train,
        k=int(cfg.k),
        seed=int(cfg.seed),
        n_init=int(cfg.solver.n_init),
        max_iter=int(cfg.solver.max_iter),
        tol=float(cfg.solver.tol),
        X_val=X_valid,
        method_params=method_params,
        device=device,
        dtype=dtype,
    )
    A_valid = spec.infer("torch")(
        torch,
        _normalized_validation(X_valid, result),
        result["normalization_archetypes"],
        max_iter=result["inference_max_iter"],
        tol=result["inference_tol"],
        step_scale=result["inference_step_scale"],
        device=device,
        dtype=dtype,
    )
    if runtime_metadata is not None:
        result["extras"]["solver_diagnostics"]["runtime"] = runtime_metadata
    return result, A_valid


def _select_backend(cfg: DictConfig, spec: SolverSpec) -> str:
    backend = str(cfg.solver.backend)
    if backend != "auto":
        if backend not in {"numpy", "torch"}:
            raise ValueError("solver.backend must be auto, numpy, or torch")
        return backend
    if spec.revision != "v2":
        # The explicit device chooses the new runtime without touching CUDA
        # before its deterministic-runtime preconditions are configured.
        return "numpy" if cfg.solver.device == "cpu" else "torch"
    try:
        import torch
    except ImportError:
        return "numpy"
    return "torch" if torch.cuda.is_available() else "numpy"


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def run_fit(cfg: DictConfig, *, base_dir: Path | None = None) -> Path:
    """Fit the selected solver and atomically publish the four success artifacts."""
    root = (base_dir or Path.cwd()).resolve()
    params = OmegaConf.to_container(cfg.solver.method_params, resolve=True)
    if not isinstance(params, dict):
        raise TypeError("solver.method_params must be a mapping")
    spec = resolve_solver(str(cfg.solver.id), params)
    input_path = _resolve_path(cfg.input.path, root)
    output_path = _resolve_path(cfg.output.dir, root)
    X_train, X_valid, labels_valid = _load_input(input_path, cfg.input["keys"])
    if int(cfg.k) > X_train.shape[0]:
        raise ValueError(f"k={int(cfg.k)} exceeds n_train={X_train.shape[0]}")

    preflight_solver(spec, X_train, X_valid, cfg, params)
    backend = _select_backend(cfg, spec)
    resolved_config = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved_config, dict):
        raise TypeError("resolved Hydra config must be a mapping")
    try:
        result, A_valid = (
            _fit_torch(cfg, X_train, X_valid, spec)
            if backend == "torch" else _fit_numpy(cfg, X_train, X_valid, spec)
        )
    except Exception as failure:
        failure_class = getattr(spec.module, "AAOptimizationFailure", None)
        if failure_class is not None and isinstance(failure, failure_class):
            try:
                evidence = write_failure_evidence(
                    output_path, failure, spec=spec, X_train=X_train,
                    input_path=input_path, resolved_config=resolved_config,
                )
                failure.add_note(f"Failure-only evidence saved to {evidence}")
            except Exception as write_error:
                failure.add_note(f"Failure evidence could not be saved: {type(write_error).__name__}: {write_error}")
        raise
    diagnostics = dict(result["extras"]["solver_diagnostics"])
    history = [dict(row) for row in diagnostics.pop("objective_history")]
    metrics = compute_all_metrics(
        X_train=X_train,
        X_valid=X_valid,
        A_train=result["coefficients_A_train"],
        A_valid=A_valid,
        Z=result["archetypes_Z"],
        labels_valid=labels_valid,
    )

    if output_path.exists():
        if not bool(cfg.output.overwrite):
            raise FileExistsError(f"output already exists: {output_path}")
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.", dir=output_path.parent))
    try:
        arrays_path = temporary / "fit_arrays.npz"
        save_fit_arrays(arrays_path, result, A_valid)
        _write_jsonl(temporary / "optimization_history.jsonl", history)
        diagnostics["objective_history_file"] = "optimization_history.jsonl"
        diagnostics["inference_state"] = inference_state(result, spec, str(cfg.solver.dtype))
        _write_json(temporary / "solver_diagnostics.json", diagnostics)

        source_hashes = spec.source_hashes()
        payload = {
            "schema_version": "neureps_aa_fit.v1" if spec.revision == "v2" else "neureps_aa_fit.v2",
            "package_version": __version__,
            "condition": {
                "dataset": str(cfg.condition.dataset),
                "model": str(cfg.condition.model),
                "k": int(cfg.k),
                "seed": int(cfg.seed),
            },
            "solver": {
                "id": spec.solver_id,
                "contract_version": spec.contract_version,
                "backend": backend,
                "device": str(result["device_used"]),
                "dtype": str(result["extras"]["dtype"]),
                "n_iter": int(result["n_iter"]),
                "termination_reason": str(diagnostics["termination_reason"]),
                "converged": bool(diagnostics["converged"]),
                "stationarity_certified": bool(diagnostics["stationarity_certified"]),
                "kkt_certified": bool(diagnostics["kkt_certified"]),
                "final_a_gradient_mapping_norm": float(diagnostics["final_a_gradient_mapping_norm"]),
                "final_b_gradient_mapping_norm": float(diagnostics["final_b_gradient_mapping_norm"]),
                "final_combined_gradient_mapping_norm": float(diagnostics["final_combined_gradient_mapping_norm"]),
            },
            "metrics": metrics,
            "resolved_config": resolved_config,
            "provenance": {
                "input_path": str(input_path),
                "input_sha256": _sha256_file(input_path),
                "resolved_config_sha256": hashlib.sha256(_canonical_json(resolved_config)).hexdigest(),
                "solver_source_sha256": source_hashes[f"{spec.module_name}.py"],
                "solver_sources_sha256": source_hashes,
                "libraries": {
                    "numpy": _version("numpy"),
                    "hydra-core": _version("hydra-core"),
                    "torch": _version("torch"),
                },
            },
            "artifacts": {
                "arrays": "fit_arrays.npz",
                "arrays_sha256": _sha256_file(arrays_path),
                "solver_diagnostics": "solver_diagnostics.json",
                "solver_diagnostics_sha256": _sha256_file(temporary / "solver_diagnostics.json"),
                "optimization_history": "optimization_history.jsonl",
            },
        }
        for key in ("solver_implementation_version", "solver_implementation_digest", "solver_dependency_identities"):
            if key in result["extras"]:
                payload["solver"][key] = result["extras"][key]
        _write_json(temporary / "metrics.json", payload)
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output_path


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    output = run_fit(cfg, base_dir=Path(get_original_cwd()))
    print(json.dumps({"output_dir": str(output), "status": "ok"}, sort_keys=True))


if __name__ == "__main__":
    main()
