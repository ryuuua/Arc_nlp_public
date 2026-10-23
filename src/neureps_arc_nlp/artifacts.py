"""Saved inference state and failure-only evidence for standalone AA fits."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .dispatch import SolverSpec, resolve_solver

INFERENCE_SCHEMA_VERSION = "neureps_aa_inference.v1"
_NORMALIZATION_KEYS = (
    "normalization_raw_scale", "normalization_center_unit",
    "normalization_centered_scale", "normalization_archetypes",
)


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _write_json(path: Path, payload: Any) -> None:
    path.write_bytes(_canonical_json(payload) + b"\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inference_state(result: Mapping[str, Any], spec: SolverSpec, dtype: str) -> dict[str, Any]:
    return {
        "schema_version": INFERENCE_SCHEMA_VERSION,
        "solver_id": spec.solver_id,
        "solver_contract_version": spec.contract_version,
        "source_sha256": spec.source_hashes(),
        "dtype": dtype,
        "max_iter": result["inference_max_iter"],
        "tol": result["inference_tol"],
        "step_scale": result["inference_step_scale"],
        "backend": "numpy",
        "coordinate_space": "affine_normalized_training_space",
    }


def save_fit_arrays(path: Path, result: Mapping[str, Any], A_valid: np.ndarray) -> None:
    arrays = {key: np.asarray(result[key]) for key in (
        "archetypes_Z", "archetype_weights_B", "coefficients_A_train", *_NORMALIZATION_KEYS,
    )}
    arrays["coefficients_A_valid"] = np.asarray(A_valid)
    np.savez_compressed(path, **arrays)


@dataclass(frozen=True)
class SavedFit:
    """A verified saved fit; transform returns mixture coefficients on CPU."""

    solver_id: str
    archetypes_Z: np.ndarray
    normalization_archetypes: np.ndarray
    normalization_center_unit: np.ndarray
    normalization_raw_scale: float
    normalization_centered_scale: float
    inference_max_iter: int
    inference_tol: float
    inference_step_scale: float
    dtype: str

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Infer coefficients using the saved training normalization and budgets."""
        values = np.asarray(X)
        if values.ndim != 2 or values.shape[1] != self.archetypes_Z.shape[1]:
            raise ValueError("X must be a matrix with the saved fit's feature count")
        if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values) or not np.isfinite(values).all():
            raise ValueError("X must contain finite real numbers")
        normalized = (
            values / self.normalization_raw_scale - self.normalization_center_unit
        ) / self.normalization_centered_scale
        return resolve_solver(self.solver_id).infer("numpy")(
            normalized, self.normalization_archetypes,
            max_iter=self.inference_max_iter, tol=self.inference_tol,
            step_scale=self.inference_step_scale, dtype=self.dtype,
        )


def transform(fit: SavedFit, X: np.ndarray) -> np.ndarray:
    """Infer coefficients with a fit returned by :func:`load_fit`."""
    return fit.transform(X)


def load_fit(directory: str | Path) -> SavedFit:
    """Load a successful fit, checking array bytes, solver identity and state.

    Older archives without inference state must be refitted; normalization is
    never estimated from held-out data. CPU replay need not be bitwise equal
    to a fit's original Torch inference.
    """
    root = Path(directory)
    metrics = json.loads((root / "metrics.json").read_text())
    if metrics.get("schema_version") not in {"neureps_aa_fit.v1", "neureps_aa_fit.v2"}:
        raise ValueError("unsupported saved fit schema")
    diagnostics_path = root / "solver_diagnostics.json"
    expected_diagnostics_hash = metrics.get("artifacts", {}).get("solver_diagnostics_sha256")
    if _sha256_file(diagnostics_path) != expected_diagnostics_hash:
        raise ValueError("saved diagnostics hash mismatch")
    diagnostics = json.loads(diagnostics_path.read_text())
    state = diagnostics.get("inference_state")
    if not isinstance(state, dict) or state.get("schema_version") != INFERENCE_SCHEMA_VERSION:
        raise ValueError("fit has no supported saved inference state; rerun the fit with version 0.3 or later")
    spec = resolve_solver(state.get("solver_id"))
    if metrics.get("solver", {}).get("id") != spec.solver_id or diagnostics.get("solver_id") != spec.solver_id:
        raise ValueError("saved solver identities disagree")
    if (
        state.get("solver_contract_version") != spec.contract_version
        or state.get("source_sha256") != spec.source_hashes()
    ):
        raise ValueError("saved solver source or contract differs from the installed implementation")
    if metrics["solver"].get("contract_version") != spec.contract_version:
        raise ValueError("saved solver contract identities disagree")
    if diagnostics.get("solver_contract_version") != spec.contract_version:
        raise ValueError("saved diagnostic contract identity disagrees")
    if state.get("backend") != "numpy" or state.get("coordinate_space") != "affine_normalized_training_space":
        raise ValueError("unsupported saved inference coordinate or backend contract")
    dtype = state.get("dtype")
    if dtype not in {"float32", "float64"} or (spec.revision == "v4" and dtype != "float64"):
        raise ValueError("invalid inference dtype")
    if metrics["solver"].get("dtype") != dtype:
        raise ValueError("saved inference and fit dtypes disagree")
    if type(state.get("max_iter")) is not int or state["max_iter"] <= 0:
        raise ValueError("invalid inference max_iter")
    for key in ("tol", "step_scale"):
        value = state.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"invalid inference {key}")
    arrays_path = root / "fit_arrays.npz"
    if _sha256_file(arrays_path) != metrics.get("artifacts", {}).get("arrays_sha256"):
        raise ValueError("saved fit array hash mismatch")
    with np.load(arrays_path, allow_pickle=False) as archive:
        required = {"archetypes_Z", "archetype_weights_B", "coefficients_A_train", "coefficients_A_valid",
                    *_NORMALIZATION_KEYS}
        if not required.issubset(archive.files):
            raise ValueError("saved fit is missing inference arrays")
        arrays = {name: np.array(archive[name], copy=True) for name in required}
    for name, values in arrays.items():
        if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values) or not np.isfinite(values).all():
            raise ValueError(f"saved array {name} must contain finite real numbers")
    Z = arrays["archetypes_Z"]
    if Z.ndim != 2 or min(Z.shape) < 1:
        raise ValueError("saved archetypes must be a nonempty matrix")
    if arrays["normalization_archetypes"].shape != Z.shape:
        raise ValueError("saved normalized archetype shape differs")
    if arrays["normalization_center_unit"].shape != (Z.shape[1],):
        raise ValueError("saved normalization center shape differs")
    for key in ("normalization_raw_scale", "normalization_centered_scale"):
        if arrays[key].shape != () or float(arrays[key]) <= 0:
            raise ValueError(f"invalid saved {key}")
    raw_scale = float(arrays["normalization_raw_scale"])
    centered_scale = float(arrays["normalization_centered_scale"])
    center = arrays["normalization_center_unit"]
    # Raw and normalized factors are computed separately by the kernels.
    # Compare in working coordinates: raw-space cancellation can amplify
    # ordinary float32 rounding by the training scale. The centering term
    # accounts for precision lost when a small spread sits on a large offset.
    with np.errstate(over="ignore", invalid="ignore"):
        normalized_Z = (Z / raw_scale - center) / centered_scale
        rounding_bound = 16 * np.finfo(dtype).eps * (1 + float(np.max(np.abs(center))) / centered_scale)
    if (
        not np.isfinite(normalized_Z).all()
        or not np.isfinite(rounding_bound)
        or not np.allclose(
            normalized_Z, arrays["normalization_archetypes"],
            rtol=16 * np.finfo(dtype).eps, atol=rounding_bound,
        )
    ):
        raise ValueError("saved raw and normalized archetypes disagree")
    B = arrays["archetype_weights_B"]
    A = arrays["coefficients_A_train"]
    Av = arrays["coefficients_A_valid"]
    if A.ndim != 2 or Av.ndim != 2 or B.shape != (Z.shape[0], A.shape[0]):
        raise ValueError("saved factor shapes disagree")
    if A.shape[1] != Z.shape[0] or Av.shape[1] != Z.shape[0]:
        raise ValueError("saved coefficient and archetype counts disagree")
    for values in (A, Av, B):
        if np.any(values < -1e-6) or not np.allclose(values.sum(axis=1), 1, atol=1e-5, rtol=0):
            raise ValueError("saved coefficients must lie on row simplices")
    for values in arrays.values():
        values.setflags(write=False)
    return SavedFit(
        solver_id=spec.solver_id, archetypes_Z=Z,
        normalization_archetypes=arrays["normalization_archetypes"],
        normalization_center_unit=arrays["normalization_center_unit"],
        normalization_raw_scale=float(arrays["normalization_raw_scale"]),
        normalization_centered_scale=float(arrays["normalization_centered_scale"]),
        inference_max_iter=state["max_iter"], inference_tol=state["tol"],
        inference_step_scale=state["step_scale"], dtype=dtype,
    )


def write_failure_evidence(
    output_path: Path, failure: Any, *, spec: SolverSpec, X_train: np.ndarray,
    input_path: Path, resolved_config: Mapping[str, Any],
) -> Path:
    """Atomically publish unique failure evidence outside successful fit paths."""
    _, normalization = spec.module.affine_normalize_numpy(np.asarray(X_train, dtype=np.float64))
    destination_root = output_path.parent / "_failed_fits"
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / f"failure-{uuid.uuid4().hex}"
    temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=destination_root))
    try:
        snapshot_path = temporary / "snapshot.npz"
        np.savez_compressed(
            snapshot_path, **failure.snapshot,
            normalization_raw_scale=np.asarray(normalization.raw_scale),
            normalization_center_unit=np.asarray(normalization.center_unit),
            normalization_centered_scale=np.asarray(normalization.centered_scale),
        )
        _write_json(temporary / "manifest.json", {
            "schema_version": "neureps_aa_failure.v1", "status": "failed", "completed": False,
            "solver_id": spec.solver_id, "solver_source_sha256": spec.source_hashes(),
            "input_sha256": _sha256_file(input_path), "input_shape": list(X_train.shape),
            "error_type": type(failure).__name__, "error": str(failure),
            "diagnostics": failure.diagnostics, "error_notes": list(getattr(failure, "__notes__", [])),
            "snapshot_coordinate_space": "affine_normalized_training_space",
            "snapshot_sha256": _sha256_file(snapshot_path), "resolved_config": dict(resolved_config),
        })
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination
