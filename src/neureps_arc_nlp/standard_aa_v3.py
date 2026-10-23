"""Dense cached projected-gradient solver for standard archetypal analysis.

The v3 selector intentionally leaves :mod:`arc_nlp.standard_aa_v2` unchanged.
It solves the same dense, full-data, row-simplex constrained factor problem,
without assuming that either factor is sparse.  Its numerical kernel removes
only redundant work: the projected preview is the first backtracking
candidate, and accepted archetypes/residuals are carried to the next block
update.  Full reconstruction matrices are deliberately not retained.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating[Any]]

SOLVER_ID = "aa_projected_gradient_v3"
SOLVER_CONTRACT_VERSION = "aa_b_update_v3"
DIAGNOSTICS_SCHEMA_VERSION = "arc_nlp_standard_aa_diagnostics_v3"
OPTIMIZATION_SCOPE = "full_data_l2_factor_problem"
SOLVER_IMPLEMENTATION_VERSION = "standard_aa_v3_dense_pg_cache_v2"
SOLVER_IMPLEMENTATION_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

_EXACT_GRAM_MAX_BYTES = 128 * 1024 * 1024
_POWER_ITERATIONS = 64
_FINAL_B_CANONICALIZATION_SCHEMA_VERSION = "arc_nlp.standard_aa_v3.final_b_canonicalization.v1"
_RESTART_DIAGNOSTICS_SCHEMA_VERSION = "arc_nlp.standard_aa_v3.restarts.v1"
_FINAL_B_CANONICALIZATION_TRIGGER = 1e-5
_FINAL_B_CANONICALIZATION_MAX_ERROR = 1e-4
DEFAULT_OBJECTIVE_RTOL = 1e-5
DEFAULT_OBJECTIVE_PATIENCE = 3
DEFAULT_MIN_OUTER_ITERATIONS = 5
_METHOD_PARAM_NAMES = frozenset(
    {
        "b_update",
        "a_max_iter",
        "a_tol",
        "b_max_iter",
        "b_tol",
        "a_step_scale",
        "b_step_scale",
        "backtracking_factor",
        "max_backtracks",
        "objective_patience",
        "min_outer_iterations",
        "force_max_iterations",
        "record_history",
    }
)


@dataclass(frozen=True)
class StandardAAV3Config:
    a_max_iter: int = 300
    a_tol: float = 1e-6
    b_max_iter: int = 300
    b_tol: float = 1e-6
    a_step_scale: float = 1.0
    b_step_scale: float = 1.0
    backtracking_factor: float = 0.5
    max_backtracks: int = 20
    objective_patience: int = DEFAULT_OBJECTIVE_PATIENCE
    min_outer_iterations: int = DEFAULT_MIN_OUTER_ITERATIONS
    force_max_iterations: bool = False


@dataclass(frozen=True)
class _BlockUpdate:
    value: Any
    objective: float
    iterations: int
    backtracks: int
    accepted_step: float
    available: bool
    archetypes: Any | None = None
    residual: Any | None = None


@dataclass(frozen=True)
class AffineNormalization:
    raw_scale: float
    center_unit: FloatArray
    centered_scale: float
    is_constant: bool


def is_v3_selector(method_params: Any | None) -> bool:
    value = _method_param_value(method_params, "b_update", None)
    return canonicalize_b_update_selector(value) == SOLVER_ID


def canonicalize_b_update_selector(value: Any) -> str:
    """Normalize a selector while requiring a literal string for v3."""

    if type(value) is str:
        return value.strip().lower()
    selector = str(value).strip().lower()
    if selector == SOLVER_ID:
        raise TypeError(
            f"b_update must be a literal string to select {SOLVER_ID!r}; string-coercible objects are unsupported."
        )
    return selector


def validate_method_params(method_params: Any | None) -> StandardAAV3Config:
    """Strictly validate the opt-in v3 numerical parameter surface."""

    params = _method_param_mapping(method_params)
    unknown = sorted(set(params) - _METHOD_PARAM_NAMES)
    if unknown:
        raise ValueError(f"Unsupported aa_projected_gradient_v3 method parameters: {unknown}.")
    if "b_update" in params:
        selector = canonicalize_b_update_selector(params["b_update"])
        if selector != SOLVER_ID:
            raise ValueError(f"b_update must select {SOLVER_ID!r} for the v3 solver.")
    if "record_history" in params:
        _strict_bool(params["record_history"], name="record_history")
    if "force_max_iterations" in params:
        _strict_bool(params["force_max_iterations"], name="force_max_iterations")
    return StandardAAV3Config(
        a_max_iter=_strict_positive_int(params.get("a_max_iter", 300), name="a_max_iter"),
        a_tol=_strict_positive_float(params.get("a_tol", 1e-6), name="a_tol"),
        b_max_iter=_strict_positive_int(params.get("b_max_iter", 300), name="b_max_iter"),
        b_tol=_strict_positive_float(params.get("b_tol", 1e-6), name="b_tol"),
        a_step_scale=_strict_positive_float(params.get("a_step_scale", 1.0), name="a_step_scale"),
        b_step_scale=_strict_positive_float(params.get("b_step_scale", 1.0), name="b_step_scale"),
        backtracking_factor=_strict_exact_float(
            params.get("backtracking_factor", 0.5),
            name="backtracking_factor",
            expected=0.5,
        ),
        max_backtracks=_strict_bounded_nonnegative_int(
            params.get("max_backtracks", 20), name="max_backtracks", maximum=20
        ),
        objective_patience=_strict_positive_int(
            params.get("objective_patience", DEFAULT_OBJECTIVE_PATIENCE),
            name="objective_patience",
        ),
        min_outer_iterations=_strict_positive_int(
            params.get("min_outer_iterations", DEFAULT_MIN_OUTER_ITERATIONS),
            name="min_outer_iterations",
        ),
        force_max_iterations=_strict_bool(params.get("force_max_iterations", False), name="force_max_iterations"),
    )


def preflight_fit_inputs(
    X_train: Any,
    *,
    k: Any,
    seed: Any,
    n_init: Any,
    max_iter: Any,
    tol: Any,
    X_val: Any | None,
    method_params: Any | None,
    dtype: Any,
) -> None:
    """Validate a v3 fit without importing or configuring Torch."""

    validate_method_params(method_params)
    k_value, _seed, _n_init, _max_iter, _rtol = _validate_fit_scalars(
        k=k, seed=seed, n_init=n_init, max_iter=max_iter, tol=tol
    )
    if not isinstance(dtype, str) or dtype not in {"float32", "float64"}:
        raise ValueError("AA v3 dtype must be 'float32' or 'float64'.")
    work_dtype = np.float64 if dtype == "float64" else np.float32
    X_raw = _validate_real_finite_matrix(X_train, name="X_train", dtype=work_dtype)
    _validate_k(k_value, n_samples=int(X_raw.shape[0]))
    normalization = _affine_normalization_metadata_numpy(X_raw)
    _validate_optional_validation_matrix(
        X_val,
        feature_count=int(X_raw.shape[1]),
        dtype=work_dtype,
        normalization=normalization,
    )


# These helpers are an intentionally frozen fork of the v2 numerical contract.
# Keeping the definitions local prevents later v2 maintenance from silently
# changing the opt-in v3 implementation, and conversely leaves v2 immutable.
def affine_normalize_numpy(
    X: FloatArray,
) -> tuple[FloatArray, AffineNormalization]:
    """Return a translation- and scalar-scale-equivariant working matrix."""

    Xf = np.asarray(X)
    raw_scale = float(np.max(np.abs(Xf)))
    if not np.isfinite(raw_scale):
        raise ValueError("X_train raw normalization scale must be finite.")
    if raw_scale == 0.0:
        return np.zeros_like(Xf), AffineNormalization(
            raw_scale=1.0,
            center_unit=np.zeros(Xf.shape[1], dtype=Xf.dtype),
            centered_scale=1.0,
            is_constant=True,
        )
    if bool(np.all(Xf == Xf[0])):
        return np.zeros_like(Xf), AffineNormalization(
            raw_scale=raw_scale,
            center_unit=np.asarray(Xf[0] / raw_scale, dtype=Xf.dtype),
            centered_scale=1.0,
            is_constant=True,
        )
    scaled = Xf / np.asarray(raw_scale, dtype=Xf.dtype)
    center_unit = np.mean(scaled, axis=0, dtype=Xf.dtype)
    centered = scaled - center_unit
    centered_scale = float(np.max(np.abs(centered)))
    if not np.isfinite(centered_scale):
        raise ValueError("X_train centered normalization scale must be finite.")
    if centered_scale == 0.0:
        return np.zeros_like(centered), AffineNormalization(
            raw_scale=raw_scale,
            center_unit=np.asarray(center_unit, dtype=Xf.dtype),
            centered_scale=1.0,
            is_constant=True,
        )
    normalized = centered / np.asarray(centered_scale, dtype=Xf.dtype)
    if not np.isfinite(normalized).all():
        raise ValueError("X_train normalization produced non-finite values.")
    return normalized, AffineNormalization(
        raw_scale=raw_scale,
        center_unit=np.asarray(center_unit, dtype=Xf.dtype),
        centered_scale=centered_scale,
        is_constant=False,
    )


def _affine_normalization_metadata_numpy(X: FloatArray) -> AffineNormalization:
    """Compute normalization metadata in bounded row chunks for preflight."""

    Xf = np.asarray(X)
    chunk_rows = max(1, min(int(Xf.shape[0]), 65_536))
    raw_scale = 0.0
    is_constant = True
    first_row = Xf[0]
    for start in range(0, int(Xf.shape[0]), chunk_rows):
        chunk = Xf[start : start + chunk_rows]
        raw_scale = max(raw_scale, float(np.max(np.abs(chunk))))
        is_constant = is_constant and bool(np.all(chunk == first_row))
    if not np.isfinite(raw_scale):
        raise ValueError("X_train raw normalization scale must be finite.")
    if raw_scale == 0.0:
        return AffineNormalization(
            raw_scale=1.0,
            center_unit=np.zeros(Xf.shape[1], dtype=Xf.dtype),
            centered_scale=1.0,
            is_constant=True,
        )
    if is_constant:
        return AffineNormalization(
            raw_scale=raw_scale,
            center_unit=np.asarray(first_row / raw_scale, dtype=Xf.dtype),
            centered_scale=1.0,
            is_constant=True,
        )
    center_sum = np.zeros(Xf.shape[1], dtype=np.float64)
    for start in range(0, int(Xf.shape[0]), chunk_rows):
        scaled_chunk = np.asarray(Xf[start : start + chunk_rows] / raw_scale, dtype=Xf.dtype)
        if not np.isfinite(scaled_chunk).all():
            raise ValueError("X_train scaling produced non-finite values.")
        center_sum += np.sum(scaled_chunk, axis=0, dtype=np.float64)
    center_unit = np.asarray(center_sum / float(Xf.shape[0]), dtype=Xf.dtype)
    centered_scale = 0.0
    for start in range(0, int(Xf.shape[0]), chunk_rows):
        scaled_chunk = np.asarray(Xf[start : start + chunk_rows] / raw_scale, dtype=Xf.dtype)
        centered_scale = max(centered_scale, float(np.max(np.abs(scaled_chunk - center_unit))))
    if not np.isfinite(centered_scale):
        raise ValueError("X_train centered normalization scale must be finite.")
    if centered_scale == 0.0:
        centered_scale = 1.0
        is_constant = True
    return AffineNormalization(
        raw_scale=raw_scale,
        center_unit=center_unit,
        centered_scale=centered_scale,
        is_constant=is_constant,
    )


def fit_standard_aa_v3_numpy(
    X_train: FloatArray,
    *,
    k: int,
    seed: int,
    n_init: int,
    max_iter: int,
    tol: float,
    X_val: FloatArray | None,
    method_params: Any | None,
    dtype: str,
) -> dict[str, Any]:
    """Fit dense v3 standard AA with NumPy on CPU."""

    config = validate_method_params(method_params)
    k_value, seed_value, n_init_value, max_iter_value, objective_rtol = _validate_fit_scalars(
        k=k, seed=seed, n_init=n_init, max_iter=max_iter, tol=tol
    )
    if not isinstance(dtype, str) or dtype not in {"float32", "float64"}:
        raise ValueError("AA v3 dtype must be 'float32' or 'float64'.")
    work_dtype = np.float64 if dtype == "float64" else np.float32
    X_raw = _validate_real_finite_matrix(X_train, name="X_train", dtype=work_dtype)
    _validate_k(k_value, n_samples=int(X_raw.shape[0]))
    X_normalized, normalization = affine_normalize_numpy(X_raw)
    _validate_optional_validation_matrix(
        X_val,
        feature_count=int(X_raw.shape[1]),
        dtype=work_dtype,
        normalization=normalization,
    )
    if normalization.is_constant:
        return _constant_solution(
            X_raw,
            k=k_value,
            seed=seed_value,
            n_init=n_init_value,
            max_iter=max_iter_value,
            objective_rtol=objective_rtol,
            config=config,
            normalization=normalization,
            device_used="cpu",
            dtype_used=str(X_raw.dtype),
        )

    restart_rng = np.random.default_rng(seed_value)
    x_lipschitz, spectral_method = _data_lipschitz_numpy(X_normalized)
    summaries: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_A_numpy: FloatArray | None = None
    best_B_numpy: FloatArray | None = None
    best_optimizer_objective = float("inf")
    for restart in range(n_init_value):
        init_seed = int(restart_rng.integers(0, 2**31 - 1)) + restart
        B = _initial_archetype_weights_numpy(X_normalized, k=k_value, seed=init_seed)
        A = np.full(
            (X_normalized.shape[0], k_value),
            1.0 / float(k_value),
            dtype=X_normalized.dtype,
        )
        result = _fit_one_numpy(
            X_normalized,
            A,
            B,
            max_iter=max_iter_value,
            objective_rtol=objective_rtol,
            config=config,
            x_lipschitz=x_lipschitz,
        )
        result["data_lipschitz"] = x_lipschitz
        result["data_lipschitz_method"] = spectral_method
        optimizer_objective = float(result["objective"])
        result["restart_index"] = restart
        result["init_seed"] = init_seed
        summaries.append(_restart_summary(restart, init_seed, result))
        if optimizer_objective < best_optimizer_objective:
            best_optimizer_objective = optimizer_objective
            best = {key: value for key, value in result.items() if key not in {"A", "B", "Z", "residual"}}
            best_A_numpy = cast(FloatArray, np.asarray(result["A"]))
            best_B_numpy = cast(FloatArray, np.asarray(result["B"]))
    if best is None or best_A_numpy is None or best_B_numpy is None:
        raise RuntimeError("aa_projected_gradient_v3 did not produce a NumPy solution.")
    best["restart_summaries"] = summaries
    best["restarts_requested"] = n_init_value
    return _build_payload(
        X_raw=X_raw,
        seed=seed_value,
        X_normalized=X_normalized,
        A=best_A_numpy,
        B=best_B_numpy,
        normalization=normalization,
        max_iter=max_iter_value,
        objective_rtol=objective_rtol,
        config=config,
        result=best,
        device_used="cpu",
        dtype_used=str(X_raw.dtype),
    )


def fit_standard_aa_v3_torch(
    torch: Any,
    X_train: FloatArray,
    *,
    k: int,
    seed: int,
    n_init: int,
    max_iter: int,
    tol: float,
    X_val: FloatArray | None,
    method_params: Any | None,
    device: Any,
    dtype: Any,
) -> dict[str, Any]:
    """Fit dense v3 standard AA with Torch on CPU or CUDA."""

    config = validate_method_params(method_params)
    k_value, seed_value, n_init_value, max_iter_value, objective_rtol = _validate_fit_scalars(
        k=k, seed=seed, n_init=n_init, max_iter=max_iter, tol=tol
    )
    numpy_dtype = np.float64 if dtype == torch.float64 else np.float32
    X_raw = _validate_real_finite_matrix(X_train, name="X_train", dtype=numpy_dtype)
    _validate_k(k_value, n_samples=int(X_raw.shape[0]))
    X_normalized_np, normalization = affine_normalize_numpy(X_raw)
    _validate_optional_validation_matrix(
        X_val,
        feature_count=int(X_raw.shape[1]),
        dtype=numpy_dtype,
        normalization=normalization,
    )
    X = torch.as_tensor(
        X_normalized_np if not normalization.is_constant else X_raw,
        device=device,
        dtype=dtype,
    )
    observed_device = X.device
    observed_device_text = str(observed_device)
    observed_dtype = str(X.dtype).removeprefix("torch.")
    if normalization.is_constant:
        return _constant_solution(
            X_raw,
            k=k_value,
            seed=seed_value,
            n_init=n_init_value,
            max_iter=max_iter_value,
            objective_rtol=objective_rtol,
            config=config,
            normalization=normalization,
            device_used=observed_device_text,
            dtype_used=observed_dtype,
        )

    restart_rng = np.random.default_rng(seed_value)
    x_lipschitz, spectral_method = _data_lipschitz_torch(torch, X)
    summaries: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_A_numpy: FloatArray | None = None
    best_B_numpy: FloatArray | None = None
    best_optimizer_objective = float("inf")
    for restart in range(n_init_value):
        init_seed = int(restart_rng.integers(0, 2**31 - 1)) + restart
        B_np = _initial_archetype_weights_numpy(X_normalized_np, k=k_value, seed=init_seed)
        B = torch.as_tensor(B_np, device=observed_device, dtype=X.dtype)
        A = torch.full(
            (X.shape[0], k_value),
            1.0 / float(k_value),
            device=observed_device,
            dtype=X.dtype,
        )
        result = _fit_one_torch(
            torch,
            X,
            A,
            B,
            max_iter=max_iter_value,
            objective_rtol=objective_rtol,
            config=config,
            x_lipschitz=x_lipschitz,
        )
        result["data_lipschitz"] = x_lipschitz
        result["data_lipschitz_method"] = spectral_method
        optimizer_objective = float(result["objective"])
        result["restart_index"] = restart
        result["init_seed"] = init_seed
        summaries.append(_restart_summary(restart, init_seed, result))
        if optimizer_objective < best_optimizer_objective:
            best_optimizer_objective = optimizer_objective
            best = {key: value for key, value in result.items() if key not in {"A", "B", "Z", "residual"}}
            best_A_numpy = cast(FloatArray, np.asarray(result["A"].detach().cpu().numpy()))
            best_B_numpy = cast(FloatArray, np.asarray(result["B"].detach().cpu().numpy()))
    if best is None or best_A_numpy is None or best_B_numpy is None:
        raise RuntimeError("aa_projected_gradient_v3 did not produce a Torch solution.")
    best["restart_summaries"] = summaries
    best["restarts_requested"] = n_init_value
    return _build_payload(
        X_raw=X_raw,
        seed=seed_value,
        X_normalized=X_normalized_np,
        A=best_A_numpy,
        B=best_B_numpy,
        normalization=normalization,
        max_iter=max_iter_value,
        objective_rtol=objective_rtol,
        config=config,
        result=best,
        device_used=observed_device_text,
        dtype_used=observed_dtype,
    )


def infer_standard_aa_v3_numpy(
    X: FloatArray,
    Z: FloatArray,
    *,
    max_iter: int,
    tol: float,
    step_scale: float,
    dtype: str,
) -> FloatArray:
    """Infer dense row-simplex coefficients in saved v3 coordinates."""

    if not isinstance(dtype, str) or dtype not in {"float32", "float64"}:
        raise ValueError("AA v3 dtype must be 'float32' or 'float64'.")
    work_dtype = np.float64 if dtype == "float64" else np.float32
    targets = _validate_real_finite_matrix(X, name="X", dtype=work_dtype)
    archetypes = _validate_real_finite_matrix(Z, name="Z", dtype=work_dtype)
    if targets.shape[1] != archetypes.shape[1]:
        raise ValueError("X and Z must have the same feature dimension.")
    iteration_count = _strict_positive_int(max_iter, name="inference_max_iter")
    tolerance = _strict_positive_float(tol, name="inference_tol")
    step_scale_value = _strict_positive_float(step_scale, name="inference_step_scale")
    coefficients = cast(
        FloatArray,
        np.full(
            (targets.shape[0], archetypes.shape[0]),
            1.0 / float(archetypes.shape[0]),
            dtype=work_dtype,
        ),
    )
    residual = coefficients @ archetypes
    residual -= targets
    objective = _objective_from_residual_numpy(residual)
    update = _update_a_numpy(
        targets,
        archetypes,
        coefficients,
        residual=residual,
        objective=objective,
        max_iter=iteration_count,
        tol=tolerance,
        initial_step=_safe_initial_step(step_scale_value, _largest_gram_eigenvalue_numpy(archetypes)),
        backtracking_factor=0.5,
        max_backtracks=20,
    )
    if not update.available:
        raise RuntimeError("AA v3 transform backtracking exhausted without a safe iterate.")
    return np.asarray(update.value, dtype=work_dtype)


def infer_standard_aa_v3_torch(
    torch: Any,
    X: FloatArray,
    Z: FloatArray,
    *,
    max_iter: int,
    tol: float,
    step_scale: float,
    device: Any,
    dtype: Any,
) -> FloatArray:
    """Torch counterpart of dense v3 coefficient inference."""

    numpy_dtype = np.float64 if dtype == torch.float64 else np.float32
    targets_np = _validate_real_finite_matrix(X, name="X", dtype=numpy_dtype)
    archetypes_np = _validate_real_finite_matrix(Z, name="Z", dtype=numpy_dtype)
    if targets_np.shape[1] != archetypes_np.shape[1]:
        raise ValueError("X and Z must have the same feature dimension.")
    iteration_count = _strict_positive_int(max_iter, name="inference_max_iter")
    tolerance = _strict_positive_float(tol, name="inference_tol")
    step_scale_value = _strict_positive_float(step_scale, name="inference_step_scale")
    targets = torch.as_tensor(targets_np, device=device, dtype=dtype)
    archetypes = torch.as_tensor(archetypes_np, device=device, dtype=dtype)
    coefficients = torch.full(
        (targets.shape[0], archetypes.shape[0]),
        1.0 / float(archetypes.shape[0]),
        device=device,
        dtype=dtype,
    )
    residual = coefficients @ archetypes
    residual.sub_(targets)
    objective = _objective_from_residual_torch(torch, residual)
    update = _update_a_torch(
        torch,
        targets,
        archetypes,
        coefficients,
        residual=residual,
        objective=objective,
        max_iter=iteration_count,
        tol=tolerance,
        initial_step=_safe_initial_step(step_scale_value, _largest_gram_eigenvalue_torch(torch, archetypes)),
        backtracking_factor=0.5,
        max_backtracks=20,
    )
    if not update.available:
        raise RuntimeError("AA v3 transform backtracking exhausted without a safe iterate.")
    return np.asarray(update.value.detach().cpu().numpy())


def read_standard_aa_diagnostics(artifact: Mapping[str, Any] | None) -> dict[str, Any]:
    """Read and strictly validate a v3 diagnostics payload when selected."""

    outer: Mapping[str, Any] = artifact or {}
    source = outer
    nested = outer.get("solver_diagnostics")
    if isinstance(nested, Mapping):
        source = nested
    payload = dict(source)
    selectors = [
        str(value).strip().lower()
        for value in (
            outer.get("b_update"),
            outer.get("solver_id"),
            payload.get("b_update"),
            payload.get("solver_id"),
        )
        if value not in (None, "")
    ]
    if len(set(selectors)) > 1:
        raise ValueError("Standard-AA solver identities disagree.")
    if isinstance(nested, Mapping):
        identity_pairs = (
            (
                "solver_contract_version",
                "solver_contract_version",
                "solver contract versions",
            ),
            (
                "solver_diagnostics_schema_version",
                "schema_version",
                "diagnostics schema versions",
            ),
            (
                "solver_implementation_version",
                "solver_implementation_version",
                "solver implementation versions",
            ),
            (
                "solver_implementation_digest",
                "solver_implementation_digest",
                "solver implementation digests",
            ),
        )
        for outer_key, nested_key, label in identity_pairs:
            outer_value = outer.get(outer_key)
            nested_value = payload.get(nested_key)
            if (
                outer_value not in (None, "")
                and nested_value not in (None, "")
                and str(outer_value) != str(nested_value)
            ):
                raise ValueError(f"Outer and nested Standard-AA {label} disagree.")
    if SOLVER_ID not in selectors:
        return payload
    if payload.get("schema_version") != DIAGNOSTICS_SCHEMA_VERSION:
        raise ValueError("aa_projected_gradient_v3 diagnostics require the exact v3 schema version.")
    if payload.get("solver_id") != SOLVER_ID:
        raise ValueError("v3 diagnostics solver_id mismatch.")
    if payload.get("solver_contract_version") != SOLVER_CONTRACT_VERSION:
        raise ValueError("v3 diagnostics solver_contract_version mismatch.")
    if payload.get("solver_implementation_version") != SOLVER_IMPLEMENTATION_VERSION:
        raise ValueError("v3 diagnostics solver_implementation_version mismatch.")
    if payload.get("solver_implementation_digest") != SOLVER_IMPLEMENTATION_DIGEST:
        raise ValueError("v3 diagnostics solver_implementation_digest mismatch.")
    if payload.get("optimization_scope") != OPTIMIZATION_SCOPE:
        raise ValueError(f"v3 diagnostics optimization_scope must be {OPTIMIZATION_SCOPE!r}.")
    return payload


def _fit_one_numpy(
    X: FloatArray,
    A: FloatArray,
    B: FloatArray,
    *,
    max_iter: int,
    objective_rtol: float,
    config: StandardAAV3Config,
    x_lipschitz: float,
) -> dict[str, Any]:
    Z = B @ X
    residual = A @ Z
    residual -= X
    objective = _objective_from_residual_numpy(residual)
    history = [_history_row(iteration=0, objective=objective, relative=None, plateau_count=0)]
    plateau_count = 0
    termination_reason = "max_iterations"
    n_iter = 0
    for iteration in range(1, max_iter + 1):
        a_update = _BlockUpdate(A, objective, 0, 0, 0.0, True, Z, residual)
        if iteration == 1:
            a_update = _update_a_numpy(
                X,
                Z,
                A,
                residual=residual,
                objective=objective,
                max_iter=config.a_max_iter,
                tol=config.a_tol,
                initial_step=_safe_initial_step(config.a_step_scale, _largest_gram_eigenvalue_numpy(Z)),
                backtracking_factor=config.backtracking_factor,
                max_backtracks=config.max_backtracks,
            )
            if not a_update.available:
                raise RuntimeError("AA v3 A-update backtracking exhausted without a safe iterate.")
            A = np.asarray(a_update.value)
            residual = np.asarray(a_update.residual)
            objective = a_update.objective
        b_update = _update_b_numpy(
            X,
            A,
            B,
            archetypes=Z,
            residual=residual,
            objective=objective,
            max_iter=config.b_max_iter,
            tol=config.b_tol,
            initial_step=_safe_initial_step(
                config.b_step_scale,
                _largest_gram_eigenvalue_numpy(A) * x_lipschitz,
            ),
            backtracking_factor=config.backtracking_factor,
            max_backtracks=config.max_backtracks,
        )
        if not b_update.available:
            raise RuntimeError("AA v3 B-update backtracking exhausted without a safe iterate.")
        B = np.asarray(b_update.value)
        Z = np.asarray(b_update.archetypes)
        residual = np.asarray(b_update.residual)
        objective = b_update.objective
        a_lipschitz = _largest_gram_eigenvalue_numpy(Z)
        polished_a_update = _update_a_numpy(
            X,
            Z,
            A,
            residual=residual,
            objective=objective,
            max_iter=config.a_max_iter,
            tol=config.a_tol,
            initial_step=_safe_initial_step(config.a_step_scale, a_lipschitz),
            backtracking_factor=config.backtracking_factor,
            max_backtracks=config.max_backtracks,
        )
        if not polished_a_update.available:
            raise RuntimeError("AA v3 final A-update backtracking exhausted without a safe iterate.")
        A = np.asarray(polished_a_update.value)
        residual = np.asarray(polished_a_update.residual)
        objective = polished_a_update.objective
        a_update = _combined_block_update(a_update, polished_a_update)
        previous_objective = float(history[-1]["objective"])
        relative = signed_relative_objective_improvement(previous_objective, objective)
        plateau_count = plateau_count + 1 if 0.0 <= relative <= objective_rtol else 0
        n_iter = iteration
        history.append(
            _history_row(
                iteration=iteration,
                objective=objective,
                relative=relative,
                plateau_count=plateau_count,
                a_update=a_update,
                b_update=b_update,
            )
        )
        if (
            not config.force_max_iterations
            and iteration >= config.min_outer_iterations
            and plateau_count >= config.objective_patience
        ):
            termination_reason = "objective_plateau"
            break
    return {
        "A": A,
        "B": B,
        "Z": Z,
        "residual": residual,
        "objective": objective,
        "n_iter": n_iter,
        "termination_reason": termination_reason,
        "history": history,
    }


def _fit_one_torch(
    torch: Any,
    X: Any,
    A: Any,
    B: Any,
    *,
    max_iter: int,
    objective_rtol: float,
    config: StandardAAV3Config,
    x_lipschitz: float,
) -> dict[str, Any]:
    Z = B @ X
    residual = A @ Z
    residual.sub_(X)
    objective = _objective_from_residual_torch(torch, residual)
    history = [_history_row(iteration=0, objective=objective, relative=None, plateau_count=0)]
    plateau_count = 0
    termination_reason = "max_iterations"
    n_iter = 0
    for iteration in range(1, max_iter + 1):
        a_update = _BlockUpdate(A, objective, 0, 0, 0.0, True, Z, residual)
        if iteration == 1:
            a_update = _update_a_torch(
                torch,
                X,
                Z,
                A,
                residual=residual,
                objective=objective,
                max_iter=config.a_max_iter,
                tol=config.a_tol,
                initial_step=_safe_initial_step(config.a_step_scale, _largest_gram_eigenvalue_torch(torch, Z)),
                backtracking_factor=config.backtracking_factor,
                max_backtracks=config.max_backtracks,
            )
            if not a_update.available:
                raise RuntimeError("AA v3 A-update backtracking exhausted without a safe iterate.")
            A = a_update.value
            residual = a_update.residual
            objective = a_update.objective
        b_update = _update_b_torch(
            torch,
            X,
            A,
            B,
            archetypes=Z,
            residual=residual,
            objective=objective,
            max_iter=config.b_max_iter,
            tol=config.b_tol,
            initial_step=_safe_initial_step(
                config.b_step_scale,
                _largest_gram_eigenvalue_torch(torch, A) * x_lipschitz,
            ),
            backtracking_factor=config.backtracking_factor,
            max_backtracks=config.max_backtracks,
        )
        if not b_update.available:
            raise RuntimeError("AA v3 B-update backtracking exhausted without a safe iterate.")
        B = b_update.value
        Z = b_update.archetypes
        residual = b_update.residual
        objective = b_update.objective
        polished_a_update = _update_a_torch(
            torch,
            X,
            Z,
            A,
            residual=residual,
            objective=objective,
            max_iter=config.a_max_iter,
            tol=config.a_tol,
            initial_step=_safe_initial_step(config.a_step_scale, _largest_gram_eigenvalue_torch(torch, Z)),
            backtracking_factor=config.backtracking_factor,
            max_backtracks=config.max_backtracks,
        )
        if not polished_a_update.available:
            raise RuntimeError("AA v3 final A-update backtracking exhausted without a safe iterate.")
        A = polished_a_update.value
        residual = polished_a_update.residual
        objective = polished_a_update.objective
        a_update = _combined_block_update(a_update, polished_a_update)
        previous_objective = float(history[-1]["objective"])
        relative = signed_relative_objective_improvement(previous_objective, objective)
        plateau_count = plateau_count + 1 if 0.0 <= relative <= objective_rtol else 0
        n_iter = iteration
        history.append(
            _history_row(
                iteration=iteration,
                objective=objective,
                relative=relative,
                plateau_count=plateau_count,
                a_update=a_update,
                b_update=b_update,
            )
        )
        if (
            not config.force_max_iterations
            and iteration >= config.min_outer_iterations
            and plateau_count >= config.objective_patience
        ):
            termination_reason = "objective_plateau"
            break
    return {
        "A": A,
        "B": B,
        "Z": Z,
        "residual": residual,
        "objective": objective,
        "n_iter": n_iter,
        "termination_reason": termination_reason,
        "history": history,
    }


def _update_a_numpy(
    X: FloatArray,
    Z: FloatArray,
    A: FloatArray,
    *,
    residual: FloatArray,
    objective: float,
    max_iter: int,
    tol: float,
    initial_step: float,
    backtracking_factor: float,
    max_backtracks: int,
) -> _BlockUpdate:
    current = A
    current_residual = residual
    current_objective = objective
    effective_tol = max(float(tol), 16.0 * float(np.finfo(current.dtype).eps))
    total_backtracks = 0
    accepted_step = initial_step
    for iteration in range(1, max_iter + 1):
        gradient = current_residual @ Z.T
        raw_preview = current - initial_step * gradient
        preview = _project_rows_to_simplex_numpy(raw_preview)
        preview_delta = _relative_iterate_change_numpy(preview, current)
        if preview_delta <= effective_tol:
            return _BlockUpdate(
                current,
                current_objective,
                iteration - 1,
                total_backtracks,
                initial_step,
                True,
                Z,
                current_residual,
            )
        update = _backtracked_a_candidate_numpy(
            X,
            Z,
            current,
            gradient,
            preview=preview,
            preview_delta=preview_delta,
            preview_raw_finite=bool(np.isfinite(raw_preview).all()),
            current_residual=current_residual,
            current_objective=current_objective,
            initial_step=initial_step,
            backtracking_factor=backtracking_factor,
            max_backtracks=max_backtracks,
            no_op_tol=effective_tol,
        )
        if update is None:
            return _BlockUpdate(
                current,
                current_objective,
                iteration - 1,
                total_backtracks + max_backtracks + 1,
                0.0,
                False,
                Z,
                current_residual,
            )
        (
            candidate,
            candidate_objective,
            accepted_step,
            backtracks,
            candidate_residual,
            delta,
        ) = update
        total_backtracks += backtracks
        current = candidate
        current_objective = candidate_objective
        current_residual = candidate_residual
        if delta <= effective_tol:
            return _BlockUpdate(
                current,
                current_objective,
                iteration,
                total_backtracks,
                accepted_step,
                True,
                Z,
                current_residual,
            )
    return _BlockUpdate(
        current,
        current_objective,
        max_iter,
        total_backtracks,
        accepted_step,
        True,
        Z,
        current_residual,
    )


def _update_b_numpy(
    X: FloatArray,
    A: FloatArray,
    B: FloatArray,
    *,
    archetypes: FloatArray,
    residual: FloatArray,
    objective: float,
    max_iter: int,
    tol: float,
    initial_step: float,
    backtracking_factor: float,
    max_backtracks: int,
) -> _BlockUpdate:
    current = B
    current_z = archetypes
    current_residual = residual
    current_objective = objective
    effective_tol = max(float(tol), 16.0 * float(np.finfo(current.dtype).eps))
    total_backtracks = 0
    accepted_step = initial_step
    for iteration in range(1, max_iter + 1):
        gradient = (A.T @ current_residual) @ X.T
        raw_preview = current - initial_step * gradient
        preview = _project_rows_to_simplex_numpy(raw_preview)
        preview_delta = _relative_iterate_change_numpy(preview, current)
        if preview_delta <= effective_tol:
            return _BlockUpdate(
                current,
                current_objective,
                iteration - 1,
                total_backtracks,
                initial_step,
                True,
                current_z,
                current_residual,
            )
        update = _backtracked_b_candidate_numpy(
            X,
            A,
            current,
            gradient,
            preview=preview,
            preview_delta=preview_delta,
            preview_raw_finite=bool(np.isfinite(raw_preview).all()),
            current_z=current_z,
            current_residual=current_residual,
            current_objective=current_objective,
            initial_step=initial_step,
            backtracking_factor=backtracking_factor,
            max_backtracks=max_backtracks,
            no_op_tol=effective_tol,
        )
        if update is None:
            return _BlockUpdate(
                current,
                current_objective,
                iteration - 1,
                total_backtracks + max_backtracks + 1,
                0.0,
                False,
                current_z,
                current_residual,
            )
        (
            candidate,
            candidate_objective,
            accepted_step,
            backtracks,
            candidate_z,
            candidate_residual,
            delta,
        ) = update
        total_backtracks += backtracks
        current = candidate
        current_objective = candidate_objective
        current_z = candidate_z
        current_residual = candidate_residual
        if delta <= effective_tol:
            return _BlockUpdate(
                current,
                current_objective,
                iteration,
                total_backtracks,
                accepted_step,
                True,
                current_z,
                current_residual,
            )
    return _BlockUpdate(
        current,
        current_objective,
        max_iter,
        total_backtracks,
        accepted_step,
        True,
        current_z,
        current_residual,
    )


def _backtracked_a_candidate_numpy(
    X: FloatArray,
    Z: FloatArray,
    current: FloatArray,
    gradient: FloatArray,
    *,
    preview: FloatArray,
    preview_delta: float,
    preview_raw_finite: bool,
    current_residual: FloatArray,
    current_objective: float,
    initial_step: float,
    backtracking_factor: float,
    max_backtracks: int,
    no_op_tol: float,
) -> tuple[FloatArray, float, float, int, FloatArray, float] | None:
    step = initial_step
    allowance = _objective_allowance_numpy(current.dtype, current_objective)
    for backtrack in range(max_backtracks + 1):
        if backtrack == 0:
            if not preview_raw_finite:
                step *= backtracking_factor
                continue
            candidate = preview
            delta = preview_delta
        else:
            raw_candidate = current - step * gradient
            if not np.isfinite(raw_candidate).all():
                step *= backtracking_factor
                continue
            candidate = _project_rows_to_simplex_numpy(raw_candidate)
            delta = _relative_iterate_change_numpy(candidate, current)
        candidate_residual = candidate @ Z
        candidate_residual -= X
        candidate_objective = _objective_from_residual_numpy(candidate_residual)
        if candidate_objective <= current_objective:
            return candidate, candidate_objective, step, backtrack, candidate_residual, delta
        if (
            np.array_equal(candidate, current) or delta <= no_op_tol
        ) and candidate_objective - current_objective <= allowance:
            return current, current_objective, step, backtrack, current_residual, 0.0
        step *= backtracking_factor
    return None


def _backtracked_b_candidate_numpy(
    X: FloatArray,
    A: FloatArray,
    current: FloatArray,
    gradient: FloatArray,
    *,
    preview: FloatArray,
    preview_delta: float,
    preview_raw_finite: bool,
    current_z: FloatArray,
    current_residual: FloatArray,
    current_objective: float,
    initial_step: float,
    backtracking_factor: float,
    max_backtracks: int,
    no_op_tol: float,
) -> tuple[FloatArray, float, float, int, FloatArray, FloatArray, float] | None:
    step = initial_step
    allowance = _objective_allowance_numpy(current.dtype, current_objective)
    for backtrack in range(max_backtracks + 1):
        if backtrack == 0:
            if not preview_raw_finite:
                step *= backtracking_factor
                continue
            candidate = preview
            delta = preview_delta
        else:
            raw_candidate = current - step * gradient
            if not np.isfinite(raw_candidate).all():
                step *= backtracking_factor
                continue
            candidate = _project_rows_to_simplex_numpy(raw_candidate)
            delta = _relative_iterate_change_numpy(candidate, current)
        candidate_z = candidate @ X
        candidate_residual = A @ candidate_z
        candidate_residual -= X
        candidate_objective = _objective_from_residual_numpy(candidate_residual)
        if candidate_objective <= current_objective:
            return (
                candidate,
                candidate_objective,
                step,
                backtrack,
                candidate_z,
                candidate_residual,
                delta,
            )
        if (
            np.array_equal(candidate, current) or delta <= no_op_tol
        ) and candidate_objective - current_objective <= allowance:
            return current, current_objective, step, backtrack, current_z, current_residual, 0.0
        step *= backtracking_factor
    return None


def _update_a_torch(
    torch: Any,
    X: Any,
    Z: Any,
    A: Any,
    *,
    residual: Any,
    objective: float,
    max_iter: int,
    tol: float,
    initial_step: float,
    backtracking_factor: float,
    max_backtracks: int,
) -> _BlockUpdate:
    current = A
    current_residual = residual
    current_objective = objective
    effective_tol = max(float(tol), 16.0 * float(torch.finfo(current.dtype).eps))
    total_backtracks = 0
    accepted_step = initial_step
    for iteration in range(1, max_iter + 1):
        gradient = current_residual @ Z.T
        raw_preview = current - initial_step * gradient
        preview = _project_rows_to_simplex_torch(torch, raw_preview)
        preview_delta = _relative_iterate_change_torch(torch, preview, current)
        if preview_delta <= effective_tol:
            return _BlockUpdate(
                current,
                current_objective,
                iteration - 1,
                total_backtracks,
                initial_step,
                True,
                Z,
                current_residual,
            )
        update = _backtracked_a_candidate_torch(
            torch,
            X,
            Z,
            current,
            gradient,
            preview=preview,
            preview_delta=preview_delta,
            preview_raw_finite=bool(torch.isfinite(raw_preview).all().detach().cpu()),
            current_residual=current_residual,
            current_objective=current_objective,
            initial_step=initial_step,
            backtracking_factor=backtracking_factor,
            max_backtracks=max_backtracks,
            no_op_tol=effective_tol,
        )
        if update is None:
            return _BlockUpdate(
                current,
                current_objective,
                iteration - 1,
                total_backtracks + max_backtracks + 1,
                0.0,
                False,
                Z,
                current_residual,
            )
        (
            candidate,
            candidate_objective,
            accepted_step,
            backtracks,
            candidate_residual,
            delta,
        ) = update
        total_backtracks += backtracks
        current = candidate
        current_objective = candidate_objective
        current_residual = candidate_residual
        if delta <= effective_tol:
            return _BlockUpdate(
                current,
                current_objective,
                iteration,
                total_backtracks,
                accepted_step,
                True,
                Z,
                current_residual,
            )
    return _BlockUpdate(
        current,
        current_objective,
        max_iter,
        total_backtracks,
        accepted_step,
        True,
        Z,
        current_residual,
    )


def _update_b_torch(
    torch: Any,
    X: Any,
    A: Any,
    B: Any,
    *,
    archetypes: Any,
    residual: Any,
    objective: float,
    max_iter: int,
    tol: float,
    initial_step: float,
    backtracking_factor: float,
    max_backtracks: int,
) -> _BlockUpdate:
    current = B
    current_z = archetypes
    current_residual = residual
    current_objective = objective
    effective_tol = max(float(tol), 16.0 * float(torch.finfo(current.dtype).eps))
    total_backtracks = 0
    accepted_step = initial_step
    for iteration in range(1, max_iter + 1):
        gradient = (A.T @ current_residual) @ X.T
        raw_preview = current - initial_step * gradient
        preview = _project_rows_to_simplex_torch(torch, raw_preview)
        preview_delta = _relative_iterate_change_torch(torch, preview, current)
        if preview_delta <= effective_tol:
            return _BlockUpdate(
                current,
                current_objective,
                iteration - 1,
                total_backtracks,
                initial_step,
                True,
                current_z,
                current_residual,
            )
        update = _backtracked_b_candidate_torch(
            torch,
            X,
            A,
            current,
            gradient,
            preview=preview,
            preview_delta=preview_delta,
            preview_raw_finite=bool(torch.isfinite(raw_preview).all().detach().cpu()),
            current_z=current_z,
            current_residual=current_residual,
            current_objective=current_objective,
            initial_step=initial_step,
            backtracking_factor=backtracking_factor,
            max_backtracks=max_backtracks,
            no_op_tol=effective_tol,
        )
        if update is None:
            return _BlockUpdate(
                current,
                current_objective,
                iteration - 1,
                total_backtracks + max_backtracks + 1,
                0.0,
                False,
                current_z,
                current_residual,
            )
        (
            candidate,
            candidate_objective,
            accepted_step,
            backtracks,
            candidate_z,
            candidate_residual,
            delta,
        ) = update
        total_backtracks += backtracks
        current = candidate
        current_objective = candidate_objective
        current_z = candidate_z
        current_residual = candidate_residual
        if delta <= effective_tol:
            return _BlockUpdate(
                current,
                current_objective,
                iteration,
                total_backtracks,
                accepted_step,
                True,
                current_z,
                current_residual,
            )
    return _BlockUpdate(
        current,
        current_objective,
        max_iter,
        total_backtracks,
        accepted_step,
        True,
        current_z,
        current_residual,
    )


def _backtracked_a_candidate_torch(
    torch: Any,
    X: Any,
    Z: Any,
    current: Any,
    gradient: Any,
    *,
    preview: Any,
    preview_delta: float,
    preview_raw_finite: bool,
    current_residual: Any,
    current_objective: float,
    initial_step: float,
    backtracking_factor: float,
    max_backtracks: int,
    no_op_tol: float,
) -> tuple[Any, float, float, int, Any, float] | None:
    step = initial_step
    allowance = _objective_allowance_torch(torch, current.dtype, current_objective)
    for backtrack in range(max_backtracks + 1):
        if backtrack == 0:
            if not preview_raw_finite:
                step *= backtracking_factor
                continue
            candidate = preview
            delta = preview_delta
        else:
            raw_candidate = current - step * gradient
            if not bool(torch.isfinite(raw_candidate).all().detach().cpu()):
                step *= backtracking_factor
                continue
            candidate = _project_rows_to_simplex_torch(torch, raw_candidate)
            delta = _relative_iterate_change_torch(torch, candidate, current)
        candidate_residual = candidate @ Z
        candidate_residual.sub_(X)
        candidate_objective = _objective_from_residual_torch(torch, candidate_residual)
        if candidate_objective <= current_objective:
            return candidate, candidate_objective, step, backtrack, candidate_residual, delta
        if (
            bool(torch.equal(candidate, current)) or delta <= no_op_tol
        ) and candidate_objective - current_objective <= allowance:
            return current, current_objective, step, backtrack, current_residual, 0.0
        step *= backtracking_factor
    return None


def _backtracked_b_candidate_torch(
    torch: Any,
    X: Any,
    A: Any,
    current: Any,
    gradient: Any,
    *,
    preview: Any,
    preview_delta: float,
    preview_raw_finite: bool,
    current_z: Any,
    current_residual: Any,
    current_objective: float,
    initial_step: float,
    backtracking_factor: float,
    max_backtracks: int,
    no_op_tol: float,
) -> tuple[Any, float, float, int, Any, Any, float] | None:
    step = initial_step
    allowance = _objective_allowance_torch(torch, current.dtype, current_objective)
    for backtrack in range(max_backtracks + 1):
        if backtrack == 0:
            if not preview_raw_finite:
                step *= backtracking_factor
                continue
            candidate = preview
            delta = preview_delta
        else:
            raw_candidate = current - step * gradient
            if not bool(torch.isfinite(raw_candidate).all().detach().cpu()):
                step *= backtracking_factor
                continue
            candidate = _project_rows_to_simplex_torch(torch, raw_candidate)
            delta = _relative_iterate_change_torch(torch, candidate, current)
        candidate_z = candidate @ X
        candidate_residual = A @ candidate_z
        candidate_residual.sub_(X)
        candidate_objective = _objective_from_residual_torch(torch, candidate_residual)
        if candidate_objective <= current_objective:
            return (
                candidate,
                candidate_objective,
                step,
                backtrack,
                candidate_z,
                candidate_residual,
                delta,
            )
        if (
            bool(torch.equal(candidate, current)) or delta <= no_op_tol
        ) and candidate_objective - current_objective <= allowance:
            return current, current_objective, step, backtrack, current_z, current_residual, 0.0
        step *= backtracking_factor
    return None


def _combined_block_update(first: _BlockUpdate, second: _BlockUpdate) -> _BlockUpdate:
    return _BlockUpdate(
        value=second.value,
        objective=second.objective,
        iterations=int(first.iterations) + int(second.iterations),
        backtracks=int(first.backtracks) + int(second.backtracks),
        accepted_step=second.accepted_step,
        available=bool(first.available and second.available),
        archetypes=second.archetypes,
        residual=second.residual,
    )


def _project_rows_to_simplex_numpy(values: FloatArray) -> FloatArray:
    U = np.sort(values, axis=1)[:, ::-1]
    cssv = np.cumsum(U, axis=1) - 1.0
    idx = np.arange(1, values.shape[1] + 1, dtype=values.dtype)
    condition = U - cssv / idx > 0
    rho = condition.sum(axis=1) - 1
    theta = cssv[np.arange(values.shape[0]), rho] / (rho.astype(values.dtype) + 1.0)
    return cast(FloatArray, np.maximum(values - theta[:, None], 0.0))


def _project_rows_to_simplex_torch(torch: Any, values: Any) -> Any:
    U, _ = torch.sort(values, dim=1, descending=True)
    cssv = torch.cumsum(U, dim=1) - 1.0
    idx = torch.arange(
        1,
        values.shape[1] + 1,
        device=values.device,
        dtype=values.dtype,
    )
    condition = U - cssv / idx > 0
    rho = condition.sum(dim=1) - 1
    theta = cssv[torch.arange(values.shape[0], device=values.device), rho] / (rho.to(values.dtype) + 1.0)
    return torch.clamp(values - theta[:, None], min=0.0)


def _objective_from_residual_numpy(residual: FloatArray) -> float:
    value = float(0.5 * np.sum(residual * residual, dtype=np.float64))
    if not np.isfinite(value):
        raise FloatingPointError("aa_projected_gradient_v3 objective became non-finite.")
    return value


def _objective_from_residual_torch(torch: Any, residual: Any) -> float:
    value = float((0.5 * torch.sum(residual * residual)).detach().cpu())
    if not np.isfinite(value):
        raise FloatingPointError("aa_projected_gradient_v3 objective became non-finite.")
    return value


def _objective_allowance_numpy(dtype: np.dtype[Any], objective: float) -> float:
    return 64.0 * float(np.finfo(dtype).eps) * max(1.0, abs(objective))


def _objective_allowance_torch(torch: Any, dtype: Any, objective: float) -> float:
    return 64.0 * float(torch.finfo(dtype).eps) * max(1.0, abs(objective))


def _largest_gram_eigenvalue_numpy(matrix: FloatArray) -> float:
    gram = matrix @ matrix.T if matrix.shape[0] <= matrix.shape[1] else matrix.T @ matrix
    if gram.shape == (1, 1):
        return max(0.0, float(gram[0, 0]))
    result = max(0.0, float(np.linalg.eigvalsh(gram)[-1]))
    if not np.isfinite(result):
        raise FloatingPointError("aa_projected_gradient_v3 Lipschitz estimate became non-finite.")
    return result


def _largest_gram_eigenvalue_torch(torch: Any, matrix: Any) -> float:
    gram = matrix @ matrix.T if matrix.shape[0] <= matrix.shape[1] else matrix.T @ matrix
    if tuple(gram.shape) == (1, 1):
        value = gram.reshape(())
    else:
        value = torch.linalg.eigvalsh(gram).amax()
    result = float(torch.clamp(value, min=0).detach().cpu())
    if not np.isfinite(result):
        raise FloatingPointError("aa_projected_gradient_v3 Lipschitz estimate became non-finite.")
    return result


def _data_lipschitz_numpy(matrix: FloatArray) -> tuple[float, str]:
    smaller_dim = int(min(matrix.shape))
    gram_bytes = smaller_dim * smaller_dim * int(matrix.dtype.itemsize)
    if gram_bytes <= _EXACT_GRAM_MAX_BYTES:
        return _largest_gram_eigenvalue_numpy(matrix), "exact_smaller_gram"
    return _matrix_free_power_lipschitz_numpy(matrix), (
        f"deterministic_matrix_free_power_iteration_{_POWER_ITERATIONS}"
    )


def _data_lipschitz_torch(torch: Any, matrix: Any) -> tuple[float, str]:
    smaller_dim = int(min(matrix.shape))
    gram_bytes = smaller_dim * smaller_dim * int(matrix.element_size())
    if gram_bytes <= _EXACT_GRAM_MAX_BYTES:
        return _largest_gram_eigenvalue_torch(torch, matrix), "exact_smaller_gram"
    return _matrix_free_power_lipschitz_torch(torch, matrix), (
        f"deterministic_matrix_free_power_iteration_{_POWER_ITERATIONS}"
    )


def _matrix_free_power_lipschitz_numpy(matrix: FloatArray) -> float:
    use_rows = matrix.shape[0] <= matrix.shape[1]
    dimension = int(matrix.shape[0] if use_rows else matrix.shape[1])
    vector = np.random.default_rng(0).standard_normal(dimension).astype(matrix.dtype)
    vector /= np.linalg.norm(vector)

    def apply(value: FloatArray) -> FloatArray:
        return matrix @ (matrix.T @ value) if use_rows else matrix.T @ (matrix @ value)

    for _ in range(_POWER_ITERATIONS):
        updated = apply(vector)
        norm = float(np.linalg.norm(updated))
        if not np.isfinite(norm) or norm == 0.0:
            raise FloatingPointError("Matrix-free data spectral estimate failed.")
        vector = updated / norm
    value = float(np.dot(vector, apply(vector)))
    if not np.isfinite(value) or value <= 0.0:
        raise FloatingPointError("Matrix-free data spectral estimate is invalid.")
    return value


def _matrix_free_power_lipschitz_torch(torch: Any, matrix: Any) -> float:
    use_rows = matrix.shape[0] <= matrix.shape[1]
    dimension = int(matrix.shape[0] if use_rows else matrix.shape[1])
    vector_np = np.random.default_rng(0).standard_normal(dimension)
    vector = torch.as_tensor(vector_np, device=matrix.device, dtype=matrix.dtype)
    vector = vector / torch.linalg.norm(vector)

    def apply(value: Any) -> Any:
        return matrix @ (matrix.T @ value) if use_rows else matrix.T @ (matrix @ value)

    for _ in range(_POWER_ITERATIONS):
        updated = apply(vector)
        norm = torch.linalg.norm(updated)
        norm_value = float(norm.detach().cpu())
        if not np.isfinite(norm_value) or norm_value == 0.0:
            raise FloatingPointError("Matrix-free data spectral estimate failed.")
        vector = updated / norm
    value = float(torch.dot(vector, apply(vector)).detach().cpu())
    if not np.isfinite(value) or value <= 0.0:
        raise FloatingPointError("Matrix-free data spectral estimate is invalid.")
    return value


def _initial_archetype_weights_numpy(X: FloatArray, *, k: int, seed: int) -> FloatArray:
    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, X.shape[0]))
    selected = [first]
    distance = np.sum((X - X[first]) ** 2, axis=1)
    distance[first] = -1.0
    for _ in range(1, k):
        next_index = int(np.argmax(distance))
        selected.append(next_index)
        new_distance = np.sum((X - X[next_index]) ** 2, axis=1)
        distance = np.minimum(distance, new_distance)
        distance[np.asarray(selected, dtype=np.int64)] = -1.0
    B = np.zeros((k, X.shape[0]), dtype=X.dtype)
    B[np.arange(k), np.asarray(selected, dtype=np.int64)] = 1.0
    return B


def _safe_initial_step(step_scale: float, lipschitz: float) -> float:
    if lipschitz <= 0.0:
        return float(step_scale)
    step = float(step_scale) / float(lipschitz)
    if not np.isfinite(step) or step <= 0.0:
        raise FloatingPointError("aa_projected_gradient_v3 step estimate is invalid.")
    return step


def signed_relative_objective_improvement(previous: float, current: float) -> float:
    previous_value = float(previous)
    current_value = float(current)
    if not np.isfinite(previous_value) or not np.isfinite(current_value):
        raise ValueError("Objective values must be finite.")
    if previous_value == 0.0:
        return 0.0 if current_value == 0.0 else float("-inf")
    return (previous_value - current_value) / abs(previous_value)


def _relative_iterate_change_numpy(candidate: FloatArray, current: FloatArray) -> float:
    numerator = float(np.linalg.norm(candidate - current))
    denominator = float(np.linalg.norm(current))
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


def _relative_iterate_change_torch(torch: Any, candidate: Any, current: Any) -> float:
    numerator = float(torch.linalg.norm(candidate - current).detach().cpu())
    denominator = float(torch.linalg.norm(current).detach().cpu())
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


def _canonicalize_final_b(B: FloatArray) -> tuple[FloatArray, dict[str, object]]:
    B64 = np.asarray(B, dtype=np.float64)
    if B64.ndim != 2 or 0 in B64.shape:
        raise ValueError("AA v3 final B must be a non-empty matrix.")
    if not np.isfinite(B64).all():
        raise FloatingPointError("AA v3 final B canonicalization received non-finite values.")
    minimum_before = float(np.min(B64))
    if minimum_before < 0.0:
        raise FloatingPointError("AA v3 final B canonicalization does not repair negative values.")
    row_sums_before = np.sum(B64, axis=1, dtype=np.float64)
    if not np.isfinite(row_sums_before).all() or bool(np.any(row_sums_before <= 0.0)):
        raise FloatingPointError("AA v3 final B canonicalization requires finite positive row sums.")
    errors_before = np.abs(row_sums_before - 1.0)
    max_error_before = float(np.max(errors_before))
    if max_error_before > _FINAL_B_CANONICALIZATION_MAX_ERROR:
        raise FloatingPointError(
            "AA v3 final B row-sum error exceeds the final canonicalization ceiling "
            f"{_FINAL_B_CANONICALIZATION_MAX_ERROR:g}: {max_error_before:g}."
        )
    repair_mask = errors_before > _FINAL_B_CANONICALIZATION_TRIGGER
    repaired_count = int(np.count_nonzero(repair_mask))
    max_abs_change = 0.0
    if repaired_count:
        row_maxima = np.max(B64, axis=1)
        scale_delta = np.abs(1.0 / row_sums_before[repair_mask] - 1.0)
        max_abs_change = float(np.max(row_maxima[repair_mask] * scale_delta))
        B64 = B64.copy()
        np.divide(B64, row_sums_before[:, None], out=B64, where=repair_mask[:, None])
    row_sums_after = np.sum(B64, axis=1, dtype=np.float64)
    max_error_after = float(np.max(np.abs(row_sums_after - 1.0)))
    if not np.isfinite(B64).all() or float(np.min(B64)) < 0.0 or max_error_after > _FINAL_B_CANONICALIZATION_TRIGGER:
        raise FloatingPointError("AA v3 final B canonicalization produced an unsafe matrix.")
    return B64, {
        "schema_version": _FINAL_B_CANONICALIZATION_SCHEMA_VERSION,
        "method": "float64_row_renormalization_v1",
        "trigger_row_sum_error": float(_FINAL_B_CANONICALIZATION_TRIGGER),
        "max_repairable_row_sum_error": float(_FINAL_B_CANONICALIZATION_MAX_ERROR),
        "applied": bool(repaired_count),
        "row_count": int(B64.shape[0]),
        "violating_row_count_before": repaired_count,
        "repaired_row_count": repaired_count,
        "minimum_before": minimum_before,
        "minimum_after": float(np.min(B64)),
        "max_row_sum_error_before": max_error_before,
        "max_row_sum_error_after": max_error_after,
        "max_abs_change": max_abs_change,
    }


def _returned_state_normalized_objective(
    X_normalized: FloatArray, A: FloatArray, B: FloatArray
) -> tuple[float, dict[str, object]]:
    A64 = np.asarray(A, dtype=np.float64)
    B64, canonicalization = _canonicalize_final_b(B)
    X64 = np.asarray(X_normalized, dtype=np.float64)
    residual = A64 @ (B64 @ X64) - X64
    return _objective_from_residual_numpy(residual), canonicalization


def _restart_summary(
    restart: int,
    init_seed: int,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "restart_index": int(restart),
        "init_seed": int(init_seed),
        "optimizer_final_normalized_objective": float(result["objective"]),
        "outer_iterations": int(result["n_iter"]),
        "termination_reason": str(result["termination_reason"]),
    }


def _build_payload(
    *,
    X_raw: FloatArray,
    seed: int,
    X_normalized: FloatArray,
    A: FloatArray,
    B: FloatArray,
    normalization: AffineNormalization,
    max_iter: int,
    objective_rtol: float,
    config: StandardAAV3Config,
    result: Mapping[str, Any],
    device_used: str,
    dtype_used: str,
) -> dict[str, Any]:
    A64 = np.asarray(A, dtype=np.float64)
    if not np.isfinite(A64).all():
        raise FloatingPointError("aa_projected_gradient_v3 produced non-finite factors.")
    B64, final_b_canonicalization = _canonicalize_final_b(B)
    X_normalized64 = np.asarray(X_normalized, dtype=np.float64)
    X_raw64 = np.asarray(X_raw, dtype=np.float64)
    Z_normalized = B64 @ X_normalized64
    normalized_residual = A64 @ Z_normalized
    normalized_residual -= X_normalized64
    returned_normalized_objective = _objective_from_residual_numpy(normalized_residual)
    Z_raw = B64 @ X_raw64
    raw_residual = A64 @ Z_raw
    raw_residual -= X_raw64
    returned_public_objective = _objective_from_residual_numpy(raw_residual)
    train_mse = 2.0 * returned_public_objective / float(X_raw.shape[0])
    if not np.isfinite(Z_raw).all() or not np.isfinite(train_mse):
        raise FloatingPointError("aa_projected_gradient_v3 produced non-finite public factors.")

    public_scale = (float(normalization.raw_scale) * float(normalization.centered_scale)) ** 2
    if not np.isfinite(public_scale):
        raise FloatingPointError("AA v3 public objective scale is non-finite.")
    raw_history = [dict(row) for row in result["history"]]
    if not raw_history:
        raise RuntimeError("AA v3 optimization history must not be empty.")
    optimizer_terminal_objective = float(raw_history[-1]["objective"])
    raw_history[-1]["objective"] = returned_normalized_objective
    if len(raw_history) > 1:
        previous = float(raw_history[-2]["objective"])
        final_relative = signed_relative_objective_improvement(previous, returned_normalized_objective)
        prior_plateau = int(raw_history[-2].get("plateau_count", 0))
        raw_history[-1]["plateau_count"] = prior_plateau + 1 if 0.0 <= final_relative <= objective_rtol else 0
    else:
        final_relative = 0.0
    raw_history[-1]["relative_improvement"] = final_relative
    raw_history[-1]["signed_relative_objective_improvement"] = final_relative
    history: list[dict[str, Any]] = []
    for index, raw_row in enumerate(raw_history):
        row = dict(raw_row)
        normalized_objective = float(row["objective"])
        public_objective = (
            returned_public_objective if index == len(raw_history) - 1 else normalized_objective * public_scale
        )
        if not np.isfinite(public_objective):
            raise FloatingPointError("AA v3 public objective history is non-finite.")
        row["normalized_objective"] = normalized_objective
        row["objective"] = public_objective
        row["train_mse"] = 2.0 * public_objective / float(X_raw.shape[0])
        history.append(row)
    exact_monotone = all(
        float(current["normalized_objective"]) <= float(previous["normalized_objective"])
        for previous, current in zip(history, history[1:])
    )
    epsilon = np.finfo(np.float32 if dtype_used == "float32" else np.float64).eps
    monotonic_allowance = (
        64.0
        * float(epsilon)
        * max(
            1.0,
            max(abs(float(row["normalized_objective"])) for row in history),
        )
    )
    monotone_with_allowance = all(
        float(current["normalized_objective"]) <= float(previous["normalized_objective"]) + monotonic_allowance
        for previous, current in zip(history, history[1:])
    )
    if not monotone_with_allowance:
        raise FloatingPointError("AA v3 returned-state reconciliation violates monotone objective acceptance.")

    termination_reason = str(result["termination_reason"])
    if (
        termination_reason == "objective_plateau"
        and int(raw_history[-1].get("plateau_count", 0)) < config.objective_patience
    ):
        termination_reason = "returned_state_reconciliation"
    status = _optimization_status(termination_reason)
    gradient_mapping = _gradient_mapping_diagnostics_numpy(
        X_normalized64,
        A64,
        B64,
        Z=Z_normalized,
        residual=normalized_residual,
        x_lipschitz=float(result["data_lipschitz"]),
    )
    diagnostics = _build_diagnostics(
        status=status,
        termination_reason=termination_reason,
        n_iter=int(result["n_iter"]),
        max_iter=max_iter,
        objective_rtol=objective_rtol,
        config=config,
        history=history,
        final_relative=final_relative,
        normalization=normalization,
        data_lipschitz=float(result["data_lipschitz"]),
        data_lipschitz_method=str(result["data_lipschitz_method"]),
        gradient_mapping=gradient_mapping,
        restart_summaries=list(result["restart_summaries"]),
        restarts_requested=int(result["restarts_requested"]),
        selected_restart_index=int(result["restart_index"]),
        selected_init_seed=int(result["init_seed"]),
    )
    diagnostics.update(
        {
            "final_b_canonicalization": final_b_canonicalization,
            "returned_state_objective_coherent": True,
            "returned_state_normalized_objective": returned_normalized_objective,
            "returned_state_public_objective": returned_public_objective,
            "optimizer_terminal_normalized_objective_before_reconciliation": optimizer_terminal_objective,
            "terminal_normalized_objective_reconciliation_delta": returned_normalized_objective
            - optimizer_terminal_objective,
            "objective_history_exactly_monotone": exact_monotone,
            "objective_history_monotone_with_numeric_allowance": monotone_with_allowance,
            "objective_monotonicity_numeric_allowance": monotonic_allowance,
        }
    )
    center_unit = np.asarray(normalization.center_unit, dtype=np.float64)
    extras = {
        "dtype": dtype_used,
        "b_update": SOLVER_ID,
        "solver_id": SOLVER_ID,
        "solver_contract_version": SOLVER_CONTRACT_VERSION,
        "solver_diagnostics_schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "solver_implementation_version": SOLVER_IMPLEMENTATION_VERSION,
        "solver_implementation_digest": SOLVER_IMPLEMENTATION_DIGEST,
        "solver_diagnostics": diagnostics,
        "optimization_history": history,
    }
    return {
        "archetypes_Z": Z_raw,
        "coefficients_A_train": A64,
        "archetype_weights_B": B64,
        "train_mse": train_mse,
        "n_iter": int(result["n_iter"]),
        "device_used": device_used,
        "normalization_center": center_unit * float(normalization.raw_scale),
        "normalization_scale": float(normalization.raw_scale) * float(normalization.centered_scale),
        "normalization_raw_scale": float(normalization.raw_scale),
        "normalization_center_unit": center_unit,
        "normalization_centered_scale": float(normalization.centered_scale),
        "normalization_archetypes": Z_normalized,
        "inference_max_iter": int(config.a_max_iter),
        "inference_tol": float(config.a_tol),
        "inference_step_scale": float(config.a_step_scale),
        "inference_seed": int(seed),
        "extras": extras,
    }


def _constant_solution(
    X_raw: FloatArray,
    *,
    k: int,
    seed: int,
    n_init: int,
    max_iter: int,
    objective_rtol: float,
    config: StandardAAV3Config,
    normalization: AffineNormalization,
    device_used: str,
    dtype_used: str,
) -> dict[str, Any]:
    n_samples = int(X_raw.shape[0])
    A = np.full((n_samples, k), 1.0 / float(k), dtype=np.float64)
    B = np.full((k, n_samples), 1.0 / float(n_samples), dtype=np.float64)
    Z = B @ np.asarray(X_raw, dtype=np.float64)
    history = [_history_row(iteration=0, objective=0.0, relative=0.0, plateau_count=0)]
    history[0]["normalized_objective"] = 0.0
    history[0]["train_mse"] = 0.0
    gradient_mapping = _zero_gradient_mapping()
    diagnostics = _build_diagnostics(
        status=_optimization_status("constant_exact"),
        termination_reason="constant_exact",
        n_iter=0,
        max_iter=max_iter,
        objective_rtol=objective_rtol,
        config=config,
        history=history,
        final_relative=0.0,
        normalization=normalization,
        data_lipschitz=0.0,
        data_lipschitz_method="constant_exact_short_circuit",
        gradient_mapping=gradient_mapping,
        restart_summaries=[],
        restarts_requested=n_init,
        selected_restart_index=None,
        selected_init_seed=None,
    )
    diagnostics.update(
        {
            "returned_state_objective_coherent": True,
            "returned_state_normalized_objective": 0.0,
            "returned_state_public_objective": 0.0,
            "objective_history_exactly_monotone": True,
            "objective_history_monotone_with_numeric_allowance": True,
            "objective_monotonicity_numeric_allowance": 0.0,
            "final_b_canonicalization": _canonicalize_final_b(B)[1],
        }
    )
    extras = {
        "dtype": dtype_used,
        "b_update": SOLVER_ID,
        "solver_id": SOLVER_ID,
        "solver_contract_version": SOLVER_CONTRACT_VERSION,
        "solver_diagnostics_schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "solver_implementation_version": SOLVER_IMPLEMENTATION_VERSION,
        "solver_implementation_digest": SOLVER_IMPLEMENTATION_DIGEST,
        "solver_diagnostics": diagnostics,
        "optimization_history": history,
    }
    return {
        "archetypes_Z": Z,
        "coefficients_A_train": A,
        "archetype_weights_B": B,
        "train_mse": 0.0,
        "n_iter": 0,
        "device_used": device_used,
        "normalization_center": np.asarray(normalization.center_unit, dtype=np.float64)
        * float(normalization.raw_scale),
        "normalization_scale": float(normalization.raw_scale) * float(normalization.centered_scale),
        "normalization_raw_scale": float(normalization.raw_scale),
        "normalization_center_unit": np.asarray(normalization.center_unit, dtype=np.float64),
        "normalization_centered_scale": float(normalization.centered_scale),
        "normalization_archetypes": np.zeros_like(Z, dtype=np.float64),
        "inference_max_iter": int(config.a_max_iter),
        "inference_tol": float(config.a_tol),
        "inference_step_scale": float(config.a_step_scale),
        "inference_seed": int(seed),
        "extras": extras,
    }


def _gradient_mapping_diagnostics_numpy(
    X: FloatArray,
    A: FloatArray,
    B: FloatArray,
    *,
    Z: FloatArray,
    residual: FloatArray,
    x_lipschitz: float,
) -> dict[str, float]:
    a_step = _safe_initial_step(1.0, _largest_gram_eigenvalue_numpy(Z))
    a_gradient = residual @ Z.T
    a_projected = _project_rows_to_simplex_numpy(A - a_step * a_gradient)
    a_mapping_norm = float(np.linalg.norm((A - a_projected) / a_step))
    b_gradient = (A.T @ residual) @ X.T
    b_step = _safe_initial_step(1.0, _largest_gram_eigenvalue_numpy(A) * float(x_lipschitz))
    b_projected = _project_rows_to_simplex_numpy(B - b_step * b_gradient)
    b_mapping_norm = float(np.linalg.norm((B - b_projected) / b_step))
    combined = float(np.hypot(a_mapping_norm, b_mapping_norm))
    if not all(np.isfinite(value) for value in (a_mapping_norm, b_mapping_norm, combined)):
        raise FloatingPointError("AA v3 final gradient-mapping diagnostics are non-finite.")
    return {
        "a_gradient_mapping_norm": a_mapping_norm,
        "b_gradient_mapping_norm": b_mapping_norm,
        "gradient_mapping_norm": combined,
        "a_gradient_mapping_step": float(a_step),
        "b_gradient_mapping_step": float(b_step),
        "final_a_gradient_mapping_norm": a_mapping_norm,
        "final_b_gradient_mapping_norm": b_mapping_norm,
        "final_combined_gradient_mapping_norm": combined,
        "final_projected_gradient_norm": combined,
        "final_a_gradient_mapping_step": float(a_step),
        "final_b_gradient_mapping_step": float(b_step),
    }


def _zero_gradient_mapping() -> dict[str, float]:
    return {
        "a_gradient_mapping_norm": 0.0,
        "b_gradient_mapping_norm": 0.0,
        "gradient_mapping_norm": 0.0,
        "a_gradient_mapping_step": 1.0,
        "b_gradient_mapping_step": 1.0,
        "final_a_gradient_mapping_norm": 0.0,
        "final_b_gradient_mapping_norm": 0.0,
        "final_combined_gradient_mapping_norm": 0.0,
        "final_projected_gradient_norm": 0.0,
        "final_a_gradient_mapping_step": 1.0,
        "final_b_gradient_mapping_step": 1.0,
    }


def _build_diagnostics(
    *,
    status: str,
    termination_reason: str,
    n_iter: int,
    max_iter: int,
    objective_rtol: float,
    config: StandardAAV3Config,
    history: list[dict[str, Any]],
    final_relative: Any,
    normalization: AffineNormalization,
    data_lipschitz: float,
    data_lipschitz_method: str,
    gradient_mapping: Mapping[str, float],
    restart_summaries: list[dict[str, Any]],
    restarts_requested: int,
    selected_restart_index: int | None,
    selected_init_seed: int | None,
) -> dict[str, Any]:
    plateau_reached = termination_reason == "objective_plateau"
    constant_exact = termination_reason == "constant_exact"
    stopping_satisfied = bool(plateau_reached or constant_exact)
    public_scale = (float(normalization.raw_scale) * float(normalization.centered_scale)) ** 2
    return {
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "solver_id": SOLVER_ID,
        "solver_contract_version": SOLVER_CONTRACT_VERSION,
        "solver_implementation_version": SOLVER_IMPLEMENTATION_VERSION,
        "solver_implementation_digest": SOLVER_IMPLEMENTATION_DIGEST,
        "dense_factorization": True,
        "sparsity_assumption": "none",
        "solution_available": True,
        "status": status,
        "optimization_status": status,
        "solution_status": status,
        "termination_reason": termination_reason,
        "objective_termination_reason": termination_reason,
        "stopping_criterion_satisfied": stopping_satisfied,
        "converged": stopping_satisfied,
        "objective_plateau_reached": plateau_reached,
        "optimization_scope": OPTIMIZATION_SCOPE,
        "convergence_scope": OPTIMIZATION_SCOPE,
        "objective_scope": OPTIMIZATION_SCOPE,
        "objective_value_space": "original_training_space",
        "normalized_objective_value_space": "affine_normalized_training_space",
        "relative_objective_evaluation_space": "affine_normalized_training_space",
        "public_objective_scale": float(public_scale),
        "convergence_criterion": (
            "constant_exact_short_circuit"
            if constant_exact
            else (
                "fixed_outer_iteration_budget" if config.force_max_iterations else "signed_relative_objective_plateau"
            )
        ),
        "convergence_claim": (
            "exact_constant_solution"
            if constant_exact
            else (
                "full_iteration_budget_without_convergence_claim"
                if config.force_max_iterations
                else "objective_plateau_only"
            )
        ),
        "objective_rtol": float(objective_rtol),
        "objective_patience": int(config.objective_patience),
        "min_outer_iterations": int(config.min_outer_iterations),
        "force_max_iterations": bool(config.force_max_iterations),
        "max_outer_iterations": int(max_iter),
        "outer_iterations": int(n_iter),
        "objective_history": history,
        "final_relative_objective_improvement": final_relative,
        "final_signed_relative_objective_improvement": final_relative,
        "final_relative_objective_change": final_relative,
        "stationarity_certified": False,
        "kkt_certified": False,
        "gradient_mapping_used_for_stopping": False,
        "gradient_mapping_diagnostic_computed": True,
        "gradient_mapping_metric_space": "affine_normalized_training_space",
        "gradient_mapping_definition": "G_eta(x)=(x-P_C(x-eta*grad_f(x)))/eta",
        "gradient_mapping_norm_kind": "combined_frobenius",
        "a_gradient_mapping_constraint": "row_simplex",
        "b_gradient_mapping_constraint": "row_simplex",
        "gradient_mapping_evaluation_point": "returned_affine_normalized_A_B",
        "gradient_mapping_evaluated_at": "final_returned_state",
        "gradient_mapping_objective": "affine_normalized_training_l2_objective",
        **dict(gradient_mapping),
        "data_lipschitz": float(data_lipschitz),
        "data_lipschitz_method": data_lipschitz_method,
        "exact_gram_max_bytes": int(_EXACT_GRAM_MAX_BYTES),
        "normalization_strategy": "raw_max_abs_then_unit_mean_then_centered_max_abs_v1",
        "normalization_raw_scale": float(normalization.raw_scale),
        "normalization_centered_scale": float(normalization.centered_scale),
        "translation_invariant": True,
        "scalar_scale_invariant": True,
        "restart_diagnostics_schema_version": _RESTART_DIAGNOSTICS_SCHEMA_VERSION,
        "n_init_requested": int(restarts_requested),
        "restarts_completed": len(restart_summaries),
        "selected_restart_index": selected_restart_index,
        "selected_init_seed": selected_init_seed,
        "restart_selection_metric": "optimizer_final_normalized_objective",
        "restart_summaries": restart_summaries,
        "cache_strategy": "accepted_dense_z_residual_cache_v1",
        "persistent_reconstruction_cache": False,
        "simplex_projection_numerical_contract": "standard_aa_v2_native_dtype_cumsum_v1",
        "preview_reused_as_backtrack_zero": True,
    }


def _history_row(
    *,
    iteration: int,
    objective: float,
    relative: float | None,
    plateau_count: int,
    a_update: _BlockUpdate | None = None,
    b_update: _BlockUpdate | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "step": int(iteration),
        "iteration": int(iteration),
        "objective": float(objective),
        "relative_improvement": relative,
        "signed_relative_objective_improvement": relative,
        "plateau_count": int(plateau_count),
    }
    if a_update is not None:
        row.update(
            {
                "a_iterations": int(a_update.iterations),
                "a_backtracks": int(a_update.backtracks),
                "a_accepted_step": float(a_update.accepted_step),
            }
        )
    if b_update is not None:
        row.update(
            {
                "b_iterations": int(b_update.iterations),
                "b_backtracks": int(b_update.backtracks),
                "b_accepted_step": float(b_update.accepted_step),
            }
        )
    return row


def _optimization_status(termination_reason: str) -> str:
    if termination_reason == "objective_plateau":
        return "approximate_objective_plateau"
    if termination_reason == "constant_exact":
        return "exact_constant_solution"
    if termination_reason == "returned_state_reconciliation":
        return "approximate_returned_state_reconciliation"
    return "approximate_max_iterations"


def _validate_fit_scalars(
    *, k: Any, seed: Any, n_init: Any, max_iter: Any, tol: Any
) -> tuple[int, int, int, int, float]:
    return (
        _strict_positive_int(k, name="k"),
        _strict_int(seed, name="seed"),
        _strict_positive_int(n_init, name="n_init"),
        _strict_positive_int(max_iter, name="max_iter"),
        _strict_positive_float(tol, name="tol"),
    )


def _validate_k(k: int, *, n_samples: int) -> None:
    if k <= 1:
        raise ValueError(f"k must be >=2, got {k}")
    if k > n_samples:
        raise ValueError(f"k ({k}) must be <= number of samples ({n_samples})")


def _validate_real_finite_matrix(value: Any, *, name: str, dtype: Any) -> FloatArray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued; complex values are unsupported.")
    if raw.dtype.kind not in {"f", "i", "u"}:
        raise TypeError(f"{name} must have a real numeric dtype, got {raw.dtype}.")
    if raw.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={raw.shape}.")
    if raw.shape[0] == 0 or raw.shape[1] == 0:
        raise ValueError(f"{name} must have at least one row and one feature.")
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.asarray(raw, dtype=dtype)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _validate_optional_validation_matrix(
    value: Any,
    *,
    feature_count: int,
    dtype: Any,
    normalization: AffineNormalization,
) -> None:
    if value is None:
        return
    array = _validate_real_finite_matrix(value, name="X_val", dtype=dtype)
    if int(array.shape[1]) != int(feature_count):
        raise ValueError(f"X_val feature dimension ({array.shape[1]}) must match X_train ({feature_count}).")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        normalized = (
            array / np.asarray(normalization.raw_scale, dtype=array.dtype)
            - np.asarray(normalization.center_unit, dtype=array.dtype)
        ) / np.asarray(normalization.centered_scale, dtype=array.dtype)
    if not np.isfinite(normalized).all():
        raise ValueError("X_val must remain finite in the affine-normalized training coordinate system.")


def _method_param_value(method_params: Any | None, name: str, default: Any) -> Any:
    if method_params is None:
        return default
    if isinstance(method_params, Mapping):
        return method_params.get(name, default)
    return getattr(method_params, name, default)


def _method_param_mapping(method_params: Any | None) -> dict[str, Any]:
    if method_params is None:
        return {}
    if isinstance(method_params, Mapping):
        if any(not isinstance(key, str) for key in method_params):
            raise TypeError("AA v3 method parameter names must be strings.")
        return dict(method_params)
    try:
        params = dict(vars(method_params))
        has_instance_mapping = True
    except TypeError:
        params = {}
        has_instance_mapping = False
    for cls in type(method_params).__mro__:
        for key, class_value in vars(cls).items():
            if (
                key.startswith("_")
                or key in params
                or callable(class_value)
                or isinstance(class_value, (classmethod, staticmethod))
            ):
                continue
            if isinstance(class_value, property) and key not in _METHOD_PARAM_NAMES:
                params[key] = class_value
                continue
            try:
                params[key] = getattr(method_params, key)
            except AttributeError:
                continue
    if not has_instance_mapping and not params:
        raise TypeError("AA v3 method_params must be a mapping or attribute object.")
    return params


def _strict_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {value!r}.")
    return int(value)


def _strict_positive_int(value: Any, *, name: str) -> int:
    result = _strict_int(value, name=name)
    if result <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}.")
    return result


def _strict_bounded_nonnegative_int(value: Any, *, name: str, maximum: int) -> int:
    result = _strict_int(value, name=name)
    if result < 0 or result > maximum:
        raise ValueError(f"{name} must be between 0 and {maximum}, got {value!r}.")
    return result


def _strict_positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, got {value!r}.")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}.")
    return result


def _strict_exact_float(value: Any, *, name: str, expected: float) -> float:
    result = _strict_positive_float(value, name=name)
    if result != expected:
        raise ValueError(f"{name} must be exactly {expected}, got {value!r}.")
    return result


def _strict_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise TypeError(f"{name} must be a boolean, got {value!r}.")
