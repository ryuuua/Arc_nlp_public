"""Affine-normalized projected-gradient solver for standard archetypal analysis.

The v2 selector is intentionally separate from both the historical projected
least-squares update and the unversioned ``aa_projected_gradient`` solver.  It
implements the same numerical contract for NumPy and Torch without changing
the ordinary AA default.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating[Any]]

SOLVER_ID = "aa_projected_gradient_v2"
SOLVER_CONTRACT_VERSION = "aa_b_update_v2"
V1_SOLVER_ID = "aa_projected_gradient"
V1_SOLVER_CONTRACT_VERSION = "aa_b_update_v1"
DIAGNOSTICS_SCHEMA_VERSION = "arc_nlp_standard_aa_diagnostics_v2"
V1_DIAGNOSTICS_SCHEMA_VERSION = "arc_nlp_standard_aa_diagnostics_v1"
OPTIMIZATION_SCOPE = "full_data_l2_factor_problem"
_EXACT_GRAM_MAX_BYTES = 128 * 1024 * 1024
_POWER_ITERATIONS = 64

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
        "record_history",
    }
)


@dataclass(frozen=True)
class StandardAAV2Config:
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


@dataclass(frozen=True)
class _BlockUpdate:
    value: Any
    objective: float
    iterations: int
    backtracks: int
    accepted_step: float
    available: bool


@dataclass(frozen=True)
class AffineNormalization:
    raw_scale: float
    center_unit: FloatArray
    centered_scale: float
    is_constant: bool


def is_v2_selector(method_params: Any | None) -> bool:
    value = _method_param_value(method_params, "b_update", None)
    return canonicalize_b_update_selector(value) == SOLVER_ID


def canonicalize_b_update_selector(value: Any) -> str:
    """Normalize a standard-AA selector without coercing an object into v2.

    Historical v1 and legacy selectors accepted string-coercible values, so
    non-v2 values retain that compatibility.  The opt-in v2 boundary is
    deliberately stricter: only a built-in string may select the v2 solver.
    """

    if type(value) is str:
        return value.strip().lower()
    selector = str(value).strip().lower()
    if selector == SOLVER_ID:
        raise TypeError(
            f"b_update must be a literal string to select {SOLVER_ID!r}; "
            "string-coercible objects are unsupported."
        )
    return selector


def validate_method_params(method_params: Any | None) -> StandardAAV2Config:
    """Strictly validate the opt-in v2 numerical parameter surface."""

    params = _method_param_mapping(method_params)
    unknown = sorted(set(params) - _METHOD_PARAM_NAMES)
    if unknown:
        raise ValueError(f"Unsupported aa_projected_gradient_v2 method parameters: {unknown}.")
    if "b_update" in params:
        selector = canonicalize_b_update_selector(params["b_update"])
        if selector != SOLVER_ID:
            raise ValueError(f"b_update must select {SOLVER_ID!r} for the v2 solver.")
    if "record_history" in params:
        _strict_bool(params["record_history"], name="record_history")

    return StandardAAV2Config(
        a_max_iter=_strict_positive_int(
            params.get("a_max_iter", 300),
            name="a_max_iter",
        ),
        a_tol=_strict_positive_float(
            params.get("a_tol", 1e-6),
            name="a_tol",
        ),
        b_max_iter=_strict_positive_int(
            params.get("b_max_iter", 300),
            name="b_max_iter",
        ),
        b_tol=_strict_positive_float(
            params.get("b_tol", 1e-6),
            name="b_tol",
        ),
        a_step_scale=_strict_positive_float(
            params.get("a_step_scale", 1.0),
            name="a_step_scale",
        ),
        b_step_scale=_strict_positive_float(
            params.get("b_step_scale", 1.0),
            name="b_step_scale",
        ),
        backtracking_factor=_strict_exact_float(
            params.get("backtracking_factor", 0.5),
            name="backtracking_factor",
            expected=0.5,
        ),
        max_backtracks=_strict_bounded_nonnegative_int(
            params.get("max_backtracks", 20),
            name="max_backtracks",
            maximum=20,
        ),
        objective_patience=_strict_positive_int(
            params.get("objective_patience", DEFAULT_OBJECTIVE_PATIENCE),
            name="objective_patience",
        ),
        min_outer_iterations=_strict_positive_int(
            params.get("min_outer_iterations", DEFAULT_MIN_OUTER_ITERATIONS),
            name="min_outer_iterations",
        ),
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
    """Validate a v2 fit without importing, resolving, or configuring Torch."""

    validate_method_params(method_params)
    k_value, _seed_value, _n_init_value, _max_iter_value, _objective_rtol = _validate_fit_scalars(
        k=k,
        seed=seed,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
    )
    if not isinstance(dtype, str) or dtype not in {"float32", "float64"}:
        raise ValueError("AA v2 dtype must be 'float32' or 'float64'.")
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


def fit_standard_aa_v2_numpy(
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
    """Fit v2 standard AA with NumPy on CPU."""

    config = validate_method_params(method_params)
    k_value, seed_value, n_init_value, max_iter_value, objective_rtol = _validate_fit_scalars(
        k=k,
        seed=seed,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
    )
    if not isinstance(dtype, str) or dtype not in {"float32", "float64"}:
        raise ValueError("AA v2 dtype must be 'float32' or 'float64'.")
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
            max_iter=max_iter_value,
            objective_rtol=objective_rtol,
            config=config,
            normalization=normalization,
            device_used="cpu",
            dtype_used=str(X_raw.dtype),
        )

    best: dict[str, Any] | None = None
    restart_rng = np.random.default_rng(seed_value)
    x_lipschitz, spectral_method = _data_lipschitz_numpy(X_normalized)
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
        if best is None or float(result["objective"]) < float(best["objective"]):
            best = result

    if best is None:
        raise RuntimeError("aa_projected_gradient_v2 did not produce a NumPy solution.")
    return _build_payload(
        X_raw=X_raw,
        seed=seed_value,
        X_normalized=X_normalized,
        A=np.asarray(best["A"]),
        B=np.asarray(best["B"]),
        normalization=normalization,
        max_iter=max_iter_value,
        objective_rtol=objective_rtol,
        config=config,
        result=best,
        device_used="cpu",
        dtype_used=str(X_raw.dtype),
    )


def fit_standard_aa_v2_torch(
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
    """Fit v2 standard AA with Torch on CPU or CUDA."""

    config = validate_method_params(method_params)
    k_value, seed_value, n_init_value, max_iter_value, objective_rtol = _validate_fit_scalars(
        k=k,
        seed=seed,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
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

    X = torch.as_tensor(X_normalized_np if not normalization.is_constant else X_raw, device=device, dtype=dtype)
    observed_device = X.device
    observed_device_text = str(observed_device)
    observed_dtype = str(X.dtype).removeprefix("torch.")

    if normalization.is_constant:
        n_samples = int(X.shape[0])
        A = torch.full(
            (n_samples, k_value),
            1.0 / float(k_value),
            device=observed_device,
            dtype=X.dtype,
        )
        B = torch.full(
            (k_value, n_samples),
            1.0 / float(n_samples),
            device=observed_device,
            dtype=X.dtype,
        )
        Z = B @ X
        reconstructed = A @ Z
        if not bool(torch.isfinite(reconstructed).all().detach().cpu()):
            raise FloatingPointError("AA v2 constant Torch solution produced non-finite values.")
        return _constant_solution(
            cast(FloatArray, np.asarray(X.detach().cpu().numpy())),
            k=k_value,
            seed=seed_value,
            max_iter=max_iter_value,
            objective_rtol=objective_rtol,
            config=config,
            normalization=normalization,
            device_used=observed_device_text,
            dtype_used=observed_dtype,
            factor_arrays=(
                cast(FloatArray, np.asarray(A.detach().cpu().numpy())),
                cast(FloatArray, np.asarray(B.detach().cpu().numpy())),
                cast(FloatArray, np.asarray(Z.detach().cpu().numpy())),
            ),
        )

    best: dict[str, Any] | None = None
    restart_rng = np.random.default_rng(seed_value)
    x_lipschitz, spectral_method = _data_lipschitz_torch(torch, X)
    for restart in range(n_init_value):
        init_seed = int(restart_rng.integers(0, 2**31 - 1)) + restart
        B_np = _initial_archetype_weights_numpy(
            X_normalized_np,
            k=k_value,
            seed=init_seed,
        )
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
        if best is None or float(result["objective"]) < float(best["objective"]):
            best = result

    if best is None:
        raise RuntimeError("aa_projected_gradient_v2 did not produce a Torch solution.")
    return _build_payload(
        X_raw=X_raw,
        seed=seed_value,
        X_normalized=X_normalized_np,
        A=np.asarray(best["A"].detach().cpu().numpy()),
        B=np.asarray(best["B"].detach().cpu().numpy()),
        normalization=normalization,
        max_iter=max_iter_value,
        objective_rtol=objective_rtol,
        config=config,
        result=best,
        device_used=observed_device_text,
        dtype_used=observed_dtype,
    )


def infer_standard_aa_v2_numpy(
    X: FloatArray,
    Z: FloatArray,
    *,
    max_iter: int,
    tol: float,
    step_scale: float,
    dtype: str,
) -> FloatArray:
    """Infer simplex coefficients in the saved v2 training coordinates."""

    if not isinstance(dtype, str) or dtype not in {"float32", "float64"}:
        raise ValueError("AA v2 dtype must be 'float32' or 'float64'.")
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
    objective = _objective_from_z_numpy(targets, coefficients, archetypes)
    update = _update_a_numpy(
        targets,
        archetypes,
        coefficients,
        objective=objective,
        max_iter=iteration_count,
        tol=tolerance,
        initial_step=_safe_initial_step(
            step_scale_value,
            _largest_gram_eigenvalue_numpy(archetypes),
        ),
        backtracking_factor=0.5,
        max_backtracks=20,
    )
    if not update.available:
        raise RuntimeError("AA v2 transform backtracking exhausted without a safe iterate.")
    return np.asarray(update.value, dtype=work_dtype)


def infer_standard_aa_v2_torch(
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
    """Torch counterpart of v2 coefficient inference."""

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
    objective = _objective_from_z_torch_value(torch, targets, coefficients, archetypes)
    update = _update_a_torch(
        torch,
        targets,
        archetypes,
        coefficients,
        objective=objective,
        max_iter=iteration_count,
        tol=tolerance,
        initial_step=_safe_initial_step(
            step_scale_value,
            _largest_gram_eigenvalue_torch(torch, archetypes),
        ),
        backtracking_factor=0.5,
        max_backtracks=20,
    )
    if not update.available:
        raise RuntimeError("AA v2 transform backtracking exhausted without a safe iterate.")
    return np.asarray(update.value.detach().cpu().numpy())


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
        centered_scale = max(
            centered_scale,
            float(np.max(np.abs(scaled_chunk - center_unit))),
        )
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


def read_standard_aa_diagnostics(artifact: Mapping[str, Any] | None) -> dict[str, Any]:
    """Read diagnostics, treating historical unversioned true-AA as v1.

    This compatibility reader never upgrades an unversioned artifact to v2.
    """

    outer: Mapping[str, Any] = artifact or {}
    outer_selectors = [
        _canonical_diagnostic_selector(outer[key])
        for key in ("b_update", "solver_id")
        if outer.get(key) not in (None, "")
    ]
    if len(set(outer_selectors)) > 1:
        raise ValueError("Outer standard-AA solver identities disagree.")
    outer_selector = outer_selectors[0] if outer_selectors else ""
    source = outer
    nested = outer.get("solver_diagnostics")
    if isinstance(nested, Mapping):
        source = nested
    payload = dict(source)
    schema = payload.get("schema_version")
    nested_selectors = [
        _canonical_diagnostic_selector(payload[key])
        for key in ("b_update", "solver_id")
        if payload.get(key) not in (None, "")
    ]
    if len(set(nested_selectors)) > 1:
        raise ValueError("Nested standard-AA solver identities disagree.")
    selector = nested_selectors[0] if nested_selectors else ""
    if isinstance(nested, Mapping):
        if outer_selector and selector and outer_selector != selector:
            raise ValueError("Outer and nested standard-AA solver identities disagree.")
        outer_contract = outer.get("solver_contract_version")
        nested_contract = payload.get("solver_contract_version")
        if outer_contract not in (None, "") and nested_contract not in (None, ""):
            if str(outer_contract) != str(nested_contract):
                raise ValueError("Outer and nested standard-AA solver contract versions disagree.")
        outer_schema = outer.get("solver_diagnostics_schema_version")
        if outer_schema not in (None, "") and schema not in (None, "") and outer_schema != schema:
            raise ValueError("Outer and nested standard-AA diagnostics schema versions disagree.")
    if SOLVER_ID in {outer_selector, selector} and schema != DIAGNOSTICS_SCHEMA_VERSION:
        raise ValueError("aa_projected_gradient_v2 diagnostics require the exact v2 schema version.")
    if schema is None:
        interpreted_selector = selector or outer_selector
        if interpreted_selector in {
            V1_SOLVER_ID,
            "aa_projected_gradient_v1",
            "true_aa_projected_gradient",
            "fixed_a_projected_gradient",
        }:
            payload["schema_version"] = V1_DIAGNOSTICS_SCHEMA_VERSION
            payload.setdefault("solver_id", V1_SOLVER_ID)
            payload.setdefault("solver_contract_version", V1_SOLVER_CONTRACT_VERSION)
            payload["artifact_version_interpretation"] = "unversioned_v1"
        return payload
    if schema != DIAGNOSTICS_SCHEMA_VERSION:
        return payload
    if payload.get("solver_id") != SOLVER_ID:
        raise ValueError("v2 diagnostics solver_id mismatch.")
    if payload.get("solver_contract_version") != SOLVER_CONTRACT_VERSION:
        raise ValueError("v2 diagnostics solver_contract_version mismatch.")
    if payload.get("optimization_scope") != OPTIMIZATION_SCOPE:
        raise ValueError(f"v2 diagnostics optimization_scope must be exactly {OPTIMIZATION_SCOPE!r}.")
    return payload


def _canonical_diagnostic_selector(value: Any) -> str:
    selector = str(value).strip().lower()
    if selector in {
        V1_SOLVER_ID,
        "aa_projected_gradient_v1",
        "true_aa_projected_gradient",
        "fixed_a_projected_gradient",
    }:
        return V1_SOLVER_ID
    if selector in {"legacy_projected_ls", "legacy", "projected_ls"}:
        return "legacy_projected_ls"
    return selector


def _fit_one_numpy(
    X: FloatArray,
    A: FloatArray,
    B: FloatArray,
    *,
    max_iter: int,
    objective_rtol: float,
    config: StandardAAV2Config,
    x_lipschitz: float,
) -> dict[str, Any]:
    objective = _objective_numpy(X, A, B)
    history = [_history_row(iteration=0, objective=objective, relative=None, plateau_count=0)]
    plateau_count = 0
    termination_reason = "max_iterations"
    n_iter = 0

    for iteration in range(1, max_iter + 1):
        a_update = _BlockUpdate(A, objective, 0, 0, 0.0, True)
        if iteration == 1:
            Z = B @ X
            a_lipschitz = _largest_gram_eigenvalue_numpy(Z)
            a_update = _update_a_numpy(
                X,
                Z,
                A,
                objective=objective,
                max_iter=config.a_max_iter,
                tol=config.a_tol,
                initial_step=_safe_initial_step(config.a_step_scale, a_lipschitz),
                backtracking_factor=config.backtracking_factor,
                max_backtracks=config.max_backtracks,
            )
            if not a_update.available:
                raise RuntimeError("AA v2 A-update backtracking exhausted without a safe iterate.")
            A = np.asarray(a_update.value)
            objective = a_update.objective
        b_lipschitz = _largest_gram_eigenvalue_numpy(A) * x_lipschitz
        b_update = _update_b_numpy(
            X,
            A,
            B,
            objective=objective,
            max_iter=config.b_max_iter,
            tol=config.b_tol,
            initial_step=_safe_initial_step(config.b_step_scale, b_lipschitz),
            backtracking_factor=config.backtracking_factor,
            max_backtracks=config.max_backtracks,
        )
        if not b_update.available:
            raise RuntimeError("AA v2 B-update backtracking exhausted without a safe iterate.")
        B = np.asarray(b_update.value)
        objective = b_update.objective
        polished_a_update = _update_a_numpy(
            X,
            B @ X,
            A,
            objective=objective,
            max_iter=config.a_max_iter,
            tol=config.a_tol,
            initial_step=_safe_initial_step(
                config.a_step_scale,
                _largest_gram_eigenvalue_numpy(B @ X),
            ),
            backtracking_factor=config.backtracking_factor,
            max_backtracks=config.max_backtracks,
        )
        if not polished_a_update.available:
            raise RuntimeError("AA v2 final A-update backtracking exhausted without a safe iterate.")
        A = np.asarray(polished_a_update.value)
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
        if iteration >= config.min_outer_iterations and plateau_count >= config.objective_patience:
            termination_reason = "objective_plateau"
            break

    return {
        "A": A,
        "B": B,
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
    config: StandardAAV2Config,
    x_lipschitz: float,
) -> dict[str, Any]:
    objective = _objective_torch_value(torch, X, A, B)
    history = [_history_row(iteration=0, objective=objective, relative=None, plateau_count=0)]
    plateau_count = 0
    termination_reason = "max_iterations"
    n_iter = 0

    for iteration in range(1, max_iter + 1):
        a_update = _BlockUpdate(A, objective, 0, 0, 0.0, True)
        if iteration == 1:
            Z = B @ X
            a_lipschitz = _largest_gram_eigenvalue_torch(torch, Z)
            a_update = _update_a_torch(
                torch,
                X,
                Z,
                A,
                objective=objective,
                max_iter=config.a_max_iter,
                tol=config.a_tol,
                initial_step=_safe_initial_step(config.a_step_scale, a_lipschitz),
                backtracking_factor=config.backtracking_factor,
                max_backtracks=config.max_backtracks,
            )
            if not a_update.available:
                raise RuntimeError("AA v2 A-update backtracking exhausted without a safe iterate.")
            A = a_update.value
            objective = a_update.objective
        b_lipschitz = _largest_gram_eigenvalue_torch(torch, A) * x_lipschitz
        b_update = _update_b_torch(
            torch,
            X,
            A,
            B,
            objective=objective,
            max_iter=config.b_max_iter,
            tol=config.b_tol,
            initial_step=_safe_initial_step(config.b_step_scale, b_lipschitz),
            backtracking_factor=config.backtracking_factor,
            max_backtracks=config.max_backtracks,
        )
        if not b_update.available:
            raise RuntimeError("AA v2 B-update backtracking exhausted without a safe iterate.")
        B = b_update.value
        objective = b_update.objective
        polished_a_update = _update_a_torch(
            torch,
            X,
            B @ X,
            A,
            objective=objective,
            max_iter=config.a_max_iter,
            tol=config.a_tol,
            initial_step=_safe_initial_step(
                config.a_step_scale,
                _largest_gram_eigenvalue_torch(torch, B @ X),
            ),
            backtracking_factor=config.backtracking_factor,
            max_backtracks=config.max_backtracks,
        )
        if not polished_a_update.available:
            raise RuntimeError("AA v2 final A-update backtracking exhausted without a safe iterate.")
        A = polished_a_update.value
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
        if iteration >= config.min_outer_iterations and plateau_count >= config.objective_patience:
            termination_reason = "objective_plateau"
            break

    return {
        "A": A,
        "B": B,
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
    objective: float,
    max_iter: int,
    tol: float,
    initial_step: float,
    backtracking_factor: float,
    max_backtracks: int,
) -> _BlockUpdate:
    current = A
    current_objective = objective
    effective_tol = max(float(tol), 16.0 * float(np.finfo(current.dtype).eps))
    total_backtracks = 0
    accepted_step = initial_step
    for iteration in range(1, max_iter + 1):
        gradient = (current @ Z - X) @ Z.T
        preview = _project_rows_to_simplex_numpy(current - initial_step * gradient)
        if _relative_iterate_change_numpy(preview, current) <= effective_tol:
            return _BlockUpdate(current, current_objective, iteration - 1, total_backtracks, initial_step, True)
        update = _backtracked_candidate_numpy(
            current,
            gradient,
            objective_fn=lambda candidate: _objective_from_z_numpy(X, candidate, Z),
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
            )
        candidate, candidate_objective, accepted_step, backtracks = update
        total_backtracks += backtracks
        delta = _relative_iterate_change_numpy(candidate, current)
        current = candidate
        current_objective = candidate_objective
        if delta <= effective_tol:
            return _BlockUpdate(
                current,
                current_objective,
                iteration,
                total_backtracks,
                accepted_step,
                True,
            )
    return _BlockUpdate(
        current,
        current_objective,
        max_iter,
        total_backtracks,
        accepted_step,
        True,
    )


def _update_b_numpy(
    X: FloatArray,
    A: FloatArray,
    B: FloatArray,
    *,
    objective: float,
    max_iter: int,
    tol: float,
    initial_step: float,
    backtracking_factor: float,
    max_backtracks: int,
) -> _BlockUpdate:
    current = B
    current_objective = objective
    effective_tol = max(float(tol), 16.0 * float(np.finfo(current.dtype).eps))
    total_backtracks = 0
    accepted_step = initial_step
    for iteration in range(1, max_iter + 1):
        residual = A @ (current @ X) - X
        gradient = (A.T @ residual) @ X.T
        preview = _project_rows_to_simplex_numpy(current - initial_step * gradient)
        if _relative_iterate_change_numpy(preview, current) <= effective_tol:
            return _BlockUpdate(current, current_objective, iteration - 1, total_backtracks, initial_step, True)
        update = _backtracked_candidate_numpy(
            current,
            gradient,
            objective_fn=lambda candidate: _objective_numpy(X, A, candidate),
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
            )
        candidate, candidate_objective, accepted_step, backtracks = update
        total_backtracks += backtracks
        delta = _relative_iterate_change_numpy(candidate, current)
        current = candidate
        current_objective = candidate_objective
        if delta <= effective_tol:
            return _BlockUpdate(
                current,
                current_objective,
                iteration,
                total_backtracks,
                accepted_step,
                True,
            )
    return _BlockUpdate(
        current,
        current_objective,
        max_iter,
        total_backtracks,
        accepted_step,
        True,
    )


def _update_a_torch(
    torch: Any,
    X: Any,
    Z: Any,
    A: Any,
    *,
    objective: float,
    max_iter: int,
    tol: float,
    initial_step: float,
    backtracking_factor: float,
    max_backtracks: int,
) -> _BlockUpdate:
    current = A
    current_objective = objective
    effective_tol = max(float(tol), 16.0 * float(torch.finfo(current.dtype).eps))
    total_backtracks = 0
    accepted_step = initial_step
    for iteration in range(1, max_iter + 1):
        gradient = (current @ Z - X) @ Z.T
        preview = _project_rows_to_simplex_torch(torch, current - initial_step * gradient)
        if _relative_iterate_change_torch(torch, preview, current) <= effective_tol:
            return _BlockUpdate(current, current_objective, iteration - 1, total_backtracks, initial_step, True)
        update = _backtracked_candidate_torch(
            torch,
            current,
            gradient,
            objective_fn=lambda candidate: _objective_from_z_torch_value(
                torch,
                X,
                candidate,
                Z,
            ),
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
            )
        candidate, candidate_objective, accepted_step, backtracks = update
        total_backtracks += backtracks
        delta = _relative_iterate_change_torch(torch, candidate, current)
        current = candidate
        current_objective = candidate_objective
        if delta <= effective_tol:
            return _BlockUpdate(
                current,
                current_objective,
                iteration,
                total_backtracks,
                accepted_step,
                True,
            )
    return _BlockUpdate(
        current,
        current_objective,
        max_iter,
        total_backtracks,
        accepted_step,
        True,
    )


def _update_b_torch(
    torch: Any,
    X: Any,
    A: Any,
    B: Any,
    *,
    objective: float,
    max_iter: int,
    tol: float,
    initial_step: float,
    backtracking_factor: float,
    max_backtracks: int,
) -> _BlockUpdate:
    current = B
    current_objective = objective
    effective_tol = max(float(tol), 16.0 * float(torch.finfo(current.dtype).eps))
    total_backtracks = 0
    accepted_step = initial_step
    for iteration in range(1, max_iter + 1):
        residual = A @ (current @ X) - X
        gradient = (A.T @ residual) @ X.T
        preview = _project_rows_to_simplex_torch(torch, current - initial_step * gradient)
        if _relative_iterate_change_torch(torch, preview, current) <= effective_tol:
            return _BlockUpdate(current, current_objective, iteration - 1, total_backtracks, initial_step, True)
        update = _backtracked_candidate_torch(
            torch,
            current,
            gradient,
            objective_fn=lambda candidate: _objective_torch_value(torch, X, A, candidate),
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
            )
        candidate, candidate_objective, accepted_step, backtracks = update
        total_backtracks += backtracks
        delta = _relative_iterate_change_torch(torch, candidate, current)
        current = candidate
        current_objective = candidate_objective
        if delta <= effective_tol:
            return _BlockUpdate(
                current,
                current_objective,
                iteration,
                total_backtracks,
                accepted_step,
                True,
            )
    return _BlockUpdate(
        current,
        current_objective,
        max_iter,
        total_backtracks,
        accepted_step,
        True,
    )


def _backtracked_candidate_numpy(
    current: FloatArray,
    gradient: FloatArray,
    *,
    objective_fn: Any,
    current_objective: float,
    initial_step: float,
    backtracking_factor: float,
    max_backtracks: int,
    no_op_tol: float,
) -> tuple[FloatArray, float, float, int] | None:
    step = initial_step
    objective_allowance = 64.0 * float(np.finfo(current.dtype).eps) * max(1.0, abs(current_objective))
    for backtrack in range(max_backtracks + 1):
        raw_candidate = current - step * gradient
        if np.isfinite(raw_candidate).all():
            candidate = _project_rows_to_simplex_numpy(raw_candidate)
            candidate_objective = float(objective_fn(candidate))
            if np.isfinite(candidate_objective) and candidate_objective <= current_objective:
                return candidate, candidate_objective, step, backtrack
            delta = _relative_iterate_change_numpy(candidate, current)
            if (
                np.isfinite(candidate_objective)
                and (np.array_equal(candidate, current) or delta <= no_op_tol)
                and candidate_objective - current_objective <= objective_allowance
            ):
                return current, current_objective, step, backtrack
        step *= backtracking_factor
    return None


def _backtracked_candidate_torch(
    torch: Any,
    current: Any,
    gradient: Any,
    *,
    objective_fn: Any,
    current_objective: float,
    initial_step: float,
    backtracking_factor: float,
    max_backtracks: int,
    no_op_tol: float,
) -> tuple[Any, float, float, int] | None:
    step = initial_step
    objective_allowance = 64.0 * float(torch.finfo(current.dtype).eps) * max(1.0, abs(current_objective))
    for backtrack in range(max_backtracks + 1):
        raw_candidate = current - step * gradient
        if bool(torch.isfinite(raw_candidate).all().detach().cpu()):
            candidate = _project_rows_to_simplex_torch(torch, raw_candidate)
            candidate_objective = float(objective_fn(candidate))
            if np.isfinite(candidate_objective) and candidate_objective <= current_objective:
                return candidate, candidate_objective, step, backtrack
            delta = _relative_iterate_change_torch(torch, candidate, current)
            if (
                np.isfinite(candidate_objective)
                and (bool(torch.equal(candidate, current)) or delta <= no_op_tol)
                and candidate_objective - current_objective <= objective_allowance
            ):
                return current, current_objective, step, backtrack
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
    idx = torch.arange(1, values.shape[1] + 1, device=values.device, dtype=values.dtype)
    condition = U - cssv / idx > 0
    rho = condition.sum(dim=1) - 1
    theta = cssv[torch.arange(values.shape[0], device=values.device), rho] / (rho.to(values.dtype) + 1.0)
    return torch.clamp(values - theta[:, None], min=0.0)


def _objective_numpy(X: FloatArray, A: FloatArray, B: FloatArray) -> float:
    return _objective_from_z_numpy(X, A, B @ X)


def _objective_from_z_numpy(X: FloatArray, A: FloatArray, Z: FloatArray) -> float:
    residual = X - A @ Z
    value = float(0.5 * np.sum(residual * residual, dtype=np.float64))
    if not np.isfinite(value):
        raise FloatingPointError("aa_projected_gradient_v2 objective became non-finite.")
    return value


def _objective_torch_value(torch: Any, X: Any, A: Any, B: Any) -> float:
    return _objective_from_z_torch_value(torch, X, A, B @ X)


def _objective_from_z_torch_value(torch: Any, X: Any, A: Any, Z: Any) -> float:
    residual = X - A @ Z
    value = float((0.5 * torch.sum(residual * residual)).detach().cpu())
    if not np.isfinite(value):
        raise FloatingPointError("aa_projected_gradient_v2 objective became non-finite.")
    return value


def _largest_gram_eigenvalue_numpy(matrix: FloatArray) -> float:
    if matrix.shape[0] <= matrix.shape[1]:
        gram = matrix @ matrix.T
    else:
        gram = matrix.T @ matrix
    if gram.shape == (1, 1):
        return max(0.0, float(gram[0, 0]))
    return max(0.0, float(np.linalg.eigvalsh(gram)[-1]))


def _largest_gram_eigenvalue_torch(torch: Any, matrix: Any) -> float:
    gram = matrix @ matrix.T if matrix.shape[0] <= matrix.shape[1] else matrix.T @ matrix
    if tuple(gram.shape) == (1, 1):
        value = gram.reshape(())
    else:
        value = torch.linalg.eigvalsh(gram).amax()
    result = float(torch.clamp(value, min=0).detach().cpu())
    if not np.isfinite(result):
        raise FloatingPointError("aa_projected_gradient_v2 Lipschitz estimate became non-finite.")
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


def _safe_initial_step(step_scale: float, lipschitz: float) -> float:
    if lipschitz <= 0.0:
        return float(step_scale)
    step = float(step_scale) / float(lipschitz)
    if not np.isfinite(step) or step <= 0.0:
        raise FloatingPointError("aa_projected_gradient_v2 step estimate is invalid.")
    return step


def _gradient_mapping_diagnostics_numpy(
    X: FloatArray,
    A: FloatArray,
    B: FloatArray,
    *,
    x_lipschitz: float,
) -> dict[str, float]:
    Z = B @ X
    a_step = _safe_initial_step(1.0, _largest_gram_eigenvalue_numpy(Z))
    a_gradient = (A @ Z - X) @ Z.T
    a_projected = _project_rows_to_simplex_numpy(A - a_step * a_gradient)
    a_mapping_norm = float(np.linalg.norm((A - a_projected) / a_step))

    residual = A @ Z - X
    b_gradient = (A.T @ residual) @ X.T
    b_lipschitz = _largest_gram_eigenvalue_numpy(A) * float(x_lipschitz)
    b_step = _safe_initial_step(1.0, b_lipschitz)
    b_projected = _project_rows_to_simplex_numpy(B - b_step * b_gradient)
    b_mapping_norm = float(np.linalg.norm((B - b_projected) / b_step))
    combined = float(np.hypot(a_mapping_norm, b_mapping_norm))
    if not all(np.isfinite(value) for value in (a_mapping_norm, b_mapping_norm, combined)):
        raise FloatingPointError("AA v2 final gradient-mapping diagnostics are non-finite.")
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
    config: StandardAAV2Config,
    result: Mapping[str, Any],
    device_used: str,
    dtype_used: str,
) -> dict[str, Any]:
    A64 = np.asarray(A, dtype=np.float64)
    B64 = np.asarray(B, dtype=np.float64)
    Z_normalized = B64 @ np.asarray(X_normalized, dtype=np.float64)
    center_unit = np.asarray(normalization.center_unit, dtype=np.float64)
    Z_raw = B64 @ np.asarray(X_raw, dtype=np.float64)
    residual = np.asarray(X_raw, dtype=np.float64) - A64 @ Z_raw
    train_mse = float(np.mean(np.sum(residual * residual, axis=1)))
    if not all(np.isfinite(value).all() for value in (A64, B64, Z_raw)):
        raise FloatingPointError("aa_projected_gradient_v2 produced non-finite factors.")
    if not np.isfinite(train_mse):
        raise FloatingPointError("aa_projected_gradient_v2 produced a non-finite train_mse.")

    public_objective_scale = (float(normalization.raw_scale) * float(normalization.centered_scale)) ** 2
    if not np.isfinite(public_objective_scale):
        raise FloatingPointError("AA v2 public objective scale is non-finite.")
    history: list[dict[str, Any]] = []
    for raw_row in result["history"]:
        row = dict(raw_row)
        normalized_objective = float(row["objective"])
        public_objective = normalized_objective * public_objective_scale
        if not np.isfinite(public_objective):
            raise FloatingPointError("AA v2 public objective history is non-finite.")
        row["normalized_objective"] = normalized_objective
        row["objective"] = public_objective
        row["train_mse"] = 2.0 * public_objective / float(X_raw.shape[0])
        history.append(row)
    termination_reason = str(result["termination_reason"])
    status = _optimization_status(termination_reason)
    final_relative = history[-1].get("signed_relative_objective_improvement")
    gradient_mapping = _gradient_mapping_diagnostics_numpy(
        np.asarray(X_normalized, dtype=np.float64),
        A64,
        B64,
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
    )
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
        "extras": {
            "dtype": dtype_used,
            "b_update": SOLVER_ID,
            "solver_id": SOLVER_ID,
            "solver_contract_version": SOLVER_CONTRACT_VERSION,
            "solver_diagnostics_schema_version": DIAGNOSTICS_SCHEMA_VERSION,
            "solver_diagnostics": diagnostics,
            "optimization_history": history,
        },
    }


def _constant_solution(
    X_raw: FloatArray,
    *,
    k: int,
    seed: int,
    max_iter: int,
    objective_rtol: float,
    config: StandardAAV2Config,
    normalization: AffineNormalization,
    device_used: str,
    dtype_used: str,
    factor_arrays: tuple[FloatArray, FloatArray, FloatArray] | None = None,
) -> dict[str, Any]:
    n_samples = X_raw.shape[0]
    if factor_arrays is None:
        A = np.full((n_samples, k), 1.0 / float(k), dtype=np.float64)
        B = np.full((k, n_samples), 1.0 / float(n_samples), dtype=np.float64)
        Z = B @ np.asarray(X_raw, dtype=np.float64)
    else:
        A, B, Z = factor_arrays
    history = [_history_row(iteration=0, objective=0.0, relative=0.0, plateau_count=0)]
    history[0]["normalized_objective"] = 0.0
    history[0]["train_mse"] = 0.0
    gradient_mapping = {
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
    )
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
        "extras": {
            "dtype": dtype_used,
            "b_update": SOLVER_ID,
            "solver_id": SOLVER_ID,
            "solver_contract_version": SOLVER_CONTRACT_VERSION,
            "solver_diagnostics_schema_version": DIAGNOSTICS_SCHEMA_VERSION,
            "solver_diagnostics": diagnostics,
            "optimization_history": history,
        },
    }


def _build_diagnostics(
    *,
    status: str,
    termination_reason: str,
    n_iter: int,
    max_iter: int,
    objective_rtol: float,
    config: StandardAAV2Config,
    history: list[dict[str, Any]],
    final_relative: Any,
    normalization: AffineNormalization,
    data_lipschitz: float,
    data_lipschitz_method: str,
    gradient_mapping: Mapping[str, float],
) -> dict[str, Any]:
    plateau_reached = termination_reason == "objective_plateau"
    constant_exact = termination_reason == "constant_exact"
    stopping_satisfied = bool(plateau_reached or termination_reason == "constant_exact")
    public_objective_scale = (float(normalization.raw_scale) * float(normalization.centered_scale)) ** 2
    public_objective_underflow = any(
        float(row.get("normalized_objective", 0.0)) > 0.0 and float(row.get("objective", 0.0)) == 0.0 for row in history
    )
    return {
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "solver_id": SOLVER_ID,
        "solver_contract_version": SOLVER_CONTRACT_VERSION,
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
        "public_objective_scale": float(public_objective_scale),
        "public_objective_underflow_detected": bool(public_objective_underflow),
        "convergence_criterion": (
            "constant_exact_short_circuit" if constant_exact else "signed_relative_objective_plateau"
        ),
        "convergence_claim": "exact_constant_solution" if constant_exact else "objective_plateau_only",
        "objective_rtol": float(objective_rtol),
        "objective_patience": int(config.objective_patience),
        "min_outer_iterations": int(config.min_outer_iterations),
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
        "gradient_mapping_a_constraint": "row_simplex",
        "gradient_mapping_b_constraint": "row_simplex",
        **dict(gradient_mapping),
        "data_lipschitz": float(data_lipschitz),
        "data_lipschitz_method": data_lipschitz_method,
        "exact_gram_max_bytes": int(_EXACT_GRAM_MAX_BYTES),
        "normalization_strategy": "raw_max_abs_then_unit_mean_then_centered_max_abs_v1",
        "normalization_raw_scale": float(normalization.raw_scale),
        "normalization_centered_scale": float(normalization.centered_scale),
        "translation_invariant": True,
        "scalar_scale_invariant": True,
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
    if termination_reason == "backtracking_exhausted":
        return "backtracking_exhausted_approximate"
    if termination_reason == "constant_exact":
        return "exact_constant_solution"
    return "approximate_max_iterations"


def _validate_fit_scalars(
    *,
    k: Any,
    seed: Any,
    n_init: Any,
    max_iter: Any,
    tol: Any,
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
            raise TypeError("AA v2 method parameter names must be strings.")
        return dict(method_params)
    has_instance_mapping = True
    try:
        params = dict(vars(method_params))
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
                # Reject unknown descriptors without invoking arbitrary getter code.
                params[key] = class_value
                continue
            try:
                params[key] = getattr(method_params, key)
            except AttributeError:
                continue
    if not has_instance_mapping and not params:
        raise TypeError("AA v2 method_params must be a mapping or attribute object.")
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


def _strict_nonnegative_int(value: Any, *, name: str) -> int:
    result = _strict_int(value, name=name)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative, got {value!r}.")
    return result


def _strict_bounded_nonnegative_int(value: Any, *, name: str, maximum: int) -> int:
    result = _strict_nonnegative_int(value, name=name)
    if result > maximum:
        raise ValueError(f"{name} must be at most {maximum}, got {value!r}.")
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


def _strict_open_unit_float(value: Any, *, name: str) -> float:
    result = _strict_positive_float(value, name=name)
    if result >= 1.0:
        raise ValueError(f"{name} must be strictly between zero and one, got {value!r}.")
    return result


def _strict_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise TypeError(f"{name} must be a boolean, got {value!r}.")
