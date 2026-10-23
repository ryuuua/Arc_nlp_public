"""Opt-in float64 AA with bounded stationarity checks and explicit search.

The dense numerical kernels are a frozen local fork, not imports from v2/v3.
Approximate first-order checks do not certify local or global optimality.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from numbers import Integral, Real
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from .aa_stopping_policy import (
    POLICY_IMPLEMENTATION_DIGEST,
    POLICY_IMPLEMENTATION_VERSION,
    POLICY_PARAM_NAMES,
    PolicyConfig,
    resolve_policy,
    summarize_simplex_block,
)

FloatArray = NDArray[np.floating[Any]]

SOLVER_ID = "aa_projected_gradient_v4"
SOLVER_CONTRACT_VERSION = "aa_b_update_v4"
DIAGNOSTICS_SCHEMA_VERSION = "arc_nlp_standard_aa_diagnostics_v4"
OPTIMIZATION_SCOPE = "full_data_l2_factor_problem"
SOLVER_IMPLEMENTATION_VERSION = "standard_aa_v4_stationarity_search_numerical_repair_v2"
KERNEL_IMPLEMENTATION_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
SOLVER_DEPENDENCY_IDENTITIES = {
    "standard_aa_v4": {
        "version": SOLVER_IMPLEMENTATION_VERSION,
        "digest": KERNEL_IMPLEMENTATION_DIGEST,
    },
    "aa_stopping_policy": {
        "version": POLICY_IMPLEMENTATION_VERSION,
        "digest": POLICY_IMPLEMENTATION_DIGEST,
    },
}
SOLVER_IMPLEMENTATION_DIGEST = hashlib.sha256(
    json.dumps(SOLVER_DEPENDENCY_IDENTITIES, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()

_EXACT_GRAM_MAX_BYTES = 128 * 1024 * 1024
_POWER_ITERATIONS = 64
_FINAL_B_CANONICALIZATION_SCHEMA_VERSION = "arc_nlp.standard_aa_v4.final_b_canonicalization.v1"
_RESTART_DIAGNOSTICS_SCHEMA_VERSION = "arc_nlp.standard_aa_v4.restarts.v1"
_RETURNED_SIMPLEX_FEASIBILITY_TOL = 1e-8
_FINAL_B_CANONICALIZATION_TRIGGER = _RETURNED_SIMPLEX_FEASIBILITY_TOL
_FINAL_B_CANONICALIZATION_MAX_ERROR = 1e-4
DEFAULT_OBJECTIVE_RTOL = 1e-5
DEFAULT_OBJECTIVE_PATIENCE = 3
DEFAULT_MIN_OUTER_ITERATIONS = 5
_METHOD_PARAM_NAMES = (
    frozenset(
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
    | POLICY_PARAM_NAMES
)


@dataclass(frozen=True)
class StandardAAV4Config:
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
    policy: PolicyConfig = field(default_factory=PolicyConfig)


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
    numerical_diagnostics: dict[str, Any] | None = None


class AAOptimizationFailure(RuntimeError):
    """Numerical failure with replay metadata and failure-only safe CPU factors."""

    def __init__(
        self,
        message: str,
        *,
        diagnostics: dict[str, Any],
        snapshot: dict[str, FloatArray],
    ) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics
        self.snapshot = snapshot
        self.metric_whitener: dict[str, Any] | None = None

    def add_note(self, note: str) -> None:
        """Preserve diagnostic failures on Python 3.10 as well as 3.11+."""
        if not isinstance(note, str):
            raise TypeError("note must be a string")
        if not hasattr(self, "__notes__"):
            self.__notes__: list[str] = []
        self.__notes__.append(note)


@dataclass(frozen=True)
class AffineNormalization:
    raw_scale: float
    center_unit: FloatArray
    centered_scale: float
    is_constant: bool


def is_v4_selector(method_params: Any | None) -> bool:
    value = _method_param_value(method_params, "b_update", None)
    return canonicalize_b_update_selector(value) == SOLVER_ID


def canonicalize_b_update_selector(value: Any) -> str:
    """Normalize a selector while requiring a literal string for v4."""

    if type(value) is str:
        return value.strip().lower()
    selector = str(value).strip().lower()
    if selector == SOLVER_ID:
        raise TypeError(
            f"b_update must be a literal string to select {SOLVER_ID!r}; string-coercible objects are unsupported."
        )
    return selector


def validate_method_params(method_params: Any | None) -> StandardAAV4Config:
    """Strictly validate the opt-in v4 numerical parameter surface."""

    params = _method_param_mapping(method_params)
    unknown = sorted(set(params) - _METHOD_PARAM_NAMES)
    if unknown:
        raise ValueError(f"Unsupported aa_projected_gradient_v4 method parameters: {unknown}.")
    if "b_update" in params:
        selector = canonicalize_b_update_selector(params["b_update"])
        if selector != SOLVER_ID:
            raise ValueError(f"b_update must select {SOLVER_ID!r} for the v4 solver.")
    if "record_history" in params:
        _strict_bool(params["record_history"], name="record_history")
    if "force_max_iterations" in params:
        _strict_bool(params["force_max_iterations"], name="force_max_iterations")
    return StandardAAV4Config(
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
        policy=resolve_policy(params),
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
    """Validate a v4 fit without importing or configuring Torch."""

    validate_method_params(method_params)
    k_value, _seed, _n_init, _max_iter, _rtol = _validate_fit_scalars(
        k=k, seed=seed, n_init=n_init, max_iter=max_iter, tol=tol
    )
    if dtype != "float64":
        raise ValueError("AA v4 requires dtype='float64'.")
    work_dtype = np.float64
    X_raw = _validate_real_finite_matrix(X_train, name="X_train", dtype=work_dtype)
    _validate_k(k_value, n_samples=int(X_raw.shape[0]))
    normalization = _affine_normalization_metadata_numpy(X_raw)
    _validate_optional_validation_matrix(
        X_val,
        feature_count=int(X_raw.shape[1]),
        dtype=work_dtype,
        normalization=normalization,
    )


# These helpers are a local numerical fork with v4-only fail-closed guards.
# Keeping the definitions local leaves historical v2/v3 implementations intact.
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
        raise FloatingPointError(
            "Nonconstant X_train collapsed under affine normalization; float64 dynamic range is insufficient."
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
        raise FloatingPointError(
            "Nonconstant X_train collapsed under affine normalization; float64 dynamic range is insufficient."
        )
    return AffineNormalization(
        raw_scale=raw_scale,
        center_unit=center_unit,
        centered_scale=centered_scale,
        is_constant=is_constant,
    )


def fit_standard_aa_v4_numpy(
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
    """Fit independently restarted, checked float64 AA on NumPy."""
    if dtype != "float64":
        raise ValueError("AA v4 requires dtype='float64'.")
    return _fit_standard(
        X_train,
        k=k,
        seed=seed,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        X_val=X_val,
        method_params=method_params,
        torch=None,
        device=None,
    )


def fit_standard_aa_v4_torch(
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
    """Fit independently restarted, checked float64 AA on Torch CPU/CUDA."""
    if dtype != torch.float64:
        raise ValueError("AA v4 requires Torch float64.")
    return _fit_standard(
        X_train,
        k=k,
        seed=seed,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        X_val=X_val,
        method_params=method_params,
        torch=torch,
        device=device,
    )


def _factor_hash(A: FloatArray, B: FloatArray) -> str:
    digest = hashlib.sha256()
    for factor in (A, B):
        array = np.ascontiguousarray(factor, dtype="<f8")
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _to_numpy(value: Any, torch: Any | None) -> FloatArray:
    return np.asarray(value if torch is None else value.detach().cpu().numpy(), dtype=np.float64)


def _mean_square_numpy(X: FloatArray) -> float:
    """Compute the fixed data scale without an n-by-d square temporary."""
    return float(np.einsum("ij,ij->", X, X)) / X.size


def _fit_standard(
    X_train: FloatArray,
    *,
    k: int,
    seed: int,
    n_init: int,
    max_iter: int,
    tol: float,
    X_val: FloatArray | None,
    method_params: Any | None,
    torch: Any | None,
    device: Any,
) -> dict[str, Any]:
    config = validate_method_params(method_params)
    k, seed, n_init, max_iter, objective_rtol = _validate_fit_scalars(
        k=k, seed=seed, n_init=n_init, max_iter=max_iter, tol=tol
    )
    X_raw = _validate_real_finite_matrix(X_train, name="X_train", dtype=np.float64)
    _validate_k(k, n_samples=int(X_raw.shape[0]))
    X_numpy, normalization = affine_normalize_numpy(X_raw)
    _validate_optional_validation_matrix(
        X_val,
        feature_count=int(X_raw.shape[1]),
        dtype=np.float64,
        normalization=normalization,
    )
    # The backend boundary is either a validated NumPy array or a Torch tensor.
    X: Any = X_numpy if torch is None else torch.as_tensor(X_numpy, device=device, dtype=torch.float64)
    device_used = "cpu" if torch is None else str(X.device)
    if normalization.is_constant:
        return _constant_solution(
            X_raw,
            k=k,
            seed=seed,
            n_init=n_init,
            max_iter=max_iter,
            objective_rtol=objective_rtol,
            config=config,
            normalization=normalization,
            device_used=device_used,
            dtype_used="float64",
        )
    gradient_scale = _mean_square_numpy(X_numpy)
    x_lipschitz, spectral_method = _data_lipschitz_numpy(X) if torch is None else _data_lipschitz_torch(torch, X)
    restart_rng = np.random.default_rng(seed)
    perturb_rng = np.random.default_rng(np.random.SeedSequence([seed, 0xAA04]))
    summaries: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_A: FloatArray | None = None
    best_B: FloatArray | None = None
    requested = n_init + config.policy.perturbation_restarts
    for attempt_index in range(requested):
        independent = attempt_index < n_init
        if independent:
            attempt_seed = int(restart_rng.integers(0, 2**31 - 1)) + attempt_index
            initial_B = _initial_archetype_weights_numpy(X_numpy, k=k, seed=attempt_seed)
            initial_A = np.full((X_numpy.shape[0], k), 1.0 / k, dtype=np.float64)
            parent_attempt = None
        else:
            attempt_seed = int(perturb_rng.integers(0, 2**63 - 1))
            if best_A is None or best_B is None or best is None:
                raise RuntimeError("Perturbation requires a checked independent result.")
            rng = np.random.default_rng(attempt_seed)
            one_hot_A = np.zeros_like(best_A)
            one_hot_B = np.zeros_like(best_B)
            one_hot_A[np.arange(best_A.shape[0]), rng.integers(0, k, best_A.shape[0])] = 1.0
            one_hot_B[np.arange(k), rng.integers(0, best_B.shape[1], k)] = 1.0
            q = config.policy.perturbation_strength
            initial_A = (1.0 - q) * best_A + q * one_hot_A
            initial_B = (1.0 - q) * best_B + q * one_hot_B
            parent_attempt = int(best["restart_index"])
        initial_hash = _factor_hash(initial_A, initial_B)
        try:
            if torch is None:
                result = _fit_one_numpy(
                    X,
                    initial_A,
                    initial_B,
                    max_iter=max_iter,
                    objective_rtol=objective_rtol,
                    config=config,
                    x_lipschitz=x_lipschitz,
                    gradient_scale=gradient_scale,
                )
            else:
                result = _fit_one_torch(
                    torch,
                    X,
                    torch.as_tensor(initial_A, device=X.device, dtype=torch.float64),
                    torch.as_tensor(initial_B, device=X.device, dtype=torch.float64),
                    max_iter=max_iter,
                    objective_rtol=objective_rtol,
                    config=config,
                    x_lipschitz=x_lipschitz,
                    gradient_scale=gradient_scale,
                )
        except AAOptimizationFailure as exc:
            # This is failure-only enrichment, not an optimizer retry. Preserve
            # the original numerical failure even if diagnostics cannot copy.
            try:
                exc.diagnostics.update(
                    normalization={
                        "raw_scale": float(normalization.raw_scale),
                        "center_unit": normalization.center_unit.tolist(),
                        "centered_scale": float(normalization.centered_scale),
                    },
                    training_shape=list(X_numpy.shape),
                    device=device_used,
                    dtype="float64",
                    data_lipschitz=float(x_lipschitz),
                    gradient_scale=float(gradient_scale),
                    objective_rtol=float(objective_rtol),
                    effective_solver_config=asdict(config),
                    attempt={
                        "index": attempt_index,
                        "type": "independent" if independent else "perturbation",
                        "seed": attempt_seed,
                        "initial_factor_hash": initial_hash,
                        "parent_attempt_index": parent_attempt,
                    },
                    completed_attempts=summaries,
                )
            except Exception as diagnostic_error:
                exc.add_note(f"Failure metadata enrichment failed: {type(diagnostic_error).__name__}")
            raise
        A = _to_numpy(result["A"], torch)
        B, canonicalization = _canonicalize_final_b(_to_numpy(result["B"], torch))
        # Evaluate exactly the public candidate before ranking starts. No stale
        # optimizer objective or pre-canonicalization stopping flag is used.
        _validate_returned_factors(A, B)
        Z = B @ X_numpy
        residual = A @ Z - X_numpy
        checked_objective = _objective_from_residual_numpy(residual)
        checked = _stationarity_check_numpy(
            X_numpy,
            A,
            B,
            Z=Z,
            residual=residual,
            x_lipschitz=x_lipschitz,
            policy=config.policy,
            gradient_scale=gradient_scale,
        )
        result["diagnostic_checks"] += 1
        result["final_check"] = checked
        result["optimizer_final_normalized_objective"] = float(result["objective"])
        result["objective"] = checked_objective
        result["candidate_canonicalization"] = canonicalization
        result["termination_reason"] = _checked_reason(
            result["termination_reason"],
            checked,
            n_iter=result["n_iter"],
            max_iter=max_iter,
            config=config,
        )
        result.update(
            data_lipschitz=x_lipschitz,
            data_lipschitz_method=spectral_method,
            restart_index=attempt_index,
            init_seed=attempt_seed,
            initial_factor_hash=initial_hash,
            gradient_scale=gradient_scale,
        )
        summary = {
            "restart_index": attempt_index,
            "attempt_type": "independent" if independent else "perturbation",
            "init_seed": attempt_seed,
            "parent_attempt_index": parent_attempt,
            "initial_factor_hash": initial_hash,
            "optimizer_final_normalized_objective": result["optimizer_final_normalized_objective"],
            "returned_normalized_objective": checked_objective,
            "stationarity_check_passed": checked["passed"],
            "termination_reason": result["termination_reason"],
            "outer_iterations": int(result["n_iter"]),
            "a_inner_iterations": int(result["a_inner_iterations"]),
            "b_inner_iterations": int(result["b_inner_iterations"]),
            "diagnostic_checks": int(result["diagnostic_checks"]),
            "a_refinement_stage": int(result["a_refinement_stage"]),
            "b_refinement_stage": int(result["b_refinement_stage"]),
            "refinements": int(result["refinements"]),
        }
        summaries.append(summary)
        if best is None or checked_objective < float(best["objective"]):
            best = {key: value for key, value in result.items() if key not in {"A", "B", "Z", "residual"}}
            best_A, best_B = A.copy(), B.copy()
    if best is None or best_A is None or best_B is None:
        raise RuntimeError("AA v4 did not produce a checked result.")
    best["restart_summaries"] = summaries
    best["restarts_requested"] = n_init
    best["search_totals"] = {
        "total_outer_iterations": sum(row["outer_iterations"] for row in summaries),
        "total_a_inner_iterations": sum(row["a_inner_iterations"] for row in summaries),
        "total_b_inner_iterations": sum(row["b_inner_iterations"] for row in summaries),
        "total_diagnostic_checks": sum(row["diagnostic_checks"] for row in summaries),
        "total_refinements": sum(row["refinements"] for row in summaries),
    }
    return _build_payload(
        X_raw=X_raw,
        seed=seed,
        X_normalized=X_numpy,
        A=best_A,
        B=best_B,
        normalization=normalization,
        max_iter=max_iter,
        objective_rtol=objective_rtol,
        config=config,
        result=best,
        device_used=device_used,
        dtype_used="float64",
    )


def _validate_returned_factors(A: FloatArray, B: FloatArray) -> None:
    for name, factor in (("A", A), ("B", B)):
        if not np.isfinite(factor).all() or float(np.min(factor)) < 0.0:
            raise FloatingPointError(f"AA v4 returned invalid {name} coefficients.")
        if float(np.max(np.abs(np.sum(factor, axis=1) - 1.0))) > _RETURNED_SIMPLEX_FEASIBILITY_TOL:
            raise FloatingPointError(f"AA v4 returned infeasible {name} coefficients.")


def _checked_reason(
    reason: str,
    check: Mapping[str, Any],
    *,
    n_iter: int,
    max_iter: int,
    config: StandardAAV4Config,
) -> str:
    if config.force_max_iterations:
        return "max_iterations"
    if check["passed"] and n_iter >= config.min_outer_iterations:
        return "first_order_stationary"
    if reason == "first_order_stationary":
        return "max_iterations" if n_iter >= max_iter else "stalled_nonstationary"
    return str(reason)


def infer_standard_aa_v4_numpy(
    X: FloatArray,
    Z: FloatArray,
    *,
    max_iter: int,
    tol: float,
    step_scale: float,
    dtype: str,
) -> FloatArray:
    """Infer dense row-simplex coefficients in saved v4 coordinates."""

    if dtype != "float64":
        raise ValueError("AA v4 requires dtype='float64'.")
    work_dtype = np.float64
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
        raise RuntimeError("AA v4 transform backtracking exhausted without a safe iterate.")
    return np.asarray(update.value, dtype=work_dtype)


def infer_standard_aa_v4_torch(
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
    """Torch counterpart of dense v4 coefficient inference."""

    if dtype != torch.float64:
        raise ValueError("AA v4 requires Torch float64.")
    numpy_dtype = np.float64
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
        raise RuntimeError("AA v4 transform backtracking exhausted without a safe iterate.")
    return np.asarray(update.value.detach().cpu().numpy())


def read_standard_aa_diagnostics(artifact: Mapping[str, Any] | None) -> dict[str, Any]:
    """Read and strictly validate a v4 diagnostics payload when selected."""

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
        raise ValueError("aa_projected_gradient_v4 diagnostics require the exact v4 schema version.")
    if payload.get("solver_id") != SOLVER_ID:
        raise ValueError("v4 diagnostics solver_id mismatch.")
    if payload.get("solver_contract_version") != SOLVER_CONTRACT_VERSION:
        raise ValueError("v4 diagnostics solver_contract_version mismatch.")
    if payload.get("solver_implementation_version") != SOLVER_IMPLEMENTATION_VERSION:
        raise ValueError("v4 diagnostics solver_implementation_version mismatch.")
    if payload.get("solver_implementation_digest") != SOLVER_IMPLEMENTATION_DIGEST:
        raise ValueError("v4 diagnostics solver_implementation_digest mismatch.")
    if payload.get("solver_dependency_identities") != SOLVER_DEPENDENCY_IDENTITIES:
        raise ValueError("v4 diagnostics solver dependency identities mismatch.")
    outer_dependencies = outer.get("solver_dependency_identities")
    if outer_dependencies is not None and outer_dependencies != payload["solver_dependency_identities"]:
        raise ValueError("Outer and nested Standard-AA solver dependency identities disagree.")
    recorded_policy = payload.get("stopping_policy")
    if not isinstance(recorded_policy, Mapping) or set(recorded_policy) != set(POLICY_PARAM_NAMES):
        raise ValueError("v4 diagnostics require the complete stopping policy.")
    if dict(recorded_policy) != asdict(resolve_policy(recorded_policy)):
        raise ValueError("v4 diagnostics stopping policy is not canonical.")
    if payload.get("termination_reason") not in {
        "first_order_stationary",
        "stalled_nonstationary",
        "max_iterations",
        "constant_exact",
    }:
        raise ValueError("v4 diagnostics termination reason is unsupported.")
    if type(payload.get("stationarity_check_passed")) is not bool:
        raise ValueError("v4 diagnostics require a numerical stationarity check flag.")
    for key in ("stationarity_certified", "kkt_certified", "global_optimality_certified"):
        if payload.get(key) is not False:
            raise ValueError(f"v4 diagnostics must not claim {key}.")
    final_check = payload.get("final_stationarity_check")
    if not isinstance(final_check, Mapping) or final_check.get("passed") is not payload["stationarity_check_passed"]:
        raise ValueError("v4 diagnostics final stationarity check disagrees with its summary.")
    if payload.get("optimization_scope") != OPTIMIZATION_SCOPE:
        raise ValueError(f"v4 diagnostics optimization_scope must be {OPTIMIZATION_SCOPE!r}.")
    return payload


def _stationarity_check_numpy(
    X: FloatArray,
    A: FloatArray,
    B: FloatArray,
    *,
    Z: FloatArray,
    residual: FloatArray,
    x_lipschitz: float,
    policy: PolicyConfig,
    gradient_scale: float | None = None,
) -> dict[str, Any]:
    scale = _mean_square_numpy(X) if gradient_scale is None else gradient_scale
    a_gradient = residual @ Z.T
    a_step = _safe_initial_step(1.0, _largest_gram_eigenvalue_numpy(Z))
    a_projected = _project_rows_to_simplex_numpy(A - a_step * a_gradient)
    a_check = summarize_simplex_block(
        A,
        a_gradient,
        a_projected,
        gradient_scale=scale,
        policy=policy,
    )
    b_gradient = (A.T @ residual) @ X.T
    b_step = _safe_initial_step(1.0, _largest_gram_eigenvalue_numpy(A) * x_lipschitz)
    b_projected = _project_rows_to_simplex_numpy(B - b_step * b_gradient)
    b_check = summarize_simplex_block(
        B,
        b_gradient,
        b_projected,
        gradient_scale=X.shape[0] * scale,
        policy=policy,
    )
    return {"a": a_check, "b": b_check, "passed": bool(a_check["passed"] and b_check["passed"])}


def _stationarity_check_torch(
    torch: Any,
    X: Any,
    A: Any,
    B: Any,
    *,
    Z: Any,
    residual: Any,
    x_lipschitz: float,
    policy: PolicyConfig,
    gradient_scale: float | None = None,
) -> dict[str, Any]:
    scale = (
        float((torch.linalg.vector_norm(X) ** 2).detach().cpu()) / X.numel()
        if gradient_scale is None
        else gradient_scale
    )
    a_gradient = residual @ Z.T
    a_step = _safe_initial_step(1.0, _largest_gram_eigenvalue_torch(torch, Z))
    a_projected = _project_rows_to_simplex_torch(torch, A - a_step * a_gradient)
    a_check = summarize_simplex_block(
        A,
        a_gradient,
        a_projected,
        gradient_scale=scale,
        policy=policy,
        torch=torch,
    )
    b_gradient = (A.T @ residual) @ X.T
    b_step = _safe_initial_step(1.0, _largest_gram_eigenvalue_torch(torch, A) * x_lipschitz)
    b_projected = _project_rows_to_simplex_torch(torch, B - b_step * b_gradient)
    b_check = summarize_simplex_block(
        B,
        b_gradient,
        b_projected,
        gradient_scale=X.shape[0] * scale,
        policy=policy,
        torch=torch,
    )
    return {"a": a_check, "b": b_check, "passed": bool(a_check["passed"] and b_check["passed"])}


def _advance_policy(
    check: Mapping[str, Any] | None,
    *,
    iteration: int,
    max_iter: int,
    plateau_count: int,
    a_stage: int,
    b_stage: int,
    config: StandardAAV4Config,
) -> tuple[str | None, int, int, int]:
    if config.force_max_iterations or iteration < config.min_outer_iterations or check is None:
        return None, a_stage, b_stage, plateau_count
    if check["passed"]:
        return "first_order_stationary", a_stage, b_stage, plateau_count
    if iteration >= max_iter or plateau_count < config.objective_patience:
        return None, a_stage, b_stage, plateau_count
    next_a, next_b = a_stage, b_stage
    if not check["a"]["passed"] and a_stage < config.policy.max_refinements:
        next_a += 1
    if not check["b"]["passed"] and b_stage < config.policy.max_refinements:
        next_b += 1
    if (next_a, next_b) == (a_stage, b_stage):
        return "stalled_nonstationary", a_stage, b_stage, plateau_count
    return None, next_a, next_b, 0


def _raise_block_failure(
    message: str,
    update: _BlockUpdate,
    *,
    A: Any,
    B: Any,
    Z: Any,
    block_stage: str,
    iteration: int,
    a_stage: int,
    b_stage: int,
    config: StandardAAV4Config,
    max_iter: int,
    history: list[dict[str, Any]],
    torch: Any | None,
) -> None:
    diagnostics: dict[str, Any] = {
        "schema_version": "arc_nlp.standard_aa_v4.optimization_failure.v1",
        "coordinate_space": "affine_normalized_training_space",
        "solver_id": SOLVER_ID,
        "solver_implementation_version": SOLVER_IMPLEMENTATION_VERSION,
        "solver_implementation_digest": SOLVER_IMPLEMENTATION_DIGEST,
        "block_stage": block_stage,
        "outer_iteration": iteration,
        "a_refinement_stage": a_stage,
        "b_refinement_stage": b_stage,
        "outer_iteration_budget": max_iter,
        "effective_inner_iteration_budget": (
            config.b_max_iter * config.policy.refinement_multiplier**b_stage
            if block_stage == "b"
            else config.a_max_iter * config.policy.refinement_multiplier**a_stage
        ),
        "accepted_inner_iterations": update.iterations,
        "snapshot_normalized_objective": float(update.objective) if np.isfinite(update.objective) else None,
        "total_block_backtracks": update.backtracks,
        "accepted_history": history,
        "history_coordinate_space": "affine_normalized_training_space",
        "objective_definition": "0.5 * sum((A @ Z - X_working)**2)",
        "history_covers": "completed_outer_iterations_before_failed_block",
        "snapshot_state": "last_accepted_inner_iterate",
        "stationarity_certified": False,
        "kkt_certified": False,
        "global_optimality_certified": False,
    }
    failure = AAOptimizationFailure(message, diagnostics=diagnostics, snapshot={})
    # No CPU factor copies occur on the successful block path. A diagnostic
    # failure must never replace the optimization failure being reported.
    try:
        diagnostics.update(update.numerical_diagnostics or {})
        safe_A = A if block_stage == "b" else update.value
        safe_B = update.value if block_stage == "b" else B
        safe_Z = Z if update.archetypes is None else update.archetypes
        snapshot = {
            "A": _to_numpy(safe_A, torch).copy(),
            "B": _to_numpy(safe_B, torch).copy(),
            "Z": _to_numpy(safe_Z, torch).copy(),
        }
        a_evidence = _simplex_evidence(snapshot["A"], None)
        b_evidence = _simplex_evidence(snapshot["B"], None)
        diagnostics["snapshot_evidence"] = {
            "a": a_evidence,
            "b": b_evidence,
            "z_finite": _finite_array(snapshot["Z"], None),
            "residual_finite": update.residual is not None and _finite_array(update.residual, torch),
            "objective_finite_nonnegative": bool(np.isfinite(update.objective) and update.objective >= 0.0),
        }
        safe = bool(
            a_evidence["simplex_feasible"]
            and b_evidence["simplex_feasible"]
            and diagnostics["snapshot_evidence"]["z_finite"]
            and diagnostics["snapshot_evidence"]["residual_finite"]
            and diagnostics["snapshot_evidence"]["objective_finite_nonnegative"]
            and diagnostics.get("status") not in {"invalid_current", "invalid_current_objective"}
        )
        diagnostics["snapshot_safe"] = safe
        if safe:
            failure.snapshot = snapshot
        else:
            diagnostics["snapshot_state"] = "invalid_state_not_saved"
    except Exception as diagnostic_error:
        failure.add_note(f"Failure snapshot capture failed: {type(diagnostic_error).__name__}")
        diagnostics["snapshot_capture_error"] = type(diagnostic_error).__name__
    raise failure


def _fit_one_numpy(
    X: FloatArray,
    A: FloatArray,
    B: FloatArray,
    *,
    max_iter: int,
    objective_rtol: float,
    config: StandardAAV4Config,
    x_lipschitz: float,
    gradient_scale: float | None = None,
) -> dict[str, Any]:
    gradient_scale = _mean_square_numpy(X) if gradient_scale is None else gradient_scale
    Z = B @ X
    residual = A @ Z
    residual -= X
    objective = _objective_from_residual_numpy(residual)
    history = [_history_row(iteration=0, objective=objective, relative=None, plateau_count=0)]
    plateau_count = 0
    termination_reason = "max_iterations"
    n_iter = 0
    a_stage = b_stage = 0
    a_iterations = b_iterations = diagnostic_checks = 0
    final_check = None
    for iteration in range(1, max_iter + 1):
        a_update = _BlockUpdate(A, objective, 0, 0, 0.0, True, Z, residual)
        if iteration == 1:
            a_update = _update_a_numpy(
                X,
                Z,
                A,
                residual=residual,
                objective=objective,
                max_iter=config.a_max_iter * config.policy.refinement_multiplier**a_stage,
                tol=config.a_tol if a_stage == 0 else 0.0,
                initial_step=_safe_initial_step(config.a_step_scale, _largest_gram_eigenvalue_numpy(Z)),
                backtracking_factor=config.backtracking_factor,
                max_backtracks=config.max_backtracks,
            )
            if not a_update.available:
                _raise_block_failure(
                    "AA v4 A-update backtracking exhausted without a safe iterate.",
                    a_update,
                    A=A,
                    B=B,
                    Z=Z,
                    block_stage="initial_a",
                    iteration=iteration,
                    a_stage=a_stage,
                    b_stage=b_stage,
                    config=config,
                    max_iter=max_iter,
                    history=history,
                    torch=None,
                )
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
            max_iter=config.b_max_iter * config.policy.refinement_multiplier**b_stage,
            tol=config.b_tol if b_stage == 0 else 0.0,
            initial_step=_safe_initial_step(
                config.b_step_scale,
                _largest_gram_eigenvalue_numpy(A) * x_lipschitz,
            ),
            backtracking_factor=config.backtracking_factor,
            max_backtracks=config.max_backtracks,
        )
        if not b_update.available:
            _raise_block_failure(
                "AA v4 B-update backtracking exhausted without a safe iterate.",
                b_update,
                A=A,
                B=B,
                Z=Z,
                block_stage="b",
                iteration=iteration,
                a_stage=a_stage,
                b_stage=b_stage,
                config=config,
                max_iter=max_iter,
                history=history,
                torch=None,
            )
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
            max_iter=config.a_max_iter * config.policy.refinement_multiplier**a_stage,
            tol=config.a_tol if a_stage == 0 else 0.0,
            initial_step=_safe_initial_step(config.a_step_scale, a_lipschitz),
            backtracking_factor=config.backtracking_factor,
            max_backtracks=config.max_backtracks,
        )
        if not polished_a_update.available:
            _raise_block_failure(
                "AA v4 final A-update backtracking exhausted without a safe iterate.",
                polished_a_update,
                A=A,
                B=B,
                Z=Z,
                block_stage="polish_a",
                iteration=iteration,
                a_stage=a_stage,
                b_stage=b_stage,
                config=config,
                max_iter=max_iter,
                history=history,
                torch=None,
            )
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
        a_iterations += a_update.iterations
        b_iterations += b_update.iterations
        should_check = (
            iteration % config.policy.stationarity_check_interval == 0
            or 0.0 <= relative <= objective_rtol
            or iteration == max_iter
        )
        check = None
        if should_check:
            check = _stationarity_check_numpy(
                X,
                A,
                B,
                Z=Z,
                residual=residual,
                x_lipschitz=x_lipschitz,
                policy=config.policy,
                gradient_scale=gradient_scale,
            )
            final_check = check
            diagnostic_checks += 1
            history[-1]["stationarity_check"] = check
        reason, a_stage, b_stage, plateau_count = _advance_policy(
            check,
            iteration=iteration,
            max_iter=max_iter,
            plateau_count=plateau_count,
            a_stage=a_stage,
            b_stage=b_stage,
            config=config,
        )
        history[-1]["plateau_count"] = plateau_count
        history[-1]["a_refinement_stage"] = a_stage
        history[-1]["b_refinement_stage"] = b_stage
        if reason is not None:
            termination_reason = reason
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
        "a_inner_iterations": a_iterations,
        "b_inner_iterations": b_iterations,
        "diagnostic_checks": diagnostic_checks,
        "a_refinement_stage": a_stage,
        "b_refinement_stage": b_stage,
        "refinements": a_stage + b_stage,
        "final_check": final_check,
    }


def _fit_one_torch(
    torch: Any,
    X: Any,
    A: Any,
    B: Any,
    *,
    max_iter: int,
    objective_rtol: float,
    config: StandardAAV4Config,
    x_lipschitz: float,
    gradient_scale: float | None = None,
) -> dict[str, Any]:
    if gradient_scale is None:
        gradient_scale = float((torch.linalg.vector_norm(X) ** 2).detach().cpu()) / X.numel()
    Z = B @ X
    residual = A @ Z
    residual.sub_(X)
    objective = _objective_from_residual_torch(torch, residual)
    history = [_history_row(iteration=0, objective=objective, relative=None, plateau_count=0)]
    plateau_count = 0
    termination_reason = "max_iterations"
    n_iter = 0
    a_stage = b_stage = 0
    a_iterations = b_iterations = diagnostic_checks = 0
    final_check = None
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
                max_iter=config.a_max_iter * config.policy.refinement_multiplier**a_stage,
                tol=config.a_tol if a_stage == 0 else 0.0,
                initial_step=_safe_initial_step(config.a_step_scale, _largest_gram_eigenvalue_torch(torch, Z)),
                backtracking_factor=config.backtracking_factor,
                max_backtracks=config.max_backtracks,
            )
            if not a_update.available:
                _raise_block_failure(
                    "AA v4 A-update backtracking exhausted without a safe iterate.",
                    a_update,
                    A=A,
                    B=B,
                    Z=Z,
                    block_stage="initial_a",
                    iteration=iteration,
                    a_stage=a_stage,
                    b_stage=b_stage,
                    config=config,
                    max_iter=max_iter,
                    history=history,
                    torch=torch,
                )
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
            max_iter=config.b_max_iter * config.policy.refinement_multiplier**b_stage,
            tol=config.b_tol if b_stage == 0 else 0.0,
            initial_step=_safe_initial_step(
                config.b_step_scale,
                _largest_gram_eigenvalue_torch(torch, A) * x_lipschitz,
            ),
            backtracking_factor=config.backtracking_factor,
            max_backtracks=config.max_backtracks,
        )
        if not b_update.available:
            _raise_block_failure(
                "AA v4 B-update backtracking exhausted without a safe iterate.",
                b_update,
                A=A,
                B=B,
                Z=Z,
                block_stage="b",
                iteration=iteration,
                a_stage=a_stage,
                b_stage=b_stage,
                config=config,
                max_iter=max_iter,
                history=history,
                torch=torch,
            )
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
            max_iter=config.a_max_iter * config.policy.refinement_multiplier**a_stage,
            tol=config.a_tol if a_stage == 0 else 0.0,
            initial_step=_safe_initial_step(config.a_step_scale, _largest_gram_eigenvalue_torch(torch, Z)),
            backtracking_factor=config.backtracking_factor,
            max_backtracks=config.max_backtracks,
        )
        if not polished_a_update.available:
            _raise_block_failure(
                "AA v4 final A-update backtracking exhausted without a safe iterate.",
                polished_a_update,
                A=A,
                B=B,
                Z=Z,
                block_stage="polish_a",
                iteration=iteration,
                a_stage=a_stage,
                b_stage=b_stage,
                config=config,
                max_iter=max_iter,
                history=history,
                torch=torch,
            )
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
        a_iterations += a_update.iterations
        b_iterations += b_update.iterations
        should_check = (
            iteration % config.policy.stationarity_check_interval == 0
            or 0.0 <= relative <= objective_rtol
            or iteration == max_iter
        )
        check = None
        if should_check:
            check = _stationarity_check_torch(
                torch,
                X,
                A,
                B,
                Z=Z,
                residual=residual,
                x_lipschitz=x_lipschitz,
                policy=config.policy,
                gradient_scale=gradient_scale,
            )
            final_check = check
            diagnostic_checks += 1
            history[-1]["stationarity_check"] = check
        reason, a_stage, b_stage, plateau_count = _advance_policy(
            check,
            iteration=iteration,
            max_iter=max_iter,
            plateau_count=plateau_count,
            a_stage=a_stage,
            b_stage=b_stage,
            config=config,
        )
        history[-1]["plateau_count"] = plateau_count
        history[-1]["a_refinement_stage"] = a_stage
        history[-1]["b_refinement_stage"] = b_stage
        if reason is not None:
            termination_reason = reason
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
        "a_inner_iterations": a_iterations,
        "b_inner_iterations": b_iterations,
        "diagnostic_checks": diagnostic_checks,
        "a_refinement_stage": a_stage,
        "b_refinement_stage": b_stage,
        "refinements": a_stage + b_stage,
        "final_check": final_check,
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
    effective_tol = float(tol)
    numerical_no_op_tol = _numerical_no_op_tolerance(current)
    diagnostic_record: dict[str, Any] = {
        "algorithmic_tol": effective_tol,
        "numerical_no_op_tol": numerical_no_op_tol,
        "initial_step": float(initial_step),
        "inner_iteration": 0,
        "backtracks": 0,
        "status": "starting",
    }
    initial_evidence = _simplex_evidence(current, None)
    if (
        not initial_evidence["simplex_feasible"]
        or not _finite_array(current_residual, None)
        or not _finite_array(Z, None)
        or not np.isfinite(current_objective)
        or current_objective < 0.0
    ):
        diagnostic_record.update(status="invalid_current", current_evidence=initial_evidence)
        return _BlockUpdate(
            current,
            current_objective,
            0,
            0,
            0.0,
            False,
            Z,
            current_residual,
            numerical_diagnostics=dict(diagnostic_record),
        )
    total_backtracks = 0
    accepted_step = initial_step
    for iteration in range(1, max_iter + 1):
        diagnostic_record["inner_iteration"] = iteration
        gradient = current_residual @ Z.T
        raw_preview = current - initial_step * gradient
        preview = _project_rows_to_simplex_numpy(raw_preview)
        preview_delta = _relative_iterate_change_numpy(preview, current)
        if (
            preview_delta <= max(effective_tol, numerical_no_op_tol)
            and _simplex_evidence(preview, None)["simplex_feasible"]
        ):
            diagnostic_record.update(
                status="numerical_no_op" if preview_delta <= numerical_no_op_tol else "algorithmic_tolerance",
                candidate_evidence={"displacement": float(preview_delta), "finite": True, "simplex_feasible": True},
            )
            return _BlockUpdate(
                current,
                current_objective,
                iteration - 1,
                total_backtracks,
                initial_step,
                True,
                Z,
                current_residual,
                numerical_diagnostics=dict(diagnostic_record),
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
            no_op_tol=numerical_no_op_tol,
            diagnostic_record=diagnostic_record,
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
                numerical_diagnostics=dict(diagnostic_record),
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
        if delta <= max(effective_tol, numerical_no_op_tol):
            return _BlockUpdate(
                current,
                current_objective,
                iteration,
                total_backtracks,
                accepted_step,
                True,
                Z,
                current_residual,
                numerical_diagnostics=dict(diagnostic_record),
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
        numerical_diagnostics=dict(diagnostic_record),
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
    effective_tol = float(tol)
    numerical_no_op_tol = _numerical_no_op_tolerance(current)
    diagnostic_record: dict[str, Any] = {
        "algorithmic_tol": effective_tol,
        "numerical_no_op_tol": numerical_no_op_tol,
        "initial_step": float(initial_step),
        "inner_iteration": 0,
        "backtracks": 0,
        "status": "starting",
    }
    initial_evidence = _simplex_evidence(current, None)
    if (
        not initial_evidence["simplex_feasible"]
        or not _finite_array(current_residual, None)
        or not _finite_array(current_z, None)
        or not np.isfinite(current_objective)
        or current_objective < 0.0
    ):
        diagnostic_record.update(status="invalid_current", current_evidence=initial_evidence)
        return _BlockUpdate(
            current,
            current_objective,
            0,
            0,
            0.0,
            False,
            current_z,
            current_residual,
            numerical_diagnostics=dict(diagnostic_record),
        )
    total_backtracks = 0
    accepted_step = initial_step
    for iteration in range(1, max_iter + 1):
        diagnostic_record["inner_iteration"] = iteration
        gradient = (A.T @ current_residual) @ X.T
        raw_preview = current - initial_step * gradient
        preview = _project_rows_to_simplex_numpy(raw_preview)
        preview_delta = _relative_iterate_change_numpy(preview, current)
        if (
            preview_delta <= max(effective_tol, numerical_no_op_tol)
            and _simplex_evidence(preview, None)["simplex_feasible"]
        ):
            diagnostic_record.update(
                status="numerical_no_op" if preview_delta <= numerical_no_op_tol else "algorithmic_tolerance",
                candidate_evidence={"displacement": float(preview_delta), "finite": True, "simplex_feasible": True},
            )
            return _BlockUpdate(
                current,
                current_objective,
                iteration - 1,
                total_backtracks,
                initial_step,
                True,
                current_z,
                current_residual,
                numerical_diagnostics=dict(diagnostic_record),
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
            no_op_tol=numerical_no_op_tol,
            diagnostic_record=diagnostic_record,
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
                numerical_diagnostics=dict(diagnostic_record),
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
        if delta <= max(effective_tol, numerical_no_op_tol):
            return _BlockUpdate(
                current,
                current_objective,
                iteration,
                total_backtracks,
                accepted_step,
                True,
                current_z,
                current_residual,
                numerical_diagnostics=dict(diagnostic_record),
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
        numerical_diagnostics=dict(diagnostic_record),
    )


def _numerical_no_op_tolerance(weights: Any) -> float:
    # The sorted cumulative-sum projector has dimension-dependent roundoff.
    # This guard never means first-order convergence; it only retains the
    # original feasible iterate when a displacement is numerically unresolved.
    return float(16.0 * np.finfo(np.float64).eps * max(1, int(weights.shape[1])))


def _finite_array(value: Any, torch: Any | None) -> bool:
    return bool(np.isfinite(value).all()) if torch is None else bool(torch.isfinite(value).all().detach().cpu())


def _simplex_evidence(value: Any, torch: Any | None) -> dict[str, Any]:
    finite = _finite_array(value, torch)
    if not finite:
        return {"finite": False, "simplex_feasible": False, "minimum_weight": None, "row_sum_error": None}
    if torch is None:
        minimum = float(np.min(value))
        error = float(np.max(np.abs(np.sum(value, axis=1) - 1.0)))
    else:
        minimum = float(torch.min(value).detach().cpu())
        error = float(torch.max(torch.abs(torch.sum(value, dim=1) - 1.0)).detach().cpu())
    return {
        "finite": True,
        "simplex_feasible": bool(minimum >= 0.0 and error <= _RETURNED_SIMPLEX_FEASIBILITY_TOL),
        "minimum_weight": minimum,
        "row_sum_error": error,
    }


def _squared_norm(value: Any, torch: Any | None) -> float:
    if torch is None:
        return float(np.einsum("ij,ij->", value, value))
    return float(torch.sum(value * value).detach().cpu())


def _quadratic_loss_change(
    residual: Any,
    displacement_residual: Any,
    *,
    torch: Any | None,
) -> dict[str, float | bool]:
    """Evaluate the quadratic identity with a heuristic FP64 sign guard.

    Do not form displacement_residual by subtracting two large residuals.
    Scale the sign guard by actual dot-product terms, not the residual norm:
    a large orthogonal residual contributes no cancellation to this difference.
    NumPy uses bounded product chunks; Torch reuses one product buffer in-place.
    The fixed 64-eps margin is an engineering guard, not a certified error bound
    for arbitrary sizes, reduction orders, or preceding matrix multiplications.
    """
    if torch is None:
        linear = float(np.einsum("ij,ij->", residual, displacement_residual))
        absolute_product_sum = 0.0
        # Bound the diagnostic temporary to approximately 8 MiB without copies
        # of either complete input (including non-contiguous inputs).
        chunk_rows = max(1, 1_048_576 // int(residual.shape[1]))
        for start in range(0, int(residual.shape[0]), chunk_rows):
            product = residual[start : start + chunk_rows] * displacement_residual[start : start + chunk_rows]
            np.abs(product, out=product)
            absolute_product_sum += float(np.sum(product))
    else:
        product = residual * displacement_residual
        linear = float(torch.sum(product).detach().cpu())
        product.abs_()
        absolute_product_sum = float(torch.sum(product).detach().cpu())
        del product
    displacement_squared = _squared_norm(displacement_residual, torch)
    quadratic = 0.5 * displacement_squared
    stable_delta = linear + quadratic
    error_bound = float(64.0 * np.finfo(np.float64).eps * (absolute_product_sum + quadratic))
    if not all(np.isfinite(v) for v in (linear, absolute_product_sum, quadratic, stable_delta, error_bound)):
        raise FloatingPointError("AA v4 quadratic candidate difference is non-finite.")
    # A rounded full objective tie must never authorize positive quadratic
    # ascent. Exactly flat reconstruction directions are safe as well.
    descent = stable_delta < -error_bound or (stable_delta == 0.0 and quadratic == 0.0)
    return {
        "stable_delta": stable_delta,
        "linear_delta": linear,
        "quadratic_delta": quadratic,
        "absolute_product_sum": absolute_product_sum,
        "stable_delta_error_bound": error_bound,
        "confident_nonincrease": bool(descent),
    }


def _backtracked_candidate(
    X: Any,
    fixed: Any,
    current: Any,
    gradient: Any,
    *,
    block: str,
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
    torch: Any | None,
    diagnostic_record: dict[str, Any],
) -> tuple[Any, float, float, int, Any, Any, float] | None:
    project = _project_rows_to_simplex_numpy if torch is None else lambda v: _project_rows_to_simplex_torch(torch, v)
    relative = (
        _relative_iterate_change_numpy if torch is None else lambda a, b: _relative_iterate_change_torch(torch, a, b)
    )
    objective_of = (
        _objective_from_residual_numpy if torch is None else lambda r: _objective_from_residual_torch(torch, r)
    )
    allowance = 64.0 * np.finfo(np.float64).eps * max(1.0, abs(current_objective))
    evidence = _simplex_evidence(current, torch)
    diagnostic_record.update(
        status="searching",
        current_evidence=evidence,
        initial_step=float(initial_step),
        numerical_no_op_tol=float(no_op_tol),
        max_backtracks=int(max_backtracks),
    )
    if not evidence["simplex_feasible"] or not _finite_array(current_residual, torch):
        diagnostic_record.update(status="invalid_current", candidate_evidence=evidence)
        return None
    if not np.isfinite(current_objective):
        diagnostic_record.update(status="invalid_current_objective")
        return None
    step = initial_step
    for backtrack in range(max_backtracks + 1):
        candidate_evidence: dict[str, Any] = {
            "backtrack": backtrack,
            "step": float(step),
            "stable_delta": None,
            "direct_delta": None,
            "displacement": None,
        }
        diagnostic_record.update(last_step=float(step), backtracks=backtrack, candidate_evidence=candidate_evidence)
        if backtrack == 0:
            raw_finite = preview_raw_finite
            candidate = preview
            delta = preview_delta
        else:
            raw = current - step * gradient
            raw_finite = _finite_array(raw, torch)
            if not raw_finite:
                candidate_evidence.update(finite=False, simplex_feasible=False)
                step *= backtracking_factor
                continue
            candidate = project(raw)
            delta = relative(candidate, current)
        if not raw_finite:
            candidate_evidence.update(finite=False, simplex_feasible=False)
            step *= backtracking_factor
            continue
        candidate_evidence.update(_simplex_evidence(candidate, torch), displacement=float(delta))
        if not candidate_evidence["simplex_feasible"]:
            step *= backtracking_factor
            continue
        displacement = candidate - current
        if block == "a":
            displacement_residual = displacement @ fixed
            candidate_z = current_z
            candidate_residual = candidate @ fixed - X
        else:
            displacement_z = displacement @ X
            displacement_residual = fixed @ displacement_z
            candidate_z = candidate @ X
            candidate_residual = fixed @ candidate_z - X
        if not _finite_array(candidate_residual, torch) or not _finite_array(displacement_residual, torch):
            candidate_evidence.update(finite=False)
            step *= backtracking_factor
            continue
        try:
            change = _quadratic_loss_change(
                current_residual,
                displacement_residual,
                torch=torch,
            )
            candidate_objective = objective_of(candidate_residual)
        except FloatingPointError as exc:
            candidate_evidence.update(finite=False, numerical_error=str(exc))
            step *= backtracking_factor
            continue
        candidate_evidence.update(change, candidate_objective=candidate_objective)
        candidate_evidence["direct_delta"] = float(candidate_objective - current_objective)
        coherent = candidate_objective - current_objective <= allowance
        candidate_evidence["objective_roundoff_allowance"] = float(allowance)
        candidate_evidence["objective_coherent"] = bool(coherent)
        if delta <= no_op_tol and coherent:
            diagnostic_record["status"] = "numerical_no_op"
            return current, current_objective, step, backtrack, current_z, current_residual, 0.0
        if change["confident_nonincrease"] and coherent:
            diagnostic_record["status"] = "accepted"
            return candidate, candidate_objective, step, backtrack, candidate_z, candidate_residual, delta
        step *= backtracking_factor
    diagnostic_record.update(status="candidate_exhaustion", backtracks=max_backtracks + 1)
    return None


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
    diagnostic_record: dict[str, Any] | None = None,
) -> tuple[FloatArray, float, float, int, FloatArray, float] | None:
    update = _backtracked_candidate(
        X,
        Z,
        current,
        gradient,
        block="a",
        preview=preview,
        preview_delta=preview_delta,
        preview_raw_finite=preview_raw_finite,
        current_z=Z,
        current_residual=current_residual,
        current_objective=current_objective,
        initial_step=initial_step,
        backtracking_factor=backtracking_factor,
        max_backtracks=max_backtracks,
        no_op_tol=no_op_tol,
        torch=None,
        diagnostic_record={} if diagnostic_record is None else diagnostic_record,
    )
    if update is None:
        return None
    value, objective, step, backtrack, _z, residual, delta = update
    return value, objective, step, backtrack, residual, delta


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
    diagnostic_record: dict[str, Any] | None = None,
) -> tuple[FloatArray, float, float, int, FloatArray, FloatArray, float] | None:
    return _backtracked_candidate(
        X,
        A,
        current,
        gradient,
        block="b",
        preview=preview,
        preview_delta=preview_delta,
        preview_raw_finite=preview_raw_finite,
        current_z=current_z,
        current_residual=current_residual,
        current_objective=current_objective,
        initial_step=initial_step,
        backtracking_factor=backtracking_factor,
        max_backtracks=max_backtracks,
        no_op_tol=no_op_tol,
        torch=None,
        diagnostic_record={} if diagnostic_record is None else diagnostic_record,
    )


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
    effective_tol = float(tol)
    numerical_no_op_tol = _numerical_no_op_tolerance(current)
    diagnostic_record: dict[str, Any] = {
        "algorithmic_tol": effective_tol,
        "numerical_no_op_tol": numerical_no_op_tol,
        "initial_step": float(initial_step),
        "inner_iteration": 0,
        "backtracks": 0,
        "status": "starting",
    }
    initial_evidence = _simplex_evidence(current, torch)
    if (
        not initial_evidence["simplex_feasible"]
        or not _finite_array(current_residual, torch)
        or not _finite_array(Z, torch)
        or not np.isfinite(current_objective)
        or current_objective < 0.0
    ):
        diagnostic_record.update(status="invalid_current", current_evidence=initial_evidence)
        return _BlockUpdate(
            current,
            current_objective,
            0,
            0,
            0.0,
            False,
            Z,
            current_residual,
            numerical_diagnostics=dict(diagnostic_record),
        )
    total_backtracks = 0
    accepted_step = initial_step
    for iteration in range(1, max_iter + 1):
        diagnostic_record["inner_iteration"] = iteration
        gradient = current_residual @ Z.T
        raw_preview = current - initial_step * gradient
        preview = _project_rows_to_simplex_torch(torch, raw_preview)
        preview_delta = _relative_iterate_change_torch(torch, preview, current)
        if (
            preview_delta <= max(effective_tol, numerical_no_op_tol)
            and _simplex_evidence(preview, torch)["simplex_feasible"]
        ):
            diagnostic_record.update(
                status="numerical_no_op" if preview_delta <= numerical_no_op_tol else "algorithmic_tolerance",
                candidate_evidence={"displacement": float(preview_delta), "finite": True, "simplex_feasible": True},
            )
            return _BlockUpdate(
                current,
                current_objective,
                iteration - 1,
                total_backtracks,
                initial_step,
                True,
                Z,
                current_residual,
                numerical_diagnostics=dict(diagnostic_record),
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
            no_op_tol=numerical_no_op_tol,
            diagnostic_record=diagnostic_record,
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
                numerical_diagnostics=dict(diagnostic_record),
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
        if delta <= max(effective_tol, numerical_no_op_tol):
            return _BlockUpdate(
                current,
                current_objective,
                iteration,
                total_backtracks,
                accepted_step,
                True,
                Z,
                current_residual,
                numerical_diagnostics=dict(diagnostic_record),
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
        numerical_diagnostics=dict(diagnostic_record),
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
    effective_tol = float(tol)
    numerical_no_op_tol = _numerical_no_op_tolerance(current)
    diagnostic_record: dict[str, Any] = {
        "algorithmic_tol": effective_tol,
        "numerical_no_op_tol": numerical_no_op_tol,
        "initial_step": float(initial_step),
        "inner_iteration": 0,
        "backtracks": 0,
        "status": "starting",
    }
    initial_evidence = _simplex_evidence(current, torch)
    if (
        not initial_evidence["simplex_feasible"]
        or not _finite_array(current_residual, torch)
        or not _finite_array(current_z, torch)
        or not np.isfinite(current_objective)
        or current_objective < 0.0
    ):
        diagnostic_record.update(status="invalid_current", current_evidence=initial_evidence)
        return _BlockUpdate(
            current,
            current_objective,
            0,
            0,
            0.0,
            False,
            current_z,
            current_residual,
            numerical_diagnostics=dict(diagnostic_record),
        )
    total_backtracks = 0
    accepted_step = initial_step
    for iteration in range(1, max_iter + 1):
        diagnostic_record["inner_iteration"] = iteration
        gradient = (A.T @ current_residual) @ X.T
        raw_preview = current - initial_step * gradient
        preview = _project_rows_to_simplex_torch(torch, raw_preview)
        preview_delta = _relative_iterate_change_torch(torch, preview, current)
        if (
            preview_delta <= max(effective_tol, numerical_no_op_tol)
            and _simplex_evidence(preview, torch)["simplex_feasible"]
        ):
            diagnostic_record.update(
                status="numerical_no_op" if preview_delta <= numerical_no_op_tol else "algorithmic_tolerance",
                candidate_evidence={"displacement": float(preview_delta), "finite": True, "simplex_feasible": True},
            )
            return _BlockUpdate(
                current,
                current_objective,
                iteration - 1,
                total_backtracks,
                initial_step,
                True,
                current_z,
                current_residual,
                numerical_diagnostics=dict(diagnostic_record),
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
            no_op_tol=numerical_no_op_tol,
            diagnostic_record=diagnostic_record,
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
                numerical_diagnostics=dict(diagnostic_record),
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
        if delta <= max(effective_tol, numerical_no_op_tol):
            return _BlockUpdate(
                current,
                current_objective,
                iteration,
                total_backtracks,
                accepted_step,
                True,
                current_z,
                current_residual,
                numerical_diagnostics=dict(diagnostic_record),
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
        numerical_diagnostics=dict(diagnostic_record),
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
    diagnostic_record: dict[str, Any] | None = None,
) -> tuple[Any, float, float, int, Any, float] | None:
    update = _backtracked_candidate(
        X,
        Z,
        current,
        gradient,
        block="a",
        preview=preview,
        preview_delta=preview_delta,
        preview_raw_finite=preview_raw_finite,
        current_z=Z,
        current_residual=current_residual,
        current_objective=current_objective,
        initial_step=initial_step,
        backtracking_factor=backtracking_factor,
        max_backtracks=max_backtracks,
        no_op_tol=no_op_tol,
        torch=torch,
        diagnostic_record={} if diagnostic_record is None else diagnostic_record,
    )
    if update is None:
        return None
    value, objective, step, backtrack, _z, residual, delta = update
    return value, objective, step, backtrack, residual, delta


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
    diagnostic_record: dict[str, Any] | None = None,
) -> tuple[Any, float, float, int, Any, Any, float] | None:
    return _backtracked_candidate(
        X,
        A,
        current,
        gradient,
        block="b",
        preview=preview,
        preview_delta=preview_delta,
        preview_raw_finite=preview_raw_finite,
        current_z=current_z,
        current_residual=current_residual,
        current_objective=current_objective,
        initial_step=initial_step,
        backtracking_factor=backtracking_factor,
        max_backtracks=max_backtracks,
        no_op_tol=no_op_tol,
        torch=torch,
        diagnostic_record={} if diagnostic_record is None else diagnostic_record,
    )


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
        numerical_diagnostics={
            "initial": first.numerical_diagnostics,
            "polish": second.numerical_diagnostics,
        },
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
        raise FloatingPointError("aa_projected_gradient_v4 objective became non-finite.")
    return value


def _objective_from_residual_torch(torch: Any, residual: Any) -> float:
    value = float((0.5 * torch.sum(residual * residual)).detach().cpu())
    if not np.isfinite(value):
        raise FloatingPointError("aa_projected_gradient_v4 objective became non-finite.")
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
        raise FloatingPointError("aa_projected_gradient_v4 Lipschitz estimate became non-finite.")
    return result


def _largest_gram_eigenvalue_torch(torch: Any, matrix: Any) -> float:
    gram = matrix @ matrix.T if matrix.shape[0] <= matrix.shape[1] else matrix.T @ matrix
    if tuple(gram.shape) == (1, 1):
        value = gram.reshape(())
    else:
        value = torch.linalg.eigvalsh(gram).amax()
    result = float(torch.clamp(value, min=0).detach().cpu())
    if not np.isfinite(result):
        raise FloatingPointError("aa_projected_gradient_v4 Lipschitz estimate became non-finite.")
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
        raise FloatingPointError("aa_projected_gradient_v4 step estimate is invalid.")
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
        raise ValueError("AA v4 final B must be a non-empty matrix.")
    if not np.isfinite(B64).all():
        raise FloatingPointError("AA v4 final B canonicalization received non-finite values.")
    minimum_before = float(np.min(B64))
    if minimum_before < 0.0:
        raise FloatingPointError("AA v4 final B canonicalization does not repair negative values.")
    row_sums_before = np.sum(B64, axis=1, dtype=np.float64)
    if not np.isfinite(row_sums_before).all() or bool(np.any(row_sums_before <= 0.0)):
        raise FloatingPointError("AA v4 final B canonicalization requires finite positive row sums.")
    errors_before = np.abs(row_sums_before - 1.0)
    max_error_before = float(np.max(errors_before))
    if max_error_before > _FINAL_B_CANONICALIZATION_MAX_ERROR:
        raise FloatingPointError(
            "AA v4 final B row-sum error exceeds the final canonicalization ceiling "
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
        raise FloatingPointError("AA v4 final B canonicalization produced an unsafe matrix.")
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
    config: StandardAAV4Config,
    result: Mapping[str, Any],
    device_used: str,
    dtype_used: str,
) -> dict[str, Any]:
    A64 = np.asarray(A, dtype=np.float64)
    if not np.isfinite(A64).all():
        raise FloatingPointError("aa_projected_gradient_v4 produced non-finite factors.")
    B64, final_b_canonicalization = _canonicalize_final_b(B)
    _validate_returned_factors(A64, B64)
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
        raise FloatingPointError("aa_projected_gradient_v4 produced non-finite public factors.")

    public_scale = (float(normalization.raw_scale) * float(normalization.centered_scale)) ** 2
    if not np.isfinite(public_scale):
        raise FloatingPointError("AA v4 public objective scale is non-finite.")
    raw_history = [dict(row) for row in result["history"]]
    if not raw_history:
        raise RuntimeError("AA v4 optimization history must not be empty.")
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
            raise FloatingPointError("AA v4 public objective history is non-finite.")
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
        raise FloatingPointError("AA v4 returned-state reconciliation violates monotone objective acceptance.")

    final_check = _stationarity_check_numpy(
        X_normalized64,
        A64,
        B64,
        Z=Z_normalized,
        residual=normalized_residual,
        x_lipschitz=float(result["data_lipschitz"]),
        policy=config.policy,
        gradient_scale=float(result["gradient_scale"]),
    )
    termination_reason = _checked_reason(
        str(result["termination_reason"]),
        final_check,
        n_iter=int(result["n_iter"]),
        max_iter=max_iter,
        config=config,
    )
    search_totals = dict(result["search_totals"])
    search_totals["total_diagnostic_checks"] += 1
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
        final_check=final_check,
        search_totals=search_totals,
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
            "returned_state_postchecks": 1,
            "selected_initial_factor_hash": result["initial_factor_hash"],
            "selected_a_refinement_stage": int(result["a_refinement_stage"]),
            "selected_b_refinement_stage": int(result["b_refinement_stage"]),
            "selected_a_inner_iterations": int(result["a_inner_iterations"]),
            "selected_b_inner_iterations": int(result["b_inner_iterations"]),
            "training_shape": [int(X_raw.shape[0]), int(X_raw.shape[1])],
            "selected_candidate_termination_reason": str(result["termination_reason"]),
            "selected_candidate_stationarity_check_passed": bool(result["final_check"]["passed"]),
            "returned_state_stopping_reconciled": (
                termination_reason != result["termination_reason"]
                or bool(final_check["passed"]) != bool(result["final_check"]["passed"])
            ),
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
        "solver_dependency_identities": SOLVER_DEPENDENCY_IDENTITIES,
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
    config: StandardAAV4Config,
    normalization: AffineNormalization,
    device_used: str,
    dtype_used: str,
) -> dict[str, Any]:
    n_samples = int(X_raw.shape[0])
    # Exact copies avoid summation-roundoff amplified by huge constant values.
    # Uniform simplex rows are equivalent mathematically, but not bitwise.
    A = np.zeros((n_samples, k), dtype=np.float64)
    B = np.zeros((k, n_samples), dtype=np.float64)
    A[:, 0] = 1.0
    B[:, 0] = 1.0
    Z = np.broadcast_to(np.asarray(X_raw[0], dtype=np.float64), (k, X_raw.shape[1])).copy()
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
            "training_shape": [int(X_raw.shape[0]), int(X_raw.shape[1])],
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
        "solver_dependency_identities": SOLVER_DEPENDENCY_IDENTITIES,
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
        raise FloatingPointError("AA v4 final gradient-mapping diagnostics are non-finite.")
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
    config: StandardAAV4Config,
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
    final_check: Mapping[str, Any] | None = None,
    search_totals: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    constant_exact = termination_reason == "constant_exact"
    numerical_passed = bool(constant_exact or (final_check is not None and final_check["passed"]))
    stopping_satisfied = bool(
        constant_exact
        or (termination_reason == "first_order_stationary" and numerical_passed and not config.force_max_iterations)
    )
    public_scale = (float(normalization.raw_scale) * float(normalization.centered_scale)) ** 2
    return {
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "solver_id": SOLVER_ID,
        "solver_contract_version": SOLVER_CONTRACT_VERSION,
        "solver_implementation_version": SOLVER_IMPLEMENTATION_VERSION,
        "solver_implementation_digest": SOLVER_IMPLEMENTATION_DIGEST,
        "solver_dependency_identities": SOLVER_DEPENDENCY_IDENTITIES,
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
        "objective_plateau_reached": False,
        "stationarity_check_passed": numerical_passed,
        "stopping_policy": asdict(config.policy),
        "final_stationarity_check": dict(final_check or {"passed": constant_exact, "analytic_constant_exact": True}),
        "optimization_scope": OPTIMIZATION_SCOPE,
        "convergence_scope": OPTIMIZATION_SCOPE,
        "objective_scope": OPTIMIZATION_SCOPE,
        "objective_value_space": "original_training_space",
        "normalized_objective_value_space": "affine_normalized_training_space",
        "relative_objective_evaluation_space": "affine_normalized_training_space",
        "public_objective_scale": float(public_scale),
        "convergence_criterion": "projected_gradient_and_frank_wolfe_v1",
        "stopping_mode": "fixed_outer_iteration_budget" if config.force_max_iterations else "checked_first_order",
        "convergence_claim": (
            "exact_constant_solution"
            if constant_exact
            else "numerical_first_order_check_only"
            if stopping_satisfied
            else "no_stationarity_claim"
        ),
        "objective_rtol": float(objective_rtol),
        "inner_iteration_budgets": {"a_max_iter": config.a_max_iter, "b_max_iter": config.b_max_iter},
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
        "gradient_mapping_used_for_stopping": not config.force_max_iterations and not constant_exact,
        "global_optimality_certified": False,
        "gradient_mapping_diagnostic_computed": not constant_exact,
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
        "restarts_completed": sum(row["attempt_type"] == "independent" for row in restart_summaries),
        "perturbation_restarts_requested": config.policy.perturbation_restarts,
        "attempts_requested": int(restarts_requested) + config.policy.perturbation_restarts,
        "attempts_completed": len(restart_summaries),
        "search_termination_reason": ("constant_exact_short_circuit" if constant_exact else "attempt_budget_exhausted"),
        **dict(
            search_totals
            or {
                "total_outer_iterations": 0,
                "total_a_inner_iterations": 0,
                "total_b_inner_iterations": 0,
                "total_diagnostic_checks": 0,
                "total_refinements": 0,
            }
        ),
        "selected_restart_index": selected_restart_index,
        "selected_init_seed": selected_init_seed,
        "restart_selection_metric": "returned_canonicalized_normalized_objective",
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
                "a_numerical_update": a_update.numerical_diagnostics,
            }
        )
    if b_update is not None:
        row.update(
            {
                "b_iterations": int(b_update.iterations),
                "b_backtracks": int(b_update.backtracks),
                "b_accepted_step": float(b_update.accepted_step),
                "b_numerical_update": b_update.numerical_diagnostics,
            }
        )
    return row


def _optimization_status(termination_reason: str) -> str:
    if termination_reason == "first_order_stationary":
        return "approximate_first_order_stationary"
    if termination_reason == "constant_exact":
        return "exact_constant_solution"
    if termination_reason == "stalled_nonstationary":
        return "stalled_nonstationary"
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
            raise TypeError("AA v4 method parameter names must be strings.")
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
        raise TypeError("AA v4 method_params must be a mapping or attribute object.")
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
