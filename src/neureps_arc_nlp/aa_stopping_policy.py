"""Opt-in AA search settings and row-simplex first-order diagnostics.

This module does not run an optimizer or claim a minimum. Numerical thresholds
are explicit engineering settings; historical solvers do not import this module.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np

POLICY_IMPLEMENTATION_VERSION = "aa_stopping_policy_row_simplex_v1"
POLICY_IMPLEMENTATION_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
POLICY_PARAM_NAMES = frozenset(
    {
        "stationarity_gap_tol",
        "stationarity_step_tol",
        "stationarity_check_interval",
        "max_refinements",
        "refinement_multiplier",
        "perturbation_restarts",
        "perturbation_strength",
    }
)
_FEASIBILITY_TOL = 1e-8


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, not a boolean or string.")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _integer(value: Any, name: str, *, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not a boolean or float.")
    result = int(value)
    if result < minimum or (maximum is not None and result > maximum):
        raise ValueError(f"{name} must be in [{minimum}, {maximum or 'unbounded'}].")
    return result


@dataclass(frozen=True)
class PolicyConfig:
    stationarity_gap_tol: float = 1e-6
    stationarity_step_tol: float = 1e-6
    stationarity_check_interval: int = 5
    max_refinements: int = 2
    refinement_multiplier: int = 2
    perturbation_restarts: int = 0
    perturbation_strength: float = 0.1

    def __post_init__(self) -> None:
        for name in ("stationarity_gap_tol", "stationarity_step_tol", "perturbation_strength"):
            object.__setattr__(self, name, _positive_float(getattr(self, name), name))
        if self.perturbation_strength > 1.0:
            raise ValueError("perturbation_strength must be <= 1.")
        for name, minimum, maximum in (
            ("stationarity_check_interval", 1, None),
            ("max_refinements", 0, 4),
            ("refinement_multiplier", 2, 4),
            ("perturbation_restarts", 0, None),
        ):
            object.__setattr__(
                self, name, _integer(getattr(self, name), name, minimum=minimum, maximum=maximum)
            )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_policy(params: Mapping[str, Any] | None = None) -> PolicyConfig:
    """Read new policy fields; the owning solver rejects unknown method fields."""

    if params is None:
        return PolicyConfig()
    if not isinstance(params, Mapping):
        raise TypeError("Stopping policy parameters must be a mapping.")
    return PolicyConfig(**{key: params[key] for key in POLICY_PARAM_NAMES if key in params})


def summarize_simplex_block(
    weights: Any,
    gradient: Any,
    projected: Any,
    *,
    gradient_scale: float,
    policy: PolicyConfig,
    torch: Any | None = None,
) -> dict[str, Any]:
    """Check every simplex row, without dividing by loss or gradient span.

    ``projected`` must be computed at the same weights/gradient using an
    independent finite, positive, curvature-consistent diagnostic step. The
    caller owns that step. The step-free Frank-Wolfe gap protects against a
    misleadingly small projected displacement. ``gradient_scale`` is a fixed
    positive data-derived scale, never a residual-dependent scale.

    Gradients are shifted by their row minimum before accumulation. This avoids
    cancellation from simplex-normal common offsets. No factors are repaired.
    Tiny negative coefficients fail feasibility instead of canceling a gap.
    Tensor reductions stay on their device; only scalar summaries are copied.
    """

    scale = _positive_float(gradient_scale, "gradient_scale")
    if not isinstance(policy, PolicyConfig):
        raise TypeError("policy must be a validated PolicyConfig.")
    if torch is None:
        arrays = tuple(np.asarray(value) for value in (weights, gradient, projected))
        if any(value.dtype != np.float64 for value in arrays):
            raise TypeError("AA stopping diagnostics require float64 arrays.")
        weights, gradient, projected = arrays
        if weights.ndim != 2 or 0 in weights.shape or any(value.shape != weights.shape for value in arrays):
            raise ValueError("Simplex diagnostic matrices must be nonempty and have identical shapes.")
        if not all(np.isfinite(value).all() for value in arrays):
            raise FloatingPointError("Simplex diagnostics received non-finite arrays.")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            shifted = gradient / scale
            shifted = shifted - np.min(shifted, axis=1, keepdims=True)
            gaps = np.sum(weights * shifted, axis=1, dtype=np.float64)
            steps = np.sum(np.abs(weights - projected), axis=1, dtype=np.float64)
        errors = np.abs(np.sum(weights, axis=1, dtype=np.float64) - 1.0)
        projected_errors = np.abs(np.sum(projected, axis=1, dtype=np.float64) - 1.0)
        feasible = (errors <= _FEASIBILITY_TOL) & (np.min(weights, axis=1) >= 0.0)
        projection_feasible = (projected_errors <= _FEASIBILITY_TOL) & (np.min(projected, axis=1) >= 0.0)
        passed = feasible & projection_feasible & (gaps >= 0.0)
        passed &= (gaps <= policy.stationarity_gap_tol) & (steps <= policy.stationarity_step_tol)
        summary = {
            "frank_wolfe_gap_max": float(np.max(gaps)),
            "projected_step_l1_max": float(np.max(steps)),
            "feasibility_error": float(max(np.max(errors), max(0.0, -float(np.min(weights))))),
            "projected_feasibility_error": float(
                max(np.max(projected_errors), max(0.0, -float(np.min(projected))))
            ),
            "minimum_weight": float(np.min(weights)),
            "violating_rows": int(np.count_nonzero(~passed)),
            "passed": bool(np.all(passed)),
        }
    else:
        tensor_values = (weights, gradient, projected)
        if any(not isinstance(value, torch.Tensor) or value.dtype != torch.float64 for value in tensor_values):
            raise TypeError("AA stopping diagnostics require float64 tensors.")
        if weights.ndim != 2 or 0 in weights.shape or any(value.shape != weights.shape for value in tensor_values):
            raise ValueError("Simplex diagnostic matrices must be nonempty and have identical shapes.")
        if any(value.device != weights.device for value in tensor_values):
            raise ValueError("Simplex diagnostic tensors must share a device.")
        if not all(bool(torch.isfinite(value).all().item()) for value in tensor_values):
            raise FloatingPointError("Simplex diagnostics received non-finite tensors.")
        shifted = gradient / scale
        shifted = shifted - torch.amin(shifted, dim=1, keepdim=True)
        gaps = torch.sum(weights * shifted, dim=1)
        steps = torch.sum(torch.abs(weights - projected), dim=1)
        errors = torch.abs(torch.sum(weights, dim=1) - 1.0)
        projected_errors = torch.abs(torch.sum(projected, dim=1) - 1.0)
        feasible = (errors <= _FEASIBILITY_TOL) & (torch.amin(weights, dim=1) >= 0.0)
        projection_feasible = (projected_errors <= _FEASIBILITY_TOL) & (torch.amin(projected, dim=1) >= 0.0)
        passed = feasible & projection_feasible & (gaps >= 0.0)
        passed &= (gaps <= policy.stationarity_gap_tol) & (steps <= policy.stationarity_step_tol)
        summary = {
            "frank_wolfe_gap_max": float(torch.max(gaps).item()),
            "projected_step_l1_max": float(torch.max(steps).item()),
            "feasibility_error": max(float(torch.max(errors).item()), max(0.0, -float(torch.min(weights).item()))),
            "projected_feasibility_error": max(
                float(torch.max(projected_errors).item()), max(0.0, -float(torch.min(projected).item()))
            ),
            "minimum_weight": float(torch.min(weights).item()),
            "violating_rows": int(torch.count_nonzero(~passed).item()),
            "passed": bool(torch.all(passed).item()),
        }
    summary["raw_frank_wolfe_gap_max"] = float(summary["frank_wolfe_gap_max"]) * scale
    summary["gradient_scale"] = scale
    summary["row_count"] = int(weights.shape[0])
    summary["gap_tolerance"] = policy.stationarity_gap_tol
    summary["step_tolerance"] = policy.stationarity_step_tol
    summary["feasibility_tolerance"] = _FEASIBILITY_TOL
    for name, value in summary.items():
        if isinstance(value, float) and not np.isfinite(value):
            raise FloatingPointError(f"Simplex diagnostics produced a non-finite {name}.")
    return summary
