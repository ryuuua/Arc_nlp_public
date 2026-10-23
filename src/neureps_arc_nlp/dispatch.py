"""Explicit selection of the portable, versioned AA implementations."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping


@dataclass(frozen=True)
class SolverSpec:
    """A solver's callable names and source identity, without eager Torch imports."""

    solver_id: str
    module_name: str
    revision: str

    @property
    def module(self) -> ModuleType:
        return import_module(f".{self.module_name}", __package__)

    @property
    def contract_version(self) -> str:
        return str(self.module.SOLVER_CONTRACT_VERSION)

    def fit(self, backend: str):
        return getattr(self.module, f"fit_standard_aa_{self.revision}_{backend}")

    def infer(self, backend: str):
        return getattr(self.module, f"infer_standard_aa_{self.revision}_{backend}")

    def source_hashes(self) -> dict[str, str]:
        names = [self.module_name]
        if self.revision == "v4":
            names.append("aa_stopping_policy")
        return {
            f"{name}.py": hashlib.sha256(Path(__file__).with_name(f"{name}.py").read_bytes()).hexdigest()
            for name in names
        }


_SOLVERS = {
    f"aa_projected_gradient_{revision}": SolverSpec(
        f"aa_projected_gradient_{revision}", "solver" if revision == "v2" else f"standard_aa_{revision}", revision
    )
    for revision in ("v2", "v3", "v4")
}


def resolve_solver(solver_id: str, method_params: Mapping[str, Any] | None = None) -> SolverSpec:
    """Resolve an exact ID and reject a conflicting method selector."""
    if not isinstance(solver_id, str) or solver_id not in _SOLVERS:
        raise ValueError(f"Unsupported solver.id={solver_id!r}; choose one of {sorted(_SOLVERS)}")
    spec = _SOLVERS[solver_id]
    if method_params is not None:
        selector = method_params.get("b_update", solver_id)
        if spec.module.canonicalize_b_update_selector(selector) != solver_id:
            raise ValueError("solver.id and solver.method_params.b_update must select the same solver")
    return spec


def preflight_solver(spec: SolverSpec, X_train: Any, X_valid: Any, cfg: Any, params: Mapping[str, Any]) -> None:
    """Check new solver inputs before backend discovery or Torch initialization."""
    if spec.revision == "v2":
        return
    spec.module.preflight_fit_inputs(
        X_train,
        k=cfg.k,
        seed=cfg.seed,
        n_init=cfg.solver.n_init,
        max_iter=cfg.solver.max_iter,
        tol=cfg.solver.tol,
        X_val=X_valid,
        method_params=params,
        dtype=cfg.solver.dtype,
    )
    if cfg.solver.backend not in {"auto", "numpy", "torch"}:
        raise ValueError("solver.backend must be auto, numpy, or torch")
    from .aa_runtime_v2 import normalize_torch_device_v2, strict_bool_v2

    normalize_torch_device_v2(cfg.solver.device)
    strict_bool_v2(cfg.solver.get("deterministic", True), name="deterministic")
    if cfg.solver.backend == "numpy" and cfg.solver.device != "cpu":
        raise ValueError("solver.backend=numpy requires solver.device=cpu")
