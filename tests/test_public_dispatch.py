import builtins
import hashlib
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_module

from neureps_arc_nlp import app
from neureps_arc_nlp.dispatch import resolve_solver


@pytest.mark.parametrize("revision", ["v2", "v3", "v4"])
def test_registry_owns_selected_kernel_contract_and_sources(revision) -> None:
    spec = resolve_solver(f"aa_projected_gradient_{revision}")
    assert spec.contract_version == f"aa_b_update_{revision}"
    assert callable(spec.fit("numpy")) and callable(spec.infer("torch"))
    module_path = Path(spec.module.__file__)
    assert spec.source_hashes()[module_path.name] == hashlib.sha256(module_path.read_bytes()).hexdigest()
    if revision == "v4":
        assert "aa_stopping_policy.py" in spec.source_hashes()


def test_registry_rejects_unknown_and_conflicting_selectors() -> None:
    with pytest.raises(ValueError, match="Unsupported solver.id"):
        resolve_solver("v5")
    with pytest.raises(ValueError, match="must select the same solver"):
        resolve_solver("aa_projected_gradient_v4", {"b_update": "aa_projected_gradient_v3"})


@pytest.mark.parametrize(
    "profile,override,error",
    [
        ("standard_v3", "solver.method_params.a_max_iter=0", "a_max_iter"),
        ("standard_v4", "solver.dtype=float32", "float64"),
        ("standard_v4", "solver.method_params.max_refinements=10", "max_refinements"),
    ],
)
def test_new_solver_preflight_rejects_before_torch_import(profile, override, error, monkeypatch, tmp_path) -> None:
    with initialize_config_module(version_base="1.3", config_module="neureps_arc_nlp.configs"):
        cfg = compose(config_name="config", overrides=[f"solver={profile}", "solver.backend=torch", "k=2", override])
    X = np.arange(24, dtype=np.float64).reshape(12, 2)
    monkeypatch.setattr(app, "_load_input", lambda *args: (X, X[:4], np.arange(4)))
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch":
            pytest.fail("Torch imported before validating the requested solver")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises((ValueError, TypeError), match=error):
        app.run_fit(cfg, base_dir=tmp_path)
