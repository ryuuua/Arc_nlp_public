"""Portable export identities and the optional-Torch import boundary."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from neureps_arc_nlp import standard_aa_v3, standard_aa_v4

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "provenance/source_fingerprints.json"
EXPORTED_FILES = (
    "src/neureps_arc_nlp/standard_aa_v3.py",
    "src/neureps_arc_nlp/standard_aa_v4.py",
    "src/neureps_arc_nlp/aa_stopping_policy.py",
    "src/neureps_arc_nlp/aa_runtime_v2.py",
)


@pytest.mark.parametrize("relative", EXPORTED_FILES)
def test_exported_kernel_bytes_match_reviewed_upstream_fingerprint(relative):
    manifest = json.loads(MANIFEST.read_text())
    entries = {entry["exported_path"]: entry for entry in manifest["files"]}
    assert set(entries) == set(EXPORTED_FILES)
    entry = entries[relative]
    payload = (ROOT / relative).read_bytes()
    assert entry["transformation"] == "none"
    assert len(payload) == entry["byte_count"]
    assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
    assert not Path(entry["upstream_path"]).is_absolute()
    assert not Path(entry["exported_path"]).is_absolute()


def test_runtime_solver_identities_match_exported_dependency_fingerprints():
    manifest = json.loads(MANIFEST.read_text())
    expected = manifest["numerical_identities"]
    for solver in (standard_aa_v3, standard_aa_v4):
        assert solver.SOLVER_IMPLEMENTATION_VERSION == expected[solver.SOLVER_ID]["implementation_version"]
        assert solver.SOLVER_IMPLEMENTATION_DIGEST == expected[solver.SOLVER_ID]["implementation_digest"]
    fingerprints = {entry["exported_path"]: entry["sha256"] for entry in manifest["files"]}
    dependencies = standard_aa_v4.SOLVER_DEPENDENCY_IDENTITIES
    assert dependencies["standard_aa_v4"]["digest"] == fingerprints[EXPORTED_FILES[1]]
    assert dependencies["aa_stopping_policy"]["digest"] == fingerprints[EXPORTED_FILES[2]]


def test_numpy_kernels_import_fit_and_transform_without_torch_or_private_package():
    script = """
import importlib.abc
import sys

class UnavailableOptionalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'arc_nlp'}:
            raise ImportError(f'Import deliberately unavailable: {fullname}')

sys.meta_path.insert(0, UnavailableOptionalImports())
import numpy as np
from neureps_arc_nlp import standard_aa_v3, standard_aa_v4
X = np.array([[0., 0.], [1., 0.], [0., 1.], [.2, .3], [.1, .6]])
for version, module in [(3, standard_aa_v3), (4, standard_aa_v4)]:
    fit = getattr(module, f'fit_standard_aa_v{version}_numpy')
    result = fit(X, k=3, seed=0, n_init=1, max_iter=2, tol=1e-5, X_val=None,
                 method_params={'a_max_iter': 3, 'b_max_iter': 3}, dtype='float64')
    normalized = ((X / result['normalization_raw_scale']) - result['normalization_center_unit'])
    normalized /= result['normalization_centered_scale']
    infer = getattr(module, f'infer_standard_aa_v{version}_numpy')
    A = infer(normalized, result['normalization_archetypes'],
              max_iter=result['inference_max_iter'], tol=result['inference_tol'],
              step_scale=result['inference_step_scale'], dtype='float64')
    np.testing.assert_allclose(A.sum(axis=1), 1.0, atol=1e-12)
    assert np.isfinite(A).all() and (A >= 0).all()
assert 'torch' not in sys.modules
assert 'arc_nlp' not in sys.modules
"""
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
