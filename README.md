# Arc_nlp_public

Portable archetypal analysis for studying the geometry of text embeddings.
The package fits convex archetypes, represents samples as simplex coefficients,
and reports reconstruction, coefficient-use and label-association metrics.

[Research website](https://ryuuua.github.io/Arc_nlp_public/) ·
[Conference entry](https://ryuuua.github.io/Arc_nlp_public/conference/) ·
[日本語](https://ryuuua.github.io/Arc_nlp_public/ja/index.html)

The distribution name `neureps-arc-nlp`, Python package `neureps_arc_nlp`,
and `neureps-aa` command are retained for compatibility. Version 0.3 adds
Standard v3/v4, saved-fit inference, and source-linked research documentation.

## Choose a solver

| Hydra profile | Solver | Precision | Purpose |
| --- | --- | --- | --- |
| `solver=paper_v2` (default) | `aa_projected_gradient_v2` | float32 or float64 | Preserve the existing paper/reproduction route |
| `solver=standard_v3` | `aa_projected_gradient_v3` | float32 or float64 | Dense-kernel research implementation |
| `solver=standard_v4` | `aa_projected_gradient_v4` | **float64 only** | Explicit stationarity checks, bounded refinement and restart diagnostics |

v4 is opt-in. Additional checks/refinement can cost more work; no general
speed or real-dataset superiority claim is made here. Approximate first-order
checks do not certify a local or global optimum. v2 remains byte-for-byte
pinned; the exported v3/v4 kernels and their support files have fingerprints in
[`provenance/source_fingerprints.json`](provenance/source_fingerprints.json).

## Install

Python 3.11 or newer is required. CPU execution requires NumPy and Hydra;
Torch is optional.

```bash
python -m pip install -e .
# Only for the Torch backend:
python -m pip install -e '.[gpu]'
```

`pyproject.toml` is the authoritative dependency specification. To try the
package without downloading data or models:

```bash
python examples/synthetic_fit.py
```

## Prepared input

Provide a `.npz` archive with these arrays:

```python
import numpy as np

np.savez(
    "input.npz",
    X_train=X_train,           # [n_train, n_features]
    X_valid=X_valid,           # [n_valid, n_features]
    labels_valid=labels_valid, # [n_valid] or [n_valid, n_labels]
)
```

The repository does not generate embeddings or download datasets. The 32
condition presets cover the configured 4-dataset × 8-model grid; they are names
and input conventions, not redistributed data. By default input is read from
`embeddings/`; `AA_EMBEDDING_ROOT` changes this root.

## Fit and replay

Existing commands retain their default v2 route:

```bash
neureps-aa condition=ag_news__sentence_bert seed=0
```

Select a research profile explicitly:

```bash
neureps-aa solver=standard_v3 condition=ag_news__sentence_bert
neureps-aa solver=standard_v4 condition=ag_news__sentence_bert \
  input.path=input.npz output.dir=outputs/my-v4-fit k=3 seed=0
```

The new profiles default to NumPy on CPU; v4 rejects float32. For Torch use
`solver.backend=torch solver.device=cpu` or an available CUDA device such as
`solver.device=cuda:0`. Explicit CUDA failures do not silently become CPU fits.
CUDA numerical behavior requires validation on the target hardware.

Load a newly saved fit and infer coefficients for another matrix using the
training normalization and fitted archetypes:

```python
from neureps_arc_nlp import load_fit

model = load_fit("outputs/my-v4-fit")
A_new = model.transform(X_new)
X_reconstructed = A_new @ model.archetypes_Z
```

Old v0.2 artifacts lack the saved normalization needed for replay. They remain
valid historical fit outputs; they cannot be upgraded by inferring missing
training state from new data.

## Outputs and interpretation

Each successful fit writes four files:

```text
fit_arrays.npz              # Z, B, A_train, A_valid and saved inference state
metrics.json                # metrics, resolved config and provenance
solver_diagnostics.json     # full selected-solver diagnostics
optimization_history.jsonl  # recorded optimization iterations
```

v2 keeps `outputs/<dataset>/<model>/seed_<n>/`. New profiles use solver-specific
paths so they do not replace v2 results. `output.dir` overrides the destination;
existing outputs are refused unless overwrite is explicitly requested.

Metrics include reconstruction MSE/R², coefficient concentration/effective K,
validation purity and normalized label association. Provenance identifies the
input, resolved configuration, selected solver source and library versions.
v4 also retains stopping-policy, refinement, restart and dependency identities.
A numerical optimization failure produces separate failure evidence and raises
its original exception; it does not create successful metrics.

`converged`, the reported termination reason, and v4's approximate stationarity
check have distinct meanings. Neither `converged` nor a passed check certifies
an optimum. The diagnostic certification flags remain separate.

Generated outputs record resolved input paths and configuration. They are
ignored by Git and are not part of the website or source distribution.

## Documentation and checks

The website builds solely from this repository's allowlisted documentation and
source files; it requires no access to a development repository or experiments.

```bash
python -m pip install -e '.[test,docs]'
python -m pytest -q -p no:cacheprovider
make research-site
make research-site-check
make research-site-serve
```

The Pages workflow checks the exact reviewed source-manifest digest before an
explicit deployment from `main`. See [`docs/site/README.md`](docs/site/README.md)
for build and publication details.

The broader research site discusses additional directions; implementations of
Robust/Scalable/Mahalanobis AA, model alignment, embedding acquisition, campaign
execution, W&B, databases and cluster infrastructure are not included here.

[`NOTICE.md`](NOTICE.md) describes distribution scope. No independent
open-source license has been specified; public visibility does not grant
additional license permissions.
