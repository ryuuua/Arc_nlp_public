# Arc_nlp_public

This is a minimal archetypal-analysis fitting repository for the NeurReps
submission. It contains one current solver, `aa_projected_gradient_v2`, a Hydra
entry point for fitting one condition, logging for the primary metrics, and 32
condition configurations covering the configured 4-dataset x 8-model grid.

The distribution name `neureps-arc-nlp`, Python package `neureps_arc_nlp`,
and `neureps-aa` command are retained for compatibility.

It does not include figures, analysis or aggregation scripts, experiment
launchers, dataset acquisition, embedding generation, W&B, databases, Docker,
or Slurm. Input embeddings must be prepared in advance.

## Install

CPU execution requires only NumPy and Hydra.

```bash
python -m pip install -e .
```

Install the optional extra only when using Torch or CUDA.

```bash
python -m pip install -e '.[gpu]'
```

`pyproject.toml` is the authoritative dependency specification.

## Input contract

Provide one `.npz` archive per condition. The required array names are fixed.

```python
import numpy as np

np.savez(
    "embeddings/ag_news__sentence_bert.npz",
    X_train=X_train,          # shape: [n_train, n_features]
    X_valid=X_valid,          # shape: [n_valid, n_features]
    labels_valid=labels_valid # shape: [n_valid] or [n_valid, n_labels]
)
```

This repository does not contain embedding-generation code. By default, inputs
are read from `embeddings/`. Set `AA_EMBEDDING_ROOT` to use another root.

## Run one fit

```bash
neureps-aa condition=ag_news__sentence_bert seed=0
```

Example with explicit input and output paths:

```bash
neureps-aa \
  condition=ag_news__sentence_bert \
  input.path=/absolute/path/to/input.npz \
  output.dir=/absolute/path/to/output \
  seed=0
```

`solver.backend=auto` selects Torch when CUDA-capable Torch is available and
otherwise selects NumPy. To select a backend explicitly, use
`solver.backend=torch solver.device=cuda` or `solver.backend=numpy`.

The 32 configuration names are under
`src/neureps_arc_nlp/configs/condition/`. Select any of them with
`condition=<filename without .yaml>`. The only shared configuration file is
`src/neureps_arc_nlp/configs/config.yaml`.

## Output

Each fit writes four files.

```text
outputs/<dataset>/<model>/seed_<n>/
  fit_arrays.npz             # Z, B, A_train, A_valid
  metrics.json               # reconstruction, coefficient, label metrics
  solver_diagnostics.json    # termination/convergence and gradient diagnostics
  optimization_history.jsonl # one row per outer iteration
```

`metrics.json` records:

- train and validation reconstruction MSE, R^2, and paper variance explained;
- mean maximum coefficient, effective K, active-anchor count, dominant share,
  and 80/90/95% contribution-rank counts;
- validation usage-weighted purity, `I(Y;J)/H(Y)`, label entropy, and global
  top-label probability;
- solver ID, iteration count, termination reason, converged flag, and
  stationarity/KKT certification flags; and
- the resolved Hydra configuration and its hash, the input hash, the solver
  source hash, and NumPy/Torch versions.

Run outputs record resolved input paths and configuration. Review them for
local paths and identifiers before sharing; generated outputs are not part of
this source distribution.

`converged` indicates only that the solver stopping rule was satisfied. Solver
diagnostics retain `stationarity_certified=false` and `kkt_certified=false` as
separate fields.

## Scope

- Conditions: 4 datasets x 8 models = 32
- Default: `K=90`, seeds are selected one at a time
- Solver: `aa_projected_gradient_v2` only
- Input: prepared embedding matrices and validation labels
- No figure generation, embedding generation, campaign execution, or result aggregation

`NOTICE.md` describes the included solver and distribution scope. This
repository currently has no independent open-source license. Distribution
terms must be confirmed before public release.
