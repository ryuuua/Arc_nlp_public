# Getting started

The portable package runs on prepared arrays. A first CPU example needs no model download, text dataset, GPU, private embedding cache, or experiment service.

## Install from this repository

Use Python 3.11 or newer and a source checkout of [Arc_nlp_public](https://github.com/ryuuua/Arc_nlp_public).

```bash
git clone https://github.com/ryuuua/Arc_nlp_public.git
cd Arc_nlp_public
python -m pip install -e .
```

The distribution name is `neureps-arc-nlp`, the import package is `neureps_arc_nlp`, and the fitting command is `neureps-aa`. These names are retained for compatibility. No PyPI release is implied by this source-installation command.

NumPy and Hydra are the CPU runtime dependencies. Install the `gpu` extra only when you intend to use Torch:

```bash
python -m pip install -e '.[gpu]'
```

The source repository's `pyproject.toml` is authoritative for dependencies. Documentation building has its own small dependency set and does not import the scientific runtime.

## Try a complete example

```bash
python examples/synthetic_fit.py --solver paper_v2
python examples/synthetic_fit.py --solver standard_v3
python examples/synthetic_fit.py --solver standard_v4
```

Each invocation uses an explicitly selected solver and a separate output directory under `artifacts/examples/`. The script refuses to replace existing outputs. See the [tutorial](tutorial.html) for the data, saved files, and replay API.

## Use prepared embeddings

A fitting input is one NPZ archive containing `X_train`, `X_valid`, and `labels_valid`. Matrices must have the same feature dimension; Validation labels must align with Validation rows. Label metrics require nonzero label entropy.

```python
import numpy as np

np.savez("input.npz", X_train=X_train, X_valid=X_valid,
         labels_valid=labels_valid)
```

Run a condition using the existing command:

```bash
neureps-aa condition=ag_news__sentence_bert input.path=input.npz solver.backend=numpy
```

The 32 condition configurations cover a named 4-dataset × 8-model grid. They describe input conditions; they do not download datasets or generate embeddings. Existing filenames and the `AA_EMBEDDING_ROOT` input-root variable remain supported.

## Select a solver deliberately

Use `solver=paper_v2` (the default), `solver=standard_v3`, or `solver=standard_v4`. The v4 profile selects NumPy CPU and float64. Use the [solver guide](solvers.html) before changing budgets or backends.
