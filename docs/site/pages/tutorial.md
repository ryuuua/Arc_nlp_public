# A first AA analysis

This tutorial runs entirely with the portable package. It generates points inside a triangle, fits three archetypes on Train, and evaluates held-out observations. No external dataset or model service is involved.

## Generate, fit, and save

From the installed source checkout, run the maintained example:

```bash
python examples/synthetic_fit.py --solver standard_v3
```

The script `examples/synthetic_fit.py` generates deterministic float64 arrays, composes the distributed Hydra configuration, and calls [run_fit](api/app.html#run_fit). It writes to `artifacts/examples/synthetic_standard_v3/` and prints replay reconstruction metrics. Existing outputs are preserved; pass a fresh `--output` directory for another run.

The package default is still v2. This command explicitly selects v3. To exercise the v4 diagnostic policy on the same synthetic construction:

```bash
python examples/synthetic_fit.py --solver standard_v4
```

This bounded example disables extra refinement and perturbation restarts and uses an explicit NumPy float64 configuration. It verifies the input/output and replay path; it is not a benchmark or evidence that one solver is universally better.

## Reload the fitted representation

A completed output contains fitted arrays and the state needed for inference. New rows must be in the same input coordinate frame used for fitting.

```python
from neureps_arc_nlp import load_fit
from neureps_arc_nlp.metrics import compute_reconstruction_metrics

fitted = load_fit("artifacts/examples/synthetic_standard_v3")
A_new = fitted.transform(X_new)
Z = fitted.archetypes_Z
X_reconstructed = A_new @ Z
metrics = compute_reconstruction_metrics(X_new, A_new, Z)
print(metrics["mse"])
```

Here `X_new` is your matrix of new observations with the training feature dimension. To replay the synthetic Validation rows exactly, load them from the example's sibling `synthetic_standard_v3-input.npz` archive using NumPy with `allow_pickle=False`.

[load_fit](api/artifacts.html#load_fit) returns [SavedFit](api/artifacts.html#SavedFit). Its `transform` method uses the stored solver identity, training normalization and inference parameters. It does not refit B or choose a new solver from today's defaults. Replay runs on NumPy CPU; results need not be bitwise identical to original Torch inference. Older outputs without the versioned inference state must be refitted before using this loader.

## Understand the factors

Z contains K points in the original input frame. A_train represents each training observation as a mixture of Z. B expresses each archetype as a mixture of training rows. B's columns therefore retain the training row order.

The synthetic generating triangle is not the same as the training sample's convex hull. AA constrains archetypes to the latter; it need not recover the original three generating vertices.

## Inspect the output

| File | Purpose |
|---|---|
| `fit_arrays.npz` | Archetypes, training weights, Train/Validation coefficients and training normalization arrays |
| `metrics.json` | Reconstruction, coefficient usage, label relationships, configuration and provenance |
| `solver_diagnostics.json` | Termination, numerical diagnostics and the versioned `inference_state` |
| `optimization_history.jsonl` | Recorded outer-iteration history |

The [reproducibility guide](reproducibility.html) explains which identities matter. A completed fit or a passed numerical stopping test is not a global-optimality certificate.

## Controlled comparisons

Keep the data, split, K, initial seed, dtype and backend fixed when changing the solver. Record actual work and termination details in addition to nominal budgets. Reconstruction error is only one property of a representation; [evaluation](evaluation.html) covers the others.
