# Solvers and convergence

The portable package preserves **Standard v2 as its default** for reproducibility of the existing fitting workflow. V3 and v4 are available through explicit profiles. Adding them does not rewrite the interpretation of earlier v2 artifacts.

## Select a profile

| Profile | Role | Configuration |
|---|---|---|
| `paper_v2` | Existing paper-oriented baseline and default | Solver ID `aa_projected_gradient_v2` |
| `standard_v3` | Explicit dense Standard AA implementation for research comparisons | Solver ID `aa_projected_gradient_v3` |
| `standard_v4` | Explicit stationarity diagnostics and bounded search policy | Solver ID `aa_projected_gradient_v4`; NumPy CPU, float64 profile |

```bash
neureps-aa solver=standard_v3 solver.backend=numpy
neureps-aa solver=standard_v4
```

Provide an input NPZ and condition as described in [Getting started](getting-started.html). The example profile's backend is a documented choice, not evidence that every device and precision combination was validated.

## Why v4 is opt-in

V4 adds approximate first-order checks, bounded refinement and optional perturbation restarts. These can add work beyond an initial budget. It requires float64. More diagnostics do not establish universal speed or solution-quality superiority.

A valid comparison holds inputs, objective, precision, initial factors and computational accounting consistent. Compare actual iterations, extra search, wall time and feasibility along with the objective. This site does not claim a broad real-data advantage from a small synthetic example.

## Stopping has several meanings

A loss plateau, a numerical stationarity check, and global optimality are different claims. Inspect the actual `termination_reason`, stopping flags and available first-order quantities in the saved solver diagnostics.

The package preserves explicit certification fields. Passing its approximate checks does not certify local or global optimality. Reaching a work limit likewise does not prove that no better factors exist.

## Bounded search settings

V4's [PolicyConfig](api/aa_stopping_policy.html#PolicyConfig) defines tolerances, check cadence, maximum refinements, and optional perturbation restarts. Zero perturbation restarts disables that extra search. Keep these settings with the result, because changing them changes the executed procedure.

For a small controlled smoke example, `examples/synthetic_fit.py --solver standard_v4` explicitly disables refinement and restarts. A production study should choose and report its own budget.

## Source and compatibility

The [v2 implementation](api/solver.html) remains available for replay. [V3](api/standard_aa_v3.html) and [v4](api/standard_aa_v4.html) are separately identified source modules. Saved-fit loading uses the recorded identity rather than the current profile default.

Robust losses, scalable approximations, Mahalanobis AA and spherical dictionaries are broader research directions, not additional solver choices in this portable package.
