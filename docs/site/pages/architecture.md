# From source to result

The portable distribution keeps numerical kernels, experiment I/O, replay, and documentation separate. Every API source view on this site is read from files contained in this repository.

<div class="equation">Prepared arrays → Fit A, B, Z → Save state → Infer new A → Evaluate</div>

## Package map

| Responsibility | Source |
|---|---|
| Hydra entry point, input validation, fit outputs | `neureps_arc_nlp/app.py` |
| Version selection and numerical dispatch | `neureps_arc_nlp/dispatch.py` |
| Saved-fit state and inference replay | `neureps_arc_nlp/artifacts.py` |
| Preserved v2 kernel | `neureps_arc_nlp/solver.py` |
| Explicit v3 and v4 kernels | `neureps_arc_nlp/standard_aa_v3.py`, `standard_aa_v4.py` |
| V4 stopping and search policy | `neureps_arc_nlp/aa_stopping_policy.py` |
| Shared solver runtime helper | `neureps_arc_nlp/aa_runtime_v2.py` |
| Reconstruction, usage, label metrics | `neureps_arc_nlp/metrics.py` |
| Distributed condition and solver profiles | `neureps_arc_nlp/configs/` |

The source root is `src/`. These paths are package modules, not imports from a separate development checkout.

## Follow a fit

[run_fit](api/app.html#run_fit) accepts a resolved Hydra configuration, loads the prepared NPZ, selects the requested numerical route, computes held-out coefficients and metrics, then writes the output contract.

[load_fit](api/artifacts.html#load_fit) reconstructs a [SavedFit](api/artifacts.html#SavedFit) using versioned saved state. `transform` uses the stored coordinate frame and solver inference parameters. It does not rerun dictionary fitting or change the original solver.

## Preserve mathematical behavior

Keep numerical changes separately identified from structural cleanup. Signatures, dtype rules, errors, factor interpretation and artifact schemas are contracts. The v2 kernel remains available as the historical baseline; v3/v4 receive explicit identities.

## Documentation is a separate build

`docs/site/site.json` lists each authored page and API symbol that may appear. `scripts/docs/` parses Python syntax trees without importing the package. Local templates, styles and search assets produce a static site with relative links, including the Japanese and conference routes.

The site does not recursively publish runtime outputs, datasets, environments or research reports. A source manifest records exact inputs and selected code excerpts.
