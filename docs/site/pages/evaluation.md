# Evaluation and interpretation

Reconstruction, coefficient concentration and label relationships describe different properties. The portable metrics return dictionaries with explicit field names.

## Reconstruction

For R = X − AZ, [compute_reconstruction_metrics](api/metrics.html#compute_reconstruction_metrics) reports:

| Field | Definition |
|---|---|
| `mse` | Σᵢ ‖Rᵢ‖² / n: mean squared Euclidean error per observation |
| `r2` | 1 − SSE / Σᵢ ‖Xᵢ − mean(X)‖²: centered reconstruction score |
| `paper_variance_explained` | 1 − SSE / ‖X‖²_F: uncentered energy explanation |
| `relative_mse` | `mse / max(var(X), 1e-12)` under NumPy's matrix-wide variance convention |

`mse` sums feature errors before averaging observations. It differs from the mean over all matrix entries by the feature dimension. Centered R² and the uncentered score have different denominators; inspect the source's explicit near-zero conventions for degenerate inputs.

## Coefficient usage

[compute_coefficient_metrics](api/metrics.html#compute_coefficient_metrics) reports mean maximum weight, mean order-2 effective K, active anchors, dominant assignment share, and the number of archetypes needed for 80/90/95% of mean contribution.

For a normalized row, order-2 effective participation is 1 / Σⱼ Aⱼ². It can be much smaller than nominal K. This is not a count of semantic concepts and does not prove that dictionary elements are nonredundant.

## Label relationships

[compute_label_metrics](api/metrics.html#compute_label_metrics) reports usage-weighted purity and normalized label–coefficient mutual information, along with label entropy and global top-label probability. It accepts single-label arrays or nonnegative multilabel membership matrices without empty rows.

These are descriptive relationships between Validation coefficients and labels. They are not classifier accuracy. The metric requires nonzero label entropy; a single constant class cannot provide that information quantity.

## Research comparisons beyond the package

Comparing original X, coefficients A and reconstructed AZ through a downstream readout requires a separately specified classifier, split and selection protocol. The portable package does not include the development project's linear probes, Hungarian matching, continuity analysis, or visualization workflows.

Across embedding models, raw Z coordinates may be incomparable even when their dimensions match. A correspondence learned from Train coefficients and an independently fitted coordinate map should be evaluated on held-out samples.

## K and interpretation

Select K and other settings using designated development data, then freeze them before Test. A plateau in an incomplete optimization sweep is descriptive rather than proof of a sufficient K. Failure to detect a predictive-performance difference is not proof of equivalence.
