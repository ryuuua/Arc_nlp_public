# Research directions

The project asks which properties of an embedding representation can be expressed by a compact, interpretable dictionary. Implementation availability and scientific evidence are separate questions.

## Available in this portable distribution

Standard AA supplies a convex-mixture model with versioned v2/v3/v4 solvers, prepared-array fitting, saved-fit inference, and reconstruction, usage and label metrics. These are the methods described by this site's [API reference](api/index.html).

The profile default remains v2. V3 and v4 support explicit research comparisons; their inclusion is not a ranking or a claim of universal performance superiority.

## Information retained by the dictionary

One direction compares original embeddings, coefficients and reconstructions using readouts and geometric diagnostics. Another asks whether a small K is sufficient when predictive information, sensitivity to data changes and human interpretation are considered together.

A small reconstruction error does not establish human interpretability or a minimal sufficient K. Those conclusions require dedicated evidence and an evaluation protocol.

## Correspondence across models and datasets

Shared-dataset dictionaries and cross-model comparisons investigate whether independently learned representations organize the same observations similarly. Correspondence learning, coordinate transformation and held-out evaluation should remain separate steps.

These comparison and visualization pipelines are not included in the portable package. The descriptions here explain research questions rather than additional callable APIs.

## Broader implementations and exploratory models

| Direction | Research question | Distribution boundary |
|---|---|---|
| Robust and scalable AA | How do robust losses and approximations affect learned factors? | Separate development implementations; not portable solver choices |
| Mahalanobis AA | What does a Train-derived covariance metric preserve? | Separate model and whitening contract; not included |
| Structured vertex/edge/triangle assignments | Can richer assignment structure describe the data? | Separate implementation; not proof of recovered topology |
| Exact Train-span reduction | Can computation be reduced while preserving Euclidean predictions? | Research workflow; distinguish exact reduction from truncated PCA |
| Near-convex dictionary expansion | What reconstruction is gained by expanding the candidate hull? | Changes constraints and may increase magnitude or cancellation |
| Spherical sparse dictionaries | Can local nonnegative mixtures of directions preserve useful structure? | Independent model and numerical workflow |

This page does not publish campaign-specific datasets, unpublished tables, operational records, or source code absent from this repository.

## Evidence boundaries

A working prototype does not establish general superiority. A stopped run does not establish stationarity, and approximate stationarity does not establish global optimality. Reused evaluation data do not become independent confirmation by being repackaged.
