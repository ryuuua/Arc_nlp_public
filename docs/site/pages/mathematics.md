# The mathematical model

Archetypal analysis describes observations through a dictionary whose elements are themselves mixtures of training observations. This gives both sides of the factorization a geometric interpretation.

## Two convex mixtures

Let X_train contain n training observations in d dimensions. For K archetypes, A is n × K and B is K × n.

<div class="equation" role="math" aria-label="Z equals B X train; X train is approximately A Z">Z = B X_train, &nbsp; X_train ≈ A Z</div>

Every row of A and B lies on a probability simplex: entries are nonnegative and the row sums to one. A tells us how an observation combines archetypes; B tells us which training observations contribute to each archetype.

## The optimization problem

Standard AA minimizes the mean squared Euclidean reconstruction error:

<div class="equation" role="math" aria-label="Minimize the squared Frobenius norm of X minus A B X divided by n">minimize &nbsp; ‖X_train − A B X_train‖²_F / n</div>

The objective is convex in either A or B with the other fixed, but it is generally nonconvex jointly. Alternating approximate block updates can reach different solutions from different initializations. A stationary point is not necessarily a global optimum.

## What the constraints imply

<div class="equation" role="math" aria-label="The convex hull of Z is contained in the convex hull of X train">conv(Z) ⊆ conv(X_train)</div>

Archetypes are not required to coincide with individual samples. Choosing actual training observations is a different constraint. Individual mixture weights do not automatically identify semantic concepts.

## Held-out inference

After fitting, keep Z fixed and infer one simplex row of A for each new observation. New observations do not contribute columns to B. Preserve the fitted coordinate transformation and solver inference settings.

[SavedFit](api/artifacts.html#SavedFit) provides this replay path using the stored training normalization. Its public `transform` method accepts original-frame input observations.

## Coordinate changes

A common orthogonal transformation preserves Euclidean distances. A common translation and scalar scale can be accounted for exactly; this is distinct from anisotropic whitening. General whitening changes the metric, and truncated PCA discards directions.

The portable Standard solvers use internal affine normalization for numerical conditioning. This does not implement a separate covariance-weighted AA model or a general preprocessing pipeline. See [preprocessing](preprocessing.html).

## Numerical policies

V2, v3 and v4 share the Standard AA model but identify different implementation and stopping policies. [Versioned solver guidance](solvers.html) connects these policies to source. V4's approximate first-order diagnostics remain numerical checks, not local or global optimality certificates.
