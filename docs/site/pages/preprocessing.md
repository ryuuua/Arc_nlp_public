# Coordinate frames and preprocessing

A fitted representation is meaningful in a particular coordinate frame. Prepare that frame consistently before fitting, and use the same transformation on later observations.

## What the portable package does

The Standard solvers internally normalize using training-derived translation and scalar scale for numerical conditioning. Saved-fit replay retains this state. Supply original-frame inputs to `SavedFit.transform`; do not apply the solver's internal normalization a second time.

The portable distribution does not expose the development repository's preprocessing pipeline, PCA implementation, intrinsic covariance whitening, or serialized preprocessor APIs.

## External transformations

You can prepare arrays with a separately implemented transformation. Fit its statistics using Train only. For simple centering:

```python
train_mean = X_train.mean(axis=0, keepdims=True)
X_train_work = X_train - train_mean
X_valid_work = X_valid - train_mean
```

If you fit AA on `X_train_work`, later calls to `SavedFit.transform` must receive rows transformed with that same `train_mean`. The saved solver state cannot reconstruct an external transformation that you did not save.

## Different geometries

| Transformation | Consequence |
|---|---|
| No external transform | Preserve the embedding producer's coordinates and normalization |
| Train centering | Apply one common translation to every split |
| Orthogonal rotation | Preserve the Euclidean geometry |
| Truncated PCA | Discard directions; specify retained dimension and reference space |
| Covariance whitening | Change the relative cost of directions |
| Per-row normalization | Change the geometry and relative magnitudes of observations |

Whitening and spherical normalization are modeling choices. They are not neutral cleanup operations.

## Splits and row identity

Preserve sample IDs and array order. Equal matrix shapes do not establish that two embedding models describe the same samples. Keep Validation and Test outside preprocessing fitting and dictionary learning.

If Test influenced prior research choices, later analysis on those observations is exploratory. Rewriting a manifest does not make the observations unseen again.
