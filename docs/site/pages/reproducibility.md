# Reproducible fits and replay

A seed alone is not an experiment identity. Preserve the input matrices, row order, coordinate frame, numerical settings and evaluation decisions together.

## Input contract

An NPZ input supplies `X_train`, `X_valid`, and `labels_valid`. Record how those arrays were produced: model/tokenizer revisions, text and pooling recipe, normalization, sample IDs, labels and split membership. The portable package consumes these arrays; it does not reconstruct their upstream provenance for you.

## Solver identity

Keep the selected profile, versioned solver ID, K, seed, initialization count, outer/inner work limits, tolerances, backend, device and dtype. V4 refinement and restarts are part of the procedure, not invisible implementation details.

`metrics.json` records the effective configuration and input/source/dependency provenance. Saved files can contain local paths and identifiers; review runtime outputs before sharing them. The documentation build publishes curated source only.

## Saved arrays and state

`fit_arrays.npz` contains `archetypes_Z`, `archetype_weights_B`, `coefficients_A_train`, and `coefficients_A_valid`. `solver_diagnostics.json` records termination information; `optimization_history.jsonl` records the saved outer-iteration history.

Training normalization arrays are stored in the same NPZ, while the diagnostics contain the versioned `inference_state` with inference settings. Use [load_fit](api/artifacts.html#load_fit) to obtain a replay object instead of trying to infer those settings from the new batch.

B's columns refer to training observations; A's rows refer to the observations whose coefficients were inferred. Preserve those identities outside the numeric arrays when interpreting samples.

## Reuse means replaying the same procedure

Replay held-out rows and compare with the saved coefficients within the stated tolerance. Use the stored solver identity and precision. The loader checks saved array and diagnostic hashes and requires the recorded solver source to match the installed implementation; source changes can therefore require the matching earlier package version for replay. A completed output directory is not proof of convergence, stationarity, or scientific validation.

Inputs transformed before fitting need that same external transformation before replay. Saved solver normalization does not replace upstream preprocessing state.

## Selection and evaluation

Fit preprocessing and archetypes on Train. Select settings on Validation. Freeze the choice before a designated Test evaluation, and retain any history of Test reuse during research. Later analysis of already viewed data remains exploratory.
