# Public v3/v4 and research website release

Release objective (2026-10-23): publish the conference/research site from
`Arc_nlp_public`, and bring v4 and its supporting portable implementation into
this repository.

## Scope and compatibility

- Add the standalone Standard v3/v4 kernels, stopping policy and Torch runtime
  helper under `src/neureps_arc_nlp/`, with source fingerprints. Preserve the
  pinned v2 `solver.py` bytes and existing v2 default/CLI route.
- Add explicit `solver=standard_v3` and `solver=standard_v4` profiles (v4 uses
  float64), dispatch, solver-specific provenance, saved inference state and
  failure-only evidence. Keep the existing four successful output filenames.
- Add public-source-only research documentation, executable examples, API source
  explanations, a stable `conference/` entry point and GitHub Pages workflow.
- Update README/NOTICE/package version, without inventing a software license,
  scientific results, affiliations or publication metadata.

Excluded: private repository history, datasets, weights, experiment outputs,
lab infrastructure, manuscript sources, remote computation and unrelated edits.
Only explicitly selected source files are exported. Private sources remain
unchanged. Workers own disjoint files; integration and publishing are serialized.

## Validation and release gates

1. Establish the existing public test baseline in an available Python >=3.11.
2. Check copied source hashes and bounded NumPy/Torch CPU equivalence; exercise
   v2 compatibility, v4 precision rejection, diagnostics, inference replay and
   failure-only evidence.
3. Run public tests, lint on maintained adapters, the runnable tutorial and the
   documentation build/link/privacy/source-manifest checks.
4. Review the complete public diff and tracked-file inventory. Verify a clean
   export builds without access to the development repository.
5. Commit/push reviewed public changes, enable public GitHub Pages, deploy the
   reviewed site manifest, and verify the live conference URL and QR destination.

Local tests establish software behavior only; they do not certify GPU behavior,
real-data solver superiority or a global optimum. Roll back a release by reverting
the public release commit and redeploying the previous site artifact; retain all
historical fit outputs and private work.

## Local acceptance (2026-10-23)

- All 261 public tests pass with Python 3.12.6, NumPy 1.26.4 and Torch 2.8.0
  on CPU. NumPy-only imports are independently covered. The original eight
  tests passed before implementation.
- The four numerical exports are byte-identical to their selected upstream
  sources; 18 bounded upstream/public fit and inference comparisons are exact.
  `provenance/source_fingerprints.json` records inputs, settings and limits.
- The v2 source hash and defaults are preserved. Saved inference, invalid-state
  rejection, numerical failure evidence and float32 cancellation-sensitive
  normalization validation have regression coverage.
- The site builds 22 pages and validates 811 links using only this repository.
  The runnable synthetic example passes for all three solver profiles.
- Desktop (1280×900) and mobile (390×844) checks cover the conference route,
  tutorial navigation, menu, Japanese search and API source disclosure. No
  overflow or relevant console errors were observed.
- Ruff, whitespace checks and wheel packaging pass. Independent source-export,
  inference/failure-contract and mathematical-content reviews are complete.

Publication remains a separate receipt: verify GitHub CI and the live Pages
manifest after deployment. No GPU or real-corpus numerical validation is claimed.
