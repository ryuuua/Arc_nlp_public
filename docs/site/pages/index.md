# Geometry for understanding embeddings

Arc_nlp studies high-dimensional representations through **archetypal analysis**: observations become mixtures of shared archetypes. The portable package connects that mathematical model to reproducible fitting, held-out inference, and interpretable measurements.

<div class="equation" role="math" aria-label="X is approximately A Z, with Z equal to B X">X ≈ A Z &nbsp; · &nbsp; Z = B X</div>

<div class="hero-actions"><a class="button-primary" href="tutorial.html">Run a first analysis →</a><a class="button-secondary" href="mathematics.html">Explore the mathematics</a><a class="button-secondary" href="ja/index.html">日本語</a></div>

## From research question to working code

<div class="card-grid">
<a class="card" href="mathematics.html"><span class="card-label">Understand</span><h3>Convex mixtures</h3><p>What archetypes and coefficients represent, and what their constraints imply.</p><span class="card-arrow">↗</span></a>
<a class="card" href="tutorial.html"><span class="card-label">Try</span><h3>A small, complete fit</h3><p>Generate synthetic inputs, fit AA, and replay a saved model on held-out observations.</p><span class="card-arrow">↗</span></a>
<a class="card" href="solvers.html"><span class="card-label">Choose</span><h3>Versioned solvers</h3><p>Preserve the v2 baseline or explicitly select v3 and v4 numerical policies.</p><span class="card-arrow">↗</span></a>
<a class="card" href="api/index.html"><span class="card-label">Inspect</span><h3>Explanations and source</h3><p>Follow actual portable-package APIs to their signatures and selected implementation.</p><span class="card-arrow">↗</span></a>
</div>

## The representation

| Object | Meaning | Shape |
|---|---|---|
| X | Observations in one embedding space | n × d |
| A | Each observation's mixture of archetypes | n × K |
| B | Each archetype's mixture of training observations | K × n_train |
| Z | Archetypes, B X_train | K × d |

Fit on Train, infer new coefficients using fixed archetypes, and evaluate reconstruction and label relationships separately. The [mathematical guide](mathematics.html) explains the model; [evaluation](evaluation.html) explains the measurements.

## What is distributed

This repository contains the `neureps_arc_nlp` portable Python package: Standard AA v2, v3 and v4, NumPy/Torch execution where supported, a Hydra fitting command, saved-fit inference, and selected metrics. The default remains the paper-oriented v2 profile for reproducibility. V3 and v4 require explicit selection.

Broader work on robust losses, shared dictionaries, model alignment and geometric extensions belongs to the [research program](research.html). Those methods are not silently included in this portable distribution.

## Visiting from a presentation

The [presentation landing page](conference/index.html) is a compact entry point for a QR code, with links to this project, a runnable example, source, and Japanese explanations.
