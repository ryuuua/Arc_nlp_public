# Understand embeddings through archetypes

A compact route into Arc_nlp: the research question, a working example, and the code behind it.

<div class="equation" role="math" aria-label="Observations are mixtures of archetypes, which are mixtures of training observations">Observations ≈ mixtures of archetypes<br>X ≈ AZ &nbsp; · &nbsp; Z = BX</div>

**Research question:** what structure and information survive when an embedding space is represented through a small collection of archetypes?

<div class="hero-actions"><a class="button-primary" href="../index.html">Explore the project →</a><a class="button-secondary" href="../tutorial.html">Run an example</a><a class="button-secondary" href="https://github.com/ryuuua/Arc_nlp_public">Read the source ↗</a><a class="button-secondary" href="../ja/index.html">日本語</a></div>

## The model in one minute

Each observation receives nonnegative weights summing to one. Archetypes are also mixtures of training observations, giving both factors a geometric interpretation. New observations receive coefficients against the fixed fitted dictionary.

Read the [mathematical introduction](../mathematics.html), then inspect the [API and selected source](../api/index.html).

## What you can run

The portable package contains Standard AA v2/v3/v4, reproducible prepared-array fits, saved-model inference, and selected evaluation metrics. The small tutorial needs no downloaded dataset or GPU. V2 remains the compatibility default; v3 and v4 are explicit choices.

Broader [research directions](../research.html) are clearly separated from distributed features.
