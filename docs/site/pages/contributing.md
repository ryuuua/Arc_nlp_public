# Contributing and documentation

This repository distributes a focused subset of the broader Arc_nlp research project. The compatibility package name is `neureps_arc_nlp`. Make changes against the interfaces and implementation actually contained here.

## Useful changes

- Reproducible examples using the distributed API.
- Focused numerical, saved-state, or configuration regression tests.
- Clearer explanations of objectives, metrics and stopping policies.
- Small structural changes that preserve public behavior.
- Research prototypes with explicit assumptions and a distinct identity.

Preserve earlier solver contracts and user-owned results. New algorithms and changed stopping policies need explicit selectors rather than silently changing old experiments.

## Build the documentation

Install documentation dependencies in an appropriate development environment from `docs/site/requirements.txt`, then run:

```bash
make research-site
make research-site-check
make research-site-serve
```

The server exposes only `artifacts/delivery/research_site/`. The build uses local authored pages and selected source symbols; it does not import the scientific runtime or require a second repository.

The source manifest links each generated view to its inputs. A later publication can require its exact reviewed SHA-256 using `check --expected-manifest-sha256`.

## Review source explanations

API signatures, declared fields, docstrings and source excerpts come from AST parsing. Authored explanations add context but do not create new APIs or stability guarantees. Missing symbols fail the build.

## Citation and distribution terms

Use approved project/release metadata for citations and redistribution terms. This site does not invent a paper identifier, author list or open-source license. Consult the repository's current README and NOTICE before redistributing source.

## Documentation inspiration

The structure takes inspiration from [CEBRA's documentation](https://cebra.ai/docs/): research introduction, installation, examples, mathematical concepts and implementation reference. Arc_nlp's content and implementation are maintained separately.
