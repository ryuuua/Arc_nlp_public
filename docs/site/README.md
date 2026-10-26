# Arc_nlp portable research website

The authored site, selected API sources, templates, assets and builder all live
in this repository. No development checkout, embedding service, model download,
scientific runtime import, or external document source is needed for a build.

## Build and preview

Use Python 3.11 or newer with the pinned documentation dependencies in
`requirements.txt`. Install them in your chosen documentation environment when
needed; they are separate from the fitting runtime.

```bash
make research-site
make research-site-check
make research-site-test
make research-site-serve
```

Set `PYTHON` to an existing interpreter if required. The default output is
`artifacts/delivery/research_site/`. The preview binds to loopback and serves only
that generated directory on port 8765; override `RESEARCH_SITE_PORT` if needed.
Relative links support the GitHub project prefix and nested language routes.

## Publication boundary

`site.json` explicitly lists every Markdown page, API symbol, and optional static asset. Source paths
must match modules inside `src/neureps_arc_nlp/`. The build never recursively
copies source trees, datasets, generated fits, operational notes or reports.
Missing modules, symbols, or selected assets fail the build. Optional `assets` entries name UTF-8 CSS/SVG/JSON/TXT files relative to `docs/site/_static/`; they are emitted under `assets/`. Unlisted files are never copied. Traversal, symlinks, duplicate names, and collisions with built-in assets are rejected.

Python declarations, docstrings, fields and selected source excerpts are read
using AST parsing without executing the package. Authored per-symbol
`explanations` are escaped plain text and distinguished from source docstrings.
The source manifest records input, output, module and excerpt hashes. Builds are
deterministic for identical source inputs and dependency versions.

The checker verifies publication content, files, links, anchors and output
digests. A reviewed publication can additionally bind to the exact manifest:

```bash
shasum -a 256 artifacts/delivery/research_site/source-manifest.json
python scripts/docs/build_site.py check --expected-manifest-sha256 REVIEWED_DIGEST
```

Replace `REVIEWED_DIGEST` with the actual 64-character lowercase hexadecimal
digest. A changed valid build fails an earlier digest check. This option is
accepted only with `check`; ordinary checks without it still work.

## Content ownership

Package tutorials and API pages describe only distributed implementations.
Broader experimental methods remain on the research page with their availability
clearly marked. The package's default is the preserved `paper_v2` profile; v3 and
v4 are explicit alternatives. No general performance superiority is claimed.

`examples/synthetic_fit.py` exercises the distributed configuration, fit, saved
state and inference path without external inputs. Keep its tutorial aligned with
the actual package and run it when these interfaces change.

## Presentation entry

`pages/conference/index.md` builds to `conference/index.html`. Its intended
canonical QR target is:

`https://ryuuua.github.io/Arc_nlp_public/conference/`

Verify the deployed route before printing a QR. The page follows the supplied poster, "アーキタイプ分析による文埋め込みの幾何学的構造の検討", including its verified title and authors. It presents the poster findings, distinguishes the selected K=90 / seed 0 comparison from the poster's overall 32-condition / ten-initialization design, and adds the saved Validation-document visualizations. It links to `conference/profile.html`, the project documentation, and the external FerMatter site. No conference acceptance or additional license is asserted.

Conference pages use the optional `layout: "conference"`; an optional explicit `toc` list supplies IDs, titles, and heading levels for authored HTML headings. Other pages keep their existing layout and automatically derived Markdown table of contents.

The selected SVGs and aggregate JSON contain only figures and summary values. Raw embeddings, document IDs/text, fit state, execution receipts, local preview provenance, and the source PDF are excluded. Scientific metrics/figures are preserved from the reviewed K=90 companion draft. The AA fit and evaluation runtime is not part of this publication. The existing personal website remains independently managed.

Third-party attribution for the scientific colour maps and FerMatter logo accompanies those assets. The poster blue (`#004AAD`) is applied through an explicitly selected theme stylesheet.

The GitHub Actions workflow and repository README own deployment instructions.
Only the validated generated site is a publication artifact; runtime outputs
are not included.
