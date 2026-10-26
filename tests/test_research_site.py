"""Publication-boundary and source-reference contracts for the research site."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/docs"))

from build_site import build_site, check_site, main  # noqa: E402
from site_sources import SiteError, extract_symbols  # noqa: E402
from site_validation import validate_site  # noqa: E402


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    for directory in ("docs/site/pages/ja", "docs/site/_templates", "docs/site/_static", "src/neureps_arc_nlp"):
        (root / directory).mkdir(parents=True)
    for relative in (
        "docs/site/_templates/layout.html",
        "docs/site/_templates/api.html",
        "docs/site/_templates/api_index.html",
        "docs/site/_static/site.css",
        "docs/site/_static/site.js",
    ):
        shutil.copyfile(REPO / relative, root / relative)
    config = {
        "schema_version": 1,
        "title": "Arc_nlp",
        "tagline": "Research documentation",
        "repository_url": "https://github.com/example/arc_nlp",
        "pages": [
            {"slug": "index", "title": "Overview", "group": "Explore", "description": "A fixture.", "lang": "en"},
            {"slug": "ja/index", "title": "日本語", "group": "日本語", "description": "説明。", "lang": "ja"},
        ],
        "api": [
            {
                "slug": "methods",
                "module": "neureps_arc_nlp.methods",
                "source": "src/neureps_arc_nlp/methods.py",
                "title": "Fitting",
                "summary": "Fit a representation.",
                "stability": "public",
                "symbols": ["fit", "Result"],
            }
        ],
    }
    (root / "docs/site/site.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "docs/site/pages/index.md").write_text(
        "# Overview\n\n[日本語](ja/index.md#説明) · [API](api/methods.html#fit)\n\n"
        '## Reproducibility\n\nCurated content.\n\n```python\nprint("<script>escaped</script>")\n```\n',
        encoding="utf-8",
    )
    (root / "docs/site/pages/ja/index.md").write_text(
        "# 日本語\n\n## 説明\n\n[Overview](../index.md#reproducibility)\n", encoding="utf-8"
    )
    (root / "src/neureps_arc_nlp/methods.py").write_text(
        'raise RuntimeError("Production modules must never be imported")\n\n'
        'def fit(x: "Array", *, seed: int = 0) -> "Result":\n'
        '    """Fit the input. Literal <script> is documentation."""\n'
        "    return x\n\n"
        "@dataclass\n"
        "class Result:\n"
        '    """A fitted representation."""\n'
        "    values: list[float]\n"
        "    seed: int = 0\n\n"
        "def private_unselected():\n"
        '    return "UNSELECTED_IMPLEMENTATION"\n',
        encoding="utf-8",
    )
    return root


def set_config(project: Path, mutate) -> None:
    path = project / "docs/site/site.json"
    config = json.loads(path.read_text())
    mutate(config)
    path.write_text(json.dumps(config), encoding="utf-8")


def test_allowlist_ast_only_source_and_determinism(project: Path) -> None:
    (project / "docs/site/pages/private.md").write_text("NEVER_PUBLISH_PRIVATE_PAGE")
    (project / "artifacts").mkdir()
    (project / "artifacts/private.txt").write_text("NEVER_PUBLISH_ARTIFACT")
    output = project / "artifacts/delivery/site"
    result = build_site(project, output)
    assert result["pages"] == 4
    first = {p.relative_to(output).as_posix(): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    build_site(project, output)
    second = {p.relative_to(output).as_posix(): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    assert first == second
    publication = "".join(data.decode() for data in first.values())
    assert "NEVER_PUBLISH" not in publication
    assert "UNSELECTED_IMPLEMENTATION" not in publication
    api = (output / "api/methods.html").read_text()
    assert 'id="fit"' in api
    assert "seed: int=0" in api
    assert "&lt;script&gt;" in api
    assert "<script> is documentation" not in api
    assert "Production modules must never be imported" not in publication
    manifest = json.loads((output / "source-manifest.json").read_text())
    expected = hashlib.sha256((project / "src/neureps_arc_nlp/methods.py").read_bytes()).hexdigest()
    assert manifest["api"][0]["sha256"] == expected
    assert manifest["api"][0]["symbols"][0]["line_start"] == 3
    assert manifest["api"][0]["symbols"][1]["line_start"] == 7
    assert "Declared annotated fields" in api
    assert validate_site(output)["links"] > 10


def test_nested_pages_remain_relative_under_project_prefix(project: Path) -> None:
    output = project / "build"
    build_site(project, output)
    page = (output / "ja/index.html").read_text()
    assert 'href="../index.html#reproducibility"' in page
    assert 'href="../assets/site.css"' in page
    assert 'src="../assets/site.js"' in page
    assert 'data-site-root="../index.html"' in page
    assert 'href="/assets/' not in page
    assert 'href="../api/methods.html"' in page


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c.update({"unexpected": True}),
        lambda c: c.update({"schema_version": True}),
        lambda c: c["pages"][0].update({"slug": "../private"}),
        lambda c: c["pages"].append(c["pages"][0]),
        lambda c: c["api"][0].update({"source": "docs/private.py"}),
        lambda c: c["api"][0].update({"symbols": ["missing"]}),
        lambda c: c["api"][0].update({"symbols": ["fit", "fit"]}),
        lambda c: c["api"][0].update({"symbols": [{}]}),
    ],
)
def test_invalid_configuration_fails_closed(project: Path, mutation) -> None:
    set_config(project, mutation)
    with pytest.raises(SiteError):
        build_site(project, project / "build")
    assert not (project / "build").exists()


@pytest.mark.parametrize(
    "content",
    [
        "[missing](missing.html)",
        "[missing anchor](ja/index.html#missing)",
        "[root relative](/index.html)",
        "[outside](../../outside.html)",
        '<img src="https://example.com/tracker.png" alt="remote">',
        "<script>alert(1)</script>",
        '<a href="javascript:alert(1)">unsafe</a>',
        '<h2 id="same">A</h2><h2 id="same">B</h2>',
        '<img src="assets/missing.svg" onerror="alert(1)">',
    ],
)
def test_broken_links_and_unsafe_html_fail_before_publication(project: Path, content: str) -> None:
    (project / "docs/site/pages/index.md").write_text("# Overview\n\n" + content)
    with pytest.raises(SiteError):
        build_site(project, project / "build")
    assert not (project / "build").exists()


@pytest.mark.parametrize(
    "private",
    [
        "/Users/researcher/private/report.txt",
        "/home/researcher/runtime/secret.json",
        "/mnt/storage/private.npz",
        "-----BEGIN PRIVATE KEY-----",
        "ghp_" + "a" * 30,
        "192.168.30.12",
        'api_key = "abc1234567890123456789"',
    ],
)
def test_private_content_is_rejected(project: Path, private: str) -> None:
    (project / "docs/site/pages/index.md").write_text("# Overview\n\n" + private)
    with pytest.raises(SiteError, match="Publication guard"):
        build_site(project, project / "build")


def test_private_selected_source_is_rejected_but_unselected_source_is_not_published(project: Path) -> None:
    path = project / "src/neureps_arc_nlp/methods.py"
    path.write_text(path.read_text() + '\nSECRET_OUTSIDE_SELECTION = "/Users/example/private"\n')
    build_site(project, project / "build")
    path.write_text(path.read_text().replace("return x", 'return "/Users/example/private"'))
    with pytest.raises(SiteError, match="Publication guard"):
        build_site(project, project / "build")


def test_symlinks_refused_and_unowned_output_preserved(project: Path, tmp_path: Path) -> None:
    destination = project / "build"
    destination.mkdir()
    (destination / "user-owned.txt").write_text("preserve")
    with pytest.raises(SiteError, match="unowned"):
        build_site(project, destination)
    assert (destination / "user-owned.txt").read_text() == "preserve"
    source = project / "docs/site/pages/index.md"
    outside = tmp_path / "outside.md"
    source.rename(outside)
    source.symlink_to(outside)
    with pytest.raises(SiteError, match="Symlink"):
        build_site(project, project / "another-build")


def test_invalid_rebuild_preserves_previous_site_and_unknown_output(project: Path) -> None:
    output = project / "build"
    build_site(project, output)
    previous = (output / "index.html").read_bytes()
    (project / "docs/site/pages/index.md").write_text("# Overview\n[bad](missing.html)")
    with pytest.raises(SiteError):
        build_site(project, output)
    assert (output / "index.html").read_bytes() == previous
    (output / "user-note.txt").write_text("preserve")
    with pytest.raises(SiteError, match="inventory changed"):
        build_site(project, output)
    assert (output / "user-note.txt").read_text() == "preserve"


def test_source_drift_updates_signature_and_snapshot(project: Path) -> None:
    output = project / "build"
    build_site(project, output)
    before = (output / "source-manifest.json").read_text()
    source = project / "src/neureps_arc_nlp/methods.py"
    source.write_text(source.read_text().replace("seed: int = 0", "seed: int = 11"))
    build_site(project, output)
    assert "seed: int=11" in (output / "api/methods.html").read_text()
    assert before != (output / "source-manifest.json").read_text()
    config = json.loads((project / "docs/site/site.json").read_text())
    extracted = extract_symbols(project, config["api"][0])
    assert extracted["symbols"][1]["fields"][1]["default"] == "11"


def test_changed_published_file_fails_integrity_check(project: Path) -> None:
    output = project / "build"
    build_site(project, output)
    with (output / "assets/site.js").open("a") as stream:
        stream.write("// modified\n")
    with pytest.raises(SiteError, match="output has changed"):
        validate_site(output)


@pytest.mark.parametrize(
    "explanations",
    [
        ["not a mapping"],
        {"unselected": "Not a selected symbol."},
        {"fit": "   "},
        {"fit": None},
        {"fit": ["not plain text"]},
    ],
)
def test_invalid_api_explanations_are_rejected(project: Path, explanations) -> None:
    set_config(project, lambda c: c["api"][0].update({"explanations": explanations}))
    with pytest.raises(SiteError):
        build_site(project, project / "build")
    assert not (project / "build").exists()


def test_api_explanations_are_escaped_and_do_not_replace_existing_docstrings(project: Path) -> None:
    explanation = 'Authored explanation: <script>alert("text")</script> & **literal Markdown**.'
    set_config(project, lambda c: c["api"][0].update({"explanations": {"fit": explanation}}))
    output = project / "build"
    build_site(project, output)
    rendered = (output / "api/methods.html").read_text()
    assert 'class="symbol-explanation"' in rendered
    assert "Authored explanation: &lt;script&gt;" in rendered
    assert "**literal Markdown**" in rendered
    assert '<script>alert("text")</script>' not in rendered
    assert 'class="docstring">Fit the input.' in rendered
    assert rendered.index('class="symbol-explanation"') < rendered.index('class="docstring"')
    search = json.loads((output / "assets/search.json").read_text())
    assert any(explanation in page["text"] for page in search)


def test_api_explanation_suppresses_missing_docstring_placeholder(project: Path) -> None:
    source = project / "src/neureps_arc_nlp/methods.py"
    source.write_text(source.read_text().replace('    """A fitted representation."""\n', ""))
    set_config(
        project,
        lambda c: c["api"][0].update(
            {"explanations": {"Result": "Carries the fitted values and initialization seed."}}
        ),
    )
    output = project / "build"
    build_site(project, output)
    rendered = (output / "api/methods.html").read_text()
    assert "Carries the fitted values and initialization seed." in rendered
    assert "No symbol docstring is present" not in rendered


def test_private_api_explanation_is_not_publishable(project: Path) -> None:
    set_config(
        project, lambda c: c["api"][0].update({"explanations": {"fit": "Inspect /Users/example/private/run.json"}})
    )
    with pytest.raises(SiteError, match="Publication guard"):
        build_site(project, project / "build")


def test_reviewed_manifest_digest_accepts_exact_build_and_preserves_unbound_check(project: Path) -> None:
    output = project / "build"
    build_site(project, output)
    reviewed = hashlib.sha256((output / "source-manifest.json").read_bytes()).hexdigest()
    assert check_site(output, reviewed) == validate_site(output)
    assert check_site(output) == validate_site(output)


def test_rebuilt_valid_content_rejects_previous_reviewed_digest(project: Path) -> None:
    output = project / "build"
    build_site(project, output)
    reviewed = hashlib.sha256((output / "source-manifest.json").read_bytes()).hexdigest()
    page = project / "docs/site/pages/index.md"
    page.write_text(page.read_text() + "\n## New research context\n\nNewly authored content.\n")
    build_site(project, output)
    assert validate_site(output)["pages"] == 4
    with pytest.raises(SiteError, match="does not match the reviewed site"):
        check_site(output, reviewed)


def test_reviewed_digest_uses_exact_manifest_bytes(project: Path) -> None:
    output = project / "build"
    build_site(project, output)
    manifest = output / "source-manifest.json"
    reviewed = hashlib.sha256(manifest.read_bytes()).hexdigest()
    manifest.write_bytes(manifest.read_bytes() + b"\n")
    assert validate_site(output)["pages"] == 4
    with pytest.raises(SiteError, match="does not match the reviewed site"):
        check_site(output, reviewed)


@pytest.mark.parametrize("reviewed", ["", "A" * 64, "a" * 63, "g" * 64, " " + "a" * 64])
def test_malformed_reviewed_manifest_digest_rejected(project: Path, reviewed: str) -> None:
    output = project / "build"
    build_site(project, output)
    with pytest.raises(SiteError, match="exactly 64 lowercase hexadecimal"):
        check_site(output, reviewed)


def test_reviewed_digest_does_not_bypass_normal_output_validation(project: Path) -> None:
    output = project / "build"
    build_site(project, output)
    reviewed = hashlib.sha256((output / "source-manifest.json").read_bytes()).hexdigest()
    with (output / "index.html").open("a") as stream:
        stream.write("Unreviewed output change")
    with pytest.raises(SiteError, match="output has changed"):
        check_site(output, reviewed)


def test_cli_build_rejects_reviewed_manifest_flag_before_writing(project: Path, monkeypatch, capsys) -> None:
    output = project / "build"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_site.py",
            "build",
            "--repo-root",
            str(project),
            "--output",
            str(output),
            "--expected-manifest-sha256",
            "a" * 64,
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "only valid with check" in capsys.readouterr().err
    assert not output.exists()


def test_cli_check_enforces_reviewed_manifest_flag(project: Path, monkeypatch, capsys) -> None:
    output = project / "build"
    build_site(project, output)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_site.py",
            "check",
            "--repo-root",
            str(project),
            "--output",
            str(output),
            "--expected-manifest-sha256",
            "a" * 64,
        ],
    )
    assert main() == 1
    assert "does not match the reviewed site" in capsys.readouterr().err


def test_portable_site_builds_from_its_own_sources_with_conference_and_japanese_routes(tmp_path: Path) -> None:
    output = tmp_path / "portable-site"
    build_site(REPO, output)
    manifest = json.loads((output / "source-manifest.json").read_text())
    assert manifest["api"]
    assert all(entry["module"].startswith("neureps_arc_nlp.") for entry in manifest["api"])
    assert all(entry["source"].startswith("src/neureps_arc_nlp/") for entry in manifest["api"])
    assert all(not Path(relative).is_absolute() and ".." not in Path(relative).parts for relative in manifest["inputs"])
    assert all((REPO / relative).is_file() for relative in manifest["inputs"])
    conference = (output / "conference/index.html").read_text()
    assert 'href="../tutorial.html"' in conference
    assert 'href="../ja/index.html"' in conference
    assert 'href="https://github.com/ryuuua/Arc_nlp_public"' in conference
    assert (output / "ja/index.html").is_file()
    assert (output / "ja/mathematics.html").is_file()
    assert "from arc_nlp." not in (output / "tutorial.html").read_text()


def test_selected_assets_are_hashed_and_unselected_assets_are_not_published(project: Path) -> None:
    selected = {
        "poster-theme.css": "body { color: #004aad; }\n",
        "conference/figure.svg": '<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0"/></svg>\n',
        "conference/summary.json": '{"mean": 0.854}\n',
        "conference/NOTICE.txt": "Selected figure attribution.\n",
    }
    for name, content in selected.items():
        path = project / "docs/site/_static" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    (project / "docs/site/_static/private.txt").write_text("NEVER_PUBLISH /Users/example/private")
    set_config(project, lambda c: c.update(assets=list(selected)))
    page = project / "docs/site/pages/index.md"
    page.write_text(page.read_text() + "\n[Figure](assets/conference/figure.svg)\n")
    output = project / "build"
    build_site(project, output)
    manifest = json.loads((output / "source-manifest.json").read_text())
    for name, content in selected.items():
        assert (output / "assets" / name).read_text() == content
        expected = hashlib.sha256(content.encode()).hexdigest()
        assert manifest["inputs"][f"docs/site/_static/{name}"] == expected
        assert manifest["outputs"][f"assets/{name}"] == expected
    assert "docs/site/_static/private.txt" not in manifest["inputs"]
    assert not (output / "assets/private.txt").exists()
    before = (output / "source-manifest.json").read_bytes()
    build_site(project, output)
    assert (output / "source-manifest.json").read_bytes() == before
    (project / "docs/site/_static/conference/summary.json").write_text('{"mean": 0.790}\n')
    build_site(project, output)
    assert (output / "source-manifest.json").read_bytes() != before


@pytest.mark.parametrize(
    "assets",
    [
        None,
        "poster-theme.css",
        [None],
        [{}],
        [""],
        ["../private.txt"],
        ["conference/../../private.txt"],
        ["/private.txt"],
        ["conference\\private.txt"],
        ["conference//figure.svg"],
        ["conference/./figure.svg"],
        ["conference/private.js"],
        ["conference/figure.png"],
        ["conference/figure.svg", "conference/figure.svg"],
        ["conference/figure.svg", "conference/Figure.svg"],
        ["site.css"],
        ["Site.css"],
        ["site.js"],
        ["code.css"],
        ["search.json"],
    ],
)
def test_invalid_asset_allowlists_fail_closed(project: Path, assets) -> None:
    set_config(project, lambda c: c.update(assets=assets))
    with pytest.raises(SiteError):
        build_site(project, project / "build")
    assert not (project / "build").exists()


@pytest.mark.parametrize("failure", ["missing", "symlink", "private"])
def test_asset_input_failure_preserves_previous_build(project: Path, tmp_path: Path, failure: str) -> None:
    output = project / "build"
    build_site(project, output)
    previous = (output / "source-manifest.json").read_bytes()
    selected = project / "docs/site/_static/selected.txt"
    if failure == "symlink":
        outside = tmp_path / "outside.txt"
        outside.write_text("External input.")
        selected.symlink_to(outside)
    elif failure == "private":
        selected.write_text("/Users/example/private/report.txt")
    set_config(project, lambda c: c.update(assets=["selected.txt"]))
    with pytest.raises(SiteError):
        build_site(project, output)
    assert (output / "source-manifest.json").read_bytes() == previous
    assert not (output / "assets/selected.txt").exists()


def test_explicit_page_toc_supports_raw_html_headings_and_overrides_generated_toc(project: Path) -> None:
    asset = project / "docs/site/_static/conference/conference.css"
    asset.parent.mkdir(parents=True)
    asset.write_text(".conference-hub { color: #004aad; }\n")
    set_config(project, lambda c: c.update(assets=["conference/conference.css"]))
    page = project / "docs/site/pages/index.md"
    page.write_text(page.read_text() + '<h2 id="poster-summary">Poster summary</h2>\n')
    set_config(
        project,
        lambda c: c["pages"][0].update(
            layout="conference", toc=[{"id": "poster-summary", "title": "Result <summary>", "level": 2}]
        ),
    )
    output = project / "build"
    build_site(project, output)
    html = (output / "index.html").read_text()
    assert 'href="#poster-summary"' in html
    assert "Result &lt;summary&gt;" in html
    assert 'href="#reproducibility"' not in html
    assert 'id="reproducibility"' in html
    set_config(project, lambda c: c["pages"][0].update(toc=[]))
    build_site(project, output)
    assert 'href="#poster-summary"' not in (output / "index.html").read_text()


@pytest.mark.parametrize(
    "options",
    [
        {"layout": "private"},
        {"layout": None},
        {"layout": []},
        {"toc": None},
        {"toc": "reproducibility"},
        {"toc": [None]},
        {"toc": [{"id": "reproducibility", "title": "Result"}]},
        {"toc": [{"id": "reproducibility", "title": " ", "level": 2}]},
        {"toc": [{"id": "reproducibility", "title": "Result", "level": True}]},
        {"toc": [{"id": "reproducibility", "title": "Result", "level": 1}]},
        {"toc": [{"id": "bad anchor", "title": "Result", "level": 2}]},
        {"toc": [{"id": "../outside", "title": "Result", "level": 2}]},
        {"toc": [{"id": [], "title": "Result", "level": 2}]},
        {"toc": [{"id": "reproducibility", "title": "Result", "level": 2}] * 2},
    ],
)
def test_invalid_page_presentation_options_fail_closed(project: Path, options) -> None:
    set_config(project, lambda c: c["pages"][0].update(options))
    with pytest.raises(SiteError):
        build_site(project, project / "build")
    assert not (project / "build").exists()


def test_explicit_toc_cannot_bypass_anchor_validation(project: Path) -> None:
    set_config(
        project, lambda c: c["pages"][0].update(toc=[{"id": "missing", "title": "Missing", "level": 3}])
    )
    with pytest.raises(SiteError, match="Broken anchor"):
        build_site(project, project / "build")
