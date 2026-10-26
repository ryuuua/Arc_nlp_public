#!/usr/bin/env python3
"""Build/check the explicit public research site without importing neureps_arc_nlp."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path
from typing import Any

from jinja2 import DictLoader, Environment, select_autoescape
from markupsafe import Markup
from pygments.formatters import HtmlFormatter
from site_render import highlight_code, relative_url, render_markdown, search_text
from site_sources import (
    CLOUDFLARE_ANALYTICS_ASSET,
    CLOUDFLARE_ANALYTICS_PAGES,
    SiteError,
    cloudflare_analytics_contract,
    digest,
    extract_symbols,
    load_config,
    safe_path,
)
from site_validation import OWNER, OWNER_FILE, check_public_text, validate_output_destination, validate_site

TEMPLATES = ("layout.html", "api.html", "api_index.html")
ASSETS = ("site.css", "site.js")


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def build_site(root: Path, output: Path) -> dict[str, int]:
    """Validate all content in staging before replacing an owned build."""
    root = root.absolute()
    output = output.absolute()
    validate_output_destination(output, root)
    config = load_config(root)
    analytics = cloudflare_analytics_contract(config.get("cloudflare_web_analytics_token", ""))
    inputs: dict[str, str] = {}

    def read(relative: str) -> str:
        data = safe_path(root, relative).read_bytes()
        inputs[relative] = digest(data)
        return data.decode("utf-8")

    read("docs/site/site.json")
    templates = {name: read(f"docs/site/_templates/{name}") for name in TEMPLATES}
    environment = Environment(loader=DictLoader(templates), autoescape=select_autoescape(["html"]))
    environment.globals["relative_url"] = relative_url
    modules = [extract_symbols(root, module) for module in config["api"]]
    for module in modules:
        inputs[module["source"]] = module["sha256"]
        for symbol in module["symbols"]:
            check_public_text(symbol["source"], f"{module['module']}.{symbol['name']}")
            symbol["highlighted"] = Markup(highlight_code(symbol["source"], "python"))
    navigation = [dict(page, url=page["slug"] + ".html") for page in config["pages"]]
    navigation += [{"title": "API overview", "url": "api/index.html", "group": "Reference"}]
    navigation += [{"title": m["title"], "url": f"api/{m['slug']}.html", "group": "Reference"} for m in modules]
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in navigation:
        groups.setdefault(item["group"], []).append(item)
    outputs: dict[str, str] = {}
    search = []

    def emit_page(url: str, page: dict[str, Any], content: str, toc: list[dict[str, Any]]) -> None:
        outputs[url] = environment.get_template("layout.html").render(
            site=config,
            page=page,
            current=url,
            navigation_groups=groups.items(),
            content=Markup(content),
            toc=page.get("toc", toc),
            analytics=analytics if url in CLOUDFLARE_ANALYTICS_PAGES else None,
        )
        search.append({"title": page["title"], "url": url, "text": search_text(content)[:16000]})

    for page in config["pages"]:
        source = read(f"docs/site/pages/{page['slug']}.md")
        check_public_text(source, page["slug"])
        content, toc = render_markdown(source)
        emit_page(f"{page['slug']}.html", page, content, toc)
    for module in modules:
        content = environment.get_template("api.html").render(module=module)
        page = {"title": module["title"], "description": module["summary"], "lang": "en", "group": "Reference"}
        toc = [{"id": s["name"], "title": s["name"], "level": 2} for s in module["symbols"]]
        emit_page(f"api/{module['slug']}.html", page, content, toc)
    content = environment.get_template("api_index.html").render(modules=modules)
    emit_page(
        "api/index.html",
        {
            "title": "API reference",
            "description": "Selected package interfaces and advanced implementation references.",
            "lang": "en",
            "group": "Reference",
        },
        content,
        [],
    )
    for name in (*ASSETS, *config.get("assets", [])):
        outputs[f"assets/{name}"] = read(f"docs/site/_static/{name}")
    if analytics:
        outputs[analytics["loader"]] = read(f"docs/site/_static/{CLOUDFLARE_ANALYTICS_ASSET}")
    outputs["assets/code.css"] = HtmlFormatter(style="friendly").get_style_defs(".highlight, pre code") + "\n"
    outputs["assets/search.json"] = json_text(search)
    outputs[".nojekyll"] = ""
    # Builder files are provenance inputs, not copied source or publication pages.
    for name in ("build_site.py", "site_sources.py", "site_render.py", "site_validation.py"):
        relative = f"scripts/docs/{name}"
        if (root / relative).is_file():
            read(relative)
    manifest = {
        "schema_version": 1,
        "generator": OWNER,
        "dependencies": {name: version(name) for name in ("markdown-it-py", "Jinja2", "Pygments")},
        "inputs": inputs,
        "outputs": {name: digest(value.encode("utf-8")) for name, value in outputs.items()},
        "api": [
            {
                "module": module["module"],
                "source": module["source"],
                "sha256": module["sha256"],
                "symbols": [
                    {k: symbol[k] for k in ("name", "line_start", "line_end", "sha256")} for symbol in module["symbols"]
                ],
            }
            for module in modules
        ],
    }
    if analytics:
        manifest["analytics"] = analytics
    outputs["source-manifest.json"] = json_text(manifest)
    outputs[OWNER_FILE] = json_text({"owner": OWNER, "files": sorted([*outputs, OWNER_FILE])})
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".research-site-", dir=output.parent) as temporary:
        stage = Path(temporary) / "site"
        stage.mkdir()
        for name, value in outputs.items():
            path = stage / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value, encoding="utf-8")
        result = validate_site(stage)
        validate_output_destination(output, root)
        backup = Path(temporary) / "previous"
        if output.exists():
            os.replace(output, backup)
        try:
            os.replace(stage, output)
        except OSError:
            if backup.exists():
                os.replace(backup, output)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    return result


def check_site(output: Path, expected_manifest_sha256: str | None = None) -> dict[str, int]:
    """Check the site, optionally binding it to the exact manifest reviewed earlier."""
    result = validate_site(output)
    if expected_manifest_sha256 is not None:
        if not isinstance(expected_manifest_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256):
            raise SiteError("Expected manifest SHA-256 must contain exactly 64 lowercase hexadecimal characters")
        actual = digest((output / "source-manifest.json").read_bytes())
        if actual != expected_manifest_sha256:
            raise SiteError("Source manifest SHA-256 does not match the reviewed site")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "check"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=Path("artifacts/delivery/research_site"))
    parser.add_argument(
        "--expected-manifest-sha256",
        help="For check only: require the exact source-manifest.json SHA-256 reviewed earlier",
    )
    args = parser.parse_args()
    if args.command != "check" and args.expected_manifest_sha256 is not None:
        parser.error("--expected-manifest-sha256 is only valid with check")
    root = args.repo_root.absolute()
    output = args.output if args.output.is_absolute() else root / args.output
    try:
        if args.command == "build":
            result = build_site(root, output)
        else:
            validate_output_destination(output, root)
            result = check_site(output, args.expected_manifest_sha256)
    except (SiteError, OSError, UnicodeError) as exc:
        print(f"Research site {args.command} failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"Research site {args.command}: {result['pages']} pages, "
        f"{result['files']} files, {result['links']} checked links."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
