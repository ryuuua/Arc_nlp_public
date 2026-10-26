"""Explicit, import-free inputs for the research documentation build."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any


class SiteError(ValueError):
    """An input or publication contract was not satisfied."""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    """Resolve a repository-relative input without following symlinks."""
    path = PurePosixPath(relative)
    if not relative or path.is_absolute() or ".." in path.parts or "\\" in relative:
        raise SiteError(f"Unsafe relative path: {relative!r}")
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise SiteError(f"Symlink input is forbidden: {relative}")
    if not current.is_file():
        raise SiteError(f"Missing input: {relative}")
    return current


def _keys(value: dict[str, Any], expected: set[str], where: str, optional: set[str] | None = None) -> None:
    if not isinstance(value, dict) or not expected <= set(value) <= expected | (optional or set()):
        raise SiteError(f"Unexpected or missing fields in {where}; expected {sorted(expected)}")


def _text(value: Any, where: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise SiteError(f"Expected nonempty text: {where}")


def load_config(root: Path) -> dict[str, Any]:
    raw = safe_path(root, "docs/site/site.json").read_text(encoding="utf-8")
    try:
        config = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SiteError("Invalid site.json") from exc
    _keys(config, {"schema_version", "title", "tagline", "repository_url", "pages", "api"}, "site.json", {"assets"})
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise SiteError("Unsupported site schema")
    for field in ("title", "tagline", "repository_url"):
        _text(config[field], field)
    if not re.fullmatch(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", config["repository_url"]):
        raise SiteError("repository_url must be a GitHub repository URL")
    if not isinstance(config["pages"], list) or not config["pages"]:
        raise SiteError("pages must be an explicit nonempty list")
    if not isinstance(config["api"], list):
        raise SiteError("api must be an explicit list")
    assets = config.get("assets", [])
    if not isinstance(assets, list):
        raise SiteError("assets must be an explicit list")
    asset_names = {"site.css", "site.js", "code.css", "search.json"}
    for asset in assets:
        if not isinstance(asset, str) or not re.fullmatch(
            r"(?:[A-Za-z0-9][A-Za-z0-9_.-]*/)*[A-Za-z0-9][A-Za-z0-9_.-]*\.(?:css|svg|json|txt)", asset
        ):
            raise SiteError(f"Invalid publication asset: {asset!r}")
        # Case-insensitive uniqueness keeps the allowlist portable to macOS.
        if asset.casefold() in asset_names:
            raise SiteError(f"Duplicate or reserved publication asset: {asset}")
        asset_names.add(asset.casefold())
    seen: set[str] = set()
    for page in config["pages"]:
        _keys(page, {"slug", "title", "group", "description", "lang"}, "page", {"layout", "toc"})
        for field in ("slug", "title", "group", "description", "lang"):
            _text(page[field], f"page.{field}")
        slug = page["slug"]
        if not re.fullmatch(r"(?:[a-z][a-z0-9-]*/)*[a-z][a-z0-9-]*", slug):
            raise SiteError(f"Invalid page slug: {slug}")
        if slug.startswith(("api/", "assets/")) or slug in seen:
            raise SiteError(f"Duplicate or reserved page slug: {slug}")
        if page["lang"] not in {"en", "ja"}:
            raise SiteError(f"Unsupported language for {slug}")
        if "layout" in page and page["layout"] != "conference":
            raise SiteError(f"Unsupported page layout for {slug}")
        if "toc" in page:
            if not isinstance(page["toc"], list):
                raise SiteError(f"Page toc must be an explicit list: {slug}")
            anchors: set[str] = set()
            for heading in page["toc"]:
                _keys(heading, {"id", "title", "level"}, f"page toc for {slug}")
                _text(heading["title"], f"page toc title for {slug}")
                anchor = heading["id"]
                if not isinstance(anchor, str) or not re.fullmatch(r"[\w-]+", anchor) or anchor in anchors:
                    raise SiteError(f"Invalid or duplicate toc anchor for {slug}")
                if type(heading["level"]) is not int or heading["level"] not in (2, 3):
                    raise SiteError(f"Invalid toc level for {slug}")
                anchors.add(anchor)
        seen.add(slug)
    if "index" not in seen:
        raise SiteError("An index page is required")
    api_seen: set[str] = set()
    for module in config["api"]:
        _keys(
            module,
            {"slug", "module", "source", "title", "summary", "stability", "symbols"},
            "API module",
            {"explanations"},
        )
        for field in ("slug", "module", "source", "title", "summary", "stability"):
            _text(module[field], f"api.{field}")
        slug = module["slug"]
        if not re.fullmatch(r"[a-z][a-z0-9_]*", slug) or slug == "index" or slug in api_seen:
            raise SiteError(f"Duplicate or invalid API slug: {slug}")
        api_seen.add(slug)
        if module["module"] != f"neureps_arc_nlp.{slug}" or module["source"] != f"src/neureps_arc_nlp/{slug}.py":
            raise SiteError(f"API source/module must match its explicit slug: {slug}")
        if module["stability"] not in {"public", "low-level"}:
            raise SiteError(f"Invalid API category: {slug}")
        symbols = module["symbols"]
        if not isinstance(symbols, list) or not symbols:
            raise SiteError(f"Expected explicit symbols: {slug}")
        if any(not isinstance(s, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", s) for s in symbols):
            raise SiteError(f"Invalid API symbol: {slug}")
        if len(set(symbols)) != len(symbols):
            raise SiteError(f"Expected unique explicit symbols: {slug}")
        explanations = module.get("explanations", {})
        if not isinstance(explanations, dict) or not set(explanations) <= set(symbols):
            raise SiteError(f"API explanations must map selected symbols to plain text: {slug}")
        for symbol, explanation in explanations.items():
            _text(explanation, f"api.{slug}.explanations.{symbol}")
    return config


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> str:
    if isinstance(node, ast.ClassDef):
        bases = [ast.unparse(base) for base in node.bases]
        bases += [ast.unparse(keyword) for keyword in node.keywords]
        return f"class {node.name}" + (f"({', '.join(bases)})" if bases else "")
    result = f"{'async ' if isinstance(node, ast.AsyncFunctionDef) else ''}{node.name}({ast.unparse(node.args)})"
    if node.returns is not None:
        result += f" → {ast.unparse(node.returns)}"
    return result


def extract_symbols(root: Path, module: dict[str, Any]) -> dict[str, Any]:
    data = safe_path(root, module["source"]).read_bytes()
    try:
        source = data.decode("utf-8")
        tree = ast.parse(source, filename=module["source"])
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise SiteError(f"Cannot parse API source: {module['source']}") from exc
    candidates = {
        node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    symbols = []
    lines = source.splitlines()
    for name in module["symbols"]:
        if name not in candidates:
            raise SiteError(f"Missing allowlisted API symbol: {module['module']}.{name}")
        node = candidates[name]
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        code = "\n".join(lines[start - 1 : node.end_lineno]) + "\n"
        fields = []
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if (
                    isinstance(child, ast.AnnAssign)
                    and isinstance(child.target, ast.Name)
                    and not child.target.id.startswith("_")
                ):
                    fields.append(
                        {
                            "name": child.target.id,
                            "annotation": ast.unparse(child.annotation),
                            "default": ast.unparse(child.value) if child.value is not None else None,
                        }
                    )
        symbols.append(
            {
                "name": name,
                "kind": "class" if isinstance(node, ast.ClassDef) else "function",
                "signature": _signature(node),
                "docstring": ast.get_docstring(node) or "",
                "explanation": module.get("explanations", {}).get(name, ""),
                "fields": fields,
                "source": code,
                "line_start": start,
                "line_end": node.end_lineno,
                "sha256": digest(code.encode("utf-8")),
            }
        )
    return {**module, "symbols": symbols, "sha256": digest(data)}
