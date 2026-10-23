"""Validate the generated publication boundary, assets, and local links."""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

from site_sources import SiteError, digest

OWNER_FILE = ".research-site-owned.json"
OWNER = "arc_nlp.research_site.v1"

_PRIVATE_PATTERNS = (
    (r"/(?:Users|home|root|mnt|work|workspace|scratch)/[^\s<>\"']+", "private filesystem path"),
    (r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", "private key"),
    (
        r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-proj-[A-Za-z0-9_-]{16,}|hf_[A-Za-z0-9]{20,})\b",
        "credential token",
    ),
    (
        r"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b",
        "private network address",
    ),
    (
        r"\b(?:api_key|access_token|password|secret_key)\s*[=:]\s*[\"'][A-Za-z0-9_+/=-]{16,}[\"']",
        "literal credential assignment",
    ),
)


def check_public_text(text: str, location: str) -> None:
    for pattern, label in _PRIVATE_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            raise SiteError(f"Publication guard found {label} in {location}")


class HTMLInventory(HTMLParser):
    def __init__(self, location: str):
        super().__init__(convert_charrefs=True)
        self.location = location
        self.ids: set[str] = set()
        self.links: list[tuple[str, str]] = []
        self.text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag in {"iframe", "object", "embed", "base", "form"}:
            raise SiteError(f"Forbidden HTML element {tag} in {self.location}")
        if any(name.lower().startswith("on") for name in attributes):
            raise SiteError(f"Inline event handler in {self.location}")
        if tag == "script" and not attributes.get("src"):
            raise SiteError(f"Inline script in {self.location}")
        if "id" in attributes:
            anchor = attributes["id"] or ""
            if not anchor or anchor in self.ids:
                raise SiteError(f"Empty or duplicate anchor {anchor!r} in {self.location}")
            self.ids.add(anchor)
        for key in ("href", "src", "poster", "action"):
            if attributes.get(key):
                self.links.append((attributes[key] or "", key))

    handle_startendtag = handle_starttag

    def handle_data(self, data: str) -> None:
        self.text.append(data)


def output_inventory(output: Path) -> list[str]:
    result = []
    for path in output.rglob("*"):
        if path.is_symlink():
            raise SiteError("Generated output contains a symlink")
        if path.is_file():
            result.append(path.relative_to(output).as_posix())
    return sorted(result)


def validate_site(output: Path) -> dict[str, int]:
    files = output_inventory(output)
    if "index.html" not in files or "source-manifest.json" not in files:
        raise SiteError("Not a complete research site")
    try:
        manifest = json.loads((output / "source-manifest.json").read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise SiteError("Invalid source manifest") from exc
    expected = set(manifest.get("outputs", {})) | {"source-manifest.json", OWNER_FILE}
    if set(files) != expected:
        raise SiteError("Generated inventory differs from source manifest")
    for name, sha256 in manifest["outputs"].items():
        if digest((output / name).read_bytes()) != sha256:
            raise SiteError(f"Generated output has changed: {name}")
    inventories: dict[str, HTMLInventory] = {}
    for name in files:
        path = output / name
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise SiteError(f"Unexpected binary publication asset: {name}") from exc
        check_public_text(text, name)
        if name.endswith(".html"):
            parsed = HTMLInventory(name)
            parsed.feed(text)
            check_public_text("".join(parsed.text), name)
            inventories[name] = parsed
    links = 0
    for name, parsed in inventories.items():
        for value, attribute in parsed.links:
            links += 1
            parts = urlsplit(value)
            if parts.scheme:
                if parts.scheme not in {"https", "http", "mailto"} or attribute != "href":
                    raise SiteError(f"Unsafe or remote asset URL in {name}: {value}")
                continue
            if parts.netloc or parts.path.startswith("/") or "\\" in parts.path:
                raise SiteError(f"Root-relative or ambiguous URL in {name}: {value}")
            target_parts = list(PurePosixPath(name).parent.parts)
            for part in PurePosixPath(unquote(parts.path)).parts:
                if part == "..":
                    if not target_parts:
                        raise SiteError(f"Link escapes publication root in {name}: {value}")
                    target_parts.pop()
                elif part != ".":
                    target_parts.append(part)
            target = "/".join(target_parts) if parts.path else name
            if parts.path.endswith("/"):
                target += "/index.html"
            if target not in files:
                raise SiteError(f"Broken local link in {name}: {value}")
            if parts.fragment:
                if target not in inventories or unquote(parts.fragment) not in inventories[target].ids:
                    raise SiteError(f"Broken anchor in {name}: {value}")
    return {"pages": len(inventories), "files": len(files), "links": links}


def validate_output_destination(output: Path, root: Path) -> None:
    for current in (output, *output.parents):
        if current.is_symlink():
            raise SiteError("Output path may not contain symlinks")
    if output == root or output in root.parents:
        raise SiteError("Output may not replace the repository or its parents")
    for protected in ("src", "scripts", "tests", "docs", "conf", ".git"):
        if output == root / protected or root / protected in output.parents:
            raise SiteError("Output must be outside maintained source directories")
    if output.exists():
        if not output.is_dir():
            raise SiteError("Output must be a directory")
        files = output_inventory(output)
        if not files and not any(output.iterdir()):
            return
        try:
            owner = json.loads((output / OWNER_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SiteError("Refusing to replace an existing unowned output directory") from exc
        if owner != {"owner": OWNER, "files": files}:
            raise SiteError("Output ownership inventory changed; refusing to overwrite")
