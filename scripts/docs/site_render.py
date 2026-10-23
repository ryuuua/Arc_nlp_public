"""Small Markdown, syntax-highlighting, and AST-reference renderer."""

from __future__ import annotations

import html
import posixpath
import re
from typing import Any

from markdown_it import MarkdownIt
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import TextLexer, get_lexer_by_name
from pygments.util import ClassNotFound


def relative_url(current: str, target: str) -> str:
    return posixpath.relpath(target, posixpath.dirname(current) or ".")


def highlight_code(code: str, language: str = "text") -> str:
    try:
        lexer = get_lexer_by_name(language or "text")
    except ClassNotFound:
        lexer = TextLexer()
    return highlight(code, lexer, HtmlFormatter(nowrap=True))


def render_markdown(source: str) -> tuple[str, list[dict[str, Any]]]:
    parser = MarkdownIt("commonmark", {"html": True}).enable("table")
    parser.options["highlight"] = lambda code, language, attrs: highlight_code(code, language)
    tokens = parser.parse(source)
    anchors: dict[str, int] = {}
    toc = []
    for index, token in enumerate(tokens):
        if token.type == "heading_open":
            label = tokens[index + 1].content
            anchor = re.sub(r"[^\w\s-]", "", label.lower())
            anchor = re.sub(r"[\s_]+", "-", anchor).strip("-") or "section"
            duplicate = anchors.get(anchor, 0)
            anchors[anchor] = duplicate + 1
            if duplicate:
                anchor = f"{anchor}-{duplicate + 1}"
            token.attrSet("id", anchor)
            level = int(token.tag[1])
            if level in (2, 3):
                toc.append({"id": anchor, "title": label, "level": level})
        for child in token.children or []:
            if child.type == "link_open":
                href = child.attrGet("href") or ""
                if not re.match(r"^[a-z]+:", href, re.IGNORECASE):
                    child.attrSet("href", re.sub(r"\.md(?=[#?]|$)", ".html", href))
    return parser.renderer.render(tokens, parser.options, {}), toc


def search_text(rendered: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", rendered))).strip()
