#!/usr/bin/env python3
"""Convert ethics .docx files to styled HTML for LebNet TechFellows (python-docx)."""

from __future__ import annotations

import html
from pathlib import Path

from docx import Document

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{page_title} - LebNet</title>
<link rel="stylesheet" href="../css/style.css">
<script>if(localStorage.getItem('lebnet-theme')==='light')document.documentElement.classList.add('theme-light');</script>
</head>
<body>

<nav class="navbar">
  <a href="../index.html" class="navbar-brand">LebNet TechFellows</a>
  <ul class="navbar-links">
    <li><a href="../index.html" class="active">Notes</a></li>
    <li><a href="#">Notebooks</a></li>
    <li><a href="#">Ethics</a></li>
  </ul>
  <div class="navbar-right">
    <button class="theme-toggle" title="Toggle theme" aria-label="Toggle theme">
      <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" stroke-width="1.5"/><path d="M12 2 A10 10 0 0 0 12 22 Z" style="fill:var(--bg)"/><path d="M12 2 A10 10 0 0 1 12 22 Z" style="fill:var(--fg)"/></svg>
    </button>
  </div>
</nav>

<button class="sidebar-toggle" aria-label="Toggle sidebar">&#9776;</button>
<aside class="sidebar" id="sidebar">
  <div class="sidebar-title">On this page</div>
  <ul class="sidebar-nav"></ul>
</aside>

<div class="paper">
{content}
<footer class="site-footer">LebNet TechFellows</footer>
</div>

<script src="../js/theme.js"></script>
<script src="../js/sidebar.js"></script>
</body>
</html>
"""


def _style_name(paragraph) -> str:
    if paragraph.style and paragraph.style.name:
        return paragraph.style.name.strip()
    return "Normal"


def _is_heading(paragraph, level: int) -> bool:
    name = _style_name(paragraph).lower()
    return name == f"heading {level}"


def _body_after_qa_colon(text: str) -> str:
    """Strip leading 'Q:' / 'A:' (case-insensitive) and following whitespace."""
    t = text.strip()
    if len(t) >= 2 and t[1] == ":" and t[0].lower() in ("q", "a"):
        return t[2:].lstrip()
    return t


def docx_to_body_html(doc: Document) -> str:
    parts: list[str] = []
    seen_h1 = False

    for p in doc.paragraphs:
        text = p.text.strip()
        style = _style_name(p)

        if _is_heading(p, 1):
            seen_h1 = True
            parts.append(f"<h2>{html.escape(text)}</h2>\n")
            continue

        if not seen_h1:
            continue

        if _is_heading(p, 2):
            if not text:
                continue
            parts.append(f"<h3>{html.escape(text)}</h3>\n")
            continue

        if style.lower() != "normal":
            continue

        if not text:
            continue

        if len(text) >= 2 and text[1] == ":" and text[0].lower() == "q":
            inner = html.escape(_body_after_qa_colon(text))
            parts.append(
                '<div class="def"><span class="label">Question</span>'
                f"<p>{inner}</p></div>\n"
            )
            continue

        if len(text) >= 2 and text[1] == ":" and text[0].lower() == "a":
            inner = html.escape(_body_after_qa_colon(text))
            parts.append(
                '<div class="rem"><span class="label">Answer</span>'
                f"<p>{inner}</p></div>\n"
            )
            continue

    return "".join(parts)


def convert_one(docx_path: Path, out_path: Path, topic: str) -> None:
    doc = Document(str(docx_path))
    page_title = f"Ethics: {topic}"
    content = f'<h1>{html.escape(page_title)}</h1>\n'
    content += (
        '<p class="subtitle">LebNet Tech Fellows &mdash; '
        "Ethics &amp; Statistical Pitfalls</p>\n\n"
    )
    content += docx_to_body_html(doc)
    html_out = TEMPLATE.format(
        page_title=html.escape(page_title, quote=True), content=content
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_out, encoding="utf-8")
    print(f"Wrote {out_path}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    jobs = [
        (
            root / "ethics_docs" / "ethics_linear_regression.docx",
            root / "website" / "web" / "notes" / "ethics-linear-regression.html",
            "Linear Regression",
        ),
        (
            root / "ethics_docs" / "ethics_logistic_regression.docx",
            root / "website" / "web" / "notes" / "ethics-logistic-regression.html",
            "Logistic Regression",
        ),
        (
            root / "ethics_docs" / "ethics_neural_networks.docx",
            root / "website" / "web" / "notes" / "ethics-neural-networks.html",
            "Neural Networks",
        ),
    ]
    for src, dst, topic in jobs:
        if not src.is_file():
            raise SystemExit(f"Missing source: {src}")
        convert_one(src, dst, topic)


if __name__ == "__main__":
    main()
