"""
Clean HTML → structured text with section detection.
Optimized for PartSelect product pages and brand troubleshooting guides.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from bs4 import BeautifulSoup, Tag


@dataclass
class Section:
    """A logical section of content (heading + body)."""
    title: str
    body: str
    section_type: str  # "step" | "qa" | "symptom" | "general"
    level: int  # heading level (1=h1, 2=h2, ...)


# Selectors for main content (skip nav, footer, ads)
MAIN_SELECTORS = [
    "article",
    "main",
    "[role='main']",
    ".content",
    ".main-content",
    "#content",
    ".product-details",
    ".troubleshooting",
    ".repair-guide",
    "body",
]


def _get_text(el: Tag) -> str:
    """Extract visible text from element, normalized."""
    if not el:
        return ""
    text = el.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def _is_step_heading(text: str) -> bool:
    return bool(
        re.search(r"step\s*\d+|^\d+\.\s", text, re.IGNORECASE)
        or "installation" in text.lower()
        or "how to" in text.lower()
    )


def _is_qa_heading(text: str) -> bool:
    return bool(
        re.search(r"^(q\s*&\s*a|faq|question|answer)", text, re.IGNORECASE)
        or "common questions" in text.lower()
    )


def _is_symptom_heading(text: str) -> bool:
    t = text.lower()
    return any(
        x in t
        for x in (
            "symptom",
            "troubleshoot",
            "problem",
            "not working",
            "not cooling",
            "leak",
            "noise",
            "error code",
        )
    )


def _classify_section_type(title: str, body: str) -> str:
    combined = f"{title} {body}"[:500].lower()
    if _is_step_heading(title) or "step 1" in combined or "first, " in combined:
        return "step"
    if _is_qa_heading(title) or "? " in body or " q: " in combined:
        return "qa"
    if _is_symptom_heading(title) or _is_symptom_heading(body):
        return "symptom"
    return "general"


def _find_main(soup: BeautifulSoup) -> Tag | None:
    for sel in MAIN_SELECTORS:
        if sel == "body":
            return soup.find("body")
        el = soup.select_one(sel)
        if el and len(_get_text(el)) > 100:
            return el
    return soup.find("body")


def extract_sections(html: str, base_title: str = "") -> list[Section]:
    """
    Parse HTML and return a list of sections with type hints (step/qa/symptom/general).
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove noise
    for tag in soup.find_all(["script", "style", "nav", "footer", "aside", "form"]):
        tag.decompose()

    main = _find_main(soup)
    if not main:
        return []

    sections: list[Section] = []
    current_title = base_title
    current_body_parts: list[str] = []
    current_level = 0

    def flush() -> None:
        if not current_body_parts:
            return
        body = " ".join(current_body_parts).strip()
        if len(body) < 20:
            return
        section_type = _classify_section_type(current_title, body)
        sections.append(
            Section(
                title=current_title or "Content",
                body=body,
                section_type=section_type,
                level=current_level or 1,
            )
        )

    # Walk all tags in order; on heading, flush current and start new section
    for tag in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "div"]):
        if tag.name in ("h1", "h2", "h3", "h4"):
            flush()
            current_title = _get_text(tag)
            current_level = int(tag.name[1])
            current_body_parts = []
        else:
            text = _get_text(tag)
            if text:
                current_body_parts.append(text)

    flush()
    return sections


def html_to_structured_text(html: str, base_title: str = "") -> str:
    """
    Simple flatten: clean HTML to one structured text block (no section split).
    Use for documents where section detection isn't needed.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["script", "style", "nav", "footer", "aside"]):
        tag.decompose()
    main = _find_main(soup) or soup
    return _get_text(main)


def load_html_path(path: Path) -> str:
    """Read HTML file from disk."""
    return path.read_text(encoding="utf-8", errors="replace")


# Browser "Save as" comment: <!-- saved from url=(0064)https://www.partselect.com/... -->
_SAVED_FROM_URL_RE = re.compile(
    r"<!--\s*saved\s+from\s+url=\(\d+\)\s*(https?://[^\s)]+)\s*-->",
    re.IGNORECASE,
)


def extract_saved_from_url(html: str) -> str | None:
    """If the HTML was saved from a URL (e.g. browser Save Page), return that URL."""
    m = _SAVED_FROM_URL_RE.search(html[:2000])
    return m.group(1).strip() if m else None


def iter_html_sources(sources_dir: Path) -> Iterator[tuple[str, str, str]]:
    """
    Yield (source_id, url, html_content) for each .html file in sources_dir.
    url = canonical URL when present (e.g. from "saved from url=..."), else relative path.
    """
    if not sources_dir.is_dir():
        return
    for f in sorted(sources_dir.glob("**/*.html")):
        try:
            html = load_html_path(f)
            if len(html.strip()) < 50:
                continue
            canonical = extract_saved_from_url(html)
            rel = f.relative_to(sources_dir)
            url = canonical if canonical else str(rel)
            source_id = f.stem
            yield source_id, url, html
        except Exception:
            continue


def iter_text_sources(sources_dir: Path) -> Iterator[tuple[str, str, str]]:
    """
    Yield (source_id, url_or_path, plain_text) for each .txt file in sources_dir.
    Used for PDF-extracted text and other plain-text manuals.
    """
    if not sources_dir.is_dir():
        return
    for f in sorted(sources_dir.glob("**/*.txt")):
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
            if len(text.strip()) < 50:
                continue
            rel = f.relative_to(sources_dir)
            url = str(rel)
            source_id = f.stem
            yield source_id, url, text
        except Exception:
            continue
