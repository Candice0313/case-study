"""
Chunk by steps, Q&A, symptoms. One chunk = one semantic unit where possible.
Chunk metadata includes symptom_tag + section for RAG filtering (triage → retrieve only matching guides).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from html_cleaner import Section, extract_sections


def url_to_content_type(url: str) -> str | None:
    """
    Infer content_type from URL for RAG filtering: repair_guide vs part_catalog.
    When guides_only is used (troubleshoot without model), retrieval excludes part_catalog chunks.
    """
    if not url:
        return None
    path = url.split("?")[0].strip("/").lower()
    # Part catalog: /Part-*, /Parts/*, -parts (segment), category pages (ice-maker-parts, ice-makers, valves, switches)
    if "/part-" in path or "/parts/" in path or path.startswith("parts/"):
        return "part_catalog"
    if re.search(r"/ps\d+", path) or re.search(r"/part[s]?/\w+", path):
        return "part_catalog"
    if "-parts" in path or "ice-maker-parts" in path or "/ice-makers" in path or path.endswith("ice-makers"):
        return "part_catalog"
    # Repair/how-to: repair, guide, how-to, or symptom-style slugs (e.g. refrigerator-not-dispensing-water)
    if "repair" in path or "guide" in path or "how-to" in path or "troubleshoot" in path:
        return "repair_guide"
    # Known symptom slugs from url_to_symptom_tags → treat as repair guide
    symptom_slugs = (
        "freezer-too-cold", "refrigerator-too-warm", "too-warm", "running-too-long",
        "will-not-start", "not-dispensing-water", "leaking", "noisy", "light-not-working",
        "door-sweating", "not-draining", "not-filling", "not-cleaning", "not-dispensing",
        "not-drying",
    )
    if any(s in path for s in symptom_slugs):
        return "repair_guide"
    return None  # unknown: not filtered by content_type in retrieval


# URL path segment → (symptom_tag, section, appliance_type) for RAG filter. Must match app triage + source_policy.
def url_to_symptom_tags(url: str) -> dict[str, str]:
    """Infer symptom_tag, section, appliance_type from PartSelect-style URL or local filename. Merge into chunk metadata."""
    if not url:
        return {}
    path = url.split("?")[0].strip("/")
    path_lower = path.lower()
    # Normalize for local filenames: "Dishwasher Not Drying.html" → "dishwasher-not-drying"
    path_normalized = path_lower.replace(" ", "-").replace(".html", "").replace(".htm", "")
    segments = [s.lower().replace(" ", "-") for s in path.split("/")]
    last = segments[-1] if segments else ""
    last_normalized = last.replace(".html", "").replace(".htm", "")
    # appliance_type: retrieval filters by this (refrigerator | dishwasher)
    if "dishwasher" in path_lower:
        appliance_type = "dishwasher"
    elif "refrigerator" in path_lower or "fridge" in path_lower:
        appliance_type = "refrigerator"
    else:
        appliance_type = "unknown"
    # Refrigerator + dishwasher URL slug or filename → (symptom_tag, section)
    tag_map = {
        "freezer-too-cold": ("too_cold", "freezer"),
        "refrigerator-too-warm": ("too_warm", "refrigerator"),
        "too-warm": ("too_warm", "refrigerator"),
        "running-too-long": ("not_cooling", "both"),
        "will-not-start": ("not_cooling", "both"),
        "not-dispensing-water": ("ice_maker_issue", "refrigerator"),
        "leaking": ("leaking", "both"),
        "noisy": ("noise", "both"),
        "light-not-working": ("other", "refrigerator"),
        "door-sweating": ("other", "refrigerator"),
        "not-draining": ("not_draining", "both"),
        "not-filling": ("not_filling", "both"),
        "not-cleaning": ("not_cleaning", "both"),
        "not-dispensing": ("not_dispensing", "both"),
        "not-drying": ("not_drying", "both"),
    }
    symptom_tag, section = tag_map.get(last, ("other", "both"))
    if symptom_tag == "other" and last_normalized:
        symptom_tag, section = tag_map.get(last_normalized, ("other", "both"))
    if symptom_tag == "other" and path_normalized:
        for slug, (tag, sec) in tag_map.items():
            if slug in path_normalized:
                symptom_tag, section = tag, sec
                break
    if appliance_type == "dishwasher" and ("will-not-start" in last_normalized or "will-not-start" in path_normalized):
        symptom_tag, section = "will_not_start", "both"
    return {"symptom_tag": symptom_tag, "section": section, "appliance_type": appliance_type}


@dataclass
class Chunk:
    text: str
    metadata: dict[str, Any]


# Approx 4 chars per token; keep chunks within embedding window
MAX_CHARS = 3000
MIN_CHARS = 80


def _split_long_text(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Split by paragraphs first, then by sentences, so we don't cut mid-sentence."""
    if len(text) <= max_chars:
        return [text] if text.strip() else []
    parts: list[str] = []
    remaining = text
    while remaining:
        remaining = remaining.strip()
        if len(remaining) <= max_chars:
            if remaining:
                parts.append(remaining)
            break
        # Try split at paragraph
        chunk = remaining[: max_chars + 1]
        last_para = chunk.rfind("\n\n")
        if last_para > max_chars // 2:
            head, remaining = remaining[: last_para + 1].strip(), remaining[last_para + 1 :]
        else:
            last_period = chunk.rfind(". ")
            if last_period > max_chars // 2:
                head, remaining = remaining[: last_period + 1], remaining[last_period + 1 :]
            else:
                head, remaining = remaining[:max_chars], remaining[max_chars:]
        if head:
            parts.append(head)
    return parts


def sections_to_chunks(sections: list[Section], source: str, url: str) -> list[Chunk]:
    """
    Turn sections into chunks. Metadata includes section_type, title, source, url,
    and symptom_tag/section/appliance for RAG filtering.
    """
    chunks: list[Chunk] = []
    tags = url_to_symptom_tags(url)
    content_type = url_to_content_type(url)
    if content_type:
        tags = {**tags, "content_type": content_type}
    for sec in sections:
        if not sec.body or len(sec.body) < MIN_CHARS:
            continue
        meta = {
            "section_type": sec.section_type,
            "title": sec.title,
            "source": source,
            "url": url,
            **tags,
        }
        if len(sec.body) <= MAX_CHARS:
            chunks.append(Chunk(text=sec.body, metadata=meta))
        else:
            for i, part in enumerate(_split_long_text(sec.body)):
                if len(part) < MIN_CHARS:
                    continue
                part_meta = {**meta, "part_index": i}
                chunks.append(Chunk(text=part, metadata=part_meta))
    return chunks


def html_to_chunks(html: str, source: str, url: str, base_title: str = "") -> list[Chunk]:
    """
    Full pipeline: HTML → sections (step/qa/symptom/general) → chunks with metadata.
    """
    sections = extract_sections(html, base_title=base_title)
    if not sections:
        from html_cleaner import html_to_structured_text
        flat = html_to_structured_text(html, base_title=base_title)
        return _fallback_chunks(flat, source, url, base_title)
    return sections_to_chunks(sections, source=source, url=url)


def _fallback_chunks(flat: str, source: str, url: str, base_title: str) -> list[Chunk]:
    tags = url_to_symptom_tags(url)
    content_type = url_to_content_type(url)
    if content_type:
        tags = {**tags, "content_type": content_type}
    out: list[Chunk] = []
    for part in _split_long_text(flat):
        if len(part) >= MIN_CHARS:
            out.append(
                Chunk(
                    text=part,
                    metadata={
                        "section_type": "general",
                        "title": base_title or "Content",
                        "source": source,
                        "url": url,
                        **tags,
                    },
                )
            )
    return out


def plain_text_to_chunks(plain_text: str, source: str, url: str, base_title: str = "") -> list[Chunk]:
    """
    Chunk plain text (e.g. from PDF manuals). Splits by section-like headers
    (TROUBLESHOOTING, Step 1, 1. Introduction) then by length.
    """
    text = plain_text.strip()
    if not text or len(text) < MIN_CHARS:
        return []
    # Section headers: "Step N", "TROUBLESHOOTING", "1. Title", etc.
    section_pattern = re.compile(
        r"^(?:(?:Step\s+\d+[.:]?|TROUBLESHOOTING|INSTALLATION|SAFETY|WARRANTY|\d+[.)]\s+[A-Z]).*)$",
        re.IGNORECASE | re.MULTILINE,
    )
    matches = list(section_pattern.finditer(text))
    parts: list[tuple[str, str]] = []  # (title, body)
    if not matches:
        return _fallback_chunks(text, source, url, base_title or source)
    for i, m in enumerate(matches):
        title = m.group(0).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body and len(body) >= MIN_CHARS:
            parts.append((title, body))
    # Content before first header
    if matches and matches[0].start() > 0:
        head = text[: matches[0].start()].strip()
        if head and len(head) >= MIN_CHARS:
            parts.insert(0, (base_title or "Content", head))
    if not parts:
        return _fallback_chunks(text, source, url, base_title or source)
    tags = url_to_symptom_tags(url)
    content_type = url_to_content_type(url)
    if content_type:
        tags = {**tags, "content_type": content_type}
    out: list[Chunk] = []
    for title, body in parts:
        meta = {"section_type": "general", "title": title, "source": source, "url": url, **tags}
        if len(body) <= MAX_CHARS:
            out.append(Chunk(text=body, metadata=meta))
        else:
            for i, part in enumerate(_split_long_text(body)):
                if len(part) >= MIN_CHARS:
                    out.append(Chunk(text=part, metadata={**meta, "part_index": i}))
    return out
