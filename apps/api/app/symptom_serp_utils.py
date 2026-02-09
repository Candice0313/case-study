"""
Serp-based extraction of "Common parts for ... issues" list for a model's symptom page.

Used by agent when we have part_topic + model_number: try regex on Serp snippets first,
then LLM extraction on aggregated results. Keeps agent.py focused on orchestration.
"""
from __future__ import annotations

import logging
import re

from app.model_parser import parse_model_revision

logger = logging.getLogger(__name__)

# PartSelect Symptoms page slug by part_topic (for Serp query building). Keep in sync with agent._PART_TOPIC_TO_SYMPTOM_SLUG.
PART_TOPIC_TO_SYMPTOM_SLUG: dict[str, str] = {
    "ice maker": "Ice-maker-not-making-ice",
    "water filter": "Water-filter-replacement",
    "door seal": "Door-seal-gasket-replacement",
    "drain pump": "Refrigerator-not-draining",
    "thermostat": "Refrigerator-too-cold-or-not-cooling",
    "evaporator fan": "Freezer-not-cooling-evaporator-fan",
    "damper": "Refrigerator-not-cooling-damper",
    "water valve": "Water-dispenser-not-working",
    "dispenser": "Water-dispenser-not-working",
}

# Phrases that indicate a snippet fragment is a sentence, not a part name (reject such items).
_SNIPPET_SENTENCE_PHRASES = (
    "for your", "at partselect", "we have", "model diagrams", "complete guide",
    "guide to", "repair help", "videos", "official parts", "through retailers",
    "check oem", "factory-authorized", "factory authorized", "order today", "ships today",
)


def _extract_parts_list_from_snippet(snippet: str) -> list[str]:
    """
    Parse a SerpApi snippet from a PartSelect Symptoms page to extract the "Common parts for ... issues" list.
    """
    if not snippet:
        return []
    text = snippet.strip()
    lower = text.lower()
    idx = lower.find("common parts")
    if idx >= 0:
        text = text[idx:]
        lower = text.lower()
    colon_idx = text.find(":")
    if colon_idx >= 0 and colon_idx + 1 < len(text):
        text = text[colon_idx + 1 :]
    for sep in ["•", "·", " - ", " – ", " — "]:
        text = text.replace(sep, ",")
    items: list[str] = []
    for raw in text.split(","):
        name = raw.strip(" .;-–—\n\t")
        if not name or len(name) < 3:
            continue
        nl = name.lower()
        if any(bad in nl for bad in _SNIPPET_SENTENCE_PHRASES):
            continue
        if re.search(r"\b[A-Z0-9]{8,}\b", name):
            continue
        if len(name) > 50:
            continue
        items.append(name)
    seen: set[str] = set()
    uniq: list[str] = []
    for name in items:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            uniq.append(name)
    if len(uniq) < 2:
        return []
    if len(uniq) == 2 and any(u.lower() in ("oem parts", "factory parts") for u in uniq):
        return []
    return uniq[:5]


def get_parts_list_for_symptom_via_serp_sync(
    part_topic: str,
    model_number: str | None,
) -> list[str]:
    """
    Use SerpApi to extract the common-parts list for a model's symptom page.
    Tries regex on snippets first; if none valid, uses LLM on aggregated results.
    """
    from app.serp import search_serp
    from app.llm_router_planner import llm_extract_common_parts_from_serp

    part_topic = (part_topic or "").strip()
    if not part_topic:
        return []
    model_number = (model_number or "").strip() or None

    base: str | None = None
    slug: str | None = None
    if model_number:
        base, _ = parse_model_revision(model_number)
        base = (base or "").strip().upper() or None
    if part_topic:
        slug = PART_TOPIC_TO_SYMPTOM_SLUG.get(part_topic.lower()) or part_topic.title().replace(" ", "-")

    queries: list[str] = []
    if base and slug:
        queries.append(f"site:partselect.com/Models/{base}/Symptoms/{slug}/ \"Common parts\"")
        queries.append(f"site:partselect.com/Models/{base}/Symptoms {part_topic} \"Common parts\"")
    if model_number:
        queries.append(f"site:partselect.com {model_number} {part_topic} \"Common parts\"")
    queries.append(f"site:partselect.com {part_topic} \"Common parts\"")

    target_subpath: str | None = None
    if base and slug:
        target_subpath = f"/models/{base.lower()}/symptoms/{slug.lower()}/"

    all_results: list[dict] = []
    for q in queries:
        try:
            results = search_serp(q, num=6)
        except Exception as e:
            logger.debug("SerpApi failed for parts list query %s: %s", q[:60], e)
            continue
        if not results:
            continue
        all_results.extend(results)

        preferred_snippets: list[str] = []
        fallback_snippets: list[str] = []
        for r in results:
            snippet = (r.get("snippet") or "").strip()
            if not snippet:
                continue
            link = (r.get("link") or "").strip().lower()
            if not link or "partselect.com" not in link:
                continue
            if target_subpath and target_subpath in link:
                preferred_snippets.append(snippet)
            else:
                fallback_snippets.append(snippet)

        for snippet in preferred_snippets + fallback_snippets:
            names = _extract_parts_list_from_snippet(snippet)
            if names:
                logger.info("Parts list via Serp (regex) topic=%s model=%s: %s", part_topic, model_number or "", names)
                return names

    if all_results:
        try:
            names_llm = llm_extract_common_parts_from_serp(model_number or "", part_topic, all_results, max_items=5)
            if names_llm:
                logger.info("Parts list via Serp+LLM topic=%s model=%s: %s", part_topic, model_number or "", names_llm)
                return names_llm
        except Exception as e:
            logger.debug("LLM extract common parts failed: %s", e)

    return []
