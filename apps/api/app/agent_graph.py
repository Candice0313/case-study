"""
LangGraph: explicit state machine for troubleshooting (doc §4, §7).

Graph shape (simplified):
  START
    ├─[USE_LLM_ROUTER_PLANNER=1]→ llm_router → llm_planner → route_after_planner
    │                                    ├→ ask_clarify → END
    │                                    ├→ compatibility_answer → END
    │                                    ├→ part_lookup_answer → END
    │                                    ├→ parts_list_answer → END   ← only when user asks for part/model: product_cards (Suggested parts)
    │                                    └→ retrieve → normalize_compose → citation_gate → END
    └─[else]→ triage → route_after_triage
                         ├→ ask_clarify | compatibility_answer | part_lookup_answer | parts_list_answer → END
                         └→ cooling_split → ask_clarify | retrieve → …

Path semantics:
  - retrieve = embedding/RAG path (troubleshooting): get_troubleshooting() → evidence → compose answer.
    Output: answer + citations (suggested links from RAG). product_cards = [] so UI shows "Suggested links" only, not "Suggested parts".
  - parts_list_answer / part_lookup_answer = when user asks for specific part or model. Output: answer + product_cards (Suggested parts in UI).
"""
from __future__ import annotations

import asyncio
import os
from typing import Literal

from langgraph.graph import START, END, StateGraph

# Suggested parts / product_cards: cap at 5 so UI stays focused
MAX_SUGGESTED_PARTS = 5
# Suggested links (troubleshoot, no model): show more so user may find a correct guide
from app.graph_state import TroubleshootingState
from app.triage import triage, get_triage_follow_up
from app.agent_state import (
    route_path,
    get_source_policy,
    get_state_guide_link,
    state_id_from_appliance_symptom,
    compute_next_action,
)
from app.tools import get_troubleshooting, check_compatibility, part_lookup, search_parts
from app.serp import get_partselect_model_page_url
from app.evidence import (
    _brand_from_message,
    chunks_to_structured_evidence,
    drop_part_catalog_citations,
    evidence_to_answer,
    evidence_to_answer_with_claims,
    filter_chunks_to_guides_only,
)
from app import llm_router_planner
import re
from urllib.parse import quote

from app.model_parser import partselect_model_url, parse_model_revision

# Suggested parts must be refrigerator/dishwasher only (scope); drop any other appliance from Serp/DB noise
_FORBIDDEN_APPLIANCE_IN_NAME = re.compile(
    r"\b(microwave|oven|stove|washer|washing machine|dryer|lawn|garden)\b", re.IGNORECASE
)


def _filter_product_cards_by_scope(cards: list[dict]) -> list[dict]:
    """Remove cards that are clearly for out-of-scope appliances (e.g. microwave fan)."""
    if not cards:
        return cards
    out = []
    for c in cards:
        name = (c.get("name") or "").strip()
        if _FORBIDDEN_APPLIANCE_IN_NAME.search(name):
            continue
        out.append(c)
    return out


# Single source for model page URL: always /Models/{base}/ or /Models/{base}/MFGModelNumber/{rev}/, never homepage.
def _model_page_url(model_number: str) -> str:
    u = partselect_model_url((model_number or "").strip())
    if not u or u.rstrip("/") == "https://www.partselect.com":
        base, _ = parse_model_revision((model_number or "").strip())
        if base and base.strip():
            return f"https://www.partselect.com/Models/{quote(base.strip())}/"
        m = re.compile(r"[A-Z0-9][A-Z0-9\-]{4,24}", re.IGNORECASE).search(model_number or "")
        if m:
            return f"https://www.partselect.com/Models/{quote(m.group(0).strip().upper())}/"
    return u or "https://www.partselect.com/"


def _human_readable_source_title(url: str, fallback: str) -> str:
    if not url or "partselect.com" not in url.lower():
        return fallback or "Repair guide"
    path = url.rstrip("/").split("/")
    if len(path) >= 2:
        topic = path[-1].replace("-", " ").title()
        return f"PartSelect Repair Guide: {topic}"
    return fallback or "PartSelect Repair Guide"


def _citation_title(raw_title: str, url: str, fallback: str = "Guide") -> str:
    """Use link-derived title; skip generic 'About this repair' from ingested HTML."""
    r = (raw_title or "").strip()
    if r and r.lower() not in ("content", "guide", "") and not r.lower().startswith("about this repair"):
        return r
    return _human_readable_source_title(url, fallback)


def _unique_sources_from_chunks(chunks: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out = []
    for c in chunks:
        url = (c.get("source_url") or "").strip()
        raw_title = (c.get("title") or c.get("source") or "").strip()
        title = _citation_title(raw_title, url, raw_title or "Guide")
        key = (url or title)[:100]
        if key and key not in seen:
            seen.add(key)
            out.append({"url": url, "title": title})
    return out


# Symptom → keywords that should appear in a matching guide title/URL (so we prefer the right guide link)
_SYMPTOM_GUIDE_KEYWORDS: dict[str, list[str]] = {
    "ice_maker_issue": ["dispensing", "ice", "water", "not dispensing", "ice maker"],
    "leaking": ["leak"],
    "not_cooling": ["cool", "warm", "not cooling"],
    "too_warm": ["warm", "cool", "not cooling"],
    "too_cold": ["cold", "freezer", "too cold"],
    "not_draining": ["drain"],
    "not_filling": ["fill"],
    "will_not_start": ["start", "not start", "will not start"],
    "not_cleaning": ["clean"],
    "not_drying": ["dry"],
    "not_dispensing": ["dispensing", "detergent", "dispenser"],
    "noise": ["noise", "noisy"],
}


def _preferred_guide_from_state(state: dict) -> dict | None:
    """Preferred Suggested link from config/state_guide_links.json (by current_state, or by symptom→state when current_state missing)."""
    current_state = state.get("current_state") or ""
    link = get_state_guide_link(current_state) if current_state else None
    if link:
        return link
    # Fallback: derive state from symptom so we still get the right guide when current_state wasn't set
    symptom = state.get("symptom") or ""
    appliance = state.get("appliance_type") or "refrigerator"
    if symptom:
        derived_state = state_id_from_appliance_symptom(appliance, symptom)
        return get_state_guide_link(derived_state)
    return None


def _reorder_citations_by_symptom(citations: list[dict], symptom: str | None) -> list[dict]:
    """Put citations that match the user's symptom (title/url) first, so we show the relevant guide link."""
    if not symptom or not citations:
        return citations
    keywords = _SYMPTOM_GUIDE_KEYWORDS.get((symptom or "").strip().lower())
    if not keywords:
        return citations
    def score(c: dict) -> int:
        t = ((c.get("title") or "") + " " + (c.get("url") or "")).lower()
        return sum(1 for k in keywords if k in t)
    return sorted(citations, key=score, reverse=True)


# --- Nodes (receive state, return partial state update) ---

def node_triage(state: TroubleshootingState) -> TroubleshootingState:
    """S1_TRIAGE: fill diagnostic slots (TriagePlan shape). Set next_action for route_after_plan.
    Same logic as fridge: when we lack required info to answer, set need_clarify (symptom follow-up or product-info missing model/part).
    """
    message = (state.get("message") or "").strip()
    last_assistant = state.get("last_assistant_message") or None
    triage_result = triage(message, last_assistant_message=last_assistant)
    follow_up = get_triage_follow_up(triage_result)
    need_clarify = follow_up
    intent = getattr(triage_result, "intent", "troubleshoot")
    model_number = (state.get("model_number") or "").strip()
    part_number = (state.get("part_number") or "").strip()
    # Product-info intent (part_install, product_info) but missing model and part → ask for one or the other (same “missing info → clarify” as symptom follow-up)
    if not need_clarify and intent in ("part_install", "product_info") and not model_number and not part_number:
        from app.llm_router_planner import get_clarify_for_product_info
        need_clarify = get_clarify_for_product_info("part_install", ["part_number", "model_number"])
    merged = {
        **state,
        "symptom": triage_result.primary_symptom,
        "affected_section": triage_result.affected_section,
        "need_clarify": need_clarify,
        "appliance_type": getattr(triage_result, "appliance_type", "refrigerator"),
        "intent": intent,
    }
    merged["next_action"] = compute_next_action(merged, "after_triage")
    return {
        "symptom": merged["symptom"],
        "affected_section": merged["affected_section"],
        "need_clarify": merged["need_clarify"],
        "appliance_type": merged["appliance_type"],
        "intent": merged["intent"],
        "next_action": merged["next_action"],
    }


def node_cooling_split(state: TroubleshootingState) -> TroubleshootingState:
    """S3: set current_state (REF_* / DW_*). Fridge cooling-unknown → need_clarify. Set next_action for route."""
    from app.triage import TriageResult, TRIAGE_FOLLOW_UP_WARM
    triage_result = TriageResult(
        primary_symptom=state.get("symptom") or "other",
        affected_section=state.get("affected_section") or "unknown",
        need_follow_up=bool(state.get("need_clarify")),
        appliance_type=state.get("appliance_type") or "refrigerator",
    )
    current_state = route_path(triage_result)
    out = {"current_state": current_state}
    if current_state == "REF_S9_FALLBACK" and triage_result.primary_symptom in ("too_warm", "not_cooling"):
        out["need_clarify"] = TRIAGE_FOLLOW_UP_WARM
    merged = {**state, **out}
    out["next_action"] = compute_next_action(merged, "after_cooling_split")
    return out


def node_ask_clarify(state: TroubleshootingState) -> TroubleshootingState:
    """Return the clarifying question as answer and stop. Explicitly set next_action so agent can skip sources/parts."""
    return {
        "answer": state.get("need_clarify") or "Before we narrow this down, can you tell me which section is affected?",
        "next_action": "ask_clarify",
    }


# PartSelect "Find your model number" guides — summary + links by appliance
FIND_MODEL_REFRIGERATOR_URL = "https://www.partselect.com/Find-Your-Refrigerator-Model-Number/"
FIND_MODEL_DISHWASHER_URL = "https://www.partselect.com/Find-Your-Dishwasher-Model-Number/"
# Fallback "Suggested link" when troubleshoot (no model) and no guide citation survived filtering
FALLBACK_HOW_TO_FIX_URL = "https://www.partselect.com/Repair-Guide/Refrigerator-Not-Dispensing-Water/"
FALLBACK_HOW_TO_FIX_TITLE = "How to fix: Refrigerator not dispensing water / ice maker"

# Single curated list for "Suggested links" when user describes a problem with no model number.
# We will filter by appliance type and cap at 3 links in get_suggested_links_troubleshoot.
SUGGESTED_LINKS_TROUBLESHOOT = [
    {"url": "https://www.partselect.com/Repair/Refrigerator/Not-Making-Ice/", "title": "Refrigerator Not Making Ice"},
    {"url": "https://www.partselect.com/Repair-Guide/Refrigerator-Not-Dispensing-Water/", "title": "Refrigerator Not Dispensing Water"},
    {"url": "https://www.partselect.com/Repair/Refrigerator/Not-Cooling/", "title": "Refrigerator Not Cooling"},
    {"url": "https://www.partselect.com/Repair/Refrigerator/Leaking-Water/", "title": "Refrigerator Leaking Water"},
    {"url": "https://www.partselect.com/Repair/Dishwasher/Not-Draining/", "title": "Dishwasher Not Draining"},
]
FIND_MODEL_SUMMARY = (
    "Your model number is usually on a label or sticker—often inside the door, on the back, or near the crisper. "
    "PartSelect has step-by-step guides with photos to help you locate it."
)


def node_find_model_help(state: TroubleshootingState) -> TroubleshootingState:
    """Answer 'where can I find my model number': summary only in body; links only in Sources (citations). No product cards."""
    message = (state.get("message") or "").lower()
    mention_fridge = "refrigerator" in message or "fridge" in message or "freezer" in message
    mention_dishwasher = "dishwasher" in message or "dish washer" in message
    # Only narrow to one appliance when user explicitly said which; otherwise show both so Sources has 2 links
    if mention_dishwasher and not mention_fridge:
        citations = [{"url": FIND_MODEL_DISHWASHER_URL, "title": "Find your dishwasher model number"}]
    elif mention_fridge and not mention_dishwasher:
        citations = [{"url": FIND_MODEL_REFRIGERATOR_URL, "title": "Find your refrigerator model number"}]
    else:
        citations = [
            {"url": FIND_MODEL_REFRIGERATOR_URL, "title": "Find your refrigerator model number"},
            {"url": FIND_MODEL_DISHWASHER_URL, "title": "Find your dishwasher model number"},
        ]
    return {"answer": FIND_MODEL_SUMMARY, "citations": citations, "product_cards": []}


def get_suggested_links_troubleshoot(result: TroubleshootingState) -> list[dict]:
    """
    Return up to 3 curated PartSelect repair guides based on the appliance type.
    - Refrigerator issues: only refrigerator guides.
    - Dishwasher issues: only dishwasher guides.
    - Unknown/mixed: keep original ordering but still cap at 3.
    """
    links = list(SUGGESTED_LINKS_TROUBLESHOOT)
    appliance = (result.get("appliance_type") or "").strip().lower()

    if appliance == "refrigerator":
        links = [l for l in links if "/Dishwasher/" not in (l.get("url") or "")]
    elif appliance == "dishwasher":
        links = [l for l in links if "/Refrigerator/" not in (l.get("url") or "")]

    return links[:3]


# Words that are not real part numbers (e.g. from "is this part compatible")
_FAKE_PART_NUMBERS = {"COMPATIBLE", "COMPATIBILITY", "PART", "PARTS", "REPLACE", "INSTALL", "INSTALLATION", "MODEL"}


async def node_compatibility_answer(state: TroubleshootingState) -> TroubleshootingState:
    """Compatibility: explicit Yes/No/Not sure, then model page link. Requires both model and part. Uses DB for yes/no. No product cards."""
    model_number = (state.get("model_number") or "").strip()
    part_number = (state.get("part_number") or "").strip()
    if part_number and part_number.upper() in _FAKE_PART_NUMBERS:
        part_number = ""
    if not model_number or not part_number:
        return {
            "answer": "To check compatibility I need both your appliance model number and the part number. Please share them (e.g. model WDT780SAEM1 and part PS11752778).",
            "citations": [],
        }
    # Model link: canonical URL only (e.g. https://www.partselect.com/Models/003719074/)
    model_url = await asyncio.to_thread(
        get_partselect_model_page_url, model_number, state.get("appliance_type")
    )
    model_link = f"[View model {model_number} on PartSelect]({model_url})"
    result = await check_compatibility(model_number, part_number)
    if result.get("compatible"):
        return {
            "answer": f"**Yes.** This part fits your model. {model_link}",
            "citations": [],
        }
    serp_links = result.get("serp_links") or []
    if serp_links:
        link_lines = [f"[{l.get('title') or 'PartSelect'}]({l.get('link') or ''})" for l in serp_links if l.get("link")]
        if link_lines:
            return {
                "answer": f"**No / Not sure.** We don't have a fitment record for this part and model. Check on PartSelect: {model_link} (or: " + ", ".join(link_lines[:2]) + ").",
                "citations": [],
            }
    return {
        "answer": f"**No / Not sure.** We don't have a fitment record for part {part_number} and model {model_number}. Check on PartSelect: {model_link}.",
        "citations": [],
    }


def _parts_list_serp_query(model_number: str, info_type: str | None) -> str:
    """Serp query by info_type. Revision in model_number helps rank the right page."""
    base = f"site:partselect.com {model_number}"
    if info_type == "model_specs":
        return f"{base} dimensions warranty specs"
    if info_type == "model_manual":
        return f"{base} manual installation guide"
    if info_type == "model_parts":
        return f"site:partselect.com Models {model_number}"
    return f"{base} parts"


def _filter_serp_by_revision(serp_results: list, revision: str | None) -> list:
    """When user asked for a specific revision, put results that match that revision (in URL/title) first."""
    if not revision or not serp_results:
        return list(serp_results)
    rev_lower = revision.lower()
    match_revision = []
    other = []
    for r in serp_results:
        link = (r.get("link") or "").lower()
        title = (r.get("title") or "").lower()
        if f"/{rev_lower}/" in link or f"mfgmodelnumber/{rev_lower}" in link or f"({rev_lower})" in title or f"({revision})" in title:
            match_revision.append(r)
        else:
            other.append(r)
    return match_revision + other


def _is_model_specific_result(link: str, model_number: str) -> bool:
    """True if the URL is the model's page (PartSelect: /Models/MODEL/ or path contains model number)."""
    if not (link or model_number):
        return False
    link_clean = (link or "").strip().lower()
    model_lower = model_number.strip().lower()
    if model_lower not in link_clean:
        return False
    return "/models/" in link_clean or f"/models/{model_lower}" in link_clean or f"partselect.com/models/" in link_clean


async def node_parts_list_answer(state: TroubleshootingState) -> TroubleshootingState:
    """
    State machine node: "parts for model X" or model specs/manual. Reads model_number, info_type.
    Flow: search_parts → if empty: direct PartSelect model page (model_parts) or Serp for specs/manual.
    """
    model_number = (state.get("model_number") or (state.get("action_args") or {}).get("model_number") or "").strip()
    if not model_number:
        return {
            "answer": "Please share your appliance model number so I can look up parts for it.",
            "citations": [],
        }
    info_type = state.get("info_type") or "model_parts"
    _, revision = parse_model_revision(model_number)
    # User gave (03) → only show (03) link and one card; do not use DB so we never surface (04). Serp + summary is ok.
    model_url = _model_page_url(model_number)
    if revision:
        one_card = [{"name": f"{model_number} - Overview", "url": model_url, "image_url": None, "part_number": None, "price": None, "brand": None}]
        summary = ""
        try:
            from app.serp import search_serp
            from app.llm_router_planner import llm_summarize_serp_snippets_for_model
            query = _parts_list_serp_query(model_number, info_type)
            serp_results = await asyncio.to_thread(search_serp, query, num=6)
            serp_results = _filter_serp_by_revision(serp_results, revision)
            if serp_results:
                summary = await asyncio.to_thread(
                    llm_summarize_serp_snippets_for_model,
                    model_number,
                    serp_results,
                    conversation_context=state.get("conversation_context"),
                )
        except Exception:
            pass
        if summary and summary.strip():
            answer = f"{summary.strip()}\n\n[View parts for {model_number}]({model_url})"
        else:
            answer = f"PartSelect has parts lists, diagrams, and repair guides for **{model_number}**.\n\n[View parts for {model_number}]({model_url})"
        return {"answer": answer, "citations": [], "product_cards": one_card}
    message = (state.get("message") or "").strip()
    symptom_for_parts = message[:100] if message else None
    parts = await search_parts(
        model_number=model_number,
        symptom=symptom_for_parts,
        limit=MAX_SUGGESTED_PARTS,
    )
    if not parts:
        one_card = [{"name": f"{model_number} - Overview", "url": model_url, "image_url": None, "part_number": None, "price": None, "brand": None}]
        from app.serp import search_serp
        from app.llm_router_planner import llm_summarize_serp_snippets_for_model
        conversation_context = state.get("conversation_context")
        query = _parts_list_serp_query(model_number, info_type)
        serp_results = await asyncio.to_thread(search_serp, query, num=8)
        serp_results = _filter_serp_by_revision(serp_results, revision)
        summary = ""
        if serp_results:
            summary = await asyncio.to_thread(
                llm_summarize_serp_snippets_for_model,
                model_number,
                serp_results,
                conversation_context=conversation_context,
            )
        if summary and summary.strip():
            answer = f"{summary.strip()}\n\n[View parts for {model_number}]({model_url})"
        else:
            answer = f"Here is the PartSelect page for **{model_number}**: [View parts for {model_number}]({model_url}). You can browse and order there."
        return {"answer": answer, "citations": [], "product_cards": one_card}
    parts_display = parts[:MAX_SUGGESTED_PARTS]
    lines = [f"Parts for **{model_number}**:"]
    for p in parts_display:
        name = p.get("name") or "Part"
        url = p.get("url")
        ps = p.get("partselect_number") or p.get("part_number")
        if url:
            lines.append(f"- [{name}]({url})" + (f" — {ps}" if ps else ""))
        else:
            lines.append(f"- {name}" + (f" ({ps})" if ps else ""))
    if info_type == "model_parts":
        lines.append("\nWhich category or symptom are you interested in? (e.g. pump, door seal, racks) I can point you to the right part.")
    return {"answer": "\n".join(lines), "citations": [], "product_cards": parts_display}


async def node_part_lookup_answer(state: TroubleshootingState) -> TroubleshootingState:
    """
    State machine node: part lookup. Reads part_number, info_type. Sets lookup_status (hit|miss|error).
    miss → SerpApi + LLM summarize. error → do not say "not in catalog"; suggest retry or PartSelect search.
    """
    part_number = (state.get("part_number") or "").strip()
    if not part_number:
        return {
            "answer": "Please share the part number (PartSelect or manufacturer number) so I can look up details and installation info.",
            "citations": [],
        }
    info_type = state.get("info_type") or "part_price"
    installation_focus = (info_type == "part_install") or ((state.get("intent") or "").strip() == "part_install")
    try:
        part = await part_lookup(part_number)
    except Exception:
        from urllib.parse import quote
        search_url = f"https://www.partselect.com/Search.aspx?SearchTerm={quote(part_number)}"
        return {
            "answer": "I couldn't look up that part right now. Please try again in a moment or search on PartSelect: [Search for " + part_number + "](" + search_url + ")",
            "citations": [],
            "lookup_status": "error",
            "error_type": "upstream",
        }
    if not part:
        from app.serp import search_serp
        from app.llm_router_planner import llm_summarize_serp_for_part
        from app.tools import _row_from_serp
        query = f"site:partselect.com {part_number} installation" if installation_focus else f"site:partselect.com {part_number}"
        serp_results = await asyncio.to_thread(search_serp, query, num=6)
        if serp_results:
            message = (state.get("message") or "").strip() or None
            summary = await asyncio.to_thread(
                llm_summarize_serp_for_part,
                part_number,
                serp_results,
                installation_focus=installation_focus,
                user_question=message,
            )
            if summary:
                product_cards = _filter_product_cards_by_scope(
                    [_row_from_serp(r) for r in serp_results[:MAX_SUGGESTED_PARTS]]
                )
                return {"answer": summary, "citations": [], "product_cards": product_cards, "lookup_status": "miss"}
        from urllib.parse import quote
        search_url = f"https://www.partselect.com/Search.aspx?SearchTerm={quote(part_number)}"
        return {
            "answer": f"I couldn't find an exact match for **{part_number}** in our catalog. Here are the closest results on PartSelect: [Search for {part_number}]({search_url})",
            "citations": [],
            "lookup_status": "miss",
        }
    # Use part record fields first; fallback to lookup key (state part_number) so we never show "PartSelect: None" when user gave a part number
    ps = (
        part.get("partselect_number")
        or part.get("part_number")
        or part.get("manufacturer_part_number")
        or part_number
    )
    name = part.get("name") or (f"Part {ps}" if ps else "Part")
    mfr = part.get("manufacturer_part_number")
    price = part.get("price")
    diff = part.get("difficulty")
    time_est = part.get("time_estimate")
    part_url = (part.get("url") or "").strip() or None
    show_install = (info_type == "part_install") or ((state.get("intent") or "").strip() == "part_install")

    _home = "https://www.partselect.com"

    def _is_base(u: str) -> bool:
        if not (u or "").strip():
            return True
        u = (u or "").strip().rstrip("/")
        return u == _home or (u.startswith(_home) and "/Models/" not in u and "/Search.aspx" not in u and u.endswith("partselect.com"))

    # When part has no real URL, resolve part page via SERP (real .htm page), not Search.aspx
    if not part_url or _is_base(part_url):
        from app.serp import search_serp
        serp_results = await asyncio.to_thread(search_serp, f"site:partselect.com {ps or part_number}", num=8)
        url = None
        for r in (serp_results or []):
            link = (r.get("link") or "").strip()
            if not link or "partselect.com" not in link.lower() or "/Search.aspx" in link:
                continue
            if _is_base(link):
                continue
            if ".htm" in link or "/PS" in link.upper():
                url = link
                break
        if not url and serp_results:
            for r in serp_results:
                link = (r.get("link") or "").strip()
                if link and "partselect.com" in link.lower() and not _is_base(link) and "/Search.aspx" not in link:
                    url = link
                    break
        url = url or part_url
    else:
        url = part_url
    # When user asked how to install: answer is only the SERP summary (no Part/PartSelect/Installation/Details header)
    if show_install:
        from app.serp import search_serp
        from app.llm_router_planner import llm_summarize_serp_for_part
        query = f"site:partselect.com {part_number} installation"
        serp_results = await asyncio.to_thread(search_serp, query, num=6)
        if serp_results:
            message = (state.get("message") or "").strip() or None
            summary = await asyncio.to_thread(
                llm_summarize_serp_for_part,
                part_number,
                serp_results,
                installation_focus=True,
                user_question=message,
            )
            if summary and summary.strip():
                answer_text = summary.strip()
            else:
                answer_text = "See the product card below for the part page and installation steps."
        else:
            answer_text = "See the product card below for the part page and installation steps."
    else:
        lines = [f"**{name}**", f"PartSelect: {ps}" + (f" | Manufacturer: {mfr}" if mfr else "")]
        if price is not None:
            lines.append(f"Price: ${price}")
        if diff or time_est:
            lines.append(f"Difficulty: {diff or '—'} | Time: {time_est or '—'}")
        if url:
            lines.append(f"Details: [View on PartSelect]({url})")
        answer_text = "\n".join(lines)

    # Image: prefer part record; else SERP thumbnail so image actually shows (PartSelectImages URL often 404)
    image_url = (part.get("image_url") or "").strip() or None
    if not image_url and (ps or part_number):
        from app.serp import search_serp
        serp_for_image = await asyncio.to_thread(search_serp, f"site:partselect.com {ps or part_number}", num=5)
        for r in (serp_for_image or []):
            thumb = (r.get("thumbnail") or "").strip()
            if thumb and thumb.startswith("http"):
                image_url = thumb
                break
        if not image_url and (ps or part_number):
            pn = (ps or part_number).strip().upper()
            if pn.startswith("PS") or (len(pn) >= 5 and pn.replace("-", "").isalnum()):
                image_url = f"https://www.partselect.com/PartSelectImages/{pn}.jpg"
    card = {
        "part_number": ps or part_number,
        "name": name,
        "price": price,
        "url": url,
        "image_url": image_url,
        "brand": part.get("brand"),
    }
    return {"answer": answer_text, "citations": [], "product_cards": [card], "lookup_status": "hit"}


def node_llm_router(state: TroubleshootingState) -> TroubleshootingState:
    """Router: dishwasher vs refrigerator graph. Sets appliance_type."""
    message = (state.get("message") or "").strip()
    last = state.get("last_assistant_message") or None
    result = llm_router_planner.llm_router(message, last)
    return {"appliance_type": result.get("appliance", "refrigerator")}


def node_llm_planner(state: TroubleshootingState) -> TroubleshootingState:
    """Planner: next_action + action_args + info_type. Sets next_action for route."""
    message = (state.get("message") or "").strip()
    appliance = state.get("appliance_type") or "refrigerator"
    model_number = state.get("model_number")
    part_number = state.get("part_number")
    last = state.get("last_assistant_message") or None
    result = llm_router_planner.llm_planner(
        message, appliance,
        model_number=model_number,
        part_number=part_number,
        last_assistant_message=last,
    )
    # Rule override: message clearly about ice maker → ice_maker_issue so we get REF_ICE_MAKER + preferred guide
    msg_lower = (message or "").strip().lower()
    symptom = result.get("symptom", "general")
    if (appliance == "refrigerator" and symptom in ("general", "other")
            and any(k in msg_lower for k in ["ice maker", "icemaker", "ice dispenser", "not dispensing water"])):
        symptom = "ice_maker_issue"
        result = {**result, "symptom": symptom}
    # Prefer current_state from initial (LLM slots diagnostic_state); else compute from planner symptom
    current_state = state.get("current_state") or state_id_from_appliance_symptom(appliance, symptom)
    # need_clarify from planner → routing uses ask_clarify (need_clarify wins in agent_state._next_action_after_planner).
    merged = {
        **state,
        "intent": result.get("intent", "troubleshoot"),
        "symptom": result.get("symptom", "general"),
        "current_state": current_state,
        "need_clarify": result.get("clarify_question"),
        "planner_next_action": result.get("next_action", "troubleshoot"),
        "action_args": result.get("action_args") or {},
        "info_type": result.get("info_type"),
        "planner_missing_info": result.get("missing_info", []),
    }
    merged["next_action"] = compute_next_action(merged, "after_planner")
    # Fallback: if we would go to retrieve but current message asks for install/replace part without model/part in *current message*, force clarify (ignore history so dishwasher question is not answered with fridge context)
    if merged["next_action"] == "retrieve":
        from app.llm_router_planner import deterministic_planner_override
        override = deterministic_planner_override(message, None, None)  # use None so override only looks at current message
        if override and override.get("clarify_question"):
            merged["need_clarify"] = override["clarify_question"]
            merged["next_action"] = "ask_clarify"
    out = {
        "intent": merged["intent"],
        "symptom": merged["symptom"],
        "current_state": merged["current_state"],
        "need_clarify": merged["need_clarify"],
        "planner_next_action": merged["planner_next_action"],
        "action_args": merged["action_args"],
        "info_type": merged["info_type"],
        "planner_missing_info": merged["planner_missing_info"],
        "next_action": merged["next_action"],
    }
    if state.get("model_number") is not None:
        out["model_number"] = state["model_number"]
    elif (merged.get("action_args") or {}).get("model_number"):
        out["model_number"] = merged["action_args"]["model_number"]
    if state.get("part_number") is not None:
        out["part_number"] = state["part_number"]
    elif (merged.get("action_args") or {}).get("part_number"):
        out["part_number"] = merged["action_args"]["part_number"]
    return out


def _merged_policy_for_adjacent(current_state: str) -> tuple[list[str], list[str], str | None]:
    """Merge allowed/forbidden/appliance from current state + adjacent_states (second-hop fallback)."""
    policy = get_source_policy(current_state)
    allowed = set(policy.get("allowed_symptom_tags") or [])
    forbidden = set(policy.get("forbidden_symptom_tags") or [])
    appliance_type = policy.get("appliance") or None
    for adj in policy.get("adjacent_states") or []:
        p = get_source_policy(adj)
        allowed.update(p.get("allowed_symptom_tags") or [])
        forbidden.update(p.get("forbidden_symptom_tags") or [])
    return list(allowed), list(forbidden), appliance_type


async def node_retrieve(state: TroubleshootingState) -> TroubleshootingState:
    """RAG with state-based policy. If first hop returns < 2 chunks, second hop uses adjacent_states.
    When troubleshoot without model (guides_only), only repair-guide chunks are retrieved so we never surface part links."""
    message = state.get("message") or ""
    current_state = state.get("current_state") or ""
    model_number = (state.get("model_number") or "").strip() or None
    policy = get_source_policy(current_state)
    allowed = policy.get("allowed_symptom_tags") or []
    forbidden = policy.get("forbidden_symptom_tags") or []
    appliance_type = policy.get("appliance") or None
    guides_only = not bool(model_number)  # troubleshoot without model → only guide chunks, no part catalog
    chunks = await get_troubleshooting(
        symptom=message[:100],
        model_number=model_number,
        appliance_type=appliance_type,
        limit=5,
        allowed_symptom_tags=allowed if allowed else None,
        forbidden_symptom_tags=forbidden if forbidden else None,
        guides_only=guides_only,
    )
    # Second hop: if too few hits, retry with merged policy from current + adjacent_states
    min_chunks = 2
    if len(chunks) < min_chunks and policy.get("adjacent_states"):
        merged_allowed, merged_forbidden, _app = _merged_policy_for_adjacent(current_state)
        extra = await get_troubleshooting(
            symptom=message[:100],
            model_number=model_number,
            appliance_type=appliance_type,
            limit=5,
            allowed_symptom_tags=merged_allowed if merged_allowed else None,
            forbidden_symptom_tags=merged_forbidden if merged_forbidden else None,
            guides_only=guides_only,
        )
        seen_ids: set[str] = {c.get("chunk_id") for c in chunks if c.get("chunk_id")}
        for c in extra:
            cid = c.get("chunk_id")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                chunks.append(c)
    return {"evidence": chunks}


# Scenario hints for answer generator: no implicit assumption; Why = state→symptom
_SCENARIO_HINT = {
    "REF_S6_BOTH_WARM": "Both the freezer and refrigerator are not cooling (both warm).",
    "REF_ICE_MAKER": "Refrigerator ice maker not producing ice / not dispensing water. Focus on water supply, fill tube, inlet valve, and ice maker assembly.",
    "REF_S7_FREEZER_COLD_FRIDGE_WARM": (
        "The freezer is cold but the refrigerator section is warm. "
        "For this scenario, the two most common causes are (1) air damper/baffle and (2) evaporator fan; "
        "include both if supported by the evidence."
    ),
    "DW_NOT_DRAINING": "Dishwasher is not draining. Common causes: drain hose, pump, or filter.",
    "DW_LEAKING": "Dishwasher is leaking. Consider door seal, pump, hoses, and tub.",
    "DW_WILL_NOT_START": "Dishwasher will not start. Check power, door latch, and control board.",
    "DW_NOT_CLEANING": "Dishwasher is not cleaning properly. Consider spray arms, filter, and detergent.",
    "DW_NOT_FILLING": "Dishwasher will not fill with water. Check water supply and inlet valve.",
    "DW_NOT_DISPENSING": "Dishwasher is not dispensing detergent properly.",
    "DW_NOT_DRYING": "Dishwasher is not drying. Consider heating element and rinse aid.",
    "DW_NOISY": "Dishwasher is making unusual noise. Consider pump, spray arm, and mounting.",
    "DW_FALLBACK": "Dishwasher issue; use evidence to focus on the described symptom.",
}


def _citations_from_claims_and_chunks(claims: list, chunks: list[dict]) -> list[dict]:
    """Build citations programmatically from validated claim chunk_ids → chunk url/title."""
    chunk_by_id = {c.get("chunk_id"): c for c in chunks if c.get("chunk_id")}
    seen: set[str] = set()
    out = []
    for claim in claims:
        for cid in getattr(claim, "supporting_chunk_ids", []):
            if cid in seen:
                continue
            seen.add(cid)
            c = chunk_by_id.get(cid)
            if not c:
                continue
            url = (c.get("source_url") or "").strip()
            raw_title = (c.get("title") or c.get("source") or "").strip()
            title = _citation_title(raw_title, url, "Repair guide")
            out.append({"url": url, "title": title})
    return out


def _message_mentions_specific_symptom(message: str) -> bool:
    """True if the message clearly describes a specific problem (so we should not use the vague opener)."""
    if not (message or "").strip():
        return False
    m = (message or "").strip().lower()
    return any(
        phrase in m
        for phrase in (
            "ice maker", "icemaker", "not producing", "not dispensing", "not dispensing water",
            "not cooling", "not cold", "too warm", "leaking", "leak", "not draining", "not filling",
            "not starting", "won't start", "not cleaning", "not drying", "noise", "noisy",
        )
    )


def _user_message_is_vague(state: dict, message: str) -> bool:
    """Vague only when symptom is general/other AND message does not clearly describe a specific problem."""
    if state.get("symptom") not in ("general", "other"):
        return False
    if _message_mentions_specific_symptom(message):
        return False  # e.g. "ice maker isn't producing any ice" → treat as specific
    return True


async def node_normalize_compose(state: TroubleshootingState) -> TroubleshootingState:
    """Compose answer; citations from claim→chunk_id (validated) and rendered programmatically."""
    evidence = state.get("evidence") or []
    message = state.get("message") or ""
    current_state = state.get("current_state") or ""
    scenario_hint = _SCENARIO_HINT.get(current_state, "")
    if not evidence:
        return {
            "answer": "I didn't find specific results yet. Please share your part or model number, or describe the symptom.",
            "citations": [],
            "product_cards": [],
        }
    model_number = state.get("model_number") or None
    brand_hint = _brand_from_message(message)
    user_message_vague = _user_message_is_vague(state, message)
    intent = (state.get("intent") or "troubleshoot").strip()
    troubleshoot_guides_only = intent == "troubleshoot" and not (model_number or "").strip()

    # When only tutorials wanted: filter evidence so answer generator sees only guide chunks (LLM, no keywords)
    if troubleshoot_guides_only and evidence:
        evidence = await filter_chunks_to_guides_only(evidence)

    # Prefer claim→chunk_id path: LLM outputs claims with supporting_chunk_ids; we validate and render citations
    answer, claims = await evidence_to_answer_with_claims(
        evidence, message, scenario_hint=scenario_hint, model_number=model_number, brand_hint=brand_hint,
        user_message_vague=user_message_vague,
        troubleshoot_guides_only=troubleshoot_guides_only,
    )
    if answer and claims:
        if troubleshoot_guides_only:
            citations = get_suggested_links_troubleshoot(state)
        else:
            citations = _citations_from_claims_and_chunks(claims, evidence)
            if not citations:
                citations = _unique_sources_from_chunks(evidence)
        return {"answer": answer, "citations": citations, "product_cards": []}

    # Fallback: legacy compose (no claims or LLM failed)
    structured = await chunks_to_structured_evidence(evidence, message)
    if structured:
        answer = await evidence_to_answer(
            structured, message, scenario_hint=scenario_hint, model_number=model_number, brand_hint=brand_hint,
            user_message_vague=user_message_vague,
            troubleshoot_guides_only=troubleshoot_guides_only,
        )
        citations = _unique_sources_from_chunks(evidence)
    else:
        answer = (
            "I didn't find enough to build a clear answer. "
            "Please share your appliance model or part number for more specific steps."
        )
        citations = _unique_sources_from_chunks(evidence)
    if troubleshoot_guides_only:
        citations = get_suggested_links_troubleshoot(state)
    return {"answer": answer, "citations": citations, "product_cards": []}


def node_citation_gate(state: TroubleshootingState) -> TroubleshootingState:
    """Block ungrounded answers (doc §9): if we had evidence but no citations, overwrite with safe fallback. Always clear product_cards on retrieve path."""
    evidence = state.get("evidence") or []
    citations = state.get("citations") or []
    answer = state.get("answer") or ""
    if evidence and not citations and answer:
        return {
            "answer": (
                "I couldn't ground a safe answer in our guides for this. "
                "Please share your appliance model number so I can suggest the right repair steps or parts."
            ),
            "product_cards": [],
        }
    # Retrieve path must not show suggested parts; ensure product_cards cleared
    return {"product_cards": []}


# --- Routing: single switch on next_action (set by triage / cooling_split / llm_planner) ---

def route_after_triage(state: TroubleshootingState) -> Literal["ask_clarify", "compatibility_answer", "part_lookup_answer", "parts_list_answer", "find_model_help", "cooling_split"]:
    """Route by next_action only."""
    return (state.get("next_action") or "cooling_split")  # type: ignore[return-value]


def route_after_cooling_split(state: TroubleshootingState) -> Literal["ask_clarify", "retrieve"]:
    """Route by next_action only."""
    return (state.get("next_action") or "retrieve")  # type: ignore[return-value]


def route_from_start(state: TroubleshootingState) -> Literal["triage", "llm_router"]:
    """Use LLM Router+Planner when USE_LLM_ROUTER_PLANNER=1, else rule-based triage."""
    if os.environ.get("USE_LLM_ROUTER_PLANNER", "").lower() in ("1", "true", "yes"):
        return "llm_router"
    return "triage"


def route_after_planner(state: TroubleshootingState) -> Literal["ask_clarify", "compatibility_answer", "part_lookup_answer", "parts_list_answer", "find_model_help", "retrieve"]:
    """Route by next_action only."""
    return (state.get("next_action") or "retrieve")  # type: ignore[return-value]


def build_troubleshooting_graph() -> StateGraph:
    """Build and compile the graph. START → (triage | llm_router→llm_planner) → clarify/tools/retrieve → compose → citation_gate."""
    builder = StateGraph(TroubleshootingState)

    builder.add_node("triage", node_triage)
    builder.add_node("cooling_split", node_cooling_split)
    builder.add_node("llm_router", node_llm_router)
    builder.add_node("llm_planner", node_llm_planner)
    builder.add_node("ask_clarify", node_ask_clarify)
    builder.add_node("compatibility_answer", node_compatibility_answer)
    builder.add_node("part_lookup_answer", node_part_lookup_answer)
    builder.add_node("parts_list_answer", node_parts_list_answer)
    builder.add_node("find_model_help", node_find_model_help)
    builder.add_node("retrieve", node_retrieve)
    builder.add_node("normalize_compose", node_normalize_compose)
    builder.add_node("citation_gate", node_citation_gate)

    builder.add_conditional_edges(START, route_from_start, {"triage": "triage", "llm_router": "llm_router"})
    builder.add_edge("llm_router", "llm_planner")
    builder.add_conditional_edges(
        "llm_planner",
        route_after_planner,
        {
            "ask_clarify": "ask_clarify",
            "compatibility_answer": "compatibility_answer",
            "part_lookup_answer": "part_lookup_answer",
            "parts_list_answer": "parts_list_answer",
            "find_model_help": "find_model_help",
            "retrieve": "retrieve",
        },
    )
    builder.add_conditional_edges(
        "triage",
        route_after_triage,
        {
            "ask_clarify": "ask_clarify",
            "compatibility_answer": "compatibility_answer",
            "part_lookup_answer": "part_lookup_answer",
            "parts_list_answer": "parts_list_answer",
            "find_model_help": "find_model_help",
            "cooling_split": "cooling_split",
        },
    )
    builder.add_edge("ask_clarify", END)
    builder.add_edge("compatibility_answer", END)
    builder.add_edge("part_lookup_answer", END)
    builder.add_edge("parts_list_answer", END)
    builder.add_edge("find_model_help", END)
    builder.add_conditional_edges("cooling_split", route_after_cooling_split, {"ask_clarify": "ask_clarify", "retrieve": "retrieve"})
    builder.add_edge("retrieve", "normalize_compose")
    builder.add_edge("normalize_compose", "citation_gate")
    builder.add_edge("citation_gate", END)

    return builder.compile()


# Compiled graph singleton
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_troubleshooting_graph()
    return _graph
