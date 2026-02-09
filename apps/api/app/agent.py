"""
Agent: S0_SCOPE in main → (optional LLM input summary) → LangGraph state machine.
- Initial state: message, scope_label, last_assistant_message, model_number, part_number (from LLM or model_parser).
- Graph routes to a node (e.g. parts_list_answer when "parts for model X"); that node writes answer, citations, product_cards.
- Final state: we read answer, citations, product_cards and return (content, sources, product_cards).
"""
import asyncio
import logging
import os
import re
from typing import List

logger = logging.getLogger(__name__)

from app.schemas import ScopeLabel, ChatMessage
from app.tools import search_parts, _row_from_serp
from app.agent_graph import get_graph, _filter_product_cards_by_scope, get_suggested_links_troubleshoot
from app.agent_state import _current_message_asks_for_parts
from app.model_parser import extract_model_from_messages, extract_part_from_messages, partselect_model_url, parse_model_revision

def _symptom_page_url(model_number: str, part_topic: str) -> str | None:
    """Build PartSelect model-specific Symptoms page URL, e.g. .../Models/WRF535SWHZ/Symptoms/Ice-maker-not-making-ice/."""
    base, _ = parse_model_revision((model_number or "").strip())
    base = (base or "").strip().upper() if base else ""
    if not base:
        return None
    slug = _PART_TOPIC_TO_SYMPTOM_SLUG.get((part_topic or "").strip().lower()) or (part_topic or "").strip().title().replace(" ", "-")
    if not slug:
        return None
    from urllib.parse import quote
    return f"https://www.partselect.com/Models/{quote(base)}/Symptoms/{quote(slug)}/"


def _model_parts_page_url(model_number: str) -> str:
    """PartSelect model parts list (all parts for this model). Differs from Overview."""
    u = partselect_model_url(model_number or "").rstrip("/")
    if not u or u == "https://www.partselect.com":
        return u or "https://www.partselect.com/"
    return f"{u}/Parts/"


# Part topic: (display_label, search_keywords). Used to detect what part user asked about and to Serp for model-specific part page.
_PART_TOPIC_KEYWORDS: list[tuple[str, list[str]]] = [
    ("ice maker", ["ice maker", "icemaker", "ice maker part", "not making ice", "not dispensing water"]),
    ("water filter", ["water filter", "filter replacement", "water filter replacement"]),
    ("door seal", ["door seal", "gasket", "door gasket"]),
    ("drain pump", ["drain pump", "drain", "pump", "not draining"]),
    ("thermostat", ["thermostat", "temperature control"]),
    ("evaporator fan", ["evaporator fan", "evaporator", "freezer fan"]),
    ("damper", ["damper", "air damper"]),
    ("water valve", ["water valve", "inlet valve"]),
    ("dispenser", ["dispenser", "water dispenser", "ice dispenser"]),
]

# PartSelect Symptoms page slug by part_topic (context fallback when Serp doesn't return a Symptoms URL).
_PART_TOPIC_TO_SYMPTOM_SLUG: dict[str, str] = {
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

def _extract_part_topic_from_message(message: str) -> str | None:
    """If the user asked about a specific part/category, return its display label (e.g. 'ice maker'); else None."""
    m = (message or "").lower()
    if not m.strip():
        return None
    for label, keywords in _PART_TOPIC_KEYWORDS:
        if any(k in m for k in keywords):
            return label
    return None


def _is_overview_url(link: str, model_upper: str) -> bool:
    link_lower = (link or "").lower()
    return bool(
        link_lower.rstrip("/").endswith("/models/" + model_upper.lower())
        or (link_lower.rstrip("/").endswith("/" + model_upper.lower()) and "/models/" in link_lower)
    )


def _collect_part_only_from_serp_results(results: list, model_upper: str) -> list[dict]:
    """
    From Serp results, collect part pages: .htm, /PS, /Parts/, or ModelNum= in query.
    Accept links that have model in URL or in title (Serp query already scoped by model).
    """
    seen: set[str] = set()
    out: list[dict] = []
    raw = results or []
    for r in raw:
        link = (r.get("link") or r.get("url") or "").strip()
        title = (r.get("title") or "").strip()
        if not link or "partselect.com" not in link.lower():
            continue
        if _is_partselect_base_url(link):
            continue
        link_lower = link.lower()
        if _is_overview_url(link, model_upper):
            continue
        if "/symptoms/" in link_lower:
            continue
        # Part page: .htm, /ps, /parts/ (model parts list), or modelnum= in query
        is_part = (
            ".htm" in link_lower
            or "/ps" in link_lower
            or "/parts" in link_lower
            or "modelnum=" in link_lower
            or "modelnum=" in (r.get("link") or r.get("url") or "").lower()
            or ("assembly" in link_lower and model_upper.lower() in link_lower)
        )
        if not is_part:
            continue
        # Model in link or title (or accept any part page when query was model-specific)
        if model_upper not in link.upper() and model_upper not in (title or "").upper():
            continue
        norm = link.split("?")[0].rstrip("/")
        if norm in seen:
            continue
        seen.add(norm)
        out.append({"name": title or "Part", "url": link, "image_url": r.get("thumbnail") or r.get("image_url"), "part_number": None, "price": None, "brand": None})
    if raw and not out:
        logger.info("Serp part filter: raw=%s part_pages=0 sample_links=%s", len(raw), [((x.get("link") or x.get("url")) or "")[:70] for x in raw[:3]])
    return out


# Suggested parts: only Symptoms + first N real part-detail pages from Serp (no Overview, no generic Parts list).
_MAX_REAL_PARTS_FROM_SYMPTOM = 2


async def _resolve_model_links_via_serp(
    model_number: str, part_topic: str | None
) -> list[dict]:
    """
    Only parts and symptoms: 1) Symptoms page, 2) first 2 real part-detail pages from Serp for that symptom.
    No Overview, no extra links. When no part_topic: just Parts page for model.
    """
    if not (model_number or "").strip():
        return []
    base, _ = parse_model_revision((model_number or "").strip())
    if not base:
        return []
    from app.serp import search_serp
    model_upper = base.upper()
    part_topic = (part_topic or "parts").strip() or "parts"
    part_slug = part_topic.lower().replace(" ", "-")
    collected: list[dict] = []
    seen_urls: set[str] = set()

    def add_one(card: dict) -> bool:
        u = (card.get("url") or "").split("?")[0].rstrip("/")
        if u and u not in seen_urls:
            seen_urls.add(u)
            collected.append(card)
            return True
        return False

    if part_topic.strip().lower() == "parts":
        # No symptom: only Parts page for model
        parts_page = _model_parts_page_url(model_number)
        if parts_page:
            add_one({"name": f"Parts for {model_number}", "url": parts_page, "image_url": None, "part_number": None, "price": None, "brand": None})
        logger.info("Serp links (parts only): %s cards", len(collected))
        return collected

    # 1) Symptoms page (Serp or context fallback)
    query_symptoms = f"site:partselect.com/Models/{base}/Symptoms {part_topic}"
    symptom_card: dict | None = None
    symptom_slug: str | None = None
    try:
        results = await asyncio.to_thread(search_serp, query_symptoms, num=8)
        for r in (results or []):
            link = (r.get("link") or "").strip()
            title = (r.get("title") or "").strip()
            if not link or "partselect.com" not in link.lower():
                continue
            if _is_partselect_base_url(link):
                continue
            link_lower = link.lower()
            if "/symptoms/" in link_lower and model_upper.lower() in link_lower and not _is_overview_url(link, model_upper):
                symptom_card = {"name": f"Symptoms: {part_topic}", "url": link, "image_url": None, "part_number": None, "price": None, "brand": None}
                try:
                    idx = link_lower.find("/symptoms/") + len("/symptoms/")
                    rest = link_lower[idx:].strip("/").split("/")
                    if rest:
                        symptom_slug = rest[0].title().replace("_", "-")
                except Exception:
                    pass
                break
    except Exception as e:
        logger.debug("Serp Symptoms failed: %s", e)
    if not symptom_card and part_topic.strip().lower() != "parts":
        symptom_slug = _PART_TOPIC_TO_SYMPTOM_SLUG.get(part_topic.strip().lower()) or part_topic.strip().title().replace(" ", "-")
        symptom_url = f"https://www.partselect.com/Models/{base}/Symptoms/{symptom_slug}/"
        symptom_card = {"name": f"Symptoms: {part_topic}", "url": symptom_url, "image_url": None, "part_number": None, "price": None, "brand": None}
    if symptom_card:
        add_one(symptom_card)

    # 2) Part pages from Serp — try queries that often return Parts list or .htm part pages
    part_queries = [
        f"site:partselect.com {base} parts",
        f"site:partselect.com {base} {part_topic}",
        f"site:partselect.com {base} {part_topic} assembly",
        f"site:partselect.com {base} .htm {part_topic}",
        f"site:partselect.com {base} PS {part_topic}",
    ]
    part_cards: list[dict] = []
    for q in part_queries:
        if len(part_cards) >= _MAX_REAL_PARTS_FROM_SYMPTOM:
            break
        try:
            results = await asyncio.to_thread(search_serp, q, num=10)
            found = _collect_part_only_from_serp_results(results or [], model_upper)
            for c in found:
                u = (c.get("url") or "").split("?")[0].rstrip("/")
                if u and u not in seen_urls:
                    part_cards.append(c)
                    seen_urls.add(u)
                    if len(part_cards) >= _MAX_REAL_PARTS_FROM_SYMPTOM:
                        break
        except Exception as e:
            logger.debug("Serp part query %s failed: %s", q[:40], e)
    for c in part_cards[:_MAX_REAL_PARTS_FROM_SYMPTOM]:
        collected.append(c)
    # When Serp returned no part links, add model Parts list page so user has at least one product-style link
    if not part_cards and symptom_card:
        parts_page = _model_parts_page_url(model_number)
        if parts_page and parts_page.rstrip("/") not in seen_urls:
            collected.append({"name": f"Parts for {model_number}", "url": parts_page, "image_url": None, "part_number": None, "price": None, "brand": None})

    logger.info("Serp links for model=%s topic=%s: %s cards (Symptoms + parts)", base[:20], part_topic, len(collected))
    return collected


from app.evidence import _brand_from_message

# When user did not specify a brand, drop Suggested parts that are clearly brand-specific (e.g. "Official GE WR09X...", "Hotpoint Refrigerator Thermostats")
_BRAND_IN_NAME = re.compile(
    r"\b(Official\s+)?(GE|Hotpoint|Whirlpool|Samsung|Kenmore|KitchenAid|Maytag|Frigidaire|LG|Bosch|Electrolux|Admiral|General\s+Electric)\b",
    re.IGNORECASE,
)


def _product_card_image_fallback(part_number: str | None) -> str | None:
    """Return PartSelect image URL for a part number so cards can show an image when missing."""
    pn = (part_number or "").strip().upper()
    if not pn:
        return None
    if pn.startswith("PS") or (pn.replace("-", "").isalnum() and len(pn) >= 6):
        return f"https://www.partselect.com/PartSelectImages/{pn}.jpg"
    return None


def _is_partselect_base_url(url: str) -> bool:
    """True if URL is PartSelect homepage or empty (no meaningful path)."""
    u = (url or "").strip().rstrip("/")
    if not u or "partselect.com" not in u.lower():
        return True
    if u.lower() in ("https://www.partselect.com", "https://www.partselect.com/"):
        return True
    from urllib.parse import urlparse
    parsed = urlparse(u)
    path = (parsed.path or "").strip().rstrip("/")
    return path in ("", "/")


_PS_PART_NUMBER = re.compile(r"^PS\d{5,15}$", re.IGNORECASE)


def _filter_suggested_parts_base_and_off_model(
    parts: list, model_number: str | None, user_asked_for_parts: bool = False
) -> list:
    """
    Remove suggested parts that are base URLs or (when model is set) not relevant to the model.
    Keeps: non-base URLs; cards with PartSelect part number (PS...) even if url is base (main will fix to Search URL); when model set, part cards (PS) are kept.
    When user_asked_for_parts is True, keep all non-base cards (do not filter by model).
    """
    if not parts:
        return parts
    model_base = None
    if (model_number or "").strip() and not user_asked_for_parts:
        base, _ = parse_model_revision((model_number or "").strip())
        model_base = (base or "").strip().upper()
    out = []
    for p in parts:
        url = (p.get("url") or "").strip()
        name = (p.get("name") or "").strip()
        part_num = (p.get("part_number") or p.get("partselect_number") or "").strip()
        if _is_partselect_base_url(url):
            if part_num and _PS_PART_NUMBER.match(part_num):
                out.append(p)
            continue
        if model_base:
            if part_num and _PS_PART_NUMBER.match(part_num):
                out.append(p)
                continue
            url_upper = url.upper()
            name_upper = name.upper()
            if model_base not in url_upper and model_base not in name_upper:
                continue
        out.append(p)
    return out


def _filter_brand_specific_cards(parts: list, brand_hint: str | None) -> list:
    """When brand_hint is None, keep only cards whose name does not contain a brand (generic guides/parts only)."""
    if brand_hint or not parts:
        return parts
    return [p for p in parts if not _BRAND_IN_NAME.search((p.get("name") or ""))]


def _strip_brand_from_card_name(name: str | None) -> str:
    """Remove brand words from card name so we don't show brand when user didn't specify one."""
    if not (name or "").strip():
        return name or ""
    t = _BRAND_IN_NAME.sub("", (name or ""))
    t = re.sub(r"  +", " ", t)
    return t.strip()


async def _get_model_specific_parts_and_content(
    model_number: str,
    message: str,
    asked_parts_for_model: bool,
    model_in_message: bool,
    content_so_far: str,
    max_parts: int,
) -> tuple[str, list[dict]]:
    """
    When user has a model and we show model-specific reply: build parts list from Serp (model-specific
    Symptoms page + part links) + search_parts. Returns (content_to_use, parts_list). No Playwright.
    """
    part_topic = _extract_part_topic_from_message(message or "")
    # Use only Serp + search_parts: Serp finds the model-specific Symptoms page; we build cards from those links + search_parts.
    # (No Playwright; part list for "Common parts for ... issues" is parsed from Serp snippet in run_agent.)
    serp_cards = await _resolve_model_links_via_serp(model_number, part_topic or "parts")
    part_rows = await search_parts(
        symptom=(message or part_topic or "")[:100],
        model_number=model_number,
        brand=_brand_from_message(message or ""),
        limit=5,
    )
    part_cards_from_search = []
    for p in (part_rows or [])[:5]:
        u = (p.get("url") or "").strip()
        if not u or _is_partselect_base_url(u):
            continue
        pn = p.get("part_number") or p.get("partselect_number")
        img = (p.get("image_url") or "").strip() or None
        if not img and pn:
            img = _product_card_image_fallback(pn)
        part_cards_from_search.append({
            "name": (p.get("name") or "").strip() or "Part",
            "url": u,
            "image_url": img,
            "part_number": pn,
            "price": p.get("price"),
            "brand": p.get("brand"),
        })
    symptom_card = next((c for c in serp_cards if (c.get("url") or "").lower().count("/symptoms/") > 0), None)
    serp_part_cards = [c for c in serp_cards if c != symptom_card and (c.get("url") or "").strip()]
    parts = part_cards_from_search + ([] if not symptom_card else [symptom_card]) + serp_part_cards
    parts = parts[:max_parts]
    # 4) Content snippet
    if asked_parts_for_model:
        content = f"Parts and instructions for **{model_number}**.\n\nBelow you'll find commonly needed parts and links to guides and repair steps."
    elif model_in_message:
        content = (content_so_far.strip() + f"\n\nParts and instructions for **{model_number}**.").strip()
    else:
        content = content_so_far
    return content, parts


def _build_product_cards(parts: list[dict], brand_hint: str | None, max_parts: int) -> list[dict]:
    """Build API product_cards from parts list with name, url, price, image_url, part_number."""
    out = []
    for p in parts[:max_parts]:
        name = p.get("name")
        if not brand_hint and name:
            name = _strip_brand_from_card_name(name)
        url = (p.get("url") or "").strip()
        image_url = (p.get("image_url") or "").strip() or None
        part_number = p.get("part_number") or p.get("partselect_number")
        if part_number and not image_url and url and "partselect.com" in url.lower():
            image_url = _product_card_image_fallback(part_number)
        out.append({
            "name": name,
            "url": url or None,
            "price": p.get("price"),
            "image_url": image_url,
            "part_number": part_number,
            "brand": p.get("brand") if brand_hint else None,
        })
    return out


MAX_SUGGESTED_PARTS = 5


def _history_context(history: List[ChatMessage]) -> tuple[str, list[str], str | None]:
    """From chat history, return (last_assistant_message, history_user_contents, conversation_context)."""
    last_assistant = ""
    history_user_contents: List[str] = []
    if history:
        for m in reversed(history):
            if getattr(m, "role", None) == "assistant" and getattr(m, "content", None):
                last_assistant = (m.content or "").strip()
                break
        history_user_contents = [getattr(m, "content", "") or "" for m in history if getattr(m, "role", None) == "user"]
    conversation_context = None
    if history_user_contents:
        conversation_context = "\n".join(h.strip() for h in history_user_contents[-2:] if (h or "").strip())
    return last_assistant, history_user_contents, conversation_context


def _resolve_part_number_fallbacks(message: str, part_number: str | None, last_assistant: str) -> str | None:
    """Apply fallbacks: drop generic words, PS in message, compatibility from last assistant."""
    if part_number and (part_number or "").strip().upper() in {"REPLACE", "PART", "PARTS", "INSTALL", "INSTALLATION", "COMPATIBLE", "COMPATIBILITY"}:
        part_number = None
    if not part_number and (message or "").strip():
        m = re.search(r"PS\d{5,15}", (message or ""), re.IGNORECASE)
        if m:
            part_number = m.group(0).upper()
    if not part_number and re.search(r"\b(compatible|compatibility|fit|fits)\b", message or "", re.IGNORECASE) and last_assistant:
        for pat in (r"PartSelect:\s*(PS\d{5,15})", r"Part\s*#\s*(PS\d{5,15})", r"Part\s+(PS\d{5,15})"):
            m = re.search(pat, last_assistant, re.IGNORECASE)
            if m:
                part_number = m.group(1).upper()
                break
        if not part_number:
            m = re.search(r"\b(PS\d{5,15})\b", last_assistant, re.IGNORECASE)
            if m:
                part_number = m.group(1).upper()
    return part_number


def _build_initial_state(
    message: str,
    scope_label: ScopeLabel,
    last_assistant: str,
    conversation_context: str | None,
    model_number: str | None,
    part_number: str | None,
    is_symptom_only_from_llm: bool | None,
    diagnostic_state_from_slots: str | None,
) -> dict:
    """Build the initial state dict passed to the graph."""
    initial: dict = {
        "message": message or "",
        "scope_label": scope_label.value,
        "last_assistant_message": last_assistant or None,
        "model_number": model_number,
        "part_number": part_number,
        "conversation_context": conversation_context,
    }
    if is_symptom_only_from_llm is not None:
        initial["is_symptom_only"] = is_symptom_only_from_llm
    if diagnostic_state_from_slots:
        initial["current_state"] = diagnostic_state_from_slots
    return initial


def _postprocess_graph_response(
    final: dict,
    initial: dict,
    message: str,
    part_number: str | None,
) -> dict:
    """Extract content, sources, routing state, parts, and intent from graph result. Returns a context dict."""
    from app.agent_state import _message_is_symptom_only

    content = (final.get("answer") or "").strip()
    for phrase in ("About this repair: ", "About this repair:", "About this repair "):
        content = content.replace(phrase, "")
    content = content.strip()
    sources = list(final.get("citations") or [])
    is_clarify = final.get("next_action") == "ask_clarify"
    if not is_clarify and final.get("need_clarify") and content:
        cq = (final.get("need_clarify") or "").strip()
        if cq and (content.strip() == cq or content.strip().startswith(cq[:50])):
            is_clarify = True
    if is_clarify:
        sources = []

    model_number = (final.get("model_number") or (final.get("action_args") or {}).get("model_number") or initial.get("model_number") or "").strip()
    msg_lower = (message or "").lower()
    state_for_routing = {
        "message": message or "",
        "model_number": model_number,
        "part_number": (final.get("part_number") or part_number or "").strip(),
        "is_symptom_only": initial.get("is_symptom_only"),
    }
    is_symptom_only = (
        state_for_routing.get("is_symptom_only")
        if state_for_routing.get("is_symptom_only") is not None
        else _message_is_symptom_only(message or "", state_for_routing)
    )
    if is_clarify or is_symptom_only:
        parts: list = []
    else:
        parts = list(final.get("product_cards") or [])[:MAX_SUGGESTED_PARTS]

    intent = (final.get("intent") or "troubleshoot").strip()
    part_intents = ("part_install", "product_info", "compatibility")
    asked_parts_for_model = (
        bool(model_number)
        and ("parts for" in msg_lower or intent in ("product_info", "part_install"))
        and _current_message_asks_for_parts(state_for_routing)
    )
    if not asked_parts_for_model and model_number and ("part" in msg_lower or "parts" in msg_lower) and any(k in msg_lower for k in ("find", "where", "need", "want", "get", "parts for")):
        asked_parts_for_model = _current_message_asks_for_parts(state_for_routing) or (model_number.lower() in msg_lower)
    has_canonical_link = model_number and f"view parts for {model_number.lower()}" in content.lower()
    model_in_message = bool(model_number) and model_number.lower() in msg_lower

    return {
        "content": content,
        "sources": sources,
        "is_clarify": is_clarify,
        "model_number": model_number,
        "state_for_routing": state_for_routing,
        "parts": parts,
        "intent": intent,
        "asked_parts_for_model": asked_parts_for_model,
        "has_canonical_link": has_canonical_link,
        "model_in_message": model_in_message,
    }


async def run_agent(
    message: str,
    scope_label: ScopeLabel,
    history: List[ChatMessage],
) -> tuple[str, list[dict], list[dict], dict | None]:
    """
    Returns (content, sources, product_cards, None). model_context deprecated and always None.

    Flow: 1) Scope guard → 2) Graph ainvoke → 3) Parse model/intent →
    4) If model and no canonical link: _get_model_specific_parts_and_content (Serp + search_parts) →
    5) Fallback search_parts if no parts → 6) Filter parts → 7) _build_product_cards →
    8) Clear product_cards on pure troubleshoot → 9) Append "Parts often needed" when part_topic → return.
    """
    if scope_label in (ScopeLabel.OUT_OF_SCOPE, ScopeLabel.AMBIGUOUS):
        return (
            "I can only help with appliance parts and related support: finding parts, checking compatibility, installation, troubleshooting, and orders for refrigerators and dishwashers. What would you like help with?",
            [],
            [],
            None,
        )

    graph = get_graph()
    # 1) History and conversation context
    last_assistant, history_user_contents, conversation_context = _history_context(history)

    # 2) Model/part from message (+ optional LLM slots when USE_LLM_ROUTER_PLANNER)
    model_from_message = extract_model_from_messages(message or "", history_user_contents)
    has_revision_in_message = bool(model_from_message and re.search(r"\(\s*\d{1,4}\s*\)", model_from_message))
    recent_user = history_user_contents[-2:] if len(history_user_contents) > 2 else history_user_contents
    is_symptom_only_from_llm = None
    diagnostic_state_from_slots = None
    if os.environ.get("USE_LLM_ROUTER_PLANNER", "").lower() in ("1", "true", "yes"):
        from app.llm_router_planner import llm_extract_slots
        from app.agent_state import VALID_DIAGNOSTIC_STATES
        slots = await asyncio.to_thread(llm_extract_slots, message or "", recent_user)
        model_number = slots.get("model_number")
        part_number = slots.get("part_number")
        if slots.get("is_symptom_only") is not None:
            is_symptom_only_from_llm = bool(slots["is_symptom_only"])
        if slots.get("diagnostic_state") in VALID_DIAGNOSTIC_STATES:
            diagnostic_state_from_slots = slots["diagnostic_state"]
        if has_revision_in_message and model_from_message:
            model_number = model_from_message
        elif not model_number:
            model_number = model_from_message
        if not part_number:
            part_number = extract_part_from_messages(message or "", history_user_contents)
    else:
        model_number = model_from_message
        part_number = extract_part_from_messages(message or "", history_user_contents)
    part_number = _resolve_part_number_fallbacks(message or "", part_number, last_assistant)

    initial = _build_initial_state(
        message, scope_label, last_assistant, conversation_context,
        model_number, part_number, is_symptom_only_from_llm, diagnostic_state_from_slots,
    )
    final = await graph.ainvoke(initial)

    # 3) Postprocess graph result (content, sources, routing state, parts, intent)
    ctx = _postprocess_graph_response(final, initial, message or "", part_number)
    content = ctx["content"]
    sources = ctx["sources"]
    model_number = ctx["model_number"]
    state_for_routing = ctx["state_for_routing"]
    parts = ctx["parts"]
    intent = ctx["intent"]
    asked_parts_for_model = ctx["asked_parts_for_model"]
    has_canonical_link = ctx["has_canonical_link"]
    model_in_message = ctx["model_in_message"]

    # find_model_help: keep graph citations as sources only; do not replace with troubleshoot links; no product cards
    is_find_model_help = (final.get("next_action") or "").strip() == "find_model_help"
    if not ctx["is_clarify"] and not (model_number or "").strip() and content and not is_find_model_help:
        sources = get_suggested_links_troubleshoot(final)
        logger.info(
            "suggested_links: message=%s intent=%s model_number=%s citations_from_graph=%s sources_count=%s",
            (message or "")[:60], intent, model_number or "(none)", len(final.get("citations") or []), len(sources),
        )

    # 4) Model-specific parts (Serp + search_parts) when graph didn't return product_cards
    part_intents = ("part_install", "product_info", "compatibility")
    lookup_status = final.get("lookup_status")
    graph_has_parts = bool(final.get("product_cards"))
    if (
        model_number
        and not has_canonical_link
        and lookup_status not in ("hit", "miss")
        and intent != "compatibility"
        and not graph_has_parts
    ):
        content, parts = await _get_model_specific_parts_and_content(
            model_number, message, asked_parts_for_model, model_in_message, content, MAX_SUGGESTED_PARTS,
        )

    if not parts and intent != "compatibility" and (
        intent in part_intents and _current_message_asks_for_parts(state_for_routing)
        or (model_number and (asked_parts_for_model or model_in_message))
    ):
        brand_hint = _brand_from_message(message or "")
        parts = await search_parts(
            symptom=(message or "")[:100],
            model_number=final.get("model_number") or model_number,
            brand=brand_hint,
            limit=MAX_SUGGESTED_PARTS,
        )
        if parts:
            parts = parts[:MAX_SUGGESTED_PARTS]
            if not content.strip().endswith("**Relevant parts:**"):
                content += "\n\n**Relevant parts:**\n"
            for p in parts:
                content += f"- {p.get('name', '')} ({p.get('part_number', '') or p.get('partselect_number', '')})\n"

    # 5) Filter parts; overview fallback when user asked for parts but no cards
    user_asked_for_parts = asked_parts_for_model or model_in_message
    parts = _filter_suggested_parts_base_and_off_model(parts, model_number, user_asked_for_parts=user_asked_for_parts)
    brand_hint = _brand_from_message(message or "")
    parts = _filter_brand_specific_cards(parts, brand_hint)

    if (model_number or "").strip() and user_asked_for_parts and not parts:
        model_url = partselect_model_url(model_number or "").rstrip("/") or ""
        if model_url:
            parts = [{"name": f"{model_number} - Overview", "url": model_url, "image_url": None, "part_number": None, "partselect_number": None, "price": None, "brand": None}]

    # 6) Build product_cards; clear for pure troubleshoot, compatibility, or find_model_help
    product_cards = _build_product_cards(parts, brand_hint, MAX_SUGGESTED_PARTS)
    if intent == "troubleshoot" and not (model_number and (asked_parts_for_model or model_in_message)):
        product_cards = []
    if intent == "compatibility":
        product_cards = []
    if is_find_model_help:
        product_cards = []

    # 7) Symptom + model: single symptom card + "Common parts" block (Serp or link)
    part_topic_summary = _extract_part_topic_from_message(message or "")
    if part_topic_summary and (model_number or "").strip():
        # Symptom URL card: when we detect a part_topic + model, use the model-specific
        # Symptoms page as the ONLY product card. We no longer render a separate
        # "Common parts for ..." text block in the answer body.
        symptom_url_for_card = _symptom_page_url(model_number, part_topic_summary)
        if symptom_url_for_card:
            product_cards = [{
                "name": f"{part_topic_summary.title()} issues for {model_number}",
                "url": symptom_url_for_card,
                "price": None,
                "image_url": None,
                "part_number": None,
                "brand": None,
            }]

    return content.strip(), sources, product_cards, None
