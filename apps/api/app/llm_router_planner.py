"""
LLM Router + Planner: optional replacement for rule-based triage.

- Router: dishwasher vs refrigerator graph (appliance_type).
- Planner: next_action (parts_list, part_detail, compatibility, clarify, troubleshoot), missing_info, clarify_question.
- Gate: USE_LLM_ROUTER_PLANNER=1 (agent.py uses llm_extract_slots; agent_graph.py uses llm_router, llm_planner, get_clarify_for_product_info, deterministic_planner_override).

Public API:
  llm_extract_slots, llm_router, llm_planner, get_clarify_for_product_info, deterministic_planner_override,
  llm_summarize_serp_snippets_for_model, llm_summarize_serp_for_model, llm_summarize_serp_for_part, llm_extract_common_parts_from_serp.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


def _get_client():
    """OpenAI client when OPENAI_API_KEY is set; None otherwise."""
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key or key == "sk-...":
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=key)
    except Exception:
        return None


# --- Serp summarization (model / part / common parts) ---


def _fallback_summary_from_serp(serp_results: List[dict]) -> str:
    """When LLM returns empty, use first result title/snippet for one line."""
    if not serp_results:
        return ""
    first = serp_results[0]
    title = (first.get("title") or "").strip()
    snippet = (first.get("snippet") or "").strip()
    if title or snippet:
        return f"PartSelect has parts, diagrams, and repair help for this model."
    return ""


def llm_summarize_serp_snippets_for_model(
    model_number: str,
    serp_results: List[dict],
    conversation_context: Optional[str] = None,
) -> str:
    """
    One short sentence summarizing what the user can find on PartSelect for this model (from snippets).
    No links in output; used when we already have the direct model page URL to show.
    When model_number includes a revision (digits in parentheses), only mention that revision—do not suggest other variants.
    When conversation_context is set (recent user messages), focus the summary on what they asked about (e.g. ice maker, leak).
    """
    if not serp_results:
        return ""
    raw = "\n".join(
        f"Title: {r.get('title') or ''}\nSnippet: {r.get('snippet') or ''}"
        for r in serp_results[:6]
    )
    client = _get_client()
    if not client:
        return _fallback_summary_from_serp(serp_results)
    system = (
        "You get search snippets about PartSelect pages for appliance model " + model_number + ". "
        "Reply with ONE short sentence only (e.g. 'PartSelect has parts lists, diagrams, and repair guides for this model.'). "
        "Plain text only: do NOT include any URLs, markdown links, or bullet lists. English only."
    )
    # Any revision in parentheses (e.g. (03), (04), (1)) — we have the direct link; don't cite "no match" or other revisions
    if re.search(r"\(\s*\d{1,4}\s*\)", model_number):
        system += (
            " The user asked for this exact model revision (number in parentheses). We are providing the direct PartSelect page for it. "
            "Do NOT say 'couldn't find exact match', 'no exact match', or 'closest results'. Do NOT suggest or mention other revision numbers. "
            "Say that PartSelect has the guide, parts list, and repair help for this model."
        )
    if conversation_context and conversation_context.strip():
        system += (
            " The user's recent messages:\n\"\"\"\n" + conversation_context.strip()[:500] + "\n\"\"\"\n"
            " If they mentioned a specific problem or part (e.g. ice maker, water filter, leaking), focus your one sentence on that; otherwise keep it general."
        )
    try:
        resp = client.chat.completions.create(
            model=os.environ.get("PLANNER_LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Model: {model_number}\n\nSnippets:\n{raw}"},
            ],
            max_tokens=120,
        )
        text = (resp.choices[0].message.content or "").strip()
        if text:
            return text
    except Exception as e:
        logger.warning("LLM summarize Serp snippets failed: %s", e)
    return _fallback_summary_from_serp(serp_results)


def llm_summarize_serp_for_model(model_number: str, serp_results: List[dict]) -> str:
    """
    Summarize SerpApi search results (PartSelect pages for a model) into a short, user-friendly answer.
    Preserve links in markdown. Used when DB has no parts: we search via SerpApi then summarize.
    """
    if not serp_results:
        return ""
    client = _get_client()
    if not client:
        lines = [f"Parts for model **{model_number}** (from web search):"]
        for r in serp_results[:5]:
            title = (r.get("title") or "").strip() or "Link"
            link = (r.get("link") or "").strip()
            if link:
                lines.append(f"- [{title}]({link})")
        return "\n".join(lines) if lines else ""

    raw = "\n".join(
        f"Title: {r.get('title') or ''}\nLink: {r.get('link') or ''}\nSnippet: {r.get('snippet') or ''}"
        for r in serp_results[:8]
    )
    try:
        resp = client.chat.completions.create(
            model=os.environ.get("PLANNER_LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You summarize PartSelect search results for this specific appliance model into one short reply. "
                        "The user asked for parts for model " + model_number + ". Only mention links that are clearly for this model (e.g. the model's parts page or model overview). "
                        "Structure: (1) One sentence saying they can find parts for this model on PartSelect. (2) One or two markdown links [text](url) that go to this model's page(s). "
                        "Do NOT include generic links like 'Official Appliance Parts', 'Refrigerator Parts', or 'Repair Help' unless the URL is specifically for this model. "
                        "If the results include the PartSelect model parts page (URL usually contains the model number), prefer that link. Lead with a brief summary. English only. No JSON."
                    ),
                },
                {"role": "user", "content": f"Model: {model_number}\n\nSearch results:\n{raw}"},
            ],
            max_tokens=400,
        )
        text = (resp.choices[0].message.content or "").strip()
        if text:
            return text
    except Exception as e:
        logger.warning("LLM summarize Serp failed: %s", e)
    # fallback: plain list with links
    lines = [f"Parts for model **{model_number}** (from web search):"]
    for r in serp_results[:5]:
        title = (r.get("title") or "").strip() or "Link"
        link = (r.get("link") or "").strip()
        if link:
            lines.append(f"- [{title}]({link})")
    return "\n".join(lines)


def llm_extract_common_parts_from_serp(
    model_number: str,
    part_topic: str,
    serp_results: List[dict],
    max_items: int = 5,
) -> List[str]:
    """
    Use LLM router to extract a short list of common replacement part names for a given
    model + symptom/topic from SerpApi results.

    Output: up to `max_items` plain-text part names (e.g. "Ice maker assembly", "Water inlet valve").
    If the LLM cannot confidently extract a list, returns [].
    """
    from json import loads

    if not serp_results:
        return []

    client = _get_client()
    if not client:
        return []

    raw = "\n\n".join(
        f"Title: {r.get('title') or ''}\nLink: {r.get('link') or ''}\nSnippet: {r.get('snippet') or ''}"
        for r in serp_results[:8]
    )
    system_msg = (
        "You are helping a refrigerator and dishwasher parts assistant.\n"
        "Given web search results (PartSelect pages) for a specific appliance model and a symptom/topic, "
        "extract a short list of the most common replacement parts mentioned for that symptom.\n"
        "Return ONLY a JSON array of 2–5 short part names, e.g. [\"Ice maker assembly\", \"Water inlet valve\"].\n"
        "Rules:\n"
        "- Only include part names explicitly mentioned in the snippets/titles.\n"
        "- Prefer parts clearly tied to the given model number when possible.\n"
        "- Do NOT include model numbers, percentages, or long sentences.\n"
        "- If you are not confident there are at least 2 good part names, return an empty JSON array []."
    )
    user_msg = (
        f"Model number: {model_number}\n"
        f"Symptom/topic: {part_topic}\n\n"
        f"Search results:\n{raw}\n\n"
        "Reply ONLY with a JSON array of 0–5 short part names."
    )
    try:
        resp = client.chat.completions.create(
            model=os.environ.get("PLANNER_LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=200,
        )
        text = (resp.choices[0].message.content or "").strip()
        if "```" in text:
            # Strip markdown fences if present
            parts = text.split("```")
            text = parts[1] if len(parts) > 1 else parts[0]
            text = text.replace("json", "").strip()
        arr = loads(text)
        if isinstance(arr, list):
            out: List[str] = []
            seen: set[str] = set()
            for name in arr[: max_items + 2]:
                if not name:
                    continue
                s = str(name).strip()
                if not s or len(s) < 3:
                    continue
                key = s.lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append(s)
                if len(out) >= max_items:
                    break
            # Require at least 2 items to consider it a real list.
            return out if len(out) >= 2 else []
    except Exception as e:
        logger.warning("LLM extract common parts from Serp failed: %s", e)
    return []


def llm_summarize_serp_for_part(
    part_number: str,
    serp_results: List[dict],
    installation_focus: bool = False,
    user_question: Optional[str] = None,
) -> str:
    """
    Summarize SerpApi search results for a part number (PartSelect pages) into a short answer.
    Used when part_lookup finds nothing: we search via SerpApi then summarize.
    installation_focus: emphasize install guides. user_question: tailor summary to what they asked (price, specs, etc.).
    """
    if not serp_results:
        return ""
    client = _get_client()
    if not client:
        lines = [f"Part **{part_number}** (from web search):"]
        for r in serp_results[:5]:
            title = (r.get("title") or "").strip() or "Link"
            link = (r.get("link") or "").strip()
            if link:
                lines.append(f"- [{title}]({link})")
        return "\n".join(lines) if lines else ""

    raw = "\n".join(
        f"Title: {r.get('title') or ''}\nLink: {r.get('link') or ''}\nSnippet: {r.get('snippet') or ''}"
        for r in serp_results[:8]
    )
    system_extra = ""
    if installation_focus:
        system_extra = " The user asked how to install this part. Prioritize installation guides and step-by-step instructions. "
    if user_question and user_question.strip():
        q = user_question.strip()[:200]
        system_extra += f" The user asked: \"{q}\" — tailor your summary to that (e.g. if they asked about price, mention price or say to check the product page for current price; if specs/size/warranty, focus on that). "
    user_content = f"Part number: {part_number}\n\nSearch results:\n{raw}"
    if user_question and user_question.strip():
        user_content = f"User question: {user_question.strip()[:200]}\n\n{user_content}"
    try:
        resp = client.chat.completions.create(
            model=os.environ.get("PLANNER_LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You summarize PartSelect search results for a part into one short, helpful reply. "
                        "Output ONLY 1–3 summary sentences that answer what the user asked (from snippets when possible). "
                        "Do NOT add 'Here are some useful links' or any list of links. Do NOT add links at the end. "
                        "Rules: Only state facts explicitly mentioned in the snippets; do not guess price or details. "
                        + system_extra +
                        "Keep the tone friendly and direct. Always reply in English. Output only the summary text, no JSON."
                    ),
                },
                {"role": "user", "content": user_content},
            ],
            max_tokens=450,
        )
        text = (resp.choices[0].message.content or "").strip()
        if text:
            # Drop "Here are some useful links" (or similar) and everything after — keep only summary
            for pat in (
                r"\n\s*Here are (?:some )?useful links[\s\S]*",
                r"\n\s*Here are [^\n]*links?[^\n]*\s*[\s\S]*",
            ):
                text = re.sub(pat, "", text, flags=re.IGNORECASE)
            if re.search(r"here are\s+(?:some\s+)?(?:useful\s+)?links?", text, re.IGNORECASE):
                idx = re.search(r"here are\s+(?:some\s+)?(?:useful\s+)?links?", text, re.IGNORECASE).start()
                text = text[:idx]
            return text.strip()
    except Exception as e:
        logger.warning("LLM summarize Serp for part failed: %s", e)
    lines = [f"Part **{part_number}** (from web search):"]
    for r in serp_results[:5]:
        title = (r.get("title") or "").strip() or "Link"
        link = (r.get("link") or "").strip()
        if link:
            lines.append(f"- [{title}]({link})")
    return "\n".join(lines)


# --- Constants and regexes for planner ---

ROUTER_APPLIANCES = ("dishwasher", "refrigerator")
# Single next_action from planner; routing maps to graph nodes: clarify→ask_clarify, part_detail→part_lookup_answer, etc.
PLANNER_NEXT_ACTIONS = ("parts_list", "part_detail", "compatibility", "clarify", "troubleshoot", "find_model_help")
# For product_info: what the user wants (drives query + clarify text)
INFO_TYPES = (
    "model_specs", "model_manual", "model_parts",
    "part_price", "part_install", "part_compatibility",
)
PLANNER_INTENTS = ("troubleshoot", "compatibility", "part_install", "product_info")
PLANNER_TOOLS = ("get_troubleshooting", "part_lookup", "search_parts", "check_compatibility")
PLANNER_SYMPTOMS = (
    "too_warm", "not_cooling", "too_cold", "ice_maker_issue",
    "not_draining", "not_filling", "will_not_start", "not_cleaning", "not_dispensing", "not_drying",
    "leaking", "noise", "general", "other",
)


# Deterministic override: rule-first so demo/behavior is stable
_PS_PART = re.compile(r"\b(PS\d{5,15})\b", re.IGNORECASE)
_MODEL_LIKE = re.compile(r"\b([A-Z][A-Z0-9\-]{5,24})\b", re.IGNORECASE)
_INSTALL_KEYWORDS = re.compile(r"\b(install|replace|how to|installation|guide)\b", re.IGNORECASE)
_MODEL_INFO_KEYWORDS = re.compile(r"\b(dimensions?|warranty|manual|specs?|installation guide)\b", re.IGNORECASE)
_PARTS_LIST_KEYWORDS = re.compile(r"\b(parts? (for|list)|what part|which part)\b", re.IGNORECASE)

# --- When we CLARIFY (ask back) vs when we ANSWER (retrieve / parts_list / part_lookup) ---
# CLARIFY: (1) Greeting only → welcome. (2) Vague problem (appliance + no specific symptom, no model/part) → which appliance + what's going on.
#          (3) Product-info intent but missing part or model → targeted ask (part number, or model number, or both). (4) Troubleshoot with symptom=general/other → same as (2).
# ANSWER:  (1) Part number in message → part_detail. (2) Model + "parts for" / specs / manual → parts_list. (3) Model + part + "fit" → compatibility. (4) Specific symptom → retrieve (RAG).

# Greeting-only: don't ask for model/part; reply with welcome
_GREETING_ONLY = re.compile(
    r"^(hi|hello|hey|howdy|greetings?|hi there|hey there|good\s+(morning|afternoon|evening)|how\s+are\s+you|what\'?s\s+up|sup|yo)[\s\.\!]*$",
    re.IGNORECASE,
)

_APPLIANCE_WORD = re.compile(r"\b(appliance|fridge|dishwasher|refrigerator|freezer)\b", re.IGNORECASE)
_VAGUE_PROBLEM = re.compile(
    r"\b(isn't working|not working|broken|problem|issue|wrong|acting up|malfunction|something wrong|"
    r"doesn't work|won't work|not running|stopped working|not working right|not working properly)\b",
    re.IGNORECASE,
)
_SAID_REFRIGERATOR = re.compile(r"\b(refrigerator|fridge|freezer|ice\s*maker)\b", re.IGNORECASE)
_SAID_DISHWASHER = re.compile(r"\b(dishwasher)\b", re.IGNORECASE)
_FAKE_PART_WORDS = {"REPLACE", "PART", "PARTS", "INSTALL", "INSTALLATION", "HOW", "IT", "COMPATIBLE", "COMPATIBILITY"}
_PRODUCT_INTENT_KEYWORDS = re.compile(
    r"\b(install|installation|replace|how to replace|replace part|install part|put in|"
    r"price|in stock|availability|buy|cost|how much|"
    r"compatible|compatibility|fit|fits|parts for|parts list|what part|which part)\b",
    re.IGNORECASE,
)


def _rule_find_model_help(msg: str) -> Optional[dict[str, Any]]:
    if re.search(r"\b(find|where|locate|how\s+to\s+find)\b", msg, re.IGNORECASE) and re.search(r"\b(model\s*number|model\s*#)\b", msg, re.IGNORECASE):
        return {"next_action": "find_model_help", "intent": "product_info"}
    return None


def _rule_install_without_model_part(msg: str) -> Optional[dict[str, Any]]:
    def _no_model_part_in_message(m: str) -> bool:
        try:
            from app.model_parser import extract_model_number, extract_part_from_messages
            return not bool((extract_model_number(m) or "").strip()) and not bool((extract_part_from_messages(m, []) or "").strip())
        except Exception:
            return True
    if _no_model_part_in_message(msg) and re.search(r"\b(install|replace|how\s+to\s+install|replace\s+(a\s+)?part)\b", msg, re.IGNORECASE):
        missing = ["part_number", "model_number"]
        return {
            "next_action": "clarify",
            "info_type": "part_install",
            "intent": "product_info",
            "missing_info": missing,
            "clarify_question": get_clarify_for_product_info("part_install", missing),
        }
    return None


def _rule_product_intent_no_model_part(msg: str) -> Optional[dict[str, Any]]:
    try:
        from app.model_parser import extract_model_number, extract_part_from_messages
        has_model_in_msg = bool((extract_model_number(msg) or "").strip())
        has_part_in_msg = bool((extract_part_from_messages(msg, []) or "").strip())
        if not has_model_in_msg and not has_part_in_msg and _PRODUCT_INTENT_KEYWORDS.search(msg):
            missing = ["part_number", "model_number"]
            return {
                "next_action": "clarify",
                "info_type": "part_install",
                "intent": "product_info",
                "missing_info": missing,
                "clarify_question": get_clarify_for_product_info("part_install", missing),
            }
    except Exception:
        pass
    return None


def _rule_compatibility(msg: str, has_model: bool) -> Optional[dict[str, Any]]:
    if has_model and re.search(r"\b(fit|compatible|compatibility|fits)\b", msg, re.IGNORECASE):
        return {"next_action": "compatibility", "intent": "compatibility"}
    return None


def _rule_part_detail(msg: str, has_part: bool) -> Optional[dict[str, Any]]:
    if _PS_PART.search(msg) or has_part:
        if _INSTALL_KEYWORDS.search(msg):
            return {"next_action": "part_detail", "info_type": "part_install", "intent": "part_install"}
        return {"next_action": "part_detail", "info_type": "part_price", "intent": "product_info"}
    return None


def _rule_parts_list(msg: str, has_model: bool, model_number: Optional[str]) -> Optional[dict[str, Any]]:
    if not has_model:
        return None
    if (
        _PARTS_LIST_KEYWORDS.search(msg)
        or "parts for" in msg.lower()
        or "parts for model" in msg.lower()
        or (re.search(r"\b(find|where|get|need|want)\b", msg, re.IGNORECASE) and re.search(r"\bpart(s)?\b", msg, re.IGNORECASE))
    ):
        return {"next_action": "parts_list", "info_type": "model_parts", "intent": "product_info"}
    if model_number and len(msg.split()) <= 10:
        model_core = model_number.replace(" ", "").lower()
        if model_core in msg.replace(" ", "").lower():
            return {"next_action": "parts_list", "info_type": "model_parts", "intent": "product_info"}
    return None


def _rule_model_specs_manual(msg: str, has_model: bool) -> Optional[dict[str, Any]]:
    if has_model and _MODEL_INFO_KEYWORDS.search(msg):
        if re.search(r"\b(manual|installation guide|guide)\b", msg, re.IGNORECASE):
            return {"next_action": "parts_list", "info_type": "model_manual", "intent": "product_info"}
        return {"next_action": "parts_list", "info_type": "model_specs", "intent": "product_info"}
    return None


def _rule_vague_problem_clarify(msg: str, has_model: bool, has_part: bool) -> Optional[dict[str, Any]]:
    if has_model or has_part:
        return None
    clarify_q = "Which appliance is it—refrigerator or dishwasher? And what's going on? For example: not cooling, not draining, leaking, not starting, not filling, or not cleaning properly?"
    if _APPLIANCE_WORD.search(msg) and _VAGUE_PROBLEM.search(msg):
        if _SAID_REFRIGERATOR.search(msg) or _SAID_DISHWASHER.search(msg):
            return None
        return {"next_action": "clarify", "intent": "troubleshoot", "clarify_question": clarify_q}
    if _APPLIANCE_WORD.search(msg) and not (_SAID_REFRIGERATOR.search(msg) or _SAID_DISHWASHER.search(msg)):
        if len(msg.split()) <= 4 and not re.search(r"\b(part|parts|model|fix|repair|drain|leak|start|cool|fill|dry|clean|noise)\b", msg, re.IGNORECASE):
            return {"next_action": "clarify", "intent": "troubleshoot", "clarify_question": clarify_q}
    return None


def deterministic_planner_override(
    message: str,
    model_number: Optional[str],
    part_number: Optional[str],
) -> Optional[dict[str, Any]]:
    """
    Rule-first: if message clearly matches a pattern, return next_action (+ info_type) and skip LLM ambiguity.
    Returns dict with next_action, info_type?, missing_info?, clarify_question?, or None to let LLM decide.
    """
    msg = (message or "").strip()
    if not msg:
        return None
    has_model = bool((model_number or "").strip())
    has_part = bool((part_number or "").strip()) and (part_number or "").strip().upper() not in _FAKE_PART_WORDS

    r = _rule_find_model_help(msg)
    if r is not None:
        return r
    r = _rule_install_without_model_part(msg)
    if r is not None:
        return r
    r = _rule_product_intent_no_model_part(msg)
    if r is not None:
        return r
    r = _rule_compatibility(msg, has_model)
    if r is not None:
        return r
    r = _rule_part_detail(msg, has_part)
    if r is not None:
        return r
    r = _rule_parts_list(msg, has_model, model_number)
    if r is not None:
        return r
    r = _rule_model_specs_manual(msg, has_model)
    if r is not None:
        return r
    r = _rule_vague_problem_clarify(msg, has_model, has_part)
    if r is not None:
        return r
    return None


def get_clarify_for_product_info(
    info_type: Optional[str],
    missing_info: List[str],
) -> Optional[str]:
    """Targeted clarify text by info_type and what's missing. Reduces back-and-forth."""
    if not missing_info:
        return None
    need_part = "part_number" in missing_info
    need_model = "model_number" in missing_info

    if need_part and need_model:
        return (
            "I can help two ways: (1) share your model number to find the right part, or "
            "(2) share the part number if you already have it (e.g. PS12345678)."
        )
    if need_part:
        return (
            "Do you have the PartSelect part number (e.g. PS12345678)? "
            "If not, tell me your model number and what part you're looking for (e.g. 'door gasket')."
        )
    if need_model:
        return (
            "What's the model number from the sticker inside the door? I'll pull the exact parts list."
        )
    return None


# --- Slot extraction, router, planner (used by agent.py and agent_graph.py) ---


def llm_extract_slots(
    message: str,
    history_user_contents: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    LLM summarizes user input into structured slots (model_number, part_number, is_symptom_only, diagnostic_state).
    Use when USE_LLM_ROUTER_PLANNER=1 so the graph gets LLM-parsed slots.
    Returns model_number, part_number, is_symptom_only, and diagnostic_state (concrete state id for RAG/scenario).
    """
    from app.agent_state import VALID_DIAGNOSTIC_STATES
    out: dict[str, Any] = {"model_number": None, "part_number": None, "is_symptom_only": None, "diagnostic_state": None}
    client = _get_client()
    if not client:
        return out
    user = (message or "").strip() or "Hello"
    if history_user_contents:
        recent = " ".join(history_user_contents[-2:]) if len(history_user_contents) > 2 else " ".join(history_user_contents)
        if recent.strip():
            user = f"[Recent context: {recent[:200]}]\nCurrent message: {user}"
    states_list = ", ".join(VALID_DIAGNOSTIC_STATES)
    try:
        resp = client.chat.completions.create(
            model=os.environ.get("PLANNER_LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract structured slots for a repair-parts chatbot. Use BOTH the current message and the recent context. "
                        "Reply with exactly one JSON object: {\"model_number\": \"...\" or null, \"part_number\": \"...\" or null, "
                        "\"is_symptom_only\": true or false, \"diagnostic_state\": \"STATE_ID\" or null}.\n"
                        "- model_number: appliance model when the user provides one in current message OR recent context. Exact string. If nowhere, null.\n"
                        "- part_number: specific part identifier (e.g. PS123) in current message OR context. Do NOT set when they only said 'parts' or 'I need parts' — null.\n"
                        "- is_symptom_only: true ONLY when the user is JUST describing a problem/symptom and is NOT asking for parts or a model. "
                        "False when they ask for 'parts for model X', or give a model/part to look up. When in doubt, false.\n"
                        "- diagnostic_state: when the user is describing a symptom, pick the ONE state that best matches. Allowed values (use exactly): "
                        + states_list
                        + ". Refrigerator: REF_S6_BOTH_WARM = both freezer and fridge warm/not cooling; REF_S7_FREEZER_COLD_FRIDGE_WARM = freezer cold but fridge warm; REF_S8_TOO_COLD = too cold; REF_S9_FALLBACK = other fridge. Dishwasher: DW_NOT_DRAINING, DW_LEAKING, DW_WILL_NOT_START, DW_NOT_CLEANING, DW_NOT_FILLING, DW_NOT_DISPENSING, DW_NOT_DRYING, DW_NOISY, DW_FALLBACK = other. Use null when they are NOT describing a symptom (e.g. asking for parts/model)."
                    ),
                },
                {"role": "user", "content": user},
            ],
            max_tokens=180,
        )
        raw = (resp.choices[0].message.content or "").strip()
        if "```" in raw:
            raw = raw.split("```")[1].replace("json", "").strip()
        data = json.loads(raw)
        mn = data.get("model_number")
        out["model_number"] = (mn.strip() or None) if isinstance(mn, str) and mn else None
        pn = data.get("part_number")
        out["part_number"] = (pn.strip() or None) if isinstance(pn, str) and pn else None
        if out["part_number"] and out["part_number"].upper() in ("PARTS", "PART"):
            out["part_number"] = None
        iso = data.get("is_symptom_only")
        if iso is not None:
            out["is_symptom_only"] = bool(iso)
        ds = data.get("diagnostic_state")
        if isinstance(ds, str) and ds.strip() and ds.strip() in VALID_DIAGNOSTIC_STATES:
            out["diagnostic_state"] = ds.strip()
    except Exception as e:
        logger.warning("LLM extract slots failed: %s", e)
    return out


def llm_router(
    message: str,
    last_assistant_message: Optional[str] = None,
) -> dict[str, str]:
    """
    Router: which graph (appliance). Returns {"appliance": "dishwasher"|"refrigerator"}.
    On failure returns {"appliance": "refrigerator"} as safe default.
    """
    out = {"appliance": "refrigerator"}
    msg = (message or "").strip().lower()
    # Deterministic: user said dishwasher → dishwasher (avoid saying "refrigerator" for a dishwasher question)
    if re.search(r"\bdishwasher\b", msg):
        return {"appliance": "dishwasher"}
    if re.search(r"\b(?:refrigerator|fridge|freezer|ice\s*maker)\b", msg):
        return {"appliance": "refrigerator"}
    client = _get_client()
    if not client:
        return out
    user = (message or "").strip() or "Hello"
    if last_assistant_message and last_assistant_message.strip():
        user = f"Previous assistant: \"{last_assistant_message.strip()[:150]}\"\nUser: {user}"
    try:
        resp = client.chat.completions.create(
            model=os.environ.get("ROUTER_LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You decide which appliance the user is asking about. "
                        "Reply with exactly one JSON object: {\"appliance\": \"dishwasher\"} or {\"appliance\": \"refrigerator\"}. "
                        "Use dishwasher for dishwashers, washing dishes, not draining/filling/cleaning/drying. "
                        "Use refrigerator for fridge, freezer, cooling, ice maker."
                    ),
                },
                {"role": "user", "content": user},
            ],
            max_tokens=30,
        )
        raw = (resp.choices[0].message.content or "").strip()
        if "```" in raw:
            raw = raw.split("```")[1].replace("json", "").strip()
        data = json.loads(raw)
        a = (data.get("appliance") or "refrigerator").lower()
        if a == "dishwasher":
            out["appliance"] = "dishwasher"
        else:
            out["appliance"] = "refrigerator"
    except Exception as e:
        logger.warning("LLM Router failed: %s", e)
    return out


def llm_planner(
    message: str,
    appliance: str,
    model_number: Optional[str] = None,
    part_number: Optional[str] = None,
    last_assistant_message: Optional[str] = None,
) -> dict[str, Any]:
    """
    Planner: single next_action + action_args + info_type (no tools list).
    Returns: next_action, action_args, info_type, intent, symptom, missing_info, clarify_question.
    Rule-first: deterministic_planner_override runs before LLM to avoid planner jitter.
    """
    has_model = bool((model_number or "").strip())
    has_part = bool((part_number or "").strip())
    out: dict[str, Any] = {
        "next_action": "troubleshoot",
        "action_args": {},
        "info_type": None,
        "intent": "troubleshoot",
        "symptom": "general",
        "missing_info": [],
        "clarify_question": None,
    }

    # 0) Greeting only: welcome back, don't ask for model/part
    msg_stripped = (message or "").strip()
    if msg_stripped and _GREETING_ONLY.match(msg_stripped):
        out["next_action"] = "clarify"
        out["clarify_question"] = (
            "Hi! I can help with appliance parts—search by part or model number, check compatibility, or describe a problem. What do you need?"
        )
        out["action_args"] = {"model_number": model_number, "part_number": part_number}
        return out

    # 1) Deterministic override: rules before LLM
    override = deterministic_planner_override(message, model_number, part_number)
    if override:
        out["next_action"] = override.get("next_action") or "troubleshoot"
        out["info_type"] = override.get("info_type")
        out["intent"] = override.get("intent") or "product_info"
        if override.get("clarify_question"):
            out["clarify_question"] = override["clarify_question"]
        # When override matched "parts for" from message, model may be in message but not in state; extract so we don't ask for it
        effective_model = (model_number or "").strip()
        if not effective_model and out.get("next_action") == "parts_list":
            from app.model_parser import extract_model_number
            effective_model = (extract_model_number(message) or "").strip()
        out["action_args"] = {"model_number": effective_model or model_number, "part_number": part_number, "limit": 10, "query_intent": out.get("info_type")}
        has_effective_model = bool(effective_model)
        # Require slots for the chosen action; else ask
        if out["next_action"] == "part_detail" and not has_part:
            out["missing_info"] = ["part_number"]
            out["clarify_question"] = get_clarify_for_product_info(out.get("info_type"), out["missing_info"])
            out["next_action"] = "clarify"
        elif out["next_action"] == "parts_list" and not has_effective_model:
            out["missing_info"] = ["model_number"]
            out["clarify_question"] = get_clarify_for_product_info(out.get("info_type"), out["missing_info"])
            out["next_action"] = "clarify"
        elif out["next_action"] == "compatibility" and (not has_model or not has_part):
            out["missing_info"] = [x for x in ("model_number", "part_number") if (x == "model_number" and not has_model) or (x == "part_number" and not has_part)]
            out["clarify_question"] = "To check compatibility I need both your appliance model number and the part number (e.g. model WDT780SAEM1 and part PS11752778)."
            out["next_action"] = "clarify"
        return out

    # 2) LLM: single next_action + action_args + info_type
    client = _get_client()
    if not client:
        return out
    user = (message or "").strip() or "Hello"
    if last_assistant_message and last_assistant_message.strip():
        user = f"Assistant asked: \"{last_assistant_message.strip()[:200]}\"\nUser replied: {user}"
    user += f"\n[Context: appliance={appliance}, has_model_number={has_model}, has_part_number={has_part}. If true, do NOT ask for part/model; output the matching next_action and use them.]"
    try:
        resp = client.chat.completions.create(
            model=os.environ.get("PLANNER_LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a planner for a repair-parts chatbot. Output exactly one JSON object with:\n"
                        "- next_action: one of parts_list, part_detail, compatibility, clarify, troubleshoot. Pick ONE.\n"
                        "  * parts_list: user wants a list of parts for a model (e.g. 'parts for model X', 'what parts for WDT780'). Need model_number.\n"
                        "  * part_detail: user wants price/specs/install for a specific part. Need part_number.\n"
                        "  * compatibility: user asks if a part fits a model. Need both model_number and part_number.\n"
                        "  * clarify: user asked about a product (price/specs/parts list) but we lack part_number and/or model_number — we must ask.\n"
                        "  * troubleshoot: user describes a malfunction (not cooling, leaking, not working, etc.); not a product-info question.\n"
                        "- info_type: only when next_action is parts_list or part_detail or we're clarifying product_info. One of model_specs, model_manual, model_parts, part_price, part_install, part_compatibility. Use model_specs for dimensions/warranty/specs; model_manual for manual/installation guide; model_parts for 'parts for model'; part_price for price/cost; part_install for how to install; part_compatibility for fits.\n"
                        "- missing_info: array of part_number and/or model_number we still need. For clarify, set what we need. For compatibility we need both.\n"
                        "- symptom: for troubleshoot only — one of too_warm, not_cooling, too_cold, ice_maker_issue, not_draining, leaking, will_not_start, not_cleaning, not_filling, not_dispensing, not_drying, noise, general, other. Use ice_maker_issue for ice maker not making ice / not dispensing; too_warm/not_cooling for fridge not cold; too_cold for freezer too cold; use general or other only when user said just 'not working' or 'broken' with no specific symptom.\n"
                        "Examples: 'WDT780SAEM1 door gasket price?' -> part_detail, part_price. 'Dishwasher not draining' -> troubleshoot, symptom not_draining. 'Ice maker isn't producing ice' -> troubleshoot, symptom ice_maker_issue. 'Fridge not cooling' -> troubleshoot, symptom not_cooling. 'My refrigerator is not working properly' (no specific symptom) -> troubleshoot, symptom general."
                    ),
                },
                {"role": "user", "content": user},
            ],
            max_tokens=220,
        )
        raw = (resp.choices[0].message.content or "").strip()
        if "```" in raw:
            raw = raw.split("```")[1].replace("json", "").strip()
        data = json.loads(raw)
        na = (data.get("next_action") or "troubleshoot").lower()
        out["next_action"] = na if na in PLANNER_NEXT_ACTIONS else "troubleshoot"
        out["info_type"] = data.get("info_type") if data.get("info_type") in INFO_TYPES else None
        out["intent"] = "product_info" if out["next_action"] in ("parts_list", "part_detail", "compatibility", "clarify") else "troubleshoot"
        if out["next_action"] == "troubleshoot":
            symptom = (data.get("symptom") or "general").lower()
            out["symptom"] = symptom if symptom in PLANNER_SYMPTOMS else "general"
        missing = data.get("missing_info") or []
        if not isinstance(missing, list):
            missing = [missing] if isinstance(missing, str) else []
        out["missing_info"] = [m for m in missing if m in ("model_number", "part_number")]
        if has_model and "model_number" in out["missing_info"]:
            out["missing_info"] = [x for x in out["missing_info"] if x != "model_number"]
        if has_part and "part_number" in out["missing_info"]:
            out["missing_info"] = [x for x in out["missing_info"] if x != "part_number"]
        if out["next_action"] == "troubleshoot":
            out["missing_info"] = [x for x in out["missing_info"] if x != "model_number"]
        out["action_args"] = {
            "model_number": model_number,
            "part_number": part_number,
            "limit": 10,
            "query_intent": out.get("info_type"),
        }
        # Targeted clarify by info_type
        if out["missing_info"]:
            cq = get_clarify_for_product_info(out.get("info_type"), out["missing_info"])
            if cq:
                out["clarify_question"] = cq
            else:
                cq = data.get("clarify_question")
                out["clarify_question"] = (cq.strip() if isinstance(cq, str) and cq.strip() else None)
        if out["next_action"] in ("parts_list", "part_detail", "compatibility") and out["missing_info"]:
            out["next_action"] = "clarify"
        # Vague problem ("not working properly") → ask which appliance + symptom only when we don't know which appliance
        if out["next_action"] == "troubleshoot" and out["symptom"] in ("general", "other") and not out["clarify_question"]:
            if (appliance or "").strip().lower() not in ("refrigerator", "dishwasher"):
                out["clarify_question"] = (
                    "Which appliance is it—refrigerator or dishwasher? And what's going on? For example: not cooling, not draining, leaking, not starting, not filling, or not cleaning properly?"
                )
        if out["next_action"] == "troubleshoot" and out["clarify_question"]:
            out["next_action"] = "clarify"
        if out["missing_info"] and not out["clarify_question"]:
            out["clarify_question"] = get_clarify_for_product_info(out.get("info_type"), out["missing_info"]) or "Which part or model? Share the part number or model number so I can look that up."
    except Exception as e:
        logger.warning("LLM Planner failed: %s", e)
    return out
