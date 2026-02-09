"""
Scope Router: IN_SCOPE vs OUT_OF_SCOPE (and AMBIGUOUS → agent redirects).
- IN_SCOPE: run graph (planner/triage → clarify or answer). OUT_OF_SCOPE/AMBIGUOUS: return redirect, no graph.
- Clear in-scope: part/model pattern, symptom_keywords, 2+ in_scope_keywords, or follow-up in an in-scope conversation.
- When ambiguous: use LLM if available (default when OPENAI_API_KEY set); else lenient for 1 keyword + longer messages.
"""
import json
import logging
import os
import re
from app.config import load_scope_contract
from app.schemas import ScopeLabel

logger = logging.getLogger(__name__)

# Phrases that indicate the last assistant reply was in-scope (troubleshooting, parts, repair)
_LAST_IN_SCOPE_HINTS = (
    "partselect", "repair", "troubleshoot", "model number", "part number",
    "refrigerator", "dishwasher", "parts list", "compatible", "view parts",
    "suggested links", "suggested parts", "cooling", "leak", "drain", "not cooling",
)


def _classify_scope_llm(message: str) -> ScopeLabel | None:
    """
    LLM classification: decide IN_SCOPE vs OUT_OF_SCOPE.
    Returns None if disabled or API error (caller falls back to rules).
    """
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key or key == "sk-...":
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model=os.environ.get("SCOPE_LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "This agent only provides product information and assists with customer transactions: parts lookup, compatibility, installation, troubleshooting, orders for refrigerators and dishwashers. "
                        "Reply with exactly one word: IN_SCOPE or OUT_OF_SCOPE. "
                        "IN_SCOPE only if the user is clearly asking about fridge/dishwasher parts, repair, installation, compatibility, or an appliance problem. "
                        "OUT_OF_SCOPE for anything else: weather, jokes, general knowledge, other products, coding, medical, legal, politics, or any topic unrelated to appliance parts and support."
                    ),
                },
                {"role": "user", "content": (message or "").strip() or "Hello"},
            ],
            max_tokens=10,
        )
        text = (resp.choices[0].message.content or "").strip().upper()
        if "IN_SCOPE" in text:
            return ScopeLabel.IN_SCOPE
        if "OUT_OF_SCOPE" in text:
            return ScopeLabel.OUT_OF_SCOPE
    except Exception as e:
        logger.warning("Scope LLM fallback failed: %s", e)
    return None


def _last_reply_looks_in_scope(last_assistant_message: str) -> bool:
    """True if the last assistant message looks like troubleshooting/parts/repair (so follow-ups are in-scope)."""
    if not (last_assistant_message or "").strip():
        return False
    lower = last_assistant_message.strip().lower()
    return any(hint in lower for hint in _LAST_IN_SCOPE_HINTS)


def classify_scope(
    message: str,
    use_llm_fallback: bool | None = None,
    last_assistant_message: str | None = None,
) -> ScopeLabel:
    """
    IN_SCOPE vs OUT_OF_SCOPE. When last_assistant_message is provided and looks like an in-scope
    reply (e.g. troubleshooting), treat follow-up user messages as IN_SCOPE if they contain any
    in-scope keyword or look like a short clarification (conversation continuity).
    """
    contract = load_scope_contract()
    msg_lower = (message or "").strip().lower()
    words = msg_lower.split()
    # Default: use LLM when API key is set (smarter for edge cases); override with USE_LLM_SCOPE=0 to disable
    has_key = bool((os.environ.get("OPENAI_API_KEY") or "").strip() and os.environ.get("OPENAI_API_KEY", "").strip() != "sk-...")
    use_llm = use_llm_fallback if use_llm_fallback is not None else (
        os.environ.get("USE_LLM_SCOPE", "1" if has_key else "").lower() in ("1", "true", "yes")
    )

    # Hard OUT_OF_SCOPE: exact phrases and forbidden topics (so we never treat them as in-scope)
    if contract.out_of_scope_phrases:
        for phrase in contract.out_of_scope_phrases:
            if phrase.lower() in msg_lower:
                return ScopeLabel.OUT_OF_SCOPE
    for topic in contract.forbidden_topics:
        if topic in msg_lower and _looks_like_off_topic(message, topic):
            return ScopeLabel.OUT_OF_SCOPE

    # Clear IN_SCOPE signals first: if message obviously asks for parts/troubleshooting, don't let LLM override
    if re.search(contract.part_number_pattern, message, re.IGNORECASE):
        return ScopeLabel.IN_SCOPE
    if re.search(contract.model_number_pattern, message, re.IGNORECASE):
        return ScopeLabel.IN_SCOPE
    # Model number anywhere: letter-starting (LFX28968ST) or digit-starting (103200, 103200 Admiral)
    if re.search(r"\b[A-Z0-9][A-Z0-9\-]{4,24}\b", message):
        return ScopeLabel.IN_SCOPE
    # Appliance + number (e.g. "refrigerator 103200") → likely model number
    if re.search(r"\b(refrigerator|fridge|dishwasher)\b", msg_lower) and re.search(r"\b\d{4,8}\b", message):
        return ScopeLabel.IN_SCOPE
    # "Where can I find my model number" / "how to find model number" → in-scope (we answer with PartSelect guides)
    if re.search(r"\b(find|where|locate|how\s+to\s+find)\b", msg_lower) and re.search(r"\b(model\s*number|model\s*#)\b", msg_lower):
        return ScopeLabel.IN_SCOPE
    # Normalize apostrophes so "isn't" / "isn't" (curly) both match
    msg_normalized = msg_lower.replace("\u2019", "'").replace("'", "'")
    if contract.symptom_keywords:
        for k in contract.symptom_keywords:
            k_norm = (k or "").replace("\u2019", "'").replace("'", "'")
            if k_norm and k_norm in msg_normalized:
                return ScopeLabel.IN_SCOPE
    # Fallback: "appliance" + problem wording (catches "My appliance isn't working right" even if JSON missed)
    if "appliance" in msg_lower and any(
        w in msg_lower for w in ("working", "broken", "problem", "issue", "fix", "repair", "right")
    ):
        return ScopeLabel.IN_SCOPE
    keyword_hits = sum(1 for k in contract.in_scope_keywords if k in msg_lower)
    if keyword_hits >= 2:
        return ScopeLabel.IN_SCOPE

    # Conversation continuity: if last assistant reply was in-scope (troubleshooting/parts), treat follow-up as in-scope
    # when the user message has at least one in-scope keyword or is a short clarification (3–40 words)
    if last_assistant_message is not None and _last_reply_looks_in_scope(last_assistant_message):
        if keyword_hits >= 1:
            return ScopeLabel.IN_SCOPE
        if 3 <= len(words) <= 40 and any(
            w in msg_lower for w in ("freezer", "fridge", "refrigerator", "cold", "warm", "cooling", "leak", "drain", "part", "model")
        ):
            return ScopeLabel.IN_SCOPE

    # Ambiguous or single keyword: try LLM when enabled (smarter than hard OUT)
    if use_llm:
        llm_label = _classify_scope_llm(message)
        if llm_label is not None:
            return llm_label

    # Lenient for 1 keyword + longer message (likely follow-up we didn't match above): treat as IN_SCOPE
    if keyword_hits >= 1 and len(words) >= 3:
        return ScopeLabel.IN_SCOPE
    if keyword_hits >= 1:
        return ScopeLabel.OUT_OF_SCOPE

    # Very short or generic → deflect
    min_words = 2
    if contract.clarification_triggers and "min_words" in contract.clarification_triggers:
        min_words = contract.clarification_triggers["min_words"]
    if len(words) <= min_words:
        return ScopeLabel.OUT_OF_SCOPE

    # Default: no in-scope signal → deflect
    return ScopeLabel.OUT_OF_SCOPE


def _looks_like_off_topic(message: str, topic: str) -> bool:
    """Avoid false positives: 'part' in 'party', 'washer' in 'dishwasher'. Match whole words only."""
    msg_lower = message.lower()
    words = set(msg_lower.split())
    if topic in words:
        return True
    # Substring only if bounded by non-alnum (so "washer" does not match "dishwasher")
    pattern = r"(^|[^\w])" + re.escape(topic) + r"([^\w]|$)"
    return bool(re.search(pattern, msg_lower))
