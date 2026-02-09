"""
Symptom triage: runs before RAG. Classifies user question into primary_symptom + affected_section.
Determines which guides to retrieve and whether to ask one follow-up question first.
Optional: USE_LLM_TRIAGE=1 uses an LLM for appliance + symptom + intent (rules as fallback).
"""
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TriageResult:
    """Triage schema: symptom, section, appliance, intent. Used to filter RAG and route state."""
    primary_symptom: str  # too_warm | not_cooling | leaking | not_draining | will_not_start | ...
    affected_section: str  # freezer | refrigerator | both | unknown
    need_follow_up: bool = False
    appliance_type: str = "refrigerator"  # refrigerator | dishwasher | unknown
    intent: str = "troubleshoot"  # troubleshoot | compatibility | part_install | product_info


# Refrigerator symptoms (checked first when appliance is refrigerator/unknown)
REFRIGERATOR_SYMPTOM_RULES = [
    ("too_cold", ["freezer too cold", "freezer is too cold", "freezing too much", "freezer freezing"]),
    ("too_warm", ["too warm", "too hot", "not cold enough", "refrigerator warm", "fridge warm", "fridge hot", "refrigerator hot"]),
    ("not_cooling", ["not cooling", "not cool", "not cold", "won't cool", "not getting cold", "running but not cooling", "will not start"]),
    ("ice_maker_issue", ["ice maker", "ice dispenser", "not dispensing water", "water dispenser", "ice not working"]),
]
# Dishwasher symptoms (PartSelect taxonomy; checked first when appliance is dishwasher)
DISHWASHER_SYMPTOM_RULES = [
    ("not_draining", ["not draining", "won't drain", "drain", "draining", "water in bottom", "standing water"]),
    ("not_filling", ["not filling", "won't fill", "no water", "water not filling", "fill water"]),
    ("will_not_start", ["won't start", "will not start", "not starting", "doesn't start", "won't run"]),
    ("not_cleaning", ["not cleaning", "dishes dirty", "not cleaning properly", "cleaning poorly"]),
    ("not_dispensing", ["not dispensing detergent", "detergent", "dispenser", "soap not dispensing"]),
    ("not_drying", ["not drying", "not dry", "won't dry", "doesn't dry", "dishes wet", "drying", "still wet"]),
]
# Shared and general (last)
SHARED_SYMPTOM_RULES = [
    ("leaking", ["leak", "leaking", "water on floor", "water leaking", "puddle"]),
    ("noise", ["noisy", "noise", "loud", "humming", "buzzing", "rattling", "knocking"]),
    ("general", ["not working", "isn't working", "not working right", "isn't working right", "not working properly", "broken", "malfunction", "something wrong", "doesn't work", "won't work", "not running", "stopped working"]),
]

APPLIANCE_DISHWASHER = ["dishwasher", "dish washer", "dishwashing"]
APPLIANCE_REFRIGERATOR = ["refrigerator", "fridge", "freezer", "ice maker"]

# Only set affected_section when user *explicitly* says which part. Otherwise keep "unknown"
# so we ask: "Is the freezer also not cold, or is only the refrigerator section warm?"
# (Avoid treating "my refrigerator is not cooling" or "fridge too warm" as section=refrigerator.)
SECTION_BOTH = ["both", "entire", "whole unit", "whole fridge", "entire refrigerator", "both sections"]
SECTION_FREEZER = ["freezer", "freezer section", "freezer is", "freezer's", "freezer compartment", "only the freezer", "only freezer"]
SECTION_REFRIGERATOR = ["refrigerator section", "fridge section", "fresh food", "only the fridge", "only the refrigerator", "only fridge", "only refrigerator", "fridge compartment", "refrigerator compartment"]

# Exact second-triage question (highest information gain for not cooling)
TRIAGE_FOLLOW_UP_WARM = "Is the freezer cold, or is it also not cooling?"

# Keywords to parse user's reply to the freezer question → affected_section
_FREEZER_OK_REPLY = [
    "freezer is cold", "freezer cold", "freezer works", "freezer's fine", "freezer fine",
    "only fridge", "only refrigerator", "only the fridge", "only the refrigerator",
    "fridge only", "refrigerator only", "just the fridge", "just fridge",
    "freezer good", "freezer ok", "freezer okay",
]
_BOTH_WARM_REPLY = [
    "both", "both warm", "both not cold", "both sections", "neither", "neither cold",
    "freezer also not", "freezer also warm", "freezer not cold", "freezer not cooling",
    "freezer too warm", "freezer warm", "also not cooling", "also not cold",
    "both not cooling", "entire", "whole unit",
]


def parse_section_from_freezer_reply(message: str) -> Optional[str]:
    """
    Parse the user's reply to "Is the freezer cold, or is it also not cooling?"
    Returns "refrigerator" (freezer cold, fridge warm) | "both" (both warm) | None.
    """
    msg_lower = (message or "").strip().lower()
    if not msg_lower:
        return None
    if any(k in msg_lower for k in _BOTH_WARM_REPLY):
        return "both"
    if any(k in msg_lower for k in _FREEZER_OK_REPLY):
        return "refrigerator"
    return None


def _is_freezer_question(msg: Optional[str]) -> bool:
    """True if this looks like our second-triage freezer question."""
    if not msg:
        return False
    m = msg.strip().lower()
    return "freezer cold" in m and ("also not cooling" in m or "also not cold" in m)


def _detect_appliance(msg_lower: str) -> str:
    if any(k in msg_lower for k in APPLIANCE_DISHWASHER):
        return "dishwasher"
    if any(k in msg_lower for k in APPLIANCE_REFRIGERATOR):
        return "refrigerator"
    return "unknown"


# Valid values for LLM triage (must match route_path / source_policy)
TRIAGE_SYMPTOMS = (
    "too_warm", "not_cooling", "too_cold", "ice_maker_issue",
    "not_draining", "not_filling", "will_not_start", "not_cleaning", "not_dispensing", "not_drying",
    "leaking", "noise", "general", "other",
)
TRIAGE_INTENTS = ("troubleshoot", "compatibility", "part_install", "product_info")


def _is_vague_symptom_message(msg_lower: str) -> bool:
    """True if message mentions appliance + vague problem (not working, broken) so we should ask clarify."""
    if not msg_lower:
        return False
    has_appliance = any(a in msg_lower for a in ["refrigerator", "fridge", "dishwasher", "appliance"])
    has_vague = any(
        v in msg_lower for v in [
            "not working properly", "not working", "isn't working", "broken", "doesn't work", "won't work",
            "not working right", "something wrong", "malfunction",
        ]
    )
    return has_appliance and has_vague


def _is_symptom_clarify_reply(last_assistant: Optional[str], message: str) -> bool:
    """True if we just asked for symptom and user gave a short reply (e.g. 'not dry', 'leaking')."""
    if not last_assistant or not (message or "").strip():
        return False
    last = last_assistant.strip().lower()
    is_our_question = (
        "what's going on" in last or "to narrow this down" in last
        or "which section" in last or "freezer cold" in last
    )
    short_reply = len((message or "").split()) <= 8
    return is_our_question and short_reply


def _triage_with_llm(
    message: str,
    last_assistant_message: Optional[str] = None,
) -> Optional[TriageResult]:
    """
    LLM triage: appliance_type, primary_symptom, intent. Optional context from previous bot message.
    Returns None if disabled, API error, or invalid output (caller uses rules).
    """
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key or key == "sk-...":
        return None
    user_content = (message or "").strip() or "Hello"
    if last_assistant_message and last_assistant_message.strip():
        user_content = (
            "We asked the user: \""
            + last_assistant_message.strip()[:200]
            + "\"\nUser replied: "
            + user_content
        )
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model=os.environ.get("TRIAGE_LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You classify the user's message about an appliance (refrigerator or dishwasher). "
                        "If we asked 'What's going on with your dishwasher?' and they said 'not dry', reply with appliance_type dishwasher, primary_symptom not_drying. "
                        "Map short replies to the right symptom: not dry/not drying -> not_drying, leak/leaking -> leaking, won't start -> will_not_start, not draining -> not_draining, not cleaning -> not_cleaning, not filling -> not_filling, noisy/noise -> noise. "
                        "Reply with a single JSON object with these exact keys: "
                        "appliance_type (refrigerator|dishwasher|unknown), "
                        "primary_symptom (one of: too_warm, not_cooling, too_cold, ice_maker_issue, not_draining, not_filling, will_not_start, not_cleaning, not_dispensing, not_drying, leaking, noise, general, other), "
                        "intent (troubleshoot|compatibility|part_install|product_info). "
                        "Use 'general' only when they don't specify a symptom; use 'other' only when not about repair/parts."
                    ),
                },
                {"role": "user", "content": user_content},
            ],
            max_tokens=120,
        )
        raw = (resp.choices[0].message.content or "").strip()
        # Allow markdown code block
        if "```" in raw:
            raw = raw.split("```")[1].replace("json", "").strip()
        data = json.loads(raw)
        at = (data.get("appliance_type") or "unknown").lower()
        if at not in ("refrigerator", "dishwasher", "unknown"):
            at = "refrigerator"
        symptom = (data.get("primary_symptom") or "other").lower()
        if symptom not in TRIAGE_SYMPTOMS:
            symptom = "general" if at != "unknown" else "other"
        intent = (data.get("intent") or "troubleshoot").lower()
        if intent not in TRIAGE_INTENTS:
            intent = "troubleshoot"
        need_follow_up = (
            symptom == "general"
            or (symptom in ("too_warm", "not_cooling") and at == "refrigerator")
        )
        return TriageResult(
            primary_symptom=symptom,
            affected_section="unknown",
            need_follow_up=need_follow_up,
            appliance_type=at if at != "unknown" else "refrigerator",
            intent=intent,
        )
    except Exception as e:
        logger.warning("Triage LLM failed: %s", e)
    return None


def triage(message: str, last_assistant_message: Optional[str] = None) -> TriageResult:
    """
    Classify user question: appliance_type, primary_symptom, affected_section, intent.
    If last_assistant_message was the freezer question, treat current message as section reply.
    When USE_LLM_TRIAGE=1, use LLM first; on failure or missing key use rules.
    """
    msg_lower = (message or "").strip().lower()
    if not msg_lower:
        return TriageResult(primary_symptom="other", affected_section="unknown", appliance_type="unknown")

    # Reply to "Is the freezer cold, or is it also not cooling?" → rules only (conversation context)
    if last_assistant_message and _is_freezer_question(last_assistant_message):
        section = parse_section_from_freezer_reply(message)
        if section is not None:
            return TriageResult(
                primary_symptom="not_cooling",
                affected_section=section,
                need_follow_up=False,
                appliance_type="refrigerator",
            )

    # LLM triage: when enabled globally, or when user is replying to our symptom question with a short answer (e.g. "not dry")
    use_llm = os.environ.get("USE_LLM_TRIAGE", "").lower() in ("1", "true", "yes")
    if use_llm or _is_symptom_clarify_reply(last_assistant_message, message):
        llm_result = _triage_with_llm(message, last_assistant_message)
        if llm_result is not None:
            # Rule override: message clearly about ice maker → ice_maker_issue so we get REF_ICE_MAKER + preferred guide
            if (llm_result.appliance_type == "refrigerator"
                    and any(k in msg_lower for k in ["ice maker", "icemaker", "ice dispenser", "not dispensing water"])):
                llm_result = TriageResult(
                    primary_symptom="ice_maker_issue",
                    affected_section=llm_result.affected_section,
                    need_follow_up=False,
                    appliance_type=llm_result.appliance_type,
                    intent=llm_result.intent,
                )
            # Fallback: vague message must get clarify even if LLM returned specific symptom
            elif not llm_result.need_follow_up and _is_vague_symptom_message(msg_lower):
                llm_result = TriageResult(
                    primary_symptom="general",
                    affected_section=llm_result.affected_section,
                    need_follow_up=True,
                    appliance_type=llm_result.appliance_type,
                    intent=llm_result.intent,
                )
            return llm_result

    appliance_type = _detect_appliance(msg_lower)
    if appliance_type == "unknown":
        appliance_type = "refrigerator"

    # Symptom rules: appliance-specific first so e.g. "dishwasher won't start" → will_not_start
    if appliance_type == "dishwasher":
        rules_order = DISHWASHER_SYMPTOM_RULES + REFRIGERATOR_SYMPTOM_RULES + SHARED_SYMPTOM_RULES
    else:
        rules_order = REFRIGERATOR_SYMPTOM_RULES + DISHWASHER_SYMPTOM_RULES + SHARED_SYMPTOM_RULES

    primary_symptom = "other"
    for tag, keywords in rules_order:
        if any(k in msg_lower for k in keywords):
            primary_symptom = tag
            break

    affected_section = "unknown"
    if any(s in msg_lower for s in SECTION_BOTH):
        affected_section = "both"
    elif any(s in msg_lower for s in SECTION_FREEZER):
        affected_section = "freezer"
    elif any(s in msg_lower for s in SECTION_REFRIGERATOR):
        affected_section = "refrigerator"

    # Align with planner: vague symptom (general) or fridge cooling + unknown section → clarify (which appliance / what's going on).
    need_follow_up = (
        (primary_symptom in ("too_warm", "not_cooling") and affected_section == "unknown" and appliance_type == "refrigerator")
        or primary_symptom == "general"
    )
    if primary_symptom == "general" and appliance_type == "dishwasher":
        need_follow_up = True
    # Fallback: message clearly vague ("not working properly", "broken") + appliance → always ask clarify
    if not need_follow_up and _is_vague_symptom_message(msg_lower):
        need_follow_up = True
        primary_symptom = "general"

    # Intent: deterministic tools first (deep-research-report)
    intent = "troubleshoot"
    if any(k in msg_lower for k in ["compatible", "compatibility", "fit", "fits", "will this fit", "work with my model", "fits my"]):
        intent = "compatibility"
    elif any(k in msg_lower for k in ["install", "installation", "how to replace", "replace part", "install part", "put in"]):
        intent = "part_install"
    elif any(k in msg_lower for k in ["price", "in stock", "availability", "buy", "cost", "how much"]):
        intent = "product_info"

    return TriageResult(
        primary_symptom=primary_symptom,
        affected_section=affected_section,
        need_follow_up=need_follow_up,
        appliance_type=appliance_type,
        intent=intent,
    )


TRIAGE_FOLLOW_UP_GENERAL = (
    "To narrow this down, what's going on? For example: not cooling, too warm, leaking, making noise, "
    "or ice/water dispenser not working?"
)
TRIAGE_FOLLOW_UP_DISHWASHER = (
    "What's going on with your dishwasher? For example: not draining, leaking, not starting, "
    "not cleaning properly, not filling with water, or not drying?"
)


def get_triage_follow_up(triage_result: TriageResult) -> Optional[str]:
    """Return the one follow-up question when triage says we need it, else None."""
    if not triage_result.need_follow_up:
        return None
    if triage_result.primary_symptom == "general":
        if getattr(triage_result, "appliance_type", "") == "dishwasher":
            return TRIAGE_FOLLOW_UP_DISHWASHER
        return TRIAGE_FOLLOW_UP_GENERAL
    if triage_result.primary_symptom in ("too_warm", "not_cooling"):
        return TRIAGE_FOLLOW_UP_WARM
    return None
