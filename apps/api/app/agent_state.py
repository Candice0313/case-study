"""
Agent state schema and state-machine routing (LangGraph-style, no LangGraph dep).
State controls which sources RAG is allowed to retrieve; RAG is a tool, not control logic.

Unified routing: next_action is the single graph exit signal; triage and planner
both fill the same logical "plan" shape (TriagePlan); route nodes only switch(next_action).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

from app.triage import TriageResult

# --- Unified routing: single enum for "where to go next" ---
NextAction = str  # one of: ask_clarify | compatibility_answer | part_lookup_answer | parts_list_answer | cooling_split | retrieve


def _current_message_asks_for_parts(state: dict[str, Any]) -> bool:
    """
    True only if the current turn message clearly asks for parts/model (not just a symptom).
    Prevents routing to parts_list_answer when user said e.g. "refrigerator not working" and
    model_number came from history — we should go to retrieve (troubleshoot) instead.
    """
    message = (state.get("message") or "").strip().lower()
    if not message:
        return False
    # If message is purely symptom-like (state0), never treat as "asking for parts"
    if _message_is_symptom_only(message, state):
        return False
    # Explicit parts intent in current message
    if "parts for" in message or "need parts" in message or "want parts" in message or "find parts" in message:
        return True
    # "Where can I find ice maker part" / "find ... part" / "where ... part"
    if re.search(r"\b(find|where|get|need|want)\b", message) and re.search(r"\bpart(s)?\b", message):
        return True
    # Model number appears in current message (user is asking about that model)
    model_number = (state.get("model_number") or "").strip()
    if model_number and model_number.lower() in message:
        return True
    # Standalone model-like token in message (e.g. user replied "WRX986SIHZ00")
    if message and len(message) <= 30 and model_number:
        # Very short message that could be just the model number
        return message.replace(" ", "") == model_number.lower().replace(" ", "")
    return False


def _message_is_symptom_only(message: str, state: dict[str, Any]) -> bool:
    """
    True when the message is only describing a problem (state0 / troubleshooting), not asking for parts.
    When state has is_symptom_only from LLM (llm_extract_slots), use that; else use keyword rules.
    """
    if state.get("is_symptom_only") is not None:
        return bool(state["is_symptom_only"])
    if not (message or "").strip():
        return False
    # Normalize so "isnt" / "isn't" / "isn't" (curly) all match
    m = message.strip().lower().replace("\u2019", "'")
    m = m.replace("isnt ", "isn't ").replace("isnt", "isn't")
    # Must mention appliance or appliance part (ice maker = refrigerator)
    if not (
        "refrigerator" in m or "fridge" in m or "freezer" in m or "dishwasher" in m
        or "ice maker" in m or "icemaker" in m
    ):
        return False
    # Must look like a problem description (no parts intent)
    if "parts for" in m or "need parts" in m or "want parts" in m or "find parts" in m:
        return False
    model_number = (state.get("model_number") or "").strip()
    if model_number and model_number.lower() in m:
        return False  # user mentioned a model → not symptom-only
    symptom_phrases = (
        "not working", "isn't working", "isnt working", "not working right", "broken", "issue", "problem",
        "not cooling", "not cold", "too warm", "is warm", "is cold", "cold but", "warm but",
        "leak", "leaking", "not draining", "not filling",
        "won't start", "wont start", "not starting", "not cleaning", "not drying", "noise", "noisy",
        "isn't producing", "not producing", "producing any ice", "no ice", "not making ice", "not dispensing",
    )
    if any(p in m for p in symptom_phrases):
        return True
    return False


def _message_asks_find_model_number(message: str) -> bool:
    """True if user is asking where/how to find their model number (scope_router already treats this as IN_SCOPE)."""
    if not (message or "").strip():
        return False
    m = message.strip().lower()
    if not ("model" in m and ("find" in m or "where" in m or "locate" in m or "how" in m)):
        return False
    return "model number" in m or "model #" in m


def _next_action_after_triage(state: dict[str, Any]) -> NextAction:
    """Rule path after triage: need_clarify → ask; find_model_help; compatibility/part when slots present; else cooling_split."""
    if state.get("need_clarify"):
        return "ask_clarify"
    message = (state.get("message") or "").strip()
    intent = (state.get("intent") or "troubleshoot").strip()
    model_number = (state.get("model_number") or "").strip()
    part_number = (state.get("part_number") or "").strip()
    # "Where can I find my model number?" → find_model_help (same as planner path)
    if _message_asks_find_model_number(message):
        return "find_model_help"
    if intent == "compatibility" and model_number and part_number:
        return "compatibility_answer"
    # Only go to parts_list when current message asks for parts/model — not when model is from history only
    if intent in ("part_install", "product_info") and model_number and not part_number:
        if _current_message_asks_for_parts(state):
            return "parts_list_answer"
        return "cooling_split"
    if intent in ("part_install", "product_info") and part_number:
        return "part_lookup_answer"
    return "cooling_split"


def _next_action_after_cooling_split(state: dict[str, Any]) -> NextAction:
    """Rule path after cooling_split: need_clarify → ask else retrieve."""
    if state.get("need_clarify"):
        return "ask_clarify"
    return "retrieve"


def _next_action_after_planner(state: dict[str, Any]) -> NextAction:
    """Planner path: need_clarify wins; then state0 (symptom-only) → retrieve; then planner_next_action."""
    if state.get("need_clarify"):
        return "ask_clarify"
    message = (state.get("message") or "").strip()
    # State0: purely symptom message → always retrieve, never parts_list (overrides planner)
    if _message_is_symptom_only(message, state):
        return "retrieve"
    na = (state.get("planner_next_action") or "").strip()
    if na == "clarify":
        return "ask_clarify"
    if na == "compatibility":
        return "compatibility_answer"
    if na == "parts_list":
        if _current_message_asks_for_parts(state):
            return "parts_list_answer"
        return "retrieve"
    if na == "part_detail":
        return "part_lookup_answer"
    if na == "troubleshoot":
        return "retrieve"
    if na == "find_model_help":
        return "find_model_help"
    # Legacy: planner_tools
    tools = state.get("planner_tools") or []
    model_number = (state.get("model_number") or "").strip()
    part_number = (state.get("part_number") or "").strip()
    if "check_compatibility" in tools and model_number and part_number:
        return "compatibility_answer"
    if "search_parts" in tools and model_number and _current_message_asks_for_parts(state):
        return "parts_list_answer"
    if "part_lookup" in tools and part_number:
        return "part_lookup_answer"
    return "retrieve"


def compute_next_action(state: dict[str, Any], phase: str) -> NextAction:
    """
    Single entry point for route decision. phase in ("after_triage", "after_cooling_split", "after_planner").
    Route nodes should only do: switch(state["next_action"]) or switch(compute_next_action(state, phase)).
    """
    if phase == "after_triage":
        return _next_action_after_triage(state)
    if phase == "after_cooling_split":
        return _next_action_after_cooling_split(state)
    if phase == "after_planner":
        return _next_action_after_planner(state)
    return "retrieve"


# --- TriagePlan: shared schema for triage (rules) and planner (LLM). Both paths fill same shape. ---
class TriagePlan(TypedDict, total=False):
    """Standardized plan output: appliance_type, intent, symptom_label, affected_section, missing_info, clarify_question, current_state, next_action."""
    appliance_type: str
    intent: str
    symptom_label: str
    affected_section: str
    missing_info: list
    clarify_question: str | None
    current_state: str
    next_action: str


# Diagnostic states: REF = refrigerator, DW = dishwasher
# Used for RAG filtering (source_policy) and scenario hints; LLM can output one of these as diagnostic_state.
VALID_DIAGNOSTIC_STATES = (
    "REF_S6_BOTH_WARM",
    "REF_S7_FREEZER_COLD_FRIDGE_WARM",
    "REF_S8_TOO_COLD",
    "REF_ICE_MAKER",
    "REF_S9_FALLBACK",
    "DW_NOT_DRAINING",
    "DW_LEAKING",
    "DW_WILL_NOT_START",
    "DW_NOT_CLEANING",
    "DW_NOT_FILLING",
    "DW_NOT_DISPENSING",
    "DW_NOT_DRYING",
    "DW_NOISY",
    "DW_FALLBACK",
)


class DiagnosticState:
    S0_SCOPE = "S0_SCOPE"
    S1_TRIAGE = "S1_TRIAGE"
    S3_COOLING_SPLIT = "S3_COOLING_SPLIT"
    # Refrigerator
    S6_BOTH_WARM = "REF_S6_BOTH_WARM"
    S7_FREEZER_COLD_FRIDGE_WARM = "REF_S7_FREEZER_COLD_FRIDGE_WARM"
    S8_TOO_COLD = "REF_S8_TOO_COLD"
    REF_ICE_MAKER = "REF_ICE_MAKER"  # ice maker / not dispensing water → RAG only ice_maker_issue chunks
    S9_FALLBACK = "REF_S9_FALLBACK"
    # Dishwasher
    DW_NOT_DRAINING = "DW_NOT_DRAINING"
    DW_LEAKING = "DW_LEAKING"
    DW_WILL_NOT_START = "DW_WILL_NOT_START"
    DW_NOT_CLEANING = "DW_NOT_CLEANING"
    DW_NOT_FILLING = "DW_NOT_FILLING"
    DW_NOT_DISPENSING = "DW_NOT_DISPENSING"
    DW_NOT_DRYING = "DW_NOT_DRYING"
    DW_NOISY = "DW_NOISY"
    DW_FALLBACK = "DW_FALLBACK"
    #
    RETRIEVE = "RETRIEVE"
    NORMALIZE = "NORMALIZE"
    ANSWER = "ANSWER"
    CITATION_CHECK = "CITATION_CHECK"
    END = "END"


# Phase B: deterministic symptom_label → current_state (no LLM). Versionable, testable.
# LLM only outputs symptom_label (Phase A); this table is the single source of truth.
DW_SYMPTOM_TO_STATE: dict[str, str] = {
    "not_draining": DiagnosticState.DW_NOT_DRAINING,
    "leaking": DiagnosticState.DW_LEAKING,
    "will_not_start": DiagnosticState.DW_WILL_NOT_START,
    "not_cleaning": DiagnosticState.DW_NOT_CLEANING,
    "not_filling": DiagnosticState.DW_NOT_FILLING,
    "not_dispensing": DiagnosticState.DW_NOT_DISPENSING,
    "not_drying": DiagnosticState.DW_NOT_DRYING,
    "noise": DiagnosticState.DW_NOISY,
}
# Refrigerator: section-independent first; cooling split (too_warm/not_cooling) uses section in route_path.
REF_SYMPTOM_TO_STATE: dict[str, str] = {
    "too_cold": DiagnosticState.S8_TOO_COLD,
    "ice_maker_issue": DiagnosticState.REF_ICE_MAKER,
}


@dataclass
class AgentState:
    """Schema aligned with doc §5. Filled by scope_router → triage → route_path → retrieve → normalize → compose → citation_gate."""
    appliance: str = "refrigerator"
    intent: str = "troubleshoot"
    has_power: bool | None = None
    affected_section: str = "unknown"
    symptom: str = "unknown"
    model_number: str | None = None
    current_state: str = DiagnosticState.S1_TRIAGE
    evidence: list[dict] = field(default_factory=list)
    answer: str = ""
    citations: list[dict] = field(default_factory=list)
    # For API response
    product_cards: list[dict] = field(default_factory=list)
    need_clarify: str | None = None  # If set, return this as message and do not RAG


def _config_dir() -> Path | None:
    """Directory containing source_policy.json (same dir for state_guide_links.json)."""
    api_dir = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[3]
    for base in (repo_root, api_dir):
        if (base / "config" / "source_policy.json").exists():
            return base / "config"
    return None


def _load_policy() -> dict[str, Any]:
    config_dir = _config_dir()
    if config_dir is None:
        return {}
    path = config_dir / "source_policy.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


_source_policy: dict[str, Any] | None = None


def get_source_policy(state_id: str) -> dict[str, Any]:
    """Return RAG filter policy for state (allowed_symptom_tags, forbidden_symptom_tags, appliance, adjacent_states)."""
    global _source_policy
    if _source_policy is None:
        _source_policy = _load_policy()
    return _source_policy.get(state_id, {})


def _load_state_guide_links() -> dict[str, Any]:
    """Load state_id -> {url, title} from same config dir as source_policy.json."""
    config_dir = _config_dir()
    if config_dir is None:
        return {}
    path = config_dir / "state_guide_links.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


_state_guide_links: dict[str, Any] | None = None


def get_state_guide_link(state_id: str) -> dict[str, str] | None:
    """Return preferred guide link {url, title} for this state, or None. Config: config/state_guide_links.json."""
    global _state_guide_links
    if _state_guide_links is None:
        _state_guide_links = _load_state_guide_links()
    entry = _state_guide_links.get(state_id)
    if not entry or not isinstance(entry, dict):
        return None
    url = (entry.get("url") or "").strip()
    title = (entry.get("title") or "").strip()
    if url and title:
        return {"url": url, "title": title}
    return None


def symptom_label_to_state(
    appliance_type: str,
    symptom_label: str,
    section: str = "unknown",
) -> str:
    """
    Phase B: symptom_label → current_state via tables only (no LLM). Versionable, testable.
    LLM only outputs symptom_label (Phase A); this is the single source of truth for mapping.
    """
    appliance = (appliance_type or "refrigerator").lower()
    label = (symptom_label or "other").lower()

    if appliance == "dishwasher":
        return DW_SYMPTOM_TO_STATE.get(label, DiagnosticState.DW_FALLBACK)

    if label in REF_SYMPTOM_TO_STATE:
        return REF_SYMPTOM_TO_STATE[label]
    if label in ("too_warm", "not_cooling"):
        if section == "both":
            return DiagnosticState.S6_BOTH_WARM
        if section == "refrigerator":
            return DiagnosticState.S7_FREEZER_COLD_FRIDGE_WARM
        if section == "freezer":
            return DiagnosticState.S6_BOTH_WARM
        return DiagnosticState.S9_FALLBACK
    return DiagnosticState.S9_FALLBACK


def route_path(triage_result: TriageResult) -> str:
    """
    Map triage (appliance + symptom + section) to diagnostic state.
    Refrigerator: cooling split S6/S7/S8/S9. Dishwasher: symptom → DW_*.
    """
    return symptom_label_to_state(
        getattr(triage_result, "appliance_type", "refrigerator"),
        triage_result.primary_symptom,
        triage_result.affected_section,
    )


def state_id_from_appliance_symptom(appliance: str, symptom: str) -> str:
    """Phase B wrapper for Planner path (section=unknown). Use symptom_label_to_state for direct call."""
    return symptom_label_to_state(appliance or "refrigerator", symptom or "other", "unknown")
