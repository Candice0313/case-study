"""
LangGraph state schema (doc §5). TypedDict for precise, observable state.
All keys optional so nodes can return partial updates.
"""
from __future__ import annotations

from typing import TypedDict


class TroubleshootingState(TypedDict, total=False):
    """State for the troubleshooting graph. Keys are optional for partial updates."""
    # Input (set at invoke)
    message: str
    scope_label: str
    last_assistant_message: str  # When set, triage can treat current message as reply (e.g. to freezer question)
    conversation_context: str | None  # Last 1–2 user messages (raw) so summary/Serp can focus on what they asked about

    # Diagnostic slots (triage + route)
    appliance: str
    appliance_type: str  # refrigerator | dishwasher
    intent: str
    has_power: bool | None
    affected_section: str
    symptom: str
    model_number: str | None
    part_number: str | None
    current_state: str

    # RAG + answer
    evidence: list
    answer: str
    citations: list
    product_cards: list  # when set by parts_list_answer (or similar), agent uses for UI cards

    # Control: when set, graph returns this as reply and skips RAG
    need_clarify: str | None

    # LLM Router+Planner (when USE_LLM_ROUTER_PLANNER=1)
    planner_next_action: str  # parts_list | part_detail | compatibility | clarify | troubleshoot
    action_args: dict  # model_number?, part_number?, limit?, query_intent? (info_type)
    info_type: str | None  # model_specs | model_manual | model_parts | part_price | part_install | part_compatibility
    planner_tools: list  # legacy; prefer planner_next_action
    planner_missing_info: list  # e.g. ["model_number"]

    # Lookup result (for part_lookup / search_parts): hit | miss | error; error_type when error
    lookup_status: str  # hit | miss | error
    error_type: str | None  # timeout | rate_limit | upstream | ...

    # Unified routing: single exit signal; set by triage, cooling_split, or llm_planner
    next_action: str  # ask_clarify | compatibility_answer | part_lookup_answer | parts_list_answer | cooling_split | retrieve
