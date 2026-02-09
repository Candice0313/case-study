import json
from pathlib import Path
from pydantic import BaseModel, ConfigDict
from typing import Any, Dict, List, Optional


class ScopeContract(BaseModel):
    version: str
    domain: str
    allowed_appliance_types: List[str]
    allowed_intents: List[str]
    allowed_entities: dict
    forbidden_topics: List[str]
    part_number_pattern: str
    model_number_pattern: str
    in_scope_keywords: List[str]
    # Optional extended fields (ignored if missing)
    allowed_brands: Optional[List[str]] = None
    out_of_scope_phrases: Optional[List[str]] = None
    symptom_keywords: Optional[List[str]] = None
    clarification_triggers: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="ignore")


def load_scope_contract() -> ScopeContract:
    # From apps/api/app/ -> repo root is parents[3]
    path = Path(__file__).resolve().parents[3] / "config" / "scope_contract.json"
    with open(path) as f:
        data = json.load(f)
    return ScopeContract(**data)
