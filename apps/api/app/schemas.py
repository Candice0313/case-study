from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


# --- Compose output: claim → supporting_chunk_ids (validated in code; citations rendered programmatically) ---
class Claim(BaseModel):
    """Single claim with required supporting chunk refs; chunk_ids validated against current evidence."""
    text: str = Field(description="Claim text")
    supporting_chunk_ids: List[str] = Field(default_factory=list, description="Chunk IDs from current evidence only")


class ComposeOutput(BaseModel):
    """LLM compose output: answer + claims; citations built in code from validated chunk_ids."""
    answer: str = Field(description="Final answer text")
    claims: List[Claim] = Field(default_factory=list, description="Claims with supporting_chunk_ids")


# --- Structured evidence (normalizer output); LLM never sees raw chunks ---
class StructuredEvidence(BaseModel):
    """Normalized troubleshooting evidence. Answer generator uses only this."""
    symptom: str = Field(description="Short user-facing symptom phrase")
    likely_causes: List[str] = Field(description="At most 2 primary causes")
    difficulty: str = Field(description="easy | medium | hard")
    steps: List[str] = Field(description="Ordered steps to check or do")
    when_to_replace: Optional[str] = Field(default=None, description="When to replace a part vs repair")
    source_titles: List[str] = Field(default_factory=list, description="Guide titles for attribution (no repetition)")


class ScopeLabel(str, Enum):
    IN_SCOPE = "IN_SCOPE"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    AMBIGUOUS = "AMBIGUOUS"


class ChatMessage(BaseModel):
    role: str  # user | assistant | system
    content: str


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    history: Optional[List[ChatMessage]] = []


class ChatResponse(BaseModel):
    content: str
    scope_label: Optional[ScopeLabel] = None
    sources: Optional[List[dict]] = []
    product_cards: Optional[List[dict]] = []
