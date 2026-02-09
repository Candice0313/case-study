"""
Extract appliance model number from user message (and optionally from conversation).
Used to enable model-specific steps, part lookup, and conditional mechanical-thermostat content.
"""
import re
from typing import Optional
from urllib.parse import quote

# Typical appliance model: letters + digits + optional hyphen, 5–25 chars.
# Prefer token after "model", "model number", "#", etc. to reduce false positives.
_MODEL_CANDIDATE = re.compile(r"[A-Z0-9][A-Z0-9\-]{4,24}", re.IGNORECASE)
# Trigger phrases that suggest the next token (or same line) is the model
_MODEL_TRIGGER = re.compile(
    r"\b(?:model\s*(?:number|#|is|:)?|serial\s*(?:number|#|is|:)?|unit\s*(?:#|is|:)?)\s*([A-Z0-9\-]{5,25})\b",
    re.IGNORECASE,
)
# Standalone token that looks like a model (e.g. "WRF535SWBM", "GTE18GMHNR")
_STANDALONE_MODEL = re.compile(r"\b([A-Z][A-Z0-9\-]{5,24})\b", re.IGNORECASE)
# Model + revision in parentheses (e.g. "LFX28968ST (03)", "LFX28968ST (03) LG Refrigerator") — must be preserved for correct PartSelect URL
_MODEL_WITH_REVISION_IN_MSG = re.compile(r"\b([A-Z0-9][A-Z0-9\-]{4,24})\s*\(\s*(\d{1,4})\s*\)", re.IGNORECASE)


def extract_model_number(message: str) -> Optional[str]:
    """
    Extract one appliance model number from the given message.
    Returns the first plausible model number, preferring one after a trigger phrase.
    When the message contains "BASEMODEL (NN)" (revision in parentheses), returns "BASEMODEL (NN)" so PartSelect link uses the correct revision.
    """
    if not message or not message.strip():
        return None
    msg = message.strip()

    # 0) Model with revision in parentheses (e.g. "parts for LFX28968ST (03)") — return with revision so we don't lose (03)
    m_rev = _MODEL_WITH_REVISION_IN_MSG.search(msg)
    if m_rev:
        base, rev = m_rev.group(1).strip().upper(), m_rev.group(2).strip()
        if _looks_like_model(base):
            return f"{base} ({rev})"

    # 1) "my X model" (e.g. "compatible with my 003719074 model")
    m_my = re.search(r"\bmy\s+([A-Z0-9\-]{5,25})\s+model\b", msg, re.IGNORECASE)
    if m_my:
        candidate = m_my.group(1).strip().upper()
        if _looks_like_model(candidate):
            return _normalize(candidate)
    # 1b) "my X Midea Dishwasher" / "my X Dishwasher" (no "model" word)
    m_dw = re.search(r"\bmy\s+([A-Z0-9\-]{5,25})\s+(?:Midea\s+)?Dishwasher\b", msg, re.IGNORECASE)
    if m_dw:
        candidate = m_dw.group(1).strip().upper()
        if _looks_like_model(candidate):
            return _normalize(candidate)
    # 2) After trigger phrase (e.g. "model is WRF535SWBM", "model number: GTE18GMHNR")
    m = _MODEL_TRIGGER.search(msg)
    if m:
        candidate = m.group(1).strip().upper()
        if _looks_like_model(candidate):
            return _normalize(candidate)

    # 2) Whole message is a single model-like token (e.g. user replied "WRF535SWBM")
    if _MODEL_CANDIDATE.fullmatch(msg.strip()):
        return _normalize(msg.strip())

    # 4) Any candidate token in the message (first one that looks like a model)
    for m in _MODEL_CANDIDATE.finditer(msg):
        candidate = m.group(0)
        if _looks_like_model(candidate):
            return _normalize(candidate)

    return None


def _looks_like_model(s: str) -> bool:
    """Filter out common false positives."""
    if len(s) < 5 or len(s) > 25:
        return False
    s_upper = s.upper()
    # Skip words that are not model numbers
    skip = {"PART", "REPAIR", "HELP", "COOLING", "WARM", "FRIDGE", "DISHWASHER", "REFRIGERATOR"}
    if s_upper in skip:
        return False
    # Usually has at least one digit
    if not any(c.isdigit() for c in s):
        return False
    return True


def _normalize(s: str) -> str:
    """Uppercase and strip; keep hyphen."""
    return s.strip().upper()


# PartSelect URL: /Models/{base}/ or /Models/{base}/MFGModelNumber/{revision}/ (e.g. LFX28968ST (03))
# No trailing \b so "(03)" at end of string matches (Python \b does not match at end of string)
_MODEL_WITH_REVISION = re.compile(r"\b([A-Z0-9][A-Z0-9\-]{4,24})\s*\(\s*(\d{1,4})\s*\)", re.IGNORECASE)


def parse_model_revision(model_number: str) -> tuple[str, Optional[str]]:
    """
    Parse 'LFX28968ST (03)' or 'LFX28968ST (03) LG Refrigerator' into (base='LFX28968ST', revision='03').
    If no (NN) revision in string, return (normalized_model, None).
    """
    if not model_number or not isinstance(model_number, str):
        return ("", None)
    s = model_number.strip()
    m = _MODEL_WITH_REVISION.search(s)
    if m:
        return (m.group(1).strip().upper(), m.group(2).strip())
    return (s.upper(), None)


def model_number_to_base(model_number: str) -> str:
    """
    Return base model only; drop revision in parentheses (e.g. (03), (04)).
    'LFX28968ST (03) LG Refrigerator' -> 'LFX28968ST'. Use when we want to treat all revisions as the same base.
    """
    if not model_number or not model_number.strip():
        return (model_number or "").strip()
    base, revision = parse_model_revision(model_number)
    if not base:
        return model_number.strip()
    return base


def partselect_model_url(model_number: str) -> str:
    """
    Build PartSelect model page URL. When model has revision in parentheses (e.g. LFX28968ST (03)),
    use /Models/{base}/MFGModelNumber/{revision}/ so the link matches PartSelect's canonical URL.
    """
    if not model_number or not model_number.strip():
        return "https://www.partselect.com/"
    base, revision = parse_model_revision(model_number)
    if not base:
        return "https://www.partselect.com/"
    if revision:
        return f"https://www.partselect.com/Models/{quote(base)}/MFGModelNumber/{quote(revision)}/"
    return f"https://www.partselect.com/Models/{quote(base)}/"


# When extracting model from history, only look at this many most recent user messages (prioritize "this conversation").
_MODEL_HISTORY_LOOKBACK = 2


def extract_model_from_messages(current_message: str, history_user_contents: list[str]) -> Optional[str]:
    """
    Extract model number from current message first, then from the most recent user messages only.
    Only considers current message + last _MODEL_HISTORY_LOOKBACK user turns so we prefer this conversation
    and don't surface an old model (e.g. LFX28968ST) when the user has since mentioned another (e.g. 103200).
    """
    out = extract_model_number(current_message)
    if out:
        return out
    # Only look at the most recent N user messages (newest first)
    recent = list(history_user_contents)[-_MODEL_HISTORY_LOOKBACK:] if len(history_user_contents) > _MODEL_HISTORY_LOOKBACK else history_user_contents
    for content in reversed(recent):
        if isinstance(content, str) and content.strip():
            out = extract_model_number(content)
            if out:
                return out
    return None


# Part number: PS + digits, or manufacturer-style (e.g. WPW10327249)
_PART_TRIGGER = re.compile(
    r"\b(?:part\s*(?:number|#|is|:)?|partselect\s*(?:#|number)?|PS\s*)\s*([A-Z0-9\-]{4,20})\b",
    re.IGNORECASE,
)
_PART_PS = re.compile(r"\b(PS\d{5,15})\b", re.IGNORECASE)
_PART_ALPHANUM = re.compile(r"\b([A-Z][A-Z0-9\-]{4,19})\b", re.IGNORECASE)


def extract_part_number(message: str) -> Optional[str]:
    """
    Extract one part number from the given message (PartSelect or manufacturer number).
    Prefers token after "part number", "PS", etc.; then PS12345 pattern; then alphanumeric.
    """
    if not message or not message.strip():
        return None
    msg = message.strip()

    m = _PART_TRIGGER.search(msg)
    if m:
        candidate = m.group(1).strip().upper()
        if _looks_like_part(candidate):
            return _normalize_part(candidate)

    for m in _PART_PS.finditer(msg):
        return _normalize_part(m.group(1))

    for m in _PART_ALPHANUM.finditer(msg):
        candidate = m.group(1)
        if _looks_like_part(candidate):
            return _normalize_part(candidate)
    return None


def _looks_like_part(s: str) -> bool:
    if len(s) < 4 or len(s) > 20:
        return False
    s_upper = s.upper()
    skip = {"PART", "PARTS", "REPAIR", "HELP", "COOLING", "WARM", "FRIDGE", "DISHWASHER", "REFRIGERATOR", "MODEL", "COMPATIBLE", "COMPATIBILITY"}
    if s_upper in skip:
        return False
    return True


def _normalize_part(s: str) -> str:
    return s.strip().upper()


def extract_part_from_messages(current_message: str, history_user_contents: list[str]) -> Optional[str]:
    """Extract part number from current message first, then from history (most recent first)."""
    out = extract_part_number(current_message or "")
    if out:
        return out
    for content in reversed(history_user_contents or []):
        if isinstance(content, str) and content.strip():
            out = extract_part_number(content)
            if out:
                return out
    return None
