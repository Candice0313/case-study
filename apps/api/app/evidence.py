"""
Evidence pipeline: raw chunks → structured evidence → answer.
LLM never sees raw chunks; only the normalizer sees them, output is structured.
Answer generator sees only structured evidence + user question.
"""
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from app.schemas import StructuredEvidence, Claim, ComposeOutput


logger = logging.getLogger(__name__)

EXTRACTOR_MODEL = "gpt-4o-mini"
ANSWER_MODEL = "gpt-4o-mini"


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv as _load
        api_dir = Path(__file__).resolve().parents[1]
        repo_root = Path(__file__).resolve().parents[3]
        _load(api_dir / ".env", override=True)
        _load(repo_root / ".env", override=True)
    except ImportError:
        pass


def _sanitize_chunk_text(text: str) -> str:
    """Strip nav/CTA boilerplate before sending to normalizer."""
    if not text:
        return ""
    text = re.sub(
        r"\s*Want Mr\. Appliance to fix that broken [^?]+\?\s*Schedule Service\s*",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# When user did not specify a brand, strip brand names that the LLM may have copied from evidence
_BRAND_PREFIXES = re.compile(
    r"\b(GE|General Electric|Whirlpool|Samsung|LG|Maytag|KitchenAid|Bosch|Frigidaire|Kenmore|Electrolux|Hotpoint|Admiral)\s+",
    re.IGNORECASE,
)
# Phrases to replace with generic "refrigerator" / "appliance"
_BRAND_APPLIANCE_PHRASES = re.compile(
    r"\b(GE|General Electric|Whirlpool|Samsung|LG|Hotpoint|Frigidaire|Kenmore|Maytag|KitchenAid|Bosch|Electrolux|Admiral)\s+(refrigerator|fridge|dishwasher|appliance)\b",
    re.IGNORECASE,
)
# LLM sometimes outputs brand in markdown bold; \b doesn't match after * so we need a separate pattern
_BRAND_IN_MARKDOWN = re.compile(
    r"\*\*(GE|General Electric|Whirlpool|Samsung|LG|Maytag|KitchenAid|Bosch|Frigidaire|Kenmore|Electrolux|Hotpoint|Admiral)\*\*",
    re.IGNORECASE,
)


def _strip_brand_names_from_answer(text: str) -> str:
    """Remove brand names so 'If your GE refrigerator...' or 'If your **GE** refrigerator...' → 'If your refrigerator...'."""
    if not (text or "").strip():
        return text or ""
    # Remove markdown-bold-wrapped brands first (e.g. **GE**), then plain brands
    t = _BRAND_IN_MARKDOWN.sub("", text)
    t = _BRAND_APPLIANCE_PHRASES.sub(r"\2", t)
    t = _BRAND_PREFIXES.sub("", t)
    t = re.sub(r"  +", " ", t)
    return t.strip()


# --- Output contract for the answer generator: chat-friendly but sufficiently detailed ---
ANSWER_SYSTEM = """You are a repair assistant for refrigerators and dishwashers. Reply in a clear, helpful way—detailed enough that the user can follow the steps without opening another page.

Rules:
- Use short paragraphs for the intro and causes. When you give concrete steps to follow, format those steps as bullet points: each step on its own line starting with "- " (hyphen space). Do not mix steps into a long paragraph—use bullets for the step-by-step part only.
- Give a complete, actionable answer: include all relevant causes from the evidence (typically 2–4) and all concrete steps (typically 4–8). Do not cut short for brevity.
- If a part may need replacement, explain when and why, then invite the user to share their model number for the right part.
- If the user only said something vague (e.g. "broken", "not working", "is broken", "doesn't work"), do NOT present one cause as the only answer. Start with: "You didn't specify the exact issue—one common possibility is: " or "Common possibilities include: " and then give causes/steps. Never say "It sounds like your [appliance] might not be working due to [one specific cause]" when the user was vague.
- Do NOT include navigation text, CTAs, filenames, or URLs. Cite sources by title only if needed, once each.
- Do NOT assume the other scenario; only give steps for the situation the user described. Do NOT list mechanical thermostat parts (sensing bulb, control knob) as definite steps before they share a model—you may mention "on some older models, the control or sensor is worth checking once you share your model."
- End with one short line: e.g. "Share your model or part number if you want the exact part or model-specific steps." """

# When showing only tutorial/guide links (no model): answer = detailed content from the guide
TROUBLESHOOT_GUIDES_ONLY_ADDON = (
    "The user only asked about a problem (no model number yet). Your reply: short paragraph(s) for causes/context, then the steps as bullet points, then one closing line.\n"
    "- Use 1–2 short paragraphs for the intro and likely causes (with brief explanation). Then format the step-by-step instructions as bullet points: each step on its own line starting with '- '. Aim for 8–12 step bullets. Include specifics: tool names, locations, readings (e.g. ohms), order of operations. Do NOT put steps in a long paragraph—use bullets for the steps only.\n"
    "- Base your answer on the FULL repair-guide evidence: include every relevant cause and step from the guide. Do NOT summarize away detail.\n"
    "- End with one short line: 'Share your model number for part-specific help.'\n"
    "Do NOT start with 'You didn't specify the exact issue' when the user clearly described a problem. "
    "Do NOT write 'Here are', 'resources', 'links', 'parts', 'Source(s):', or ANY list of link/source titles. Cite only repair-guide chunks by id."
)

# When user said only "not working" / "broken" / "not working properly" — do not assume one symptom (e.g. cooling)
VAGUE_USER_SYSTEM_ADDON = (
    "The user did NOT specify a specific symptom (they said only 'not working', 'broken', or 'not working properly'). "
    "Do NOT write 'If your refrigerator is not cooling' or 'not cooling properly' or assume cooling, leaking, or any single cause. "
    "Start with 'You didn't specify the exact issue—common possibilities include: not cooling, not starting, leaking, making noise, etc.' "
    "and then give general steps or 1–2 possibilities. Do not present one cause (e.g. not cooling) as if the user said it."
)

# Post-process: if user never said "cool/cooling" but answer assumes "not cooling", replace that opening
_ASSUMED_COOLING_OPENING = re.compile(
    r"^(If your (?:refrigerator|fridge) is not cooling[^.]*\.)\s*",
    re.IGNORECASE,
)


# Strip "Here are ... links/parts" and everything after so answer = summary only
_REPLACEMENT_PARTS_SENTENCE = re.compile(
    r"\s*You can find various replacement parts[^.]*\.\s*",
    re.IGNORECASE,
)
# From "Here are" when followed by resources/parts/links (any wording) to end — remove lead-in and entire list
_HERE_ARE_LINKS_OR_PARTS_TO_END = re.compile(
    r"\s*Here are\s+.+?(?:resources?|parts?|links?).*$",
    re.IGNORECASE | re.DOTALL,
)
# "I recommend checking out these parts..." or "Consider these parts..." to end
_RECOMMEND_CHECK_OUT_PARTS_TO_END = re.compile(
    r"\s*(?:I recommend|Consider|You can)\s+(?:checking out|exploring?|viewing?)\s+(?:these\s+)?parts?.*$",
    re.IGNORECASE | re.DOTALL,
)
_TAIL_CHECK_THESE_LINKS = re.compile(r"\s*Check these links for[^\n]*\s*$", re.IGNORECASE)
_TAIL_RESOURCES_TROUBLESHOOT = re.compile(r"\s*These resources should help you troubleshoot[^\n]*\s*$", re.IGNORECASE)


def _paragraph_to_bullets(text: str) -> str:
    """When answer has no bullets but has 'First,' 'Next,' 'Then' or '1.' '2.', convert to bullet lines."""
    if not (text or "").strip():
        return text or ""
    t = (text or "").strip()
    if re.search(r"^\s*[-*•]\s+", t, re.MULTILINE):
        return t  # already has bullets
    # Split before "First," "Next," "Then," etc. so each step becomes a bullet
    parts = re.split(r"\s+(?=First,|Next,|Then,|Also,|Finally,)", t, flags=re.IGNORECASE)
    if len(parts) <= 1:
        parts = re.split(r"\s+(?=\d+\.\s+)", t)
    if len(parts) <= 1:
        return t
    bullet_lines = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        p = re.sub(r"^(?:First|Next|Then|Also|Finally),?\s*", "", p, flags=re.IGNORECASE).strip()
        if p:
            bullet_lines.append(f"- {p}")
    return "\n".join(bullet_lines) if bullet_lines else t


def _strip_helpful_links_block_from_answer(text: str) -> str:
    """Remove parts/links lead-in and any following list so answer is just the cause summary."""
    if not (text or "").strip():
        return text or ""
    t = (text or "").strip()
    t = _REPLACEMENT_PARTS_SENTENCE.sub(" ", t)
    t = re.sub(r"  +", " ", t).strip()
    for pattern in (_HERE_ARE_LINKS_OR_PARTS_TO_END, _RECOMMEND_CHECK_OUT_PARTS_TO_END):
        m = pattern.search(t)
        if m:
            t = t[: m.start()].rstrip()
            break
    t = _TAIL_CHECK_THESE_LINKS.sub("", t).rstrip()
    t = _TAIL_RESOURCES_TROUBLESHOOT.sub("", t).rstrip()
    return t.strip()


def _fix_assumed_cooling_in_answer(answer: str, user_message: str) -> str:
    """If user didn't say cool/cooling but answer starts with 'If your refrigerator is not cooling...', replace with vague opener."""
    if not (answer or "").strip() or not (user_message or "").strip():
        return answer or ""
    msg_lower = (user_message or "").lower()
    if "cool" in msg_lower or "cooling" in msg_lower:
        return answer or ""
    match = _ASSUMED_COOLING_OPENING.match((answer or "").strip())
    if not match:
        return answer or ""
    rest = (answer or "").strip()[len(match.group(0)) :].lstrip()
    return (
        "You didn't specify the exact issue—common possibilities include: not cooling, not starting, leaking, or making noise. "
        + (rest if rest else "Share a bit more (e.g. not cooling, not starting) so I can give targeted steps.")
    ).strip()


async def _get_client() -> AsyncOpenAI | None:
    _load_dotenv()
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key or key == "sk-...":
        logger.warning("OPENAI_API_KEY missing or placeholder — evidence pipeline disabled")
        return None
    return AsyncOpenAI(api_key=key)


async def filter_chunks_to_guides_only(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Use LLM to keep only chunks that are from repair guides / how-to (not part catalog pages).
    So the answer generator only sees guide content and won't cite or mention part catalogs.
    """
    if not chunks or len(chunks) <= 1:
        return chunks
    client = await _get_client()
    if not client:
        return chunks
    lines = []
    for i, c in enumerate(chunks):
        url = (c.get("source_url") or c.get("url") or "").strip()
        title = (c.get("title") or c.get("source") or "Untitled").strip()
        lines.append(f"{i}. {title} | {url}")
    prompt = (
        "Below are retrieved items (index, title, url). For troubleshooting-only answers we want only "
        "repair guides / how-to / troubleshooting articles, NOT part catalog or product listing pages.\n"
        "Output JSON: {\"keep_indices\": [0-based indices of items that are repair guides or how-to content]}.\n"
        "Drop part catalogs, product lists, 'X Parts', 'X Valves', 'X Switches' category pages.\n\n"
        + "\n".join(lines)
    )
    try:
        resp = await client.chat.completions.create(
            model=os.environ.get("PLANNER_LLM_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
        indices = data.get("keep_indices")
        if isinstance(indices, list):
            valid = [i for i in indices if isinstance(i, int) and 0 <= i < len(chunks)]
            if valid:
                return [chunks[i] for i in valid]
            return []  # LLM kept none → treat as no guide content, don't pass part chunks
    except Exception as e:
        logger.debug("filter_chunks_to_guides_only failed: %s", e)
    return chunks  # on parse/API failure, keep chunks so we still get an answer (strip will clean body)


async def filter_citations_to_guides_only(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Use LLM to keep only citations that are repair guides / how-to (not part catalog pages).
    """
    if not citations or len(citations) <= 1:
        return citations
    client = await _get_client()
    if not client:
        return citations
    lines = [f"{i}. {c.get('title') or 'Untitled'} | {c.get('url') or ''}" for i, c in enumerate(citations)]
    prompt = (
        "Below are candidate links (index, title, url). Keep only repair guides / how-to articles, "
        "NOT part catalog or product listing pages. Output JSON: {\"keep_indices\": [0-based indices to keep]}.\n\n"
        + "\n".join(lines)
    )
    try:
        resp = await client.chat.completions.create(
            model=os.environ.get("PLANNER_LLM_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
        indices = data.get("keep_indices")
        if isinstance(indices, list):
            valid = [i for i in indices if isinstance(i, int) and 0 <= i < len(citations)]
            # Return only guide citations; if LLM kept none, return [] so we don't show part links
            return [citations[i] for i in valid] if valid else []
    except Exception as e:
        logger.debug("filter_citations_to_guides_only failed: %s", e)
    return []  # on failure, show no links rather than part catalog links


def _looks_like_part_catalog_citation(title: str, url: str) -> bool:
    """Rule-based: treat as part catalog so we drop it when guides_only (safety net)."""
    if not title and not url:
        return False
    t = (title or "").lower()
    u = (url or "").lower()
    # Category-style titles: "X Parts", "X Valves", "X Switches", "X Ice Makers"
    if " parts" in t or t.endswith(" parts") or " valves" in t or " switches" in t:
        return True
    if " ice makers" in t or t.endswith(" ice makers") or " ice maker " in t and "parts" in t:
        return True
    # Part/product-style titles (e.g. "Dispenser Control Board", "Inlet Valve", "Water Filter")
    guide_words = ("repair", "guide", "how to", "fix", "troubleshoot", "step", "not making", "not dispensing", "diagnos")
    part_words = ("control board", "inlet valve", "water filter", " dispenser", "assembly", " kit ", " motor ", " valve ")
    if any(p in t for p in part_words) and not any(g in t for g in guide_words):
        return True
    # Short title that looks like a single part name (e.g. "Dispenser Control Board")
    if len(t) < 60 and any(w in t for w in ("board", "valve", "motor", "filter")) and not any(g in t for g in guide_words):
        return True
    # URL patterns (same as retrieval guides_only)
    if "/part-" in u or "/parts/" in u or "-parts" in u or "ice-maker-parts" in u or "ice-makers" in u or "/ps-" in u:
        return True
    return False


def drop_part_catalog_citations(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """When troubleshoot_guides_only: drop any citation that looks like part catalog (title/url). Safety net after LLM filter."""
    if not citations:
        return citations
    return [
        c for c in citations
        if not _looks_like_part_catalog_citation(c.get("title") or "", c.get("url") or "")
    ]


async def chunks_to_structured_evidence(
    chunks: list[dict[str, Any]],
    user_query: str,
) -> StructuredEvidence | None:
    """
    Normalizer: raw retrieved chunks → structured evidence.
    Only this step sees raw chunk text; output is a single structured object.
    """
    if not chunks:
        return None

    # Build input: sanitized text + unique source titles for attribution
    texts = []
    source_titles_seen: set[str] = set()
    source_titles_list: list[str] = []
    for c in chunks:
        raw = (c.get("text") or "").strip()
        if len(raw) < 40:
            continue
        texts.append(_sanitize_chunk_text(raw)[:1200])
        title = (c.get("title") or c.get("source") or "").strip()
        if title and title.lower() not in ("content", "guide", "") and title not in source_titles_seen:
            source_titles_seen.add(title)
            source_titles_list.append(title)
    if not texts:
        return None

    combined = "\n\n---\n\n".join(texts[:8])
    user_question = (user_query or "").strip() or "General troubleshooting"

    system = """You are an extractor. From the given repair guide excerpts and the user's question, output a single JSON object with these exact keys:
- symptom: short phrase summarizing the user's issue
- likely_causes: array of 1 to 3 primary causes (strings); use conditional phrasing when possible
- difficulty: one of "easy", "medium", "hard"
- steps: array of ordered steps to check or do (strings)
- when_to_replace: one sentence on when to replace a part vs repair, or null
- source_titles: array of unique guide/source names to attribute (from the excerpts), no repetition

Critical rules:
- Mechanical-thermostat-only parts (temperature sensing bulb, control knob linkage) exist only on older mechanical-control fridges; modern electronic units do not have them. Do NOT list these as definite steps when model is unknown. If you include them at all, use conditional language: "On some models with a mechanical temperature control, …" and defer detail until model is known.
- For airflow / damper / fridge-not-cooling scenarios: order steps with non-invasive first—(1) check if the refrigerator vent is blocked by food or ice, (2) confirm there is airflow at the refrigerator vent—then steps that require disconnecting power or opening the door (e.g. locate damper, inspect baffle). Do not put "disconnect refrigerator" or "open door and locate damper" as step 1.
- when_to_replace: be specific to the part and situation (e.g. "If the damper does not move freely or shows visible damage, it will typically need replacement rather than adjustment"). Do NOT use generic phrases like "Replace the part if it is damaged beyond repair." Tie replacement to the user's next step (e.g. model number helps find the correct part).

Do not include raw navigation, CTAs, or repeated phrases. Extract only factual repair content. Output only valid JSON."""

    user_msg = f"""Guide excerpts:\n{combined}\n\nUser question: {user_question}"""

    client = await _get_client()
    if not client:
        return None

    try:
        resp = await client.chat.completions.create(
            model=EXTRACTOR_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        content = (resp.choices[0].message.content or "").strip()
        data = json.loads(content)
        # Enforce max 2 causes
        causes = data.get("likely_causes") or []
        if len(causes) > 3:
            causes = causes[:3]
        data["likely_causes"] = causes
        data["source_titles"] = data.get("source_titles") or source_titles_list[:3]
        return StructuredEvidence(
            symptom=data.get("symptom", user_question[:80]),
            likely_causes=causes,
            difficulty=data.get("difficulty", "medium"),
            steps=data.get("steps") or [],
            when_to_replace=data.get("when_to_replace"),
            source_titles=data["source_titles"],
        )
    except Exception as e:
        logger.exception("Evidence normalizer failed: %s", e)
        return None


def _brand_from_message(message: str) -> str | None:
    """Extract brand from user message so we can prefer matching content."""
    if not (message or "").strip():
        return None
    m = (message or "").lower()
    for brand in ("whirlpool", "kenmore", "ge ", "general electric", "samsung", "frigidaire", "maytag", "lg ", "bosch", "kitchenaid"):
        if brand.rstrip() in m or brand.replace(" ", "") in m.replace(" ", ""):
            return brand.strip().title()
    return None


async def evidence_to_answer(
    evidence: StructuredEvidence,
    user_message: str,
    scenario_hint: str = "",
    model_number: str | None = None,
    brand_hint: str | None = None,
    user_message_vague: bool = False,
    troubleshoot_guides_only: bool = False,
) -> str:
    """
    Answer generator: structured evidence + user question → final reply.
    When scenario_hint is set, answer must not assume the other scenario; Why = this state→symptom.
    When user_message_vague is True, do not assume one symptom (e.g. not cooling); use "common possibilities include" wording.
    """
    client = await _get_client()
    if not client:
        return (
            "I found some guides but couldn’t process them right now. "
            "Please try again or share your model number for more specific help."
        )

    # Serialize evidence for the model (no raw chunk text)
    evidence_blob = {
        "symptom": evidence.symptom,
        "likely_causes": evidence.likely_causes,
        "difficulty": evidence.difficulty,
        "steps": evidence.steps,
        "when_to_replace": evidence.when_to_replace,
        "source_titles": evidence.source_titles,
    }
    evidence_str = json.dumps(evidence_blob, ensure_ascii=False, indent=0)

    system = ANSWER_SYSTEM
    if troubleshoot_guides_only:
        system = TROUBLESHOOT_GUIDES_ONLY_ADDON + "\n\n" + system
    if user_message_vague:
        system = VAGUE_USER_SYSTEM_ADDON + "\n\n" + system
    if scenario_hint:
        system = f"The user has confirmed: {scenario_hint}\n\n" + system
    if brand_hint:
        system = (
            f"The user said their appliance is {brand_hint}. "
            "Do NOT suggest, list, or cite parts/links/sources for other brands (e.g. Kenmore, KitchenAid). "
            "Only cite content that matches this brand. If the evidence is mostly for other brands, give general troubleshooting steps only and say they can search PartSelect for {brand_hint} parts—do not list other-brand links or titles.\n\n"
            + system
        )
    else:
        # User did not specify a brand — do not assume or mention any brand (evidence may be from brand-specific guides)
        system = (
            "The user did NOT specify a brand. Do NOT assume or mention any brand name (e.g. GE, Whirlpool, Samsung, LG) in your answer. "
            "Use generic wording only: 'your refrigerator', 'the refrigerator', 'the appliance'. "
            "You may summarize causes and steps from the evidence, but strip out brand names.\n\n"
            + system
        )
    if model_number:
        system = (
            f"The user has provided their appliance model number: {model_number}. "
            "You may give model-specific guidance. When the evidence supports it, "
            "you may mention mechanical temperature control parts (sensing bulb, control knob linkage) "
            "as applicable for that model—use conditional language: 'On some models with a mechanical temperature control…'\n\n"
            + system
        )

    if troubleshoot_guides_only:
        user_prompt = (
            f"Structured evidence (from the repair guide—use all of it):\n\n{evidence_str}\n\nUser question: {user_message}\n\n"
            "Write your reply: 1–2 short paragraphs for causes/context, then the steps as bullet points (each step starting with '- '). Include 8–12 step bullets. Do not skip steps from the guide. End with: 'Share your model number for part-specific help.' Do not add Source(s) or any link list."
        )
    else:
        user_prompt = f"""Structured evidence (use this only; do not add raw document text):

{evidence_str}

User question: {user_message}

Write your reply: short paragraph(s) for causes, then steps as bullet points (each step starting with '- '). Include 4–8 steps. End with one line inviting model/part number. Do not add Source(s) or link lists."""

    try:
        resp = await client.chat.completions.create(
            model=ANSWER_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        content = (resp.choices[0].message.content or "").strip()
        if not brand_hint and content:
            content = _strip_brand_names_from_answer(content)
        if user_message_vague and content:
            content = _fix_assumed_cooling_in_answer(content, user_message)
        if troubleshoot_guides_only and content:
            content = _strip_helpful_links_block_from_answer(content)
            content = _paragraph_to_bullets(content)
        return content or "I couldn’t generate a reply. Please share your model number or part number for more specific help."
    except Exception as e:
        logger.exception("Answer generator failed: %s", e)
        return (
            "I had trouble generating the answer. You can try rephrasing or share your appliance model for targeted steps."
        )


# --- Claim→chunk_id contract: LLM outputs claims with supporting_chunk_ids; we validate and render citations ---
COMPOSE_WITH_CLAIMS_SYSTEM = """You are a repair assistant. Output a single JSON object with:
- answer: clear, detailed reply. Use short paragraphs for causes/context; when you give step-by-step instructions, format those steps as bullet points (each step starting with '- '). End with one short line (e.g. 'Share your model number for part-specific help.'). No navigation/CTAs. Do NOT add 'Source(s):' or any link list. Cite by chunk id only.
- claims: array of 1–5 claims. Each claim has:
  - text: one sentence or short phrase (what you are asserting)
  - supporting_chunk_ids: array of chunk IDs that support this claim. Use ONLY IDs from the valid_chunk_ids list below.

Rules:
- Every factual claim must list at least one supporting_chunk_id from valid_chunk_ids.
- Do not invent chunk IDs; only use those provided.
- The answer must be only the troubleshooting text. Use paragraphs for intro/causes; use bullets for the steps only. When the user asked about a problem with no model number, give 1–2 paragraphs then 8–12 step bullets, then the closing line. Do NOT include 'Here are some helpful links', 'Source(s):', or list of link titles."""


async def evidence_to_answer_with_claims(
    chunks: list[dict[str, Any]],
    user_message: str,
    scenario_hint: str = "",
    model_number: str | None = None,
    brand_hint: str | None = None,
    user_message_vague: bool = False,
    troubleshoot_guides_only: bool = False,
) -> tuple[str, list[Claim]]:
    """
    Compose answer + claims with supporting_chunk_ids. Chunk IDs are validated against chunks;
    citations are then built programmatically from validated chunk_ids (caller).
    Returns (answer, claims). On failure returns ("", []); caller can fall back to legacy compose.
    """
    if not chunks:
        return ("", [])
    valid_chunk_ids = [c.get("chunk_id") for c in chunks if c.get("chunk_id")]
    if not valid_chunk_ids:
        return ("", [])

    structured = await chunks_to_structured_evidence(chunks, user_message)
    evidence_blob = (
        {
            "symptom": structured.symptom,
            "likely_causes": structured.likely_causes,
            "steps": structured.steps,
            "when_to_replace": structured.when_to_replace,
            "source_titles": structured.source_titles,
        }
        if structured
        else {}
    )
    evidence_str = (
        json.dumps(evidence_blob, ensure_ascii=False, indent=0) if evidence_blob else "No structured evidence."
    )

    system = COMPOSE_WITH_CLAIMS_SYSTEM + f"\n\nValid chunk IDs for this response (use only these): {valid_chunk_ids}"
    if troubleshoot_guides_only:
        system = TROUBLESHOOT_GUIDES_ONLY_ADDON + "\n\n" + system
    if user_message_vague:
        system = VAGUE_USER_SYSTEM_ADDON + "\n\n" + system
    if scenario_hint:
        system = f"User context: {scenario_hint}\n\n" + system
    if brand_hint:
        system = (
            f"The user said their appliance is {brand_hint}. "
            "Do NOT suggest or cite sources for other brands (e.g. Kenmore, KitchenAid). Only cite content that matches this brand; if evidence is for other brands, give general steps only and do not list other-brand links or titles.\n\n"
            + system
        )
    else:
        system = (
            "The user did NOT specify a brand. Do NOT assume or mention any brand name (e.g. GE, Whirlpool, Samsung) in the answer. "
            "Use generic wording only: 'your refrigerator', 'the refrigerator', 'the appliance'. Strip brand names from the evidence when summarizing.\n\n"
            + system
        )
    if model_number:
        system = f"User model number: {model_number}. You may give model-specific guidance.\n\n" + system

    if troubleshoot_guides_only:
        user_prompt = (
            f"Structured evidence (from the repair guide—use all of it):\n{evidence_str}\n\nUser question: {user_message}\n\n"
            "Output JSON with 'answer' and 'claims'. The answer: 1–2 short paragraphs for causes, then steps as bullet points (each step starting with '- '), 8–12 steps. Do not skip steps. End with 'Share your model number for part-specific help.' Do not add Source(s) or link list. Each claim: 'text', 'supporting_chunk_ids'."
        )
    else:
        user_prompt = (
            f"Structured evidence:\n{evidence_str}\n\nUser question: {user_message}\n\n"
            "Output JSON with 'answer' and 'claims'. The answer: paragraph(s) for causes, then steps as bullet points (each step with '- '), 4–8 steps. End with one line inviting model/part number. Do not add Source(s) or link list. Each claim: 'text', 'supporting_chunk_ids'."
        )

    client = await _get_client()
    if not client:
        return ("", [])

    try:
        resp = await client.chat.completions.create(
            model=ANSWER_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        content = (resp.choices[0].message.content or "").strip()
        data = json.loads(content)
        answer = (data.get("answer") or "").strip()
        if not brand_hint and answer:
            answer = _strip_brand_names_from_answer(answer)
        if user_message_vague and answer:
            answer = _fix_assumed_cooling_in_answer(answer, user_message)
        if troubleshoot_guides_only and answer:
            answer = _strip_helpful_links_block_from_answer(answer)
            answer = _paragraph_to_bullets(answer)
        raw_claims = data.get("claims") or []
        if not isinstance(raw_claims, list):
            raw_claims = []

        valid_set = set(valid_chunk_ids)
        claims_out: list[Claim] = []
        for c in raw_claims:
            if not isinstance(c, dict):
                continue
            text = (c.get("text") or "").strip()
            ids = c.get("supporting_chunk_ids")
            if not isinstance(ids, list):
                ids = [ids] if isinstance(ids, str) and ids else []
            ids = [str(i).strip() for i in ids if str(i).strip() in valid_set]
            if text:
                claims_out.append(Claim(text=text, supporting_chunk_ids=ids))

        return (answer or "I couldn't generate a reply.", claims_out)
    except Exception as e:
        logger.exception("Compose with claims failed: %s", e)
        return ("", [])
