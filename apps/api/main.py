"""
PartSelect Chat Agent API - FastAPI entrypoint.
Chat endpoint with optional SSE streaming.
"""
import asyncio
import re
import os
from pathlib import Path
from contextlib import asynccontextmanager

# Load .env so DATABASE_URL / OPENAI_API_KEY are set for RAG (override=True so .env wins)
try:
    from dotenv import load_dotenv
    _api_dir = Path(__file__).resolve().parent
    _repo_root = _api_dir.parent.parent
    load_dotenv(_api_dir / ".env", override=True)
    load_dotenv(_repo_root / ".env", override=True)
except ImportError:
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from app.schemas import ChatRequest, ChatResponse, ScopeLabel
from app.scope_router import classify_scope
from app.agent import run_agent

_HOMEPAGE = "https://www.partselect.com"
_MODEL_LIKE = re.compile(r"[A-Z0-9][A-Z0-9\-]{4,24}", re.IGNORECASE)


def _model_code_for_url(model_number: str | None) -> str | None:
    """Return only the model code for /Models/ URL (e.g. 003719074 from '003719074 Midea Dishwasher')."""
    if not (model_number or "").strip():
        return None
    raw = (model_number or "").strip()
    m = _MODEL_LIKE.search(raw)
    if m:
        return m.group(0).strip().upper()
    return raw.split()[0].strip().upper() if raw.split() else None


def _extract_model_from_content(content: str) -> str | None:
    """Extract model number from content (e.g. 'Parts and instructions for **WRF535SWHZ**') for base-URL resolution. Do not treat part numbers (PS + digits) as model."""
    if not (content or "").strip():
        return None
    import re
    m = re.search(r"for\s+\*\*([A-Z0-9][A-Z0-9\-]{4,25})\*\*", content, re.IGNORECASE)
    if m:
        val = m.group(1).strip().upper()
        if re.match(r"^PS\d{5,15}$", val, re.IGNORECASE):
            return None  # part number, not model
        return val
    m = re.search(r"\*\*([A-Z0-9][A-Z0-9\-]{4,25})\*\*", content)
    if m:
        val = m.group(1).strip().upper()
        if re.match(r"^PS\d{5,15}$", val, re.IGNORECASE):
            return None
        return val
    return None


def _is_partselect_base(url: str) -> bool:
    """True if url is PartSelect with no meaningful path (homepage only)."""
    from urllib.parse import urlparse
    u = (url or "").strip()
    if not u or "partselect.com" not in u.lower():
        return False
    parsed = urlparse(u)
    path = (parsed.path or "").strip().rstrip("/")
    # No path, or path is just "/" or empty → treat as base so we try Serp resolve
    if path in ("", "/"):
        return True
    # PartSelect homepage can have trailing slash; exact match
    if u.rstrip("/") == _HOMEPAGE.rstrip("/"):
        return True
    return False


def _resolve_partselect_url_by_title(title: str, cache: dict) -> str:
    """Resolve PartSelect guide URL from page title via Serp. Uses cache keyed by normalized title."""
    import logging
    _log = logging.getLogger(__name__)
    key = (title or "").strip().lower()[:120]
    if not key:
        return _HOMEPAGE + "/"
    if key in cache:
        return cache[key]
    try:
        from app.serp import search_serp
        query = f"site:partselect.com {title.strip()}"
        results = search_serp(query, num=5)
        for r in (results or []):
            link = (r.get("link") or "").strip()
            if not link or "partselect.com" not in link.lower():
                continue
            if _is_partselect_base(link):
                continue
            cache[key] = link
            _log.info("PartSelect URL resolved: %s -> %s", title[:50], link[:80])
            return link
        if not results:
            _log.debug("Serp returned no results for %s (check SERPAPI_API_KEY?)", query[:60])
    except Exception as e:
        _log.warning("Serp resolve failed for %s: %s", title[:40], e)
    cache[key] = _HOMEPAGE + "/"
    return _HOMEPAGE + "/"


def _resolve_partselect_part_page_url(part_number: str, cache: dict) -> str:
    """Resolve PartSelect part page URL via SERP (e.g. .../PS18189330-...-.htm). Prefer .htm or /PS in path; never return Search.aspx."""
    import logging
    _log = logging.getLogger(__name__)
    pn = (part_number or "").strip().upper()
    if not pn:
        return _HOMEPAGE + "/"
    cache_key = f"part:{pn}"
    if cache_key in cache:
        return cache[cache_key]
    try:
        from app.serp import search_serp
        query = f"site:partselect.com {pn}"
        results = search_serp(query, num=8)
        for r in (results or []):
            link = (r.get("link") or "").strip()
            if not link or "partselect.com" not in link.lower():
                continue
            if _is_partselect_base(link):
                continue
            if "/Search.aspx" in link:
                continue
            # Prefer real part page: .htm or /PS in path
            if ".htm" in link or "/PS" in link.upper():
                cache[cache_key] = link
                _log.info("PartSelect part URL resolved: %s -> %s", pn, link[:80])
                return link
        # Fallback: any non-base PartSelect link from SERP
        for r in (results or []):
            link = (r.get("link") or "").strip()
            if not link or "partselect.com" not in link.lower() or _is_partselect_base(link) or "/Search.aspx" in link:
                continue
            cache[cache_key] = link
            return link
    except Exception as e:
        _log.warning("Serp part URL resolve failed for %s: %s", pn, e)
    cache[cache_key] = _HOMEPAGE + "/"
    return _HOMEPAGE + "/"


def _fix_sources_partselect_urls(sources: list, cache: dict | None = None, model_number: str | None = None) -> list:
    """Replace any source url that is empty or PartSelect base with Serp-resolved URL from title, or model page when model_number given. Mutates and returns sources."""
    if not sources:
        return sources
    from urllib.parse import quote
    cache = cache if cache is not None else {}
    model_url = f"{_HOMEPAGE}/Models/{quote(model_number.strip().upper())}/" if model_number else None
    for s in sources:
        if not isinstance(s, dict):
            continue
        url = (s.get("url") or "").strip()
        title = (s.get("title") or "").strip()
        need_resolve = not url or _is_partselect_base(url)
        if need_resolve and model_url:
            s["url"] = model_url
        elif need_resolve and title:
            resolved = _resolve_partselect_url_by_title(title, cache)
            s["url"] = resolved
    return sources


def _last_assistant_content(history: list) -> str | None:
    """Last assistant message content from chat history (for scope context)."""
    if not history:
        return None
    for m in reversed(history):
        if getattr(m, "role", None) == "assistant" and getattr(m, "content", None):
            return (m.content or "").strip() or None
    return None


def _fix_model_overview_card_urls(cards: list, url_resolve_cache: dict | None = None, model_number: str | None = None) -> list:
    """Ensure no card has PartSelect homepage url; for part cards use part search URL, for model cards use /Models/{base}/, else resolve by title."""
    if not cards:
        return cards
    import re
    from urllib.parse import quote
    model_like = re.compile(r"[A-Z0-9][A-Z0-9\-]{4,24}", re.IGNORECASE)
    ps_part = re.compile(r"^PS\d{5,15}$", re.IGNORECASE)
    cache = url_resolve_cache if url_resolve_cache is not None else {}
    out = []
    for c in cards:
        c = dict(c) if isinstance(c, dict) else {}
        url = (c.get("url") or "").strip()
        name = (c.get("name") or "").strip()
        part_num = (c.get("part_number") or "").strip()
        # Only replace URL when it's missing, base (homepage), or Search.aspx for part cards; keep real part pages e.g. .../PS18189330-...-.htm
        url_is_base_or_missing = (
            not url
            or url.rstrip("/") == _HOMEPAGE
            or (part_num and ps_part.match(part_num) and "/Search.aspx" in url)
        )
        is_partselect_not_model = (
            url.startswith(_HOMEPAGE) and "/Models/" not in url
        ) or not url or url.rstrip("/") == _HOMEPAGE
        if is_partselect_not_model and part_num and ps_part.match(part_num) and url_is_base_or_missing:
            resolved = _resolve_partselect_part_page_url(part_num, cache)
            c["url"] = resolved if not _is_partselect_base(resolved) else f"{_HOMEPAGE}/Search.aspx?SearchTerm={quote(part_num)}"
        elif is_partselect_not_model and name and url_is_base_or_missing:
            raw = name.replace(" - Overview", "").strip()
            m = model_like.search(raw)
            if m:
                base = m.group(0).strip().upper()
                if not ps_part.match(base):
                    c["url"] = f"{_HOMEPAGE}/Models/{quote(base)}/"
                else:
                    c["url"] = f"{_HOMEPAGE}/Search.aspx?SearchTerm={quote(base)}"
            elif model_number and not ps_part.match((model_number or "").strip()):
                code = _model_code_for_url(model_number)
                if code:
                    c["url"] = f"{_HOMEPAGE}/Models/{quote(code)}/"
            elif part_num and ps_part.match(part_num):
                c["url"] = f"{_HOMEPAGE}/Search.aspx?SearchTerm={quote(part_num)}"
            else:
                c["url"] = _resolve_partselect_url_by_title(name, cache)
        elif is_partselect_not_model and url_is_base_or_missing and model_number and not ps_part.match((model_number or "").strip()):
            code = _model_code_for_url(model_number)
            if code:
                c["url"] = f"{_HOMEPAGE}/Models/{quote(code)}/"
        elif is_partselect_not_model and url_is_base_or_missing and part_num and ps_part.match(part_num):
            resolved = _resolve_partselect_part_page_url(part_num, cache)
            c["url"] = resolved if not _is_partselect_base(resolved) else f"{_HOMEPAGE}/Search.aspx?SearchTerm={quote(part_num)}"
        out.append(c)
    return out


def _fix_content_partselect_homepage_link(
    content: str, product_cards: list, sources: list | None = None, model_number: str | None = None
) -> str:
    """Replace PartSelect base links in markdown: use model URL from content or product_cards, never overwrite a good /Models/ link with base."""
    if not content:
        return content
    import re
    from urllib.parse import quote
    first_url = None
    # Prefer: keep existing /Models/ link from content so we never replace correct URL with base
    for path in re.findall(r"\]\(" + re.escape(_HOMEPAGE) + r"([^)]*)\)", content):
        path = (path or "").strip()
        if "/Models/" in path and path.replace("/Models/", "").strip("/").strip():
            first_url = _HOMEPAGE + (path if path.startswith("/") else "/" + path)
            break
    if not first_url and product_cards:
        for c in product_cards:
            u = (c.get("url") or "").strip()
            if u and "/Models/" in u:
                first_url = u
                break
    if not first_url and product_cards:
        name = (product_cards[0].get("name") or "").strip()
        m = re.search(r"[A-Z0-9][A-Z0-9\-]{4,24}", name, re.IGNORECASE)
        if m:
            first_url = f"{_HOMEPAGE}/Models/{quote(m.group(0).strip().upper())}/"
    if not first_url and model_number:
        code = _model_code_for_url(model_number)
        if code:
            first_url = f"{_HOMEPAGE}/Models/{quote(code)}/"
    if not first_url:
        match = re.search(r"view\s+model\s+([^\]]+?)\s+on\s+PartSelect", content, re.IGNORECASE)
        if match:
            code = _model_code_for_url(match.group(1).strip())
            if code:
                first_url = f"{_HOMEPAGE}/Models/{quote(code)}/"
        if not first_url:
            match = re.search(r"view\s+parts\s+for\s+([^\]]+?)(?:\]|\s*$)", content, re.IGNORECASE | re.DOTALL)
            if match:
                raw = match.group(1).strip().replace("**", "")
                m = re.search(r"[A-Z0-9][A-Z0-9\-]{4,24}", raw, re.IGNORECASE)
                if m:
                    first_url = f"{_HOMEPAGE}/Models/{quote(m.group(0).strip().upper())}/"
    if first_url:
        content = re.sub(
            r"\]\(" + re.escape(_HOMEPAGE) + r"/?[^)]*\)",
            "](" + first_url + ")",
            content,
        )
        return content
    # No model URL: fix each [text](base) using sources (match link text to citation title)
    if not sources:
        return content
    title_to_url = {}
    for s in sources:
        if isinstance(s, dict):
            u = (s.get("url") or "").strip()
            t = (s.get("title") or "").strip()
            if u and not _is_partselect_base(u) and t:
                key = t.lower().strip()
                title_to_url[key] = u
    if not title_to_url:
        return content
    def replace_link(match):
        text, url = match.group(1), (match.group(2) or "").strip()
        if not _is_partselect_base(url):
            return match.group(0)
        text_norm = (text or "").replace("**", "").strip().lower()
        for title_key, resolved in title_to_url.items():
            if text_norm in title_key or title_key in text_norm:
                return f"[{text}]({resolved})"
        return match.group(0)
    content = re.sub(
        r"\[([^\]]*)\]\((https://www\.partselect\.com[^)]*)\)",
        replace_link,
        content,
    )
    return content


def _filter_product_cards_base_urls(cards: list) -> list:
    """Remove any card whose URL is still PartSelect base (homepage). Don't show base URLs in Suggested parts."""
    if not cards:
        return cards
    return [c for c in cards if isinstance(c, dict) and not _is_partselect_base((c.get("url") or "").strip())]


def _apply_partselect_url_fixes(content: str, sources: list, product_cards: list) -> tuple[str, list, list]:
    """Sync helper: resolve base PartSelect URLs for sources and cards, fix content links. Run in thread to avoid blocking."""
    url_cache = {}
    model_number = _extract_model_from_content(content)
    sources = _fix_sources_partselect_urls(sources or [], url_cache, model_number)
    product_cards = _fix_model_overview_card_urls(product_cards or [], url_cache, model_number)
    product_cards = _filter_product_cards_base_urls(product_cards or [])
    content = _fix_content_partselect_homepage_link(content, product_cards, sources, model_number)
    return content, sources, product_cards


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load config, DB pool, etc.
    yield
    # Shutdown
    pass


app = FastAPI(
    title="PartSelect Chat Agent API",
    description="Domain-locked chat for refrigerator & dishwasher parts",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "partselect-agent-api"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Single-turn chat (no streaming)."""
    scope_label = classify_scope(request.message)
    content, sources, product_cards, _ = await run_agent(
        request.message, scope_label, request.history or []
    )
    content, sources, product_cards = await asyncio.to_thread(
        _apply_partselect_url_fixes, content, sources or [], product_cards or []
    )
    return ChatResponse(
        content=content,
        scope_label=scope_label,
        sources=sources,
        product_cards=product_cards,
    )


async def stream_chat(request: ChatRequest):
    """Generator for SSE: scope first, then content."""
    last = _last_assistant_content(request.history or [])
    scope_label = classify_scope(request.message, last_assistant_message=last)
    yield {"event": "scope", "data": scope_label.value}

    if scope_label == ScopeLabel.OUT_OF_SCOPE:
        content, _, _ = await run_agent(request.message, scope_label, request.history or [])
        yield {"event": "message", "data": content}
        return

    content, sources, product_cards, _ = await run_agent(
        request.message, scope_label, request.history or []
    )
    content, sources, product_cards = await asyncio.to_thread(
        _apply_partselect_url_fixes, content, sources or [], product_cards or []
    )
    yield {"event": "message", "data": content}
    if sources:
        import json
        yield {"event": "sources", "data": json.dumps(sources)}
    if product_cards:
        import json
        yield {"event": "product_cards", "data": json.dumps(product_cards)}


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat via Server-Sent Events."""
    async def event_generator():
        async for payload in stream_chat(request):
            yield {
                "event": payload["event"],
                "data": payload["data"],
            }

    return EventSourceResponse(event_generator())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
