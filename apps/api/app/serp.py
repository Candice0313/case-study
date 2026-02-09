"""
SerpApi client for on-demand search. Used when DB/cache have no part or model data,
so we don't need to pre-crawl all PartSelect pages.
Requires SERPAPI_API_KEY in env. See https://serpapi.com/search-api
"""
import logging
import os
import re
from typing import Any, List, Optional
from urllib.parse import quote

import httpx

from app.model_parser import parse_model_revision

logger = logging.getLogger(__name__)

# Fallback: extract first model-like token (e.g. LFX28968ST from "LFX28968ST (03) LG Refrigerator")
_MODEL_LIKE = re.compile(r"[A-Z0-9][A-Z0-9\-]{4,24}", re.IGNORECASE)

SERPAPI_BASE = "https://serpapi.com/search"


def search_serp(
    query: str,
    *,
    num: int = 5,
    api_key: Optional[str] = None,
    timeout: float = 15.0,
    gl: str = "us",
) -> List[dict]:
    """
    Run a Google search via SerpApi. Returns list of organic results with title, link, snippet.
    Uses organic_results[].link and .title (SerpApi standard). If SERPAPI_API_KEY is not set or request fails, returns [].
    """
    key = api_key or os.environ.get("SERPAPI_API_KEY", "").strip()
    if not key:
        logger.warning("SerpApi skipped: SERPAPI_API_KEY not set. Set env SERPAPI_API_KEY to search PartSelect for model-specific part pages.")
        return []

    params: dict[str, Any] = {
        "engine": "google",
        "q": query,
        "api_key": key,
        "num": min(num, 10),
        "gl": gl,
        "hl": "en",
    }
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(SERPAPI_BASE, params=params)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        logger.warning("SerpApi request failed: %s", e)
        return []

    status = data.get("search_metadata", {}).get("status")
    if status != "Success":
        logger.info("SerpApi status not Success: %s (query=%s)", status, query[:80])
        return []
    results = data.get("organic_results") or []
    out = []
    for r in results[:num]:
        link = (r.get("link") or r.get("url") or "").strip()
        title = (r.get("title") or "").strip()
        out.append({
            "title": title,
            "link": link,
            "snippet": r.get("snippet") or "",
            "thumbnail": r.get("thumbnail") or r.get("image") or "",
        })
    if out:
        logger.info("SerpApi OK: query=%s results=%s links=%s", query[:55], len(out), [(x.get("link") or "")[:65] for x in out[:3]])
    else:
        logger.info("SerpApi OK: query=%s organic_results=%s (empty)", query[:55], len(results))
    return out


def get_partselect_model_page_url(
    model_number: str,
    appliance_type: Optional[str] = None,
) -> str:
    """
    Return PartSelect model page URL: https://www.partselect.com/Models/{base}/.
    Uses only the model code (e.g. 003719074), never extra words like "Midea Dishwasher".
    """
    raw = (model_number or "").strip()
    if not raw:
        return "https://www.partselect.com/"
    # Prefer first model-like token so "003719074 Midea Dishwasher" -> /Models/003719074/
    m = _MODEL_LIKE.search(raw)
    if m:
        base = m.group(0).strip().upper()
    else:
        base, _ = parse_model_revision(raw)
        if base and " " in base:
            # parse_model_revision returned full string; take first token
            first = raw.split()[0] if raw.split() else ""
            if first and len(first) >= 5 and first.replace("-", "").isalnum():
                base = first.upper()
        elif not base or not base.strip():
            return "https://www.partselect.com/"
    if not base or not base.strip():
        return "https://www.partselect.com/"
    return f"https://www.partselect.com/Models/{quote(base.strip())}/"
