"""
Agent tools: part_lookup, search_parts, check_compatibility, get_troubleshooting.
On-demand lookup + cache: part/compat check cache first, then DB; tool_gate fallback when external source is unavailable.
Live PartSelect fetch on DB miss is off by default (often 403 / 0 parts); set ENABLE_LIVE_PARTS_FETCH=1 to try.
When DB/cache (and optional live fetch) have no data, SerpApi can return PartSelect search links so we don't need to pre-crawl everything; set SERPAPI_API_KEY to enable.
"""
import asyncio
import logging
import os
from typing import Any, List, Optional

from app import retrieval
from app.part_cache import (
    get_part_cached,
    set_part_cached,
    get_compat_cached,
    set_compat_cached,
    get_model_parts_cached,
    set_model_parts_cached,
    normalize_part_number,
    normalize_model_number,
)

logger = logging.getLogger(__name__)

# Fallback copy when external source is unavailable (tool_gate)
MSG_COMPAT_UNAVAILABLE = (
    "We couldn't verify compatibility right now. Please confirm on the PartSelect model page or share brand/model."
)
MSG_PART_UNAVAILABLE = (
    "We couldn't fetch part details. Please try again later or check the part number and install notes on the model page."
)


def _get_connection():
    """Reuse retrieval DB connection helper."""
    return retrieval.get_connection()


async def part_lookup(part_number: str) -> Optional[dict]:
    """
    Part detail: cache first, then DB. Can be extended to call PartSelect on cache miss.
    """
    if not (part_number or "").strip():
        return None
    raw = part_number.strip()
    norm = normalize_part_number(raw)
    if norm:
        cached = get_part_cached(raw)
        if cached is not None:
            return cached

    def _query():
        conn = _get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT part_id, part_number, partselect_number, manufacturer_part_number,
                           name, brand, category, price, stock, url, image_url
                    FROM parts
                    WHERE partselect_number = %s OR manufacturer_part_number = %s OR part_number = %s
                    LIMIT 1
                    """,
                    (raw, raw, raw),
                )
                row = cur.fetchone()
            if not row:
                return None
            cols = [c.name for c in cur.description] if cur.description else []
            out = dict(zip(cols, row)) if cols else {}
            out.setdefault("difficulty", None)
            out.setdefault("time_estimate", None)
            if not (out.get("image_url") or "").strip():
                out["image_url"] = _part_image_url_fallback(out)
            return out
        finally:
            conn.close()

    try:
        return await asyncio.to_thread(_query)
    except Exception as e:
        logger.exception("part_lookup failed: %s", e)
        return None


def _row_from_live_part(p: dict, model_number: str) -> dict:
    """Map PartSelect live-fetch part dict to search_parts row shape."""
    ps = (p.get("partselect_number") or "").strip()
    return {
        "part_id": None,
        "part_number": ps,
        "partselect_number": ps,
        "manufacturer_part_number": p.get("manufacturer_part_number"),
        "name": p.get("name") or f"Part {ps}",
        "brand": None,
        "price": p.get("price"),
        "difficulty": None,
        "time_estimate": None,
        "url": p.get("url"),
        "image_url": None,
    }


def _part_image_url_fallback(part: dict) -> Optional[str]:
    """Return image_url from part, or a PartSelect guess when missing (frontend will hide on 404)."""
    url = (part.get("image_url") or "").strip()
    if url:
        return url
    ps = (part.get("partselect_number") or part.get("part_number") or "").strip().upper()
    if ps and (ps.startswith("PS") or ps.replace("-", "").isalnum()):
        return f"https://www.partselect.com/PartSelectImages/{ps}.jpg"
    return None


def _row_from_serp(item: dict) -> dict:
    """Map SerpApi organic result to search_parts row shape (name, url, image_url from thumbnail)."""
    thumb = (item.get("thumbnail") or item.get("image") or "").strip()
    return {
        "part_id": None,
        "part_number": None,
        "partselect_number": None,
        "manufacturer_part_number": None,
        "name": (item.get("title") or item.get("link") or "PartSelect link").strip(),
        "brand": None,
        "price": None,
        "difficulty": None,
        "time_estimate": None,
        "url": (item.get("link") or "").strip(),
        "image_url": thumb or None,
        "source": "serp",
    }


def _write_fitment_and_parts(model_number: str, parts: List[dict], fit_source: str = "partselect_live") -> None:
    """Insert part_fitment and parts from live-fetch result (run in thread)."""
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            for p in parts:
                ps = (p.get("partselect_number") or "").strip()
                if not ps:
                    continue
                cur.execute(
                    """
                    INSERT INTO part_fitment (partselect_number, model_number, fit_source)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (partselect_number, model_number) DO NOTHING
                    """,
                    (ps, model_number.strip(), fit_source),
                )
                cur.execute(
                    """
                    INSERT INTO parts (part_number, partselect_number, name, manufacturer_part_number, price, url, image_url)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (part_number) DO UPDATE SET
                      name = COALESCE(EXCLUDED.name, parts.name),
                      manufacturer_part_number = COALESCE(EXCLUDED.manufacturer_part_number, parts.manufacturer_part_number),
                      price = COALESCE(EXCLUDED.price, parts.price),
                      url = COALESCE(EXCLUDED.url, parts.url),
                      image_url = COALESCE(EXCLUDED.image_url, parts.image_url)
                    """,
                    (
                        ps,
                        ps,
                        (p.get("name") or f"Part {ps}"),
                        p.get("manufacturer_part_number"),
                        p.get("price"),
                        p.get("url"),
                        p.get("image_url"),
                    ),
                )
        conn.commit()
    finally:
        conn.close()


async def search_parts(
    part_number: Optional[str] = None,
    model_number: Optional[str] = None,
    brand: Optional[str] = None,
    symptom: Optional[str] = None,
    limit: int = 5,
) -> List[dict]:
    """Search parts by part number, model, brand, or symptom. Cache/DB first; for model_number, on cache+DB miss fetches live from PartSelect and fills DB."""
    # Single part lookup
    if part_number and not model_number and not brand and not symptom:
        p = await part_lookup(part_number)
        return [p] if p else []

    # Model-based search: check cache first, then DB, then live fetch
    if model_number and not part_number and not brand:
        cached = get_model_parts_cached(model_number)
        if cached is not None:
            return cached[:limit]

    def _search():
        conn = _get_connection()
        try:
            with conn.cursor() as cur:
                if part_number:
                    cur.execute(
                        """
                        SELECT part_id, part_number, partselect_number, manufacturer_part_number,
                               name, brand, price, url, image_url
                        FROM parts
                        WHERE partselect_number = %s OR manufacturer_part_number = %s OR part_number = %s
                        LIMIT %s
                        """,
                        (part_number.strip(), part_number.strip(), part_number.strip(), limit),
                    )
                elif model_number:
                    if brand:
                        cur.execute(
                            """
                            SELECT p.part_id, p.part_number, p.partselect_number, p.manufacturer_part_number,
                                   p.name, p.brand, p.price, p.url, p.image_url
                            FROM parts p
                            JOIN part_fitment f ON f.partselect_number = p.partselect_number
                            WHERE f.model_number = %s AND p.brand ILIKE %s
                            LIMIT %s
                            """,
                            (model_number.strip(), f"%{brand.strip()}%", limit),
                        )
                    else:
                        symptom_like = (symptom or "").strip()[:80]
                        if symptom_like:
                            cur.execute(
                                """
                                SELECT p.part_id, p.part_number, p.partselect_number, p.manufacturer_part_number,
                                       p.name, p.brand, p.price, p.url, p.image_url
                                FROM parts p
                                JOIN part_fitment f ON f.partselect_number = p.partselect_number
                                WHERE f.model_number = %s AND p.name ILIKE %s
                                LIMIT %s
                                """,
                                (model_number.strip(), f"%{symptom_like}%", limit),
                            )
                        else:
                            cur.execute(
                                """
                                SELECT p.part_id, p.part_number, p.partselect_number, p.manufacturer_part_number,
                                       p.name, p.brand, p.price, p.url, p.image_url
                                FROM parts p
                                JOIN part_fitment f ON f.partselect_number = p.partselect_number
                                WHERE f.model_number = %s
                                LIMIT %s
                                """,
                                (model_number.strip(), limit),
                            )
                elif brand:
                    cur.execute(
                        """
                        SELECT part_id, part_number, partselect_number, manufacturer_part_number,
                               name, brand, price, url, image_url
                        FROM parts
                        WHERE brand ILIKE %s
                        LIMIT %s
                        """,
                        (f"%{brand.strip()}%", limit),
                    )
                else:
                    return []
                rows = cur.fetchall()
                desc = cur.description or []
                cols = [c.name for c in desc]
                out_list = [dict(zip(cols, r)) for r in rows]
                for row in out_list:
                    row.setdefault("difficulty", None)
                    row.setdefault("time_estimate", None)
                    if not (row.get("image_url") or "").strip():
                        row["image_url"] = _part_image_url_fallback(row)
                return out_list
        finally:
            conn.close()

    try:
        rows = await asyncio.to_thread(_search)
    except Exception as e:
        logger.exception("search_parts failed: %s", e)
        return []

    # On model_number and DB returned nothing: optionally try live fetch (often fails: 403 / 0 parts).
    # Set ENABLE_LIVE_PARTS_FETCH=1 to enable; otherwise we return [] and rely on ingest (--from-html / --from-csv).
    if (
        model_number
        and not part_number
        and not brand
        and len(rows) == 0
        and os.environ.get("ENABLE_LIVE_PARTS_FETCH", "").lower() in ("1", "true", "yes")
    ):
        try:
            from app.partselect_fetch import fetch_parts_for_model_sync
            live_parts = await asyncio.to_thread(fetch_parts_for_model_sync, model_number.strip(), True)
            if live_parts:
                await asyncio.to_thread(_write_fitment_and_parts, model_number.strip(), live_parts)
                result = [_row_from_live_part(p, model_number) for p in live_parts[:limit]]
                set_model_parts_cached(model_number, result)
                return result
        except ImportError:
            logger.debug("playwright not installed; skip live PartSelect fetch")
        except Exception as e:
            logger.warning("live PartSelect fetch failed for model %s: %s", model_number, e)

    # When model_number + symptom returned 0: try symptom-only search (no fitment) so we still show relevant parts.
    if (
        model_number
        and not part_number
        and not brand
        and (symptom or "").strip()
        and len(rows) == 0
    ):
        def _search_symptom_only():
            conn = _get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT part_id, part_number, partselect_number, manufacturer_part_number,
                               name, brand, price, url, image_url
                        FROM parts
                        WHERE name ILIKE %s
                        LIMIT %s
                        """,
                        (f"%{(symptom or '').strip()[:80]}%", limit),
                    )
                    rs = cur.fetchall()
                    desc = cur.description or []
                    cols = [c.name for c in desc]
                    out_list = [dict(zip(cols, r)) for r in rs]
                    for row in out_list:
                        row.setdefault("difficulty", None)
                        row.setdefault("time_estimate", None)
                        if not (row.get("image_url") or "").strip():
                            row["image_url"] = _part_image_url_fallback(row)
                    return out_list
            finally:
                conn.close()

        try:
            rows = await asyncio.to_thread(_search_symptom_only)
        except Exception as e:
            logger.debug("symptom-only search_parts fallback failed: %s", e)

    # On model_number and still no rows: SerpApi fallback (no pre-crawl needed). Set SERPAPI_API_KEY to enable.
    if (
        model_number
        and not part_number
        and not brand
        and len(rows) == 0
        and os.environ.get("SERPAPI_API_KEY", "").strip()
    ):
        try:
            from app.serp import search_serp
            symptom_bit = f" {(symptom or '').strip()[:60]}" if (symptom or "").strip() else ""
            query = f"site:partselect.com {model_number.strip()}{symptom_bit} parts".strip()
            serp_results = await asyncio.to_thread(search_serp, query, num=limit)
            if serp_results:
                return [_row_from_serp(r) for r in serp_results]
        except Exception as e:
            logger.warning("SerpApi fallback failed for model %s: %s", model_number, e)

    if model_number and not part_number and not brand and rows and not rows_from_symptom_fallback:
        set_model_parts_cached(model_number, rows)
    return rows


async def check_compatibility(model_number: str, part_number: str) -> dict:
    """
    Check if a part is compatible with a model. Uses part_fitment (source of truth).
    part_number can be PartSelect number (PS...) or manufacturer part number; we resolve to partselect_number.
    Returns: compatible (bool), partselect_number, model_number, fit_source, message.
    """
    if not (model_number or "").strip() or not (part_number or "").strip():
        return {
            "compatible": False,
            "partselect_number": None,
            "model_number": (model_number or "").strip(),
            "fit_source": None,
            "message": "Missing model number or part number.",
        }
    model_number = model_number.strip()
    part_number = part_number.strip()

    def _check():
        conn = _get_connection()
        try:
            with conn.cursor() as cur:
                # Resolve part_number to partselect_number
                cur.execute(
                    "SELECT partselect_number FROM parts WHERE partselect_number = %s OR manufacturer_part_number = %s OR part_number = %s LIMIT 1",
                    (part_number, part_number, part_number),
                )
                row = cur.fetchone()
                if not row:
                    return {
                        "compatible": False,
                        "partselect_number": part_number,
                        "model_number": model_number,
                        "fit_source": None,
                        "message": f"Part '{part_number}' not in catalog; will try Serp for verification links.",
                    }
                ps_num = row[0] or part_number
                cur.execute(
                    "SELECT fit_source FROM part_fitment WHERE partselect_number = %s AND model_number = %s LIMIT 1",
                    (ps_num, model_number),
                )
                fit = cur.fetchone()
                if fit:
                    return {
                        "compatible": True,
                        "partselect_number": ps_num,
                        "model_number": model_number,
                        "fit_source": fit[0],
                        "message": f"Part {ps_num} fits model {model_number}.",
                    }
                return {
                    "compatible": False,
                    "partselect_number": ps_num,
                    "model_number": model_number,
                    "fit_source": None,
                    "message": f"No fitment record for part {ps_num} and model {model_number}. Check model number or part.",
                }
        finally:
            conn.close()

    try:
        out = await asyncio.to_thread(_check)
        # When not compatible (or part not in catalog), add PartSelect links via Serp so user can verify.
        if not out.get("compatible") and os.environ.get("SERPAPI_API_KEY", "").strip():
            try:
                from app.serp import search_serp
                ps = out.get("partselect_number") or part_number
                q = f"site:partselect.com {model_number} {ps} compatibility"
                links = await asyncio.to_thread(search_serp, q, num=5)
                if links:
                    out["serp_links"] = [{"title": r.get("title"), "link": r.get("link")} for r in links]
            except Exception as e:
                logger.debug("SerpApi links for compatibility failed: %s", e)
        return out
    except Exception as e:
        logger.exception("check_compatibility failed: %s", e)
        return {
            "compatible": False,
            "partselect_number": None,
            "model_number": model_number,
            "fit_source": None,
            "message": f"Compatibility check failed: {e}",
        }


async def get_troubleshooting(
    symptom: Optional[str] = None,
    model_number: Optional[str] = None,
    appliance_type: Optional[str] = None,
    symptom_tag: Optional[str] = None,
    allowed_symptom_tags: Optional[List[str]] = None,
    forbidden_symptom_tags: Optional[List[str]] = None,
    limit: int = 5,
    guides_only: bool = False,
) -> List[dict]:
    """Retrieve troubleshooting steps from RAG. When guides_only=True (troubleshoot without model), only repair-guide chunks are returned (no part catalog)."""
    query = symptom or ""
    if model_number:
        query = f"{query} {model_number}".strip()
    if appliance_type:
        query = f"{query} {appliance_type}".strip()
    if not query.strip():
        query = "refrigerator dishwasher troubleshooting repair"
    return await asyncio.to_thread(
        retrieval.search_chunks,
        query,
        limit,
        symptom_tag,
        allowed_symptom_tags,
        forbidden_symptom_tags,
        appliance_type,
        guides_only,
    )
