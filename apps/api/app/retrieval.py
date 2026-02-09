
"""
RAG retrieval: embed user query with OpenAI, search chunks by cosine similarity.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from openai import OpenAI


logger = logging.getLogger(__name__)
EMBEDDING_MODEL = "text-embedding-3-small"
DIMENSIONS = 1536


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv as _load
        # app/retrieval.py -> parents[1]=api, parents[3]=repo root
        api_dir = Path(__file__).resolve().parents[1]
        repo_root = Path(__file__).resolve().parents[3]
        _load(api_dir / ".env", override=True)
        _load(repo_root / ".env", override=True)
    except ImportError:
        pass


def get_connection():
    _load_dotenv()
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL not set (add to apps/api/.env or repo root .env)")
    return psycopg.connect(url)


def embed_query(query: str) -> list[float]:
    """Single query → 1536-dim vector (same model as ingest)."""
    _load_dotenv()
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key or key == "sk-...":
        logger.warning("OPENAI_API_KEY missing or placeholder — RAG search may be poor (using zero vector)")
        return [0.0] * DIMENSIONS
    client = OpenAI(api_key=key)
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query],
        dimensions=DIMENSIONS,
    )
    return list(resp.data[0].embedding)


def search_chunks(
    query: str,
    limit: int = 5,
    symptom_tag: str | None = None,
    allowed_symptom_tags: list[str] | None = None,
    forbidden_symptom_tags: list[str] | None = None,
    appliance_type: str | None = None,
    guides_only: bool = False,
) -> list[dict[str, Any]]:
    """
    Vector search on chunks. State-based filter:
    - allowed_symptom_tags / forbidden_symptom_tags: filter by metadata.symptom_tag
    - appliance_type: when set, prefer chunks with metadata.appliance_type = value (allow NULL for backward compat)
    - guides_only: when True (troubleshoot without model), exclude chunks with metadata.content_type = 'part_catalog'
    """
    if not query or not query.strip():
        return []
    try:
        embedding = embed_query(query.strip())
    except Exception as e:
        logger.exception("RAG embed_query failed: %s", e)
        return []
    try:
        conn = get_connection()
    except Exception as e:
        logger.exception("RAG get_connection failed: %s", e)
        return []
    try:
        register_vector(conn)
        query_vec = Vector(embedding)
        allowed = allowed_symptom_tags if allowed_symptom_tags is not None else ([symptom_tag] if symptom_tag else [])
        forbidden = forbidden_symptom_tags or []

        conditions = ["c.embedding IS NOT NULL"]
        params: list[Any] = []
        if allowed:
            conditions.append("c.metadata->>'symptom_tag' = ANY(%s)")
            params.append(allowed)
        if forbidden:
            conditions.append("(c.metadata->>'symptom_tag' IS NULL OR NOT (c.metadata->>'symptom_tag' = ANY(%s)))")
            params.append(forbidden)
        if appliance_type:
            conditions.append("(c.metadata->>'appliance_type' IS NULL OR c.metadata->>'appliance_type' = %s)")
            params.append(appliance_type)
        if guides_only:
            # Exclude part-catalog chunks: by content_type (from ingest) or by URL (backward compat for existing chunks)
            # URL: /Part-, /Parts/, -parts, ice-maker-parts, ice-makers (category pages)
            conditions.append(
                "(c.metadata->>'content_type' IS NULL OR c.metadata->>'content_type' <> 'part_catalog') "
                "AND (c.metadata->>'content_type' IS NOT NULL OR ("
                "LOWER(COALESCE(c.metadata->>'url', '')) NOT LIKE '%%/part-%%' AND "
                "LOWER(COALESCE(c.metadata->>'url', '')) NOT LIKE '%%/parts/%%' AND "
                "LOWER(COALESCE(c.metadata->>'url', '')) NOT LIKE '%%-parts%%' AND "
                "LOWER(COALESCE(c.metadata->>'url', '')) NOT LIKE '%%ice-maker-parts%%' AND "
                "LOWER(COALESCE(c.metadata->>'url', '')) NOT LIKE '%%/ice-makers%%' AND "
                "LOWER(COALESCE(c.metadata->>'url', '')) NOT LIKE '%%ice-makers%%'"
                "))"
            )
        params.extend([query_vec, limit])
        where = " AND ".join(conditions)
        cur = conn.execute(
            f"""
            SELECT c.chunk_id, c.text, c.metadata
            FROM chunks c
            WHERE {where}
            ORDER BY c.embedding <=> %s
            LIMIT %s
            """,
            tuple(params),
        )
        rows = cur.fetchall()
        # Fallback: if strict filter returns nothing, retry with appliance_type only (so "dishwasher do not dry" still gets dishwasher chunks)
        if not rows and (allowed or forbidden) and appliance_type:
            conditions_fallback = ["c.embedding IS NOT NULL", "(c.metadata->>'appliance_type' IS NULL OR c.metadata->>'appliance_type' = %s)"]
            if guides_only:
                conditions_fallback.append(
                    "(c.metadata->>'content_type' IS NULL OR c.metadata->>'content_type' <> 'part_catalog') "
                    "AND (c.metadata->>'content_type' IS NOT NULL OR ("
                    "LOWER(COALESCE(c.metadata->>'url', '')) NOT LIKE '%%/part-%%' AND "
                    "LOWER(COALESCE(c.metadata->>'url', '')) NOT LIKE '%%/parts/%%' AND "
                    "LOWER(COALESCE(c.metadata->>'url', '')) NOT LIKE '%%-parts%%' AND "
                    "LOWER(COALESCE(c.metadata->>'url', '')) NOT LIKE '%%ice-maker-parts%%' AND "
                    "LOWER(COALESCE(c.metadata->>'url', '')) NOT LIKE '%%/ice-makers%%' AND "
                    "LOWER(COALESCE(c.metadata->>'url', '')) NOT LIKE '%%ice-makers%%'"
                    "))"
                )
            params_fb = [appliance_type, query_vec, limit]
            cur = conn.execute(
                """
                SELECT c.chunk_id, c.text, c.metadata
                FROM chunks c
                WHERE """ + " AND ".join(conditions_fallback) + """
                ORDER BY c.embedding <=> %s
                LIMIT %s
                """,
                tuple(params_fb),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    out = []
    for chunk_id, text, metadata in rows:
        meta = metadata or {}
        out.append({
            "chunk_id": chunk_id,
            "text": text,
            "title": meta.get("title", ""),
            "source_url": meta.get("url", ""),
            "source": meta.get("source", ""),
        })
    return out
