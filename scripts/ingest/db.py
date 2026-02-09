"""
Store documents and chunks (with embeddings) in PostgreSQL + pgvector.
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Any, Iterator

import psycopg
from pgvector.psycopg import register_vector


def get_connection():
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL not set")
    return psycopg.connect(url)


@contextmanager
def db_connection():
    conn = get_connection()
    try:
        register_vector(conn)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def ensure_vector_extension(conn) -> None:
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")


def insert_document(conn, source: str, url: str, raw_text: str) -> int:
    """Insert one row into documents; return doc_id."""
    row = conn.execute(
        """
        INSERT INTO documents (source, url, raw_text)
        VALUES (%s, %s, %s)
        RETURNING doc_id
        """,
        (source, url, raw_text),
    ).fetchone()
    return row[0]


def insert_chunk(conn, doc_id: int, text: str, metadata: dict[str, Any], embedding: list[float]) -> None:
    """Insert one chunk with embedding. metadata stored as JSONB."""
    conn.execute(
        """
        INSERT INTO chunks (doc_id, text, metadata, embedding)
        VALUES (%s, %s, %s, %s)
        """,
        (doc_id, text, json.dumps(metadata), embedding),
    )


def delete_documents_by_source(conn, source: str) -> int:
    """Delete all documents (and their chunks) with given source. Return count of docs deleted."""
    # Delete chunks first (FK: chunks.doc_id → documents.doc_id)
    conn.execute(
        "DELETE FROM chunks WHERE doc_id IN (SELECT doc_id FROM documents WHERE source = %s)",
        (source,),
    )
    cur = conn.execute(
        "DELETE FROM documents WHERE source = %s RETURNING doc_id",
        (source,),
    )
    return len(cur.fetchall())
