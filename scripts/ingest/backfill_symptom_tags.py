#!/usr/bin/env python3
"""
Backfill chunk metadata with symptom_tag, section, appliance from document URL.
Run from scripts/ingest (with .env). No re-embedding. After this, RAG filter by symptom_tag will work.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

scripts_ingest = Path(__file__).resolve().parent
repo_root = scripts_ingest.parents[1]
sys.path.insert(0, str(scripts_ingest))
os.chdir(scripts_ingest)

try:
    from dotenv import load_dotenv
    load_dotenv(scripts_ingest / ".env")
    load_dotenv(repo_root / ".env")
except ImportError:
    pass

from chunker import url_to_symptom_tags
from db import db_connection, get_connection


def main() -> None:
    conn = get_connection()
    try:
        cur = conn.execute("SELECT doc_id, url FROM documents WHERE url IS NOT NULL AND url != ''")
        rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        print("No documents with URL found. Run ingest first.")
        return

    with db_connection() as conn:
        for doc_id, url in rows:
            tags = url_to_symptom_tags(url)
            if not tags:
                continue
            conn.execute(
                "UPDATE chunks SET metadata = COALESCE(metadata, '{}') || %s::jsonb WHERE doc_id = %s",
                (json.dumps(tags), doc_id),
            )
            print(f"  doc_id={doc_id} url={url[:60]}... -> {tags}")
    print("Done. Chunks now have symptom_tag for RAG filtering.")


if __name__ == "__main__":
    main()
