#!/usr/bin/env python3
"""
Update documents.url and chunks.metadata->url from "saved from url" in HTML files.
Run from scripts/ingest with .env and sources in ./sources. No re-embedding.
"""
from __future__ import annotations

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

from html_cleaner import extract_saved_from_url, load_html_path
from db import db_connection


def main() -> None:
    sources_dir = Path(os.environ.get("SOURCES_DIR", str(scripts_ingest / "sources")))
    if not sources_dir.is_dir():
        print(f"SOURCES_DIR not found: {sources_dir}")
        sys.exit(1)

    updates: list[tuple[str, str]] = []
    for f in sorted(sources_dir.glob("**/*.html")):
        try:
            html = load_html_path(f)
            if len(html.strip()) < 50:
                continue
            canonical = extract_saved_from_url(html)
            if not canonical:
                print(f"  Skip (no saved-from URL): {f.name}")
                continue
            source_id = f.stem
            updates.append((source_id, canonical))
        except Exception as e:
            print(f"  Error reading {f.name}: {e}")

    if not updates:
        print("No HTML files with 'saved from url' found.")
        return

    with db_connection() as conn:
        for source_id, url in updates:
            cur = conn.execute(
                "UPDATE documents SET url = %s WHERE source = %s RETURNING doc_id",
                (url, source_id),
            )
            doc_ids = [r[0] for r in cur.fetchall()]
            if not doc_ids:
                print(f"  No document for source: {source_id}")
                continue
            conn.execute(
                """
                UPDATE chunks SET metadata = jsonb_set(metadata, '{url}', to_jsonb(%s::text))
                WHERE doc_id = ANY(%s)
                """,
                (url, doc_ids),
            )
            print(f"  Updated {source_id} -> {url} ({len(doc_ids)} doc(s))")
    print("Done. Sources in chat should now open the correct PartSelect URLs.")


if __name__ == "__main__":
    main()
