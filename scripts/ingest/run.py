#!/usr/bin/env python3
"""
RAG ingestion pipeline: high-quality sources → clean HTML → chunk by steps/Q&A/symptoms → embed → store.
Rule: fewer documents, higher quality > large noisy corpus.

Usage:
  From repo root:  python -m scripts.ingest.run
  From scripts/ingest:  python run.py  (with .env and sources in ./sources)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# So that chunker, db, embedder, html_cleaner resolve (from scripts/ingest or repo root)
scripts_ingest = Path(__file__).resolve().parent
repo_root = scripts_ingest.parents[1]
sys.path.insert(0, str(scripts_ingest))
sys.path.insert(0, str(repo_root))
os.chdir(scripts_ingest)

# Load .env from scripts/ingest (or repo root)
try:
    from dotenv import load_dotenv
    load_dotenv(scripts_ingest / ".env")
    load_dotenv(repo_root / ".env")
except ImportError:
    pass

from chunker import Chunk, html_to_chunks, plain_text_to_chunks
from db import db_connection, delete_documents_by_source, ensure_vector_extension, insert_chunk, insert_document
from embedder import get_embeddings
from html_cleaner import html_to_structured_text, iter_html_sources, iter_text_sources


def main() -> None:
    sources_dir = Path(os.environ.get("SOURCES_DIR", str(scripts_ingest / "sources")))
    if not sources_dir.is_dir():
        print(f"SOURCES_DIR not found: {sources_dir}")
        print("Create it and add .html/.txt files, or run fetch.py first.")
        sys.exit(1)

    doc_chunks: list[tuple[int, Chunk]] = []
    docs_raw: list[tuple[str, str, str]] = []

    def process_doc(conn, source_id: str, url: str, raw_text: str, chunks: list) -> None:
        if not chunks:
            return
        if len(raw_text) > 100_000:
            raw_text = raw_text[:100_000] + "\n...[truncated]"
        doc_id = insert_document(conn, source=source_id, url=url, raw_text=raw_text)
        for ch in chunks:
            doc_chunks.append((doc_id, ch))
        docs_raw.append((source_id, url, raw_text))
        print(f"  doc {doc_id}: {source_id} -> {len(chunks)} chunks")

    with db_connection() as conn:
        ensure_vector_extension(conn)

        for source_id, url, html in iter_html_sources(sources_dir):
            delete_documents_by_source(conn, source_id)
            chunks = html_to_chunks(html, source=source_id, url=url)
            raw_text = html_to_structured_text(html, base_title=source_id)
            process_doc(conn, source_id, url, raw_text, chunks)

        for source_id, url, text in iter_text_sources(sources_dir):
            delete_documents_by_source(conn, source_id)
            chunks = plain_text_to_chunks(text, source=source_id, url=url, base_title=source_id)
            process_doc(conn, source_id, url, text, chunks)

    if not doc_chunks:
        print("No chunks produced. Add .html files to", sources_dir)
        return

    texts = [ch.text for _, ch in doc_chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = get_embeddings(texts)

    with db_connection() as conn:
        for (doc_id, ch), emb in zip(doc_chunks, embeddings):
            insert_chunk(conn, doc_id, ch.text, ch.metadata, emb)

    print(f"Done: {len(docs_raw)} documents, {len(doc_chunks)} chunks stored.")


if __name__ == "__main__":
    main()
