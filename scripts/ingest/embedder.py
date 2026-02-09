"""
Generate embeddings via OpenAI (text-embedding-3-small, 1536 dims).
"""
from __future__ import annotations

import os
from openai import OpenAI


EMBEDDING_MODEL = "text-embedding-3-small"
DIMENSIONS = 1536


def get_embeddings(texts: list[str], client: OpenAI | None = None) -> list[list[float]]:
    """
    Embed a batch of texts. OpenAI allows up to 2048 inputs per request; we batch smaller.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    if not texts:
        return []
    client = client or OpenAI(api_key=api_key)
    batch_size = 100
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
            dimensions=DIMENSIONS,
        )
        ordered = sorted(resp.data, key=lambda x: x.index)
        for e in ordered:
            out.append(list(e.embedding))
    return out
