#!/usr/bin/env python3
"""
Fetch content for RAG: scrape HTML from URLs, download PDFs and extract text.
Output: sources/*.html and sources/*.txt for run.py to chunk and embed.

Config: sources_config.yaml (see sources_config.example.yaml).
Usage: cd scripts/ingest && python fetch.py
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import httpx
import yaml

# Optional PDF text extraction
try:
    import pdfplumber
except ImportError:
    pdfplumber = None

# Blocked/error page detection (e.g. CDN "Access Denied")
def _is_blocked_or_error_page(html: str) -> bool:
    if not html or len(html) < 200:
        return True
    lower = html.lower()
    if "access denied" in lower or "you don't have permission" in lower:
        return True
    if "<title>access denied</title>" in lower or "<h1>access denied</h1>" in lower:
        return True
    return False


# Optional: headless browser when site returns 403 to plain HTTP
def _fetch_html_playwright(url: str) -> str:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=REQUEST_TIMEOUT * 1000)
        html = page.content()
        browser.close()
    return html

SCRIPTS_INGEST = Path(__file__).resolve().parent
SOURCES_DIR = SCRIPTS_INGEST / "sources"
CONFIG_NAMES = ["sources_config.yaml", "sources_config.example.yaml"]
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
REQUEST_TIMEOUT = 30.0


def _safe_filename(name: str) -> str:
    return re.sub(r"[^\w\-.]", "_", name).strip("_") or "unnamed"


def load_config() -> dict[str, Any]:
    for n in CONFIG_NAMES:
        p = SCRIPTS_INGEST / n
        if p.is_file():
            with open(p) as f:
                return yaml.safe_load(f) or {}
    return {}


def fetch_html_url(url: str, client: httpx.Client) -> str:
    r = client.get(url)
    r.raise_for_status()
    r.encoding = r.encoding or "utf-8"
    return r.text


def fetch_pdf_to_text(url: str, client: httpx.Client) -> str:
    r = client.get(url)
    r.raise_for_status()
    if not pdfplumber:
        return "[PDF text extraction not available: install pdfplumber]"
    import io
    buf = io.BytesIO(r.content)
    text_parts = []
    with pdfplumber.open(buf) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
    return "\n\n".join(text_parts) if text_parts else ""


def main() -> None:
    config = load_config()
    if not config:
        print("No sources_config.yaml or sources_config.example.yaml found. Create from example.")
        return

    SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    delay = float(config.get("fetch_delay_seconds", 1.0))

    with httpx.Client(
        follow_redirects=True,
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        # HTML URLs: try httpx first, on 403 use headless browser
        for item in config.get("html_urls") or []:
            url = item.get("url")
            name = item.get("name") or _safe_filename(url)
            if not url:
                continue
            html = None
            try:
                html = fetch_html_url(url, client)
            except Exception as e:
                if hasattr(e, "response") and getattr(e.response, "status_code", None) == 403:
                    try:
                        print(f"  403 → using browser for {name}...")
                        html = _fetch_html_playwright(url)
                        if _is_blocked_or_error_page(html):
                            print(f"  blocked: {name} (server returned Access Denied; try saving page manually)")
                            html = None
                    except Exception as e2:
                        print(f"  skip {url}: {e2}")
                        html = None
                else:
                    print(f"  skip {url}: {e}")
            if html:
                path = SOURCES_DIR / f"{_safe_filename(name)}.html"
                path.write_text(html, encoding="utf-8")
                print(f"  saved: {_safe_filename(name)}.html")
            time.sleep(delay)

        # PDF URLs → extract text → .txt
        for item in config.get("pdf_urls") or []:
            url = item.get("url")
            name = item.get("name") or _safe_filename(url or "pdf")
            if not url:
                continue
            try:
                text = fetch_pdf_to_text(url, client)
                path = SOURCES_DIR / f"{_safe_filename(name)}.txt"
                path.write_text(text, encoding="utf-8")
                print(f"  saved: {_safe_filename(name)}.txt (from PDF)")
            except Exception as e:
                print(f"  skip {url}: {e}")
            time.sleep(delay)

    print("Fetch done. Run python run.py to chunk and embed.")


if __name__ == "__main__":
    main()
