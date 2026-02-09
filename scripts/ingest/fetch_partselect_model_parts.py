#!/usr/bin/env python3
"""
Fetch parts for each model from PartSelect model Parts pages.
Fills part_fitment (model ↔ part) and ensures parts table has minimal rows so search_parts(model_number) works.

Live fetch often gets 0 parts (page is JS-rendered). Prefer one of:

  1) --from-html: parse after saving the Parts page in the browser
     - Open e.g. https://www.partselect.com/Models/3000W10/Parts/, Save As "Web Page, HTML only" or "Complete"
     - Save as 3000W10.html (filename = model number), e.g. under sources/model_parts/
     - Run: python -m scripts.ingest.fetch_partselect_model_parts --from-html sources/model_parts/
     - Or single file: python -m scripts.ingest.fetch_partselect_model_parts --from-html sources/model_parts/3000W10.html

  2) --from-csv: import from existing (model_number, partselect_number) CSV
     - CSV header: model_number, partselect_number
     - Run: python -m scripts.ingest.fetch_partselect_model_parts --from-csv config/part_fitment.csv

  3) Live fetch (Playwright + scroll for lazy load + .mega-m__part parsing):
     python -m scripts.ingest.fetch_partselect_model_parts config/partselect_dishwasher_models.csv
     python -m scripts.ingest.fetch_partselect_model_parts config/partselect_dishwasher_models.csv --limit 2 --no-headless  # show browser

  4) Jina / Firecrawl / Bright Data (URL → markdown → extract part numbers → DB):
     python -m scripts.ingest.fetch_partselect_model_parts --via-jina config/partselect_dishwasher_models.csv [--limit N]
     python -m scripts.ingest.fetch_partselect_model_parts --via-brightdata config/partselect_dishwasher_models.csv  # bypasses 403 (Bright Data Web Unlocker; same as brightdata-mcp)
     Note: Jina/Firecrawl often get 403 on PartSelect; --via-brightdata or --from-html/--from-csv are more reliable.

  5) XML Sitemap: get model (and optional part) URLs without crawling the homepage. PartSelect may 403 sitemap.xml; save from browser and pass file path.
     python -m scripts.ingest.fetch_partselect_model_parts --from-sitemap https://www.partselect.com/sitemap.xml [--sitemap-follow-index 20] [--output-models-csv config/models_from_sitemap.csv]
     python -m scripts.ingest.fetch_partselect_model_parts --from-sitemap path/to/saved_sitemap.xml --output-models-csv config/models_from_sitemap.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import httpx
from bs4 import BeautifulSoup

SCRIPTS_INGEST = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_INGEST.parents[1]
BASE_URL = "https://www.partselect.com"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
REQUEST_TIMEOUT = 60.0  # PartSelect can be slow; networkidle often never fires
DELAY_BETWEEN_REQUESTS = 1.5

# PartSelect part number: PS + digits (e.g. PS11752778)
PART_NUMBER_RE = re.compile(r"\b(PS\d{6,})\b", re.IGNORECASE)
HREF_PS_RE = re.compile(r"(?:^|/)(PS\d{6,})(?:[-/]|$)", re.IGNORECASE)
# Saved HTML may have partselect.com/PS123 or partselect.com/PartSelect/PS123
HREF_PS_ANY_RE = re.compile(r"partselect\.com[^\"'\s]*?(PS\d{6,})", re.IGNORECASE)


def _fetch_url(url: str, client: httpx.Client) -> str:
    r = client.get(url)
    r.raise_for_status()
    r.encoding = r.encoding or "utf-8"
    return r.text


def _fetch_url_playwright(url: str) -> str:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle", timeout=int(REQUEST_TIMEOUT * 1000))
        page.wait_for_timeout(2000)
        html = page.content()
        browser.close()
    return html


def _parse_part_items_from_page(page) -> list[dict]:
    """Parse .mega-m__part from current page; return list of part dicts (no dedup)."""
    out: list[dict] = []
    items = page.query_selector_all(".mega-m__part")
    for item in items:
        try:
            name_el = item.query_selector(".mega-m__part__name")
            name = name_el.inner_text().strip() if name_el else "Unknown"
            link = ""
            if name_el:
                link = name_el.get_attribute("href") or ""
            full_link = f"https://www.partselect.com{link}" if link else ""
            raw_text = item.inner_text()
            ps_num = "N/A"
            mfg_num = "N/A"
            for line in raw_text.split("\n"):
                if "PartSelect #:" in line:
                    ps_num = line.replace("PartSelect #:", "").strip()
                elif "Manufacturer #:" in line:
                    mfg_num = line.replace("Manufacturer #:", "").strip()
            if not ps_num or ps_num == "N/A":
                if link and PART_NUMBER_RE.search(link):
                    ps_num = PART_NUMBER_RE.search(link).group(1).upper()
                else:
                    continue
            price_el = item.query_selector(".mega-m__part__price")
            price = price_el.inner_text().strip().replace("\n", "") if price_el else None
            img_el = item.query_selector(".mega-m__part__img img, .mega-m__part img, img")
            image_url = None
            if img_el:
                src = (img_el.get_attribute("src") or "").strip()
                if src:
                    image_url = f"https://www.partselect.com{src}" if src.startswith("/") else src
            out.append({
                "partselect_number": ps_num.upper(),
                "name": name,
                "manufacturer_part_number": mfg_num if mfg_num != "N/A" else None,
                "price": price,
                "url": full_link,
                "image_url": image_url,
            })
        except Exception:
            continue
    return out


def _find_next_button(page) -> bool:
    """Find and click the Next pagination button; return True on success, else False."""
    # 1) Common class/rel selectors
    for selector in (
        "a.pagination__next",
        "li.next a",
        "a[rel='next']",
        ".pagination a[aria-label='Next']",
        "a[aria-label='Next page']",
        "a.next",
    ):
        btn = page.query_selector(selector)
        if btn:
            try:
                btn.click()
                return True
            except Exception:
                pass
    # 2) By link text ("Next" or "Next »")
    try:
        for name in ("Next", "Next »", "Next »", "next"):
            link = page.get_by_role("link", name=name)
            if link.count() > 0:
                link.first.click()
                return True
    except Exception:
        pass
    # 3) Iterate all a/button with text containing Next
    for sel in ("a", "button"):
        for el in page.query_selector_all(sel):
            try:
                t = (el.inner_text() or "").strip()
                if "next" in t.lower() and len(t) < 15:
                    # Avoid clicking e.g. "Go to next section"
                    el.click()
                    return True
            except Exception:
                continue
    return False


def _base_model_url(url: str) -> str:
    """Normalize to base model URL with trailing slash, e.g. .../Models/003719074/"""
    u = (url or "").rstrip("/")
    if "/Parts" in u:
        u = u.split("/Parts")[0]
    return u + "/"


def fetch_parts_with_playwright(
    model_url: str,
    headless: bool = True,
    verbose: bool = False,
) -> list[dict]:
    """
    Pagination: first page Parts/, then Parts/?start=2, start=3, ...; Referer header to reduce 403.
    Returns [{"partselect_number", "name", ...}, ...] (deduped by partselect_number).
    """
    from playwright.sync_api import sync_playwright
    base_url = _base_model_url(model_url)
    seen_ps: set[str] = set()
    out: list[dict] = []
    start_index = 1
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            extra_http_headers={
                "Accept-Language": "en-US,en;q=0.9",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )
        page = context.new_page()
        while True:
            if start_index == 1:
                target_url = f"{base_url}Parts/"
            else:
                target_url = f"{base_url}Parts/?start={start_index}"
                page.set_extra_http_headers({"Referer": base_url})
            if verbose:
                print(f"      Parts/?start={start_index} ..." if start_index > 1 else "      Parts/ ...")
            try:
                page.goto(target_url, wait_until="domcontentloaded", timeout=int(REQUEST_TIMEOUT * 1000))
                page.wait_for_timeout(3000)
                page.mouse.wheel(0, 2000)
                page.wait_for_timeout(1000)
            except Exception as e:
                if verbose:
                    print(f"      goto failed: {e}")
                break
            parts_this_page = _parse_part_items_from_page(page)
            if not parts_this_page:
                if verbose:
                    print(f"      start={start_index}: 0 items, done.")
                break
            new_count = 0
            for p in parts_this_page:
                ps = p["partselect_number"]
                if ps not in seen_ps:
                    seen_ps.add(ps)
                    out.append(p)
                    new_count += 1
            if verbose:
                print(f"      start={start_index}: {len(parts_this_page)} items ({new_count} new), total {len(out)}")
            start_index += 1
        browser.close()
    return out


def _fetch_html(url: str, client: httpx.Client, use_playwright_on_403: bool = True) -> str:
    try:
        return _fetch_url(url, client)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 403 and use_playwright_on_403:
            print("    (403 → trying Playwright ...)")
            return _fetch_url_playwright(url)
        raise


def extract_part_numbers_from_parts_page(html: str) -> set[str]:
    """Parse a model Parts page for PartSelect part numbers (PS + digits). Works on saved HTML from browser."""
    found: set[str] = set()
    # Full text / hrefs
    for m in HREF_PS_RE.finditer(html):
        found.add(m.group(1).upper())
    for m in HREF_PS_ANY_RE.finditer(html):
        found.add(m.group(1).upper())
    for m in PART_NUMBER_RE.finditer(html):
        found.add(m.group(1).upper())
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a.get("href", "") or ""
        for m in HREF_PS_RE.finditer(href):
            found.add(m.group(1).upper())
        for m in HREF_PS_ANY_RE.finditer(href):
            found.add(m.group(1).upper())
    return found


def extract_part_numbers_from_markdown(md: str) -> set[str]:
    """Extract PartSelect part numbers (PS + digits) from markdown (e.g. from Jina Reader / Firecrawl)."""
    found: set[str] = set()
    if not md:
        return found
    for m in PART_NUMBER_RE.finditer(md):
        found.add(m.group(1).upper())
    for m in HREF_PS_ANY_RE.finditer(md):
        found.add(m.group(1).upper())
    # Markdown links: ](https://.../PS123...)
    for m in re.finditer(r"\]\([^)]*?(PS\d{6,})[^)]*\)", md, re.IGNORECASE):
        found.add(m.group(1).upper())
    return found


def _markdown_looks_blocked(md: str) -> bool:
    """True if the fetched markdown is an error page (403 / Access Denied) rather than real content."""
    if not md or len(md) < 50:
        return True
    lower = md.lower()
    if "access denied" in lower or "403" in lower or "forbidden" in lower:
        return True
    if "don't have permission" in lower or "you don't have permission" in lower:
        return True
    if "error" in lower and "reference #" in lower:
        return True
    return False


def fetch_markdown_jina(url: str, api_key: Optional[str] = None, timeout: float = 60.0) -> str:
    """
    Fetch URL via Jina Reader (r.jina.ai); returns markdown. Handles JS-rendered pages.
    Without API key: 20 RPM. With JINA_API_KEY: higher rate limit.
    """
    reader_url = f"https://r.jina.ai/{url}"
    headers = {"Accept": "text/plain"}
    if api_key and api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        r = client.get(reader_url, headers=headers)
        r.raise_for_status()
        return (r.text or "").strip()


def fetch_markdown_firecrawl(url: str, api_key: str, timeout: float = 60.0) -> str:
    """
    Fetch URL via Firecrawl /v2/scrape; returns markdown. Handles JS-rendered pages.
    Requires FIRECRAWL_API_KEY.
    """
    if not (api_key or "").strip():
        raise ValueError("Firecrawl requires FIRECRAWL_API_KEY")
    with httpx.Client(timeout=timeout) as client:
        r = client.post(
            "https://api.firecrawl.dev/v1/scrape",
            headers={
                "Authorization": f"Bearer {api_key.strip()}",
                "Content-Type": "application/json",
            },
            json={"url": url, "formats": ["markdown"]},
        )
        r.raise_for_status()
        data = r.json()
        if not data.get("success"):
            raise RuntimeError(data.get("error") or "Firecrawl scrape failed")
        out = data.get("data") or data
        md = out.get("markdown") or ""
        return (md or "").strip()


def fetch_markdown_brightdata(
    url: str,
    api_key: str,
    zone: str = "web_unlocker1",
    timeout: float = 90.0,
) -> str:
    """
    Fetch URL via Bright Data Web Unlocker API; returns markdown. Bypasses 403/anti-bot.
    Same backend as Bright Data MCP's scrape_as_markdown (https://github.com/brightdata/brightdata-mcp).
    Requires BRIGHTDATA_API_KEY; optional BRIGHTDATA_ZONE (default web_unlocker1).
    """
    if not (api_key or "").strip():
        raise ValueError("Bright Data requires BRIGHTDATA_API_KEY")
    with httpx.Client(timeout=timeout) as client:
        r = client.post(
            "https://api.brightdata.com/request",
            headers={
                "Authorization": f"Bearer {api_key.strip()}",
                "Content-Type": "application/json",
            },
            json={
                "zone": (zone or "web_unlocker1").strip(),
                "url": url,
                "format": "json",
                "method": "GET",
                "country": "us",
                "data_format": "markdown",
            },
        )
        if r.status_code >= 400:
            try:
                err_body = r.json()
            except Exception:
                err_body = r.text or ""
            msg = f"Bright Data API {r.status_code}: {err_body}"
            if r.status_code == 400:
                msg += " (If 400: create a Web Unlocker zone at https://brightdata.com/cp → Web Access APIs → Create API → Web Unlocker API; then set BRIGHTDATA_ZONE to that zone name.)"
            raise RuntimeError(msg)
        r.raise_for_status()
        data = r.json()
        # Response may be {"content": "..."} or {"markdown": "..."} or nested
        md = (
            data.get("markdown")
            or data.get("content")
            or data.get("body")
            or (data.get("data") or {}).get("markdown")
            or (data.get("data") or {}).get("content")
        )
        if isinstance(md, dict):
            md = md.get("markdown") or md.get("content") or ""
        return (md or "").strip()


# Sitemap: extract <loc> URLs (handles default namespace and no namespace)
SITEMAP_LOC_RE = re.compile(r"<(?:\w+:)?loc>\s*([^<]+)\s*</(?:\w+:)?loc>", re.IGNORECASE)


def fetch_sitemap(url_or_path: str, timeout: float = 30.0) -> str:
    """
    Load sitemap XML from a URL (HTTP GET with browser User-Agent) or from a local file path.
    PartSelect may return 403 for sitemap.xml; use a local file saved from the browser in that case.
    """
    s = (url_or_path or "").strip()
    if not s:
        return ""
    # Local file: no scheme, or path exists
    if "://" not in s:
        path = _resolve_path(s)
        if path.is_file():
            return path.read_text(encoding="utf-8", errors="replace")
    url = s if s.startswith("http") else (f"https://www.partselect.com/{s}" if s.startswith("sitemap") else f"https://{s}")
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        r = client.get(url, headers={"User-Agent": USER_AGENT})
        # Return body on 403 so caller can detect Access Denied and suggest saving from browser
        if r.status_code == 403:
            return (r.text or "").strip()
        r.raise_for_status()
        return (r.text or "").strip()


def parse_sitemap_xml(xml: str) -> tuple[list[str], list[str]]:
    """
    Parse sitemap XML. Returns (page_urls, child_sitemap_urls).
    - For <urlset>: page_urls = all <loc>; child_sitemap_urls = [].
    - For <sitemapindex>: page_urls = []; child_sitemap_urls = all <loc> (child sitemap URLs).
    """
    page_urls: list[str] = []
    child_sitemap_urls: list[str] = []
    if not xml or "Access Denied" in xml or "403" in xml:
        return page_urls, child_sitemap_urls
    locs = [m.group(1).strip() for m in SITEMAP_LOC_RE.finditer(xml) if m.group(1).strip()]
    if not locs:
        return page_urls, child_sitemap_urls
    is_index = "<sitemapindex" in xml.lower()
    if is_index:
        child_sitemap_urls = locs
    else:
        page_urls = locs
    return page_urls, child_sitemap_urls


# URL patterns: /Models/{model}/Parts/ for model; /PS12345678 or /.../PS12345678/ for part
MODEL_FROM_PARTS_URL_RE = re.compile(r"/Models/([^/?#]+)/Parts", re.IGNORECASE)
PART_FROM_URL_RE = re.compile(r"(?:^|/)(PS\d{6,})(?:[/?#]|$)", re.IGNORECASE)


def extract_models_and_parts_from_sitemap_urls(urls: list[str]) -> tuple[set[str], list[tuple[str, str]]]:
    """
    From sitemap page URLs, extract model numbers (/Models/XXX/Parts/) and part numbers (PS...).
    Returns (model_numbers, fitment_rows). fitment_rows only has entries when both model and part
    appear in the same URL (e.g. .../Models/M/Parts/PS123/).
    """
    model_numbers: set[str] = set()
    fitment_rows: list[tuple[str, str]] = []
    for u in urls:
        model_m = MODEL_FROM_PARTS_URL_RE.search(u)
        part_m = PART_FROM_URL_RE.search(u)
        model = model_m.group(1).strip() if model_m else None
        part = part_m.group(1).upper() if part_m else None
        if model:
            model_numbers.add(model)
        if model and part:
            fitment_rows.append((model, part))
    return model_numbers, fitment_rows


def load_model_numbers_from_csv(path: Path) -> list[str]:
    """Return list of model_number from CSV (header: model_number, brand, appliance_type)."""
    models: list[str] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            mn = (row.get("model_number") or "").strip()
            if mn:
                models.append(mn)
    return models


def load_fitment_from_csv(path: Path) -> list[tuple[str, str]]:
    """Load (model_number, partselect_number) from CSV with header model_number, partselect_number."""
    rows: list[tuple[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            mn = (row.get("model_number") or "").strip()
            ps = (row.get("partselect_number") or "").strip().upper()
            if mn and ps:
                rows.append((mn, ps))
    return rows


def _resolve_path(p: str) -> Path:
    path = Path(p)
    if path.is_file() or path.is_dir():
        return path
    path = SCRIPTS_INGEST / p
    if path.is_file() or path.is_dir():
        return path
    return REPO_ROOT / p


def _write_fitment_to_db(fitment_rows: list[tuple[str, str]], fit_source: str = "partselect_model_parts") -> None:
    sys.path.insert(0, str(SCRIPTS_INGEST))
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from dotenv import load_dotenv
        load_dotenv(SCRIPTS_INGEST / ".env")
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass
    from db import db_connection
    with db_connection() as conn:
        for model_number, ps_num in fitment_rows:
            conn.execute(
                """
                INSERT INTO part_fitment (partselect_number, model_number, fit_source)
                VALUES (%s, %s, %s)
                ON CONFLICT (partselect_number, model_number) DO NOTHING
                """,
                (ps_num, model_number, fit_source),
            )
            conn.execute(
                """
                INSERT INTO parts (part_number, partselect_number, name)
                VALUES (%s, %s, %s)
                ON CONFLICT (part_number) DO NOTHING
                """,
                (ps_num, ps_num, f"Part {ps_num}"),
            )
    print(f"Wrote {len(fitment_rows)} part_fitment rows (+ parts table). Run search_parts(model_number=...) to list parts.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch parts per model from PartSelect, fill part_fitment")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=None,
        help="CSV with model_number column (for live fetch). Default: config/partselect_dishwasher_models.csv",
    )
    parser.add_argument(
        "--from-html",
        nargs="+",
        metavar="PATH",
        help="Parse saved HTML: file(s) or directory. Filename stem = model_number (e.g. 3000W10.html).",
    )
    parser.add_argument(
        "--from-csv",
        metavar="FITMENT_CSV",
        help="Import from CSV with columns model_number, partselect_number (no live fetch).",
    )
    parser.add_argument("--limit", type=int, default=0, help="Only process first N models (0 = all)")
    parser.add_argument("--parts-max-pages", type=int, default=20, metavar="N", help="When using --via-*: max pagination pages per model (Parts/, Parts/?start=2, ...). Default 20.")
    parser.add_argument("--delay", type=float, default=DELAY_BETWEEN_REQUESTS, help="Seconds between requests")
    parser.add_argument("--dry-run", action="store_true", help="Print only; do not write DB")
    parser.add_argument("--no-headless", action="store_true", help="Show browser window (Playwright live fetch)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-page part counts (see if pagination runs)")
    parser.add_argument(
        "--via-jina",
        action="store_true",
        help="Fetch each model Parts page via Jina Reader (r.jina.ai), parse markdown for part numbers, write to DB. Set JINA_API_KEY for higher rate limit.",
    )
    parser.add_argument(
        "--via-firecrawl",
        action="store_true",
        help="Fetch each model Parts page via Firecrawl API, parse markdown for part numbers, write to DB. Requires FIRECRAWL_API_KEY.",
    )
    parser.add_argument(
        "--via-brightdata",
        action="store_true",
        help="Fetch each model Parts page via Bright Data Web Unlocker (same as brightdata-mcp scrape_as_markdown). Bypasses 403. Requires BRIGHTDATA_API_KEY; optional BRIGHTDATA_ZONE (default web_unlocker1).",
    )
    parser.add_argument(
        "--from-sitemap",
        metavar="URL_OR_FILE",
        help="Parse sitemap (URL or local XML file). Extract model numbers from /Models/.../Parts/ URLs and optional (model,part) from URLs that contain both. PartSelect may 403 the sitemap URL; save sitemap.xml from browser and pass the file path.",
    )
    parser.add_argument(
        "--sitemap-follow-index",
        type=int,
        default=0,
        metavar="N",
        help="When using --from-sitemap: if the sitemap is an index, fetch up to N child sitemaps and aggregate URLs (default 0 = do not follow).",
    )
    parser.add_argument(
        "--output-models-csv",
        metavar="PATH",
        help="With --from-sitemap: write extracted model numbers to this CSV (header: model_number) for use with --via-jina or --from-html.",
    )
    args = parser.parse_args()

    # ---- From CSV (model_number, partselect_number) ----
    if args.from_csv:
        path = _resolve_path(args.from_csv)
        if not path.is_file():
            print(f"CSV not found: {args.from_csv}", file=sys.stderr)
            sys.exit(1)
        fitment_rows = load_fitment_from_csv(path)
        print(f"Loaded {len(fitment_rows)} (model, part) pairs from {path}")
        if args.dry_run:
            from collections import defaultdict
            by_model = defaultdict(list)
            for mn, ps in fitment_rows:
                by_model[mn].append(ps)
            for mn in sorted(by_model.keys())[:10]:
                print(f"  {mn}: {len(by_model[mn])} parts")
            if len(by_model) > 10:
                print(f"  ... and {len(by_model) - 10} more models")
            return
        _write_fitment_to_db(fitment_rows)
        return

    # ---- From HTML (saved browser pages) ----
    if args.from_html:
        files: list[tuple[str, Path]] = []  # (model_number, path)
        for p in args.from_html:
            path = _resolve_path(p)
            if path.is_file() and path.suffix.lower() in (".html", ".htm"):
                model = path.stem
                files.append((model, path))
            elif path.is_dir():
                for f in sorted(path.glob("*.html")) + sorted(path.glob("*.htm")):
                    files.append((f.stem, f))
        if not files:
            print("No .html/.htm files found.", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(files)} HTML file(s)")
        if args.dry_run:
            for model, fpath in files[:20]:
                html = fpath.read_text(encoding="utf-8", errors="replace")
                parts = extract_part_numbers_from_parts_page(html)
                print(f"  {model}: {len(parts)} parts -> {sorted(parts)[:3]}{' ...' if len(parts) > 3 else ''}")
            if len(files) > 20:
                print(f"  ... and {len(files) - 20} more files")
            return
        fitment_rows: list[tuple[str, str]] = []
        for model, fpath in files:
            html = fpath.read_text(encoding="utf-8", errors="replace")
            for ps in extract_part_numbers_from_parts_page(html):
                fitment_rows.append((model, ps))
        if not fitment_rows:
            print("No part numbers extracted. Ensure saved HTML contains links like .../PS12345678/ or text PS12345678.", file=sys.stderr)
            sys.exit(1)
        _write_fitment_to_db(fitment_rows)
        return

    # ---- From sitemap (URL or local XML): extract model/part from URLs, optional DB or models CSV ----
    if args.from_sitemap:
        xml = fetch_sitemap(args.from_sitemap)
        if _markdown_looks_blocked(xml):
            print("Sitemap returned 403/Access Denied. Save sitemap.xml from your browser and pass the file path.", file=sys.stderr)
        page_urls, child_urls = parse_sitemap_xml(xml)
        if args.sitemap_follow_index and child_urls:
            follow = min(args.sitemap_follow_index, len(child_urls))
            if args.verbose:
                print(f"Fetching {follow} child sitemap(s)...")
            for i, child in enumerate(child_urls[:follow]):
                try:
                    child_xml = fetch_sitemap(child)
                    pu, _ = parse_sitemap_xml(child_xml)
                    page_urls.extend(pu)
                except Exception as e:
                    if args.verbose:
                        print(f"  {child}: {e}", file=sys.stderr)
                if (i + 1) < follow:
                    time.sleep(0.3)
        model_numbers, fitment_rows = extract_models_and_parts_from_sitemap_urls(page_urls)
        model_list = sorted(model_numbers)
        print(f"From sitemap: {len(model_list)} model(s), {len(fitment_rows)} (model, part) pair(s) from URLs.")
        if args.output_models_csv:
            out_path = _resolve_path(args.output_models_csv)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["model_number"])
                for m in model_list:
                    w.writerow([m])
            print(f"Wrote {len(model_list)} model numbers to {out_path}")
        if fitment_rows and not args.dry_run:
            _write_fitment_to_db(fitment_rows, fit_source="partselect_sitemap")
        elif fitment_rows and args.dry_run:
            print(f"(Dry run: would write {len(fitment_rows)} fitment rows. Remove --dry-run to write to DB.)")
        if args.dry_run and not args.output_models_csv:
            for m in model_list[:15]:
                print(f"  {m}")
            if len(model_list) > 15:
                print(f"  ... and {len(model_list) - 15} more. Use --output-models-csv to save.")
        return

    # ---- Via Jina / Firecrawl / Bright Data (URL → markdown → extract PS numbers → DB) ----
    if args.via_jina or args.via_firecrawl or args.via_brightdata:
        if args.via_jina:
            backend = "jina"
        elif args.via_firecrawl:
            backend = "firecrawl"
        else:
            backend = "brightdata"
        csv_path = args.csv_path or str(REPO_ROOT / "config" / "partselect_dishwasher_models.csv")
        path = _resolve_path(csv_path)
        if not path.is_file():
            print(f"CSV not found: {csv_path}", file=sys.stderr)
            sys.exit(1)
        model_numbers = load_model_numbers_from_csv(path)
        if args.limit and args.limit > 0:
            model_numbers = model_numbers[: args.limit]
        print(f"Loaded {len(model_numbers)} model numbers from {path}; backend={backend}")

        try:
            from dotenv import load_dotenv
            load_dotenv(SCRIPTS_INGEST / ".env")
            load_dotenv(REPO_ROOT / ".env")
        except ImportError:
            pass
        jina_key = os.environ.get("JINA_API_KEY", "").strip() or None
        firecrawl_key = os.environ.get("FIRECRAWL_API_KEY", "").strip() or None
        brightdata_key = os.environ.get("BRIGHTDATA_API_KEY", "").strip() or None
        brightdata_zone = os.environ.get("BRIGHTDATA_ZONE", "web_unlocker1").strip() or "web_unlocker1"
        if args.via_firecrawl and not firecrawl_key:
            print("FIRECRAWL_API_KEY is required for --via-firecrawl.", file=sys.stderr)
            sys.exit(1)
        if args.via_brightdata and not brightdata_key:
            print("BRIGHTDATA_API_KEY is required for --via-brightdata. Get one at https://brightdata.com/cp/setting/users", file=sys.stderr)
            sys.exit(1)

        # Pagination: Parts/ then Parts/?start=2, start=3, ... until 0 new parts or max_pages
        max_pages = max(1, getattr(args, "parts_max_pages", 0) or 20)
        fitment_rows: list[tuple[str, str]] = []
        blocked_warned = False
        for i, model_number in enumerate(model_numbers):
            base_url = f"{BASE_URL}/Models/{quote(model_number)}/Parts/"
            seen_ps: set[str] = set()
            page = 1
            try:
                while page <= max_pages:
                    url = f"{base_url}?start={page}" if page > 1 else base_url
                    if args.via_jina:
                        md = fetch_markdown_jina(url, api_key=jina_key)
                    elif args.via_firecrawl:
                        md = fetch_markdown_firecrawl(url, api_key=firecrawl_key)
                    else:
                        md = fetch_markdown_brightdata(url, api_key=brightdata_key, zone=brightdata_zone)
                    if _markdown_looks_blocked(md) and not blocked_warned:
                        print("  (PartSelect returned 403/Access Denied to the crawler; 0 parts expected. Use --from-html or --from-csv instead.)", file=sys.stderr)
                        blocked_warned = True
                    part_numbers = extract_part_numbers_from_markdown(md)
                    new_count = 0
                    for ps in part_numbers:
                        if ps not in seen_ps:
                            seen_ps.add(ps)
                            fitment_rows.append((model_number, ps))
                            new_count += 1
                    if args.verbose:
                        print(f"  {model_number}: page {page} -> {len(part_numbers)} parts ({new_count} new), total {len(seen_ps)}")
                    if not part_numbers or new_count == 0:
                        break
                    page += 1
                    if page <= max_pages:
                        time.sleep(max(0.3, args.delay * 0.5))
            except Exception as e:
                print(f"  {model_number}: error — {e}", file=sys.stderr)
            if (i + 1) < len(model_numbers):
                time.sleep(max(0.5, args.delay))
        print(f"Extracted {len(fitment_rows)} (model, part) pairs from {len(model_numbers)} models.")
        if args.dry_run:
            from collections import defaultdict
            by_model = defaultdict(list)
            for mn, ps in fitment_rows:
                by_model[mn].append(ps)
            for mn in sorted(by_model.keys())[:10]:
                print(f"  {mn}: {len(by_model[mn])} parts")
            if len(by_model) > 10:
                print(f"  ... and {len(by_model) - 10} more models")
            print("(Dry run. Remove --dry-run to write to DB.)")
            return
        if not fitment_rows:
            print("No part numbers extracted. Check URLs or try --verbose.", file=sys.stderr)
            sys.exit(1)
        _write_fitment_to_db(fitment_rows, fit_source=f"partselect_{backend}")
        return

    # ---- Live fetch (CSV of model numbers, Playwright) ----
    csv_path = args.csv_path or str(REPO_ROOT / "config" / "partselect_dishwasher_models.csv")
    path = _resolve_path(csv_path)
    if not path.is_file():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    model_numbers = load_model_numbers_from_csv(path)
    if args.limit and args.limit > 0:
        model_numbers = model_numbers[: args.limit]
    print(f"Loaded {len(model_numbers)} model numbers from {path}")

    headless = not args.no_headless
    if args.dry_run:
        print("(Dry run: will not write to DB)\n")
        for model in model_numbers:
            url_parts = f"{BASE_URL}/Models/{quote(model)}/Parts/"
            url_overview = f"{BASE_URL}/Models/{quote(model)}/"
            try:
                parts = fetch_parts_with_playwright(url_parts, headless=headless, verbose=args.verbose)
                if not parts:
                    parts = fetch_parts_with_playwright(url_overview, headless=headless, verbose=args.verbose)
                ps_nums = [p["partselect_number"] for p in parts]
                print(f"  {model}: {len(parts)} parts -> {ps_nums[:5]}{' ...' if len(ps_nums) > 5 else ''}")
            except Exception as e:
                print(f"  {model}: error {e}")
            time.sleep(args.delay)
        print("\nDone (dry run). Remove --dry-run to write part_fitment and parts.")
        return

    # DB
    sys.path.insert(0, str(SCRIPTS_INGEST))
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from dotenv import load_dotenv
        load_dotenv(SCRIPTS_INGEST / ".env")
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass
    from db import db_connection

    # Ensure part_fitment / parts tables exist (run schema first)
    try:
        with db_connection() as conn:
            conn.execute("SELECT 1 FROM part_fitment LIMIT 0")
    except Exception as e:
        if "part_fitment" in str(e) or "does not exist" in str(e).lower():
            print("Database tables missing. Run the schema first:", file=sys.stderr)
            print("  psql $DATABASE_URL -f apps/api/schema.sql", file=sys.stderr)
            print("  (or: createdb partselect && psql partselect -f apps/api/schema.sql)", file=sys.stderr)
            sys.exit(1)
        raise

    fit_source = "partselect_model_parts"
    total_fitments = 0
    for model_number in model_numbers:
        url_parts = f"{BASE_URL}/Models/{quote(model_number)}/Parts/"
        url_overview = f"{BASE_URL}/Models/{quote(model_number)}/"
        try:
            parts = fetch_parts_with_playwright(url_parts, headless=headless, verbose=args.verbose)
            if not parts:
                parts = fetch_parts_with_playwright(url_overview, headless=headless, verbose=args.verbose)
        except Exception as e:
            print(f"  {model_number}: error — {e}")
            time.sleep(args.delay)
            continue

        if not parts:
            print(f"  {model_number}: 0 parts (skipped)")
            time.sleep(args.delay)
            continue

        with db_connection() as conn:
            for p in parts:
                ps_num = p["partselect_number"]
                name = p.get("name") or f"Part {ps_num}"
                mfr = p.get("manufacturer_part_number")
                price = p.get("price")
                part_url = p.get("url")
                image_url = p.get("image_url")
                conn.execute(
                    """
                    INSERT INTO part_fitment (partselect_number, model_number, fit_source)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (partselect_number, model_number) DO NOTHING
                    """,
                    (ps_num, model_number, fit_source),
                )
                conn.execute(
                    """
                    INSERT INTO parts (part_number, partselect_number, name, manufacturer_part_number, url, image_url)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (part_number) DO UPDATE SET
                      name = COALESCE(EXCLUDED.name, parts.name),
                      manufacturer_part_number = COALESCE(EXCLUDED.manufacturer_part_number, parts.manufacturer_part_number),
                      url = COALESCE(EXCLUDED.url, parts.url),
                      image_url = COALESCE(EXCLUDED.image_url, parts.image_url)
                    """,
                    (ps_num, ps_num, name, mfr, part_url, image_url),
                )
            total_fitments += len(parts)
            print(f"  {model_number}: {len(parts)} parts -> part_fitment + parts")
        time.sleep(args.delay)

    print(f"\nDone: {len(model_numbers)} model(s), {total_fitments} part_fitment rows written.")
    print("Run search_parts(model_number=...) in the API to list parts for a model.")


if __name__ == "__main__":
    main()
