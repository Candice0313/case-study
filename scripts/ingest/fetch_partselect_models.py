#!/usr/bin/env python3
"""
Fetch model_number + brand from PartSelect (refrigerator or dishwasher).
Output: CSV for seed_models.py (model_number,brand,appliance_type).

Refrigerator: Parts + Models pages per brand.
Dishwasher: single list page https://www.partselect.com/Dishwasher-Models.htm (100 per page).

Usage (from repo root):
  # Refrigerator (default)
  python -m scripts.ingest.fetch_partselect_models
  python -m scripts.ingest.fetch_partselect_models -o config/partselect_refrigerator_models.csv

  # Dishwasher (all pages via Bright Data)
  python -m scripts.ingest.fetch_partselect_models --appliance dishwasher --via-brightdata --models-max-pages 500 -o config/partselect_dishwasher_models_full.csv
  # Then fetch all parts per model: python -m scripts.ingest.fetch_partselect_model_parts --via-brightdata config/partselect_dishwasher_models_full.csv

If PartSelect returns 403, the script falls back to Playwright. Install browsers once:
  playwright install chromium

Manual download: save the page(s) in your browser (Save Page As → Complete), then:
  python -m scripts.ingest.fetch_partselect_models --from-html path/to/Whirlpool-Refrigerator-Parts.htm --brand Whirlpool -o config/partselect_refrigerator_models.csv
  python -m scripts.ingest.fetch_partselect_models --from-html path/to/Dishwasher-Models.htm --appliance dishwasher -o config/partselect_dishwasher_models.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

SCRIPTS_INGEST = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_INGEST.parents[1]
BASE_URL = "https://www.partselect.com"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
REQUEST_TIMEOUT = 25.0
DELAY_BETWEEN_REQUESTS = 1.5

# Dishwasher list: 100 models per page, 447 pages. Pagination: ?start=1, ?start=2, ... ?start=447
DISHWASHER_MODELS_PAGE_PARAM = "start"
DISHWASHER_MODELS_PAGE_SIZE = 100

# Refrigerator: brand slug -> display brand name
REFRIGERATOR_BRANDS = [
    ("Whirlpool", "Whirlpool"),
    ("GE", "GE"),
    ("Samsung", "Samsung"),
    ("Frigidaire", "Frigidaire"),
]

# Dishwasher: single list page (100 models per page; 44640 total — pagination TBD)
DISHWASHER_MODELS_URL = "https://www.partselect.com/Dishwasher-Models.htm"


def _normalize_model(s: str) -> str | None:
    """Take only the model token (alphanumeric + hyphen), strip whitespace."""
    s = (s or "").strip().upper()
    # Remove trailing type text like "REFRIGERATOR" or "Side-by-side" from link text
    s = re.sub(r"\s+REFRIGERATOR\s*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+Side-by-side\s*$", "", s, flags=re.IGNORECASE)
    s = s.strip()
    if not s or len(s) < 4 or len(s) > 30:
        return None
    if not re.match(r"^[A-Z0-9\-]+$", s):
        return None
    return s


def _extract_models_from_html(html: str) -> set[str]:
    """Parse HTML for links to Models/{model_number} (with or without leading slash / full URL)."""
    soup = BeautifulSoup(html, "html.parser")
    found: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = (a["href"] or "").strip()
        # Match /Models/XXX, Models/XXX, or https://...partselect.com/Models/XXX
        m = re.search(r"(?:^|/)(?:Models/)([A-Za-z0-9\-%()]+?)/?(?:\?|/|$)", href, re.IGNORECASE)
        if not m:
            continue
        raw = m.group(1)
        # Decode URL-encoded model (e.g. %28 -> (, %29 -> ))
        try:
            from urllib.parse import unquote
            raw = unquote(raw)
        except Exception:
            pass
        model = _normalize_model(raw)
        if model:
            found.add(model)
    return found


def _normalize_model_dishwasher(s: str) -> str | None:
    """Allow parentheses and slightly longer for dishwasher model numbers (e.g. 19885(1988))."""
    s = (s or "").strip().upper()
    s = re.sub(r"\s+DISHWASHER\s*$", "", s, flags=re.IGNORECASE)
    s = s.strip()
    if not s or len(s) < 2 or len(s) > 40:
        return None
    if not re.match(r"^[A-Z0-9\-%()]+$", s):
        return None
    return s


def _extract_dishwasher_models_from_html(html: str) -> list[tuple[str, str]]:
    """
    Parse Dishwasher-Models.htm: links to /Models/XXX with text like "3000W10 General Electric Dishwasher".
    Returns list of (model_number, brand); model from href, brand from text (part before 'Dishwasher').
    """
    soup = BeautifulSoup(html, "html.parser")
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = (a["href"] or "").strip()
        m = re.search(r"(?:^|/)(?:Models/)([A-Za-z0-9\-%()]+?)/?(?:\?|/|$)", href, re.IGNORECASE)
        if not m:
            continue
        try:
            from urllib.parse import unquote
            raw_model = unquote(m.group(1))
        except Exception:
            raw_model = m.group(1)
        model = _normalize_model_dishwasher(raw_model)
        if not model:
            continue
        text = (a.get_text() or "").strip()
        # "3000W10 General Electric Dishwasher" -> brand = "General Electric"
        if "Dishwasher" not in text and "dishwasher" not in text.lower():
            brand = "Unknown"
        else:
            # Remove trailing " Dishwasher" / " dishwasher" and leading model number + space
            rest = re.sub(r"\s+Dishwasher\s*$", "", text, flags=re.IGNORECASE).strip()
            # Rest should start with model number (maybe with different casing); brand is the remainder
            rest = re.sub(r"^" + re.escape(model), "", rest, flags=re.IGNORECASE).strip()
            if rest.startswith(" "):
                rest = rest.lstrip()
            brand = rest if rest else "Unknown"
        if model not in seen:
            seen.add(model)
            rows.append((model, brand))
    return rows


def _fetch_url(url: str, client: httpx.Client) -> str:
    r = client.get(url)
    r.raise_for_status()
    r.encoding = r.encoding or "utf-8"
    return r.text


def _fetch_url_playwright(url: str) -> str:
    """Fallback when server returns 403 to plain HTTP. Wait for content to load."""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle", timeout=int(REQUEST_TIMEOUT * 1000))
        page.wait_for_timeout(2000)  # extra 2s for JS-rendered model list
        html = page.content()
        browser.close()
    return html


def _fetch_html(url: str, client: httpx.Client, use_playwright_on_403: bool = True) -> str:
    try:
        return _fetch_url(url, client)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 403 and use_playwright_on_403:
            print("    (403 -> trying Playwright ...)")
            return _fetch_url_playwright(url)
        raise


def _fetch_html_brightdata(url: str, api_key: str, zone: str = "web_unlocker1", timeout: float = 90.0) -> str:
    """Fetch URL via Bright Data Web Unlocker; return raw HTML (format=raw)."""
    import os
    try:
        from dotenv import load_dotenv
        load_dotenv(SCRIPTS_INGEST / ".env")
        load_dotenv(SCRIPTS_INGEST.parents[1] / ".env")
    except ImportError:
        pass
    key = (api_key or os.environ.get("BRIGHTDATA_API_KEY") or "").strip()
    z = (zone or os.environ.get("BRIGHTDATA_ZONE") or "web_unlocker1").strip() or "web_unlocker1"
    if not key:
        raise ValueError("BRIGHTDATA_API_KEY required for --via-brightdata")
    with httpx.Client(timeout=timeout) as client:
        r = client.post(
            "https://api.brightdata.com/request",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"zone": z, "url": url, "format": "raw", "method": "GET", "country": "us"},
        )
        if r.status_code >= 400:
            try:
                err = r.json()
            except Exception:
                err = r.text
            raise RuntimeError(f"Bright Data {r.status_code}: {err}")
        body = r.text or ""
        if body.strip().startswith("{"):
            try:
                data = r.json()
                body = data.get("content") or data.get("html") or data.get("body") or (data.get("data") if isinstance(data.get("data"), str) else None) or body
            except Exception:
                pass
        return body.strip() if isinstance(body, str) else ""


def fetch_dishwasher_models(
    client: httpx.Client,
    url: str = DISHWASHER_MODELS_URL,
    delay: float = DELAY_BETWEEN_REQUESTS,
    use_brightdata: bool = False,
    brightdata_api_key: str | None = None,
    brightdata_zone: str = "web_unlocker1",
    models_max_pages: int = 1,
    start_page: int = 1,
    initial_rows: list[tuple[str, str, str]] | None = None,
    pagination_mode: str = "start",
) -> list[tuple[str, str, str]]:
    """
    Fetch Dishwasher-Models.htm and optionally paginated pages. (447 pages total.)
    pagination_mode: "start" -> ?start=1, ?start=2, ... (default); "offset" -> ?start=0, ?start=100, ...
    Returns (model_number, brand, appliance_type). start_page: resume from this page (1-based). initial_rows: existing rows to keep.
    """
    initial_rows = initial_rows or []
    rows: list[tuple[str, str, str]] = list(initial_rows)
    seen: set[str] = {r[0] for r in initial_rows}
    base_url = url.split("?")[0].rstrip("/")
    page = start_page
    consecutive_empty = 0
    max_consecutive_empty = 3  # stop only after this many pages in a row with 0 new
    while page <= models_max_pages:
        if pagination_mode == "offset":
            val = (page - 1) * DISHWASHER_MODELS_PAGE_SIZE
            page_url = f"{base_url}?start={val}"
        else:
            # start=1, start=2, ... start=447 (page number in param "start")
            page_url = f"{base_url}?start={page}"
        try:
            if use_brightdata and brightdata_api_key:
                print(f"  Fetching {page_url} (Bright Data) ...")
                html = _fetch_html_brightdata(page_url, brightdata_api_key, brightdata_zone)
            else:
                print(f"  Fetching {page_url} ...")
                html = _fetch_html(page_url, client)
            time.sleep(delay)
            page_models = _extract_dishwasher_models_from_html(html)
            new_count = 0
            for model, brand in page_models:
                if model not in seen:
                    seen.add(model)
                    rows.append((model, brand, "dishwasher"))
                    new_count += 1
            if new_count == 0:
                consecutive_empty += 1
                dup_info = f" (page had {len(page_models)} links, all duplicates)" if page_models else " (no model links in page)"
                print(f"    -> page {page}: 0 new, total {len(rows)} models{dup_info} ({consecutive_empty}/{max_consecutive_empty} consecutive empty)")
                if consecutive_empty >= max_consecutive_empty:
                    print(f"    Stopping after {max_consecutive_empty} consecutive empty pages.")
                    break
            else:
                consecutive_empty = 0
                dup_info = f", {len(page_models) - new_count} dup" if len(page_models) > new_count else ""
                print(f"    -> page {page}: {new_count} new, total {len(rows)} models{dup_info}")
            page += 1
        except Exception as e:
            print(f"    Error: {e}", file=sys.stderr)
            consecutive_empty += 1
            if consecutive_empty >= max_consecutive_empty:
                break
            page += 1
    return rows


def fetch_refrigerator_models(
    client: httpx.Client,
    delay: float = DELAY_BETWEEN_REQUESTS,
) -> list[tuple[str, str, str]]:
    """
    Fetch Parts and Models pages for each brand; return list of (model_number, brand, appliance_type).
    """
    rows: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()  # (model_number, brand) for dedup

    for brand_slug, brand_name in REFRIGERATOR_BRANDS:
        urls = [
            f"{BASE_URL}/{brand_slug}-Refrigerator-Parts.htm",
            f"{BASE_URL}/{brand_slug}-Refrigerator-Models.htm",
        ]
        for url in urls:
            try:
                print(f"  Fetching {url} ...")
                html = _fetch_html(url, client)
                time.sleep(delay)
                models = _extract_models_from_html(html)
                for model in models:
                    key = (model, brand_name)
                    if key not in seen:
                        seen.add(key)
                        rows.append((model, brand_name, "refrigerator"))
                print(f"    -> {len(models)} models (total unique for {brand_name}: {sum(1 for r in rows if r[1]==brand_name)})")
            except Exception as e:
                print(f"    Error: {e}", file=sys.stderr)

    return rows


def parse_local_html_files(
    paths: list[Path],
    brand: str | None,
    appliance_type: str = "refrigerator",
) -> list[tuple[str, str, str]]:
    """Parse already-saved HTML files. For refrigerator pass --brand; for dishwasher brand is parsed from each link."""
    rows: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for path in paths:
        if not path.is_file():
            print(f"  Skip (not file): {path}", file=sys.stderr)
            continue
        with open(path, encoding="utf-8", errors="replace") as f:
            html = f.read()
        if appliance_type == "dishwasher":
            file_rows = _extract_dishwasher_models_from_html(html)
            for model, link_brand in file_rows:
                if model not in seen:
                    seen.add(model)
                    rows.append((model, link_brand, "dishwasher"))
            print(f"  {path.name}: {len(file_rows)} models")
        else:
            models = _extract_models_from_html(html)
            for model in models:
                if model not in seen:
                    seen.add(model)
                    rows.append((model, brand or "Unknown", appliance_type))
            print(f"  {path.name}: {len(models)} models")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch PartSelect model numbers (refrigerator or dishwasher) into CSV")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output CSV path (default: config/partselect_<appliance>_models.csv)",
    )
    parser.add_argument(
        "--appliance",
        choices=("refrigerator", "dishwasher"),
        default="refrigerator",
        help="Appliance type to fetch (default: refrigerator)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DELAY_BETWEEN_REQUESTS,
        help="Seconds between requests",
    )
    parser.add_argument(
        "--from-html",
        nargs="+",
        metavar="FILE",
        help="Use local HTML file(s) instead of live fetch. For refrigerator requires --brand.",
    )
    parser.add_argument(
        "--brand",
        type=str,
        help="Brand name when using --from-html for refrigerator (e.g. Whirlpool, GE).",
    )
    parser.add_argument(
        "--via-brightdata",
        action="store_true",
        help="Fetch dishwasher list via Bright Data Web Unlocker (avoids 403). Requires BRIGHTDATA_API_KEY. Use with --appliance dishwasher.",
    )
    parser.add_argument(
        "--models-max-pages",
        type=int,
        default=1,
        metavar="N",
        help="For dishwasher: max pages for models list (offset start=0,100,200,... 100 per page; site has 447 pages). Default 1. Use 447+ for all.",
    )
    parser.add_argument(
        "--models-start-page",
        type=int,
        default=1,
        metavar="N",
        help="For dishwasher: start from page N (resume). Use with same -o as previous run to append; existing CSV is loaded and merged.",
    )
    parser.add_argument(
        "--models-pagination",
        choices=("start", "offset"),
        default="start",
        help="Dishwasher list: 'start' = ?start=1,2,... (default); 'offset' = ?start=0,100,200,...",
    )
    parser.add_argument(
        "--seed-db",
        action="store_true",
        help="After writing CSV, also upsert rows into the models table (so TablePlus/API see them without running seed_models).",
    )
    args = parser.parse_args()

    default_output = (
        REPO_ROOT / "config" / f"partselect_{args.appliance}_models.csv"
    )
    out_path = Path(args.output or str(default_output))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.from_html:
        if args.appliance == "refrigerator" and not args.brand:
            print("When using --from-html for refrigerator you must set --brand.", file=sys.stderr)
            sys.exit(1)
        paths = []
        for p in args.from_html:
            path = Path(p)
            if not path.is_file() and not path.is_absolute():
                # From repo root: sources/foo.html -> scripts/ingest/sources/foo.html
                path = SCRIPTS_INGEST / p
            paths.append(path)
        print(f"Parsing {len(paths)} local HTML file(s) (appliance={args.appliance}) ...")
        rows = parse_local_html_files(paths, brand=args.brand, appliance_type=args.appliance)
    else:
        with httpx.Client(
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
            timeout=REQUEST_TIMEOUT,
        ) as client:
            if args.appliance == "dishwasher":
                import os
                try:
                    from dotenv import load_dotenv
                    load_dotenv(SCRIPTS_INGEST / ".env")
                    load_dotenv(REPO_ROOT / ".env")
                except ImportError:
                    pass
                bd_key = os.environ.get("BRIGHTDATA_API_KEY", "").strip() or None
                bd_zone = os.environ.get("BRIGHTDATA_ZONE", "web_unlocker1").strip() or "web_unlocker1"
                if args.via_brightdata and not bd_key:
                    print("BRIGHTDATA_API_KEY required for --via-brightdata.", file=sys.stderr)
                    sys.exit(1)
                initial_rows: list[tuple[str, str, str]] = []
                if args.models_start_page > 1 and out_path.is_file():
                    with open(out_path, newline="", encoding="utf-8") as f:
                        r = csv.DictReader(f)
                        for row in r:
                            mn = (row.get("model_number") or "").strip()
                            br = (row.get("brand") or "").strip()
                            if mn:
                                initial_rows.append((mn, br, "dishwasher"))
                    print(f"Resuming from page {args.models_start_page} (loaded {len(initial_rows)} existing models from {out_path})")
                print("Fetching PartSelect Dishwasher-Models.htm" + (" (Bright Data + pagination)" if args.via_brightdata else "") + " ...")
                rows = fetch_dishwasher_models(
                    client,
                    delay=args.delay,
                    use_brightdata=args.via_brightdata,
                    brightdata_api_key=bd_key,
                    brightdata_zone=bd_zone,
                    models_max_pages=args.models_max_pages,
                    start_page=args.models_start_page,
                    initial_rows=initial_rows if args.models_start_page > 1 else None,
                    pagination_mode=args.models_pagination,
                )
            else:
                print("Fetching PartSelect refrigerator pages (Parts + Models) ...")
                rows = fetch_refrigerator_models(client, delay=args.delay)

    if not rows:
        print("No models extracted.", file=sys.stderr)
        if args.from_html:
            missing = [p for p in paths if not p.is_file()]
            if missing:
                print("Missing file(s):", [str(p) for p in missing], file=sys.stderr)
                print("To create partselect-dishwasher-models.html: add the Dishwasher-Models URL to sources_config.yaml and run 'python fetch.py' from scripts/ingest.", file=sys.stderr)
            else:
                print("Try saving the page in your browser (Save Page As) to scripts/ingest/sources/partselect-dishwasher-models.html", file=sys.stderr)
        else:
            print("Try --from-html with local HTML (save page in browser, then run script).", file=sys.stderr)
        sys.exit(1)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model_number", "brand", "appliance_type"])
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")

    if getattr(args, "seed_db", False):
        sys.path.insert(0, str(SCRIPTS_INGEST))
        sys.path.insert(0, str(REPO_ROOT))
        try:
            from dotenv import load_dotenv
            load_dotenv(SCRIPTS_INGEST / ".env")
            load_dotenv(REPO_ROOT / ".env")
        except ImportError:
            pass
        from db import db_connection
        try:
            with db_connection() as conn:
                conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_models_model_number ON models (model_number)")
                for model_number, brand, appliance_type in rows:
                    conn.execute(
                        """
                        INSERT INTO models (model_number, brand, appliance_type)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (model_number) DO UPDATE SET
                          brand = EXCLUDED.brand,
                          appliance_type = EXCLUDED.appliance_type
                        """,
                        (model_number, brand, appliance_type),
                    )
            print(f"Seeded DB: upserted {len(rows)} rows into models table.")
        except Exception as e:
            print(f"Seed DB failed: {e}", file=sys.stderr)
            print("You can still run: python -m scripts.ingest.seed_models", out_path)
    else:
        print("To put models in DB: python -m scripts.ingest.seed_models", out_path, "  or re-run with --seed-db")


if __name__ == "__main__":
    main()
