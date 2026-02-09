"""
On-demand fetch of parts for a single model from PartSelect (Playwright).
Used only when ENABLE_LIVE_PARTS_FETCH=1; live fetch often fails (403, 0 parts). Prefer ingest --from-html/--from-csv.
"""
from __future__ import annotations

import re

from app.model_parser import partselect_model_url

# PartSelect part number: PS + digits
PART_NUMBER_RE = re.compile(r"\b(PS\d{6,})\b", re.IGNORECASE)

REQUEST_TIMEOUT_MS = 30_000
BASE_URL = "https://www.partselect.com"


def _base_model_url(url: str) -> str:
    """Normalize to base model URL with trailing slash, e.g. .../Models/003719074/"""
    u = (url or "").rstrip("/")
    if "/Parts" in u:
        u = u.split("/Parts")[0]
    return u + "/"


def _parse_part_items_from_page(page) -> list[dict]:
    """Parse .mega-m__part from current page; return list of part dicts."""
    out: list[dict] = []
    items = page.query_selector_all(".mega-m__part")
    for item in items:
        try:
            name_el = item.query_selector(".mega-m__part__name")
            name = name_el.inner_text().strip() if name_el else "Unknown"
            link = (name_el.get_attribute("href") or "") if name_el else ""
            full_link = f"{BASE_URL}{link}" if link else ""
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
                    image_url = f"{BASE_URL}{src}" if src.startswith("/") else src
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


def fetch_parts_for_model_sync(model_number: str, headless: bool = True) -> list[dict]:
    """
    Fetch parts for one model from PartSelect (sync, for use in asyncio.to_thread).
    Uses Parts/ and Parts/?start=2, ... with Referer. Returns list of part dicts
    (partselect_number, name, manufacturer_part_number, price, url).
    """
    from playwright.sync_api import sync_playwright

    model_number = (model_number or "").strip()
    if not model_number:
        return []
    base_url = partselect_model_url(model_number)
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
        try:
            while True:
                if start_index == 1:
                    target_url = f"{base_url}Parts/"
                else:
                    target_url = f"{base_url}Parts/?start={start_index}"
                    page.set_extra_http_headers({"Referer": base_url})
                try:
                    page.goto(target_url, wait_until="domcontentloaded", timeout=REQUEST_TIMEOUT_MS)
                    page.wait_for_timeout(3000)
                    page.mouse.wheel(0, 2000)
                    page.wait_for_timeout(1000)
                except Exception:
                    break
                parts_this_page = _parse_part_items_from_page(page)
                if not parts_this_page:
                    break
                for p in parts_this_page:
                    ps = p["partselect_number"]
                    if ps not in seen_ps:
                        seen_ps.add(ps)
                        out.append(p)
                start_index += 1
        finally:
            browser.close()
    return out


def fetch_parts_from_symptom_page_sync(
    model_number: str, symptom_slug: str, limit: int = 5, headless: bool = True
) -> list[dict]:
    """
    Fetch the first `limit` parts from the model's symptom page (e.g. /Models/WRF535SWHZ/Symptoms/Ice-maker-not-making-ice/).
    symptom_slug should be the path segment after /Symptoms/, e.g. "Ice-maker-not-making-ice".
    Returns list of part dicts (partselect_number, name, url, price, image_url, ...) in same shape as fetch_parts_for_model_sync.
    """
    from playwright.sync_api import sync_playwright

    model_number = (model_number or "").strip()
    symptom_slug = (symptom_slug or "").strip()
    if not model_number or not symptom_slug:
        return []
    base_url = partselect_model_url(model_number)
    symptom_url = f"{base_url.rstrip('/')}/Symptoms/{symptom_slug}/"

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
        try:
            page.goto(symptom_url, wait_until="domcontentloaded", timeout=REQUEST_TIMEOUT_MS)
            page.wait_for_timeout(3000)
            page.mouse.wheel(0, 2000)
            page.wait_for_timeout(1000)
            parts_this_page = _parse_part_items_from_page(page)
            out = (parts_this_page or [])[:limit]
        except Exception:
            out = []
        finally:
            browser.close()
    return out
