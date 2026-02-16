"""
Scrape Lebanese real estate listings from realestate.com.lb via their API.

Hits /laravel/api/member/properties with pagination, resolves IDs to names
using lookup tables from the site, and writes a clean CSV.

Usage:
    python scrape_lebanese_zillow_dataset.py
    python scrape_lebanese_zillow_dataset.py --max-pages 10   # limit pages for testing
"""

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

try:
    from colorama import Fore, Style, init
    init(autoreset=True)
except ImportError:
    Fore = type("Fore", (), {k: "" for k in ("GREEN", "YELLOW", "RED", "CYAN", "RESET")})()
    Style = type("Style", (), {"RESET_ALL": ""})()


def log(msg: str, color: str = "") -> None:
    print(f"{color}{msg}{Style.RESET_ALL}", flush=True)


BASE = "https://www.realestate.com.lb"
API = f"{BASE}/laravel/api/member/properties"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": BASE,
}

FIELDNAMES = [
    "listing_id",
    "status",
    "property_type",
    "title",
    "price_usd",
    "price_period",
    "bedrooms",
    "bathrooms",
    "area_sqm",
    "furnished",
    "community_id",
    "district_id",
    "governorate_id",
    "agency",
    "agent_name",
    "listing_url",
    "scraped_at_utc",
]


# -- Lookup builders ----------------------------------------------------------

PROPERTY_TYPES = {
    1: "Apartment", 2: "Villa", 3: "Chalet", 4: "Office Space",
    5: "Land", 6: "Duplex", 7: "Penthouse", 8: "Townhouse",
    9: "Loft", 10: "Whole Building", 11: "Shop", 12: "Warehouse",
    13: "Roof", 14: "Studio",
}

CATEGORIES = {1: "buy", 2: "rent", 3: "commercial_buy", 4: "commercial_rent"}

PRICE_TYPES = {1: "USD", 4: "USD / month", 7: "USD / year"}


def fetch_lookup_tables() -> None:
    """Try to refresh lookup tables from the site's _next/data endpoint."""
    global PROPERTY_TYPES, CATEGORIES, PRICE_TYPES
    try:
        # Get a listing page to find the build ID
        r = requests.get(f"{BASE}/en/buy-properties-lebanon", headers=HEADERS, timeout=15)
        # Extract buildId from __NEXT_DATA__
        import json, re
        m = re.search(r'"buildId"\s*:\s*"([^"]+)"', r.text)
        if not m:
            return
        build_id = m.group(1)

        r2 = requests.get(
            f"{BASE}/_next/data/{build_id}/en/buy-properties-lebanon.json",
            headers=HEADERS, timeout=15,
        )
        pp = r2.json().get("pageProps", {})

        for pt in pp.get("propertyTypes", []):
            PROPERTY_TYPES[pt["id"]] = pt["name_en"]

        for pr in pp.get("priceTypes", []):
            PRICE_TYPES[pr["id"]] = pr["name_en"]

        log(f"Loaded {len(PROPERTY_TYPES)} property types, {len(PRICE_TYPES)} price types", Fore.GREEN)
    except Exception as e:
        log(f"Could not refresh lookups (using defaults): {e}", Fore.YELLOW)


# -- API scraping --------------------------------------------------------------

def fetch_page(page: int, retries: int = 3) -> Optional[dict]:
    """Fetch a single page of listings from the API."""
    for attempt in range(retries):
        try:
            r = requests.get(
                API,
                params={"pg": page, "sort": "listing_level", "direction": "asc"},
                headers=HEADERS,
                timeout=20,
            )
            if r.status_code == 200:
                return r.json().get("data", {})
        except requests.RequestException:
            pass
        time.sleep(1 * (attempt + 1))
    return None


def parse_doc(doc: dict) -> Dict:
    """Convert one API doc into a flat row dict."""
    client = doc.get("client") or {}
    agent = doc.get("agent") or {}

    agent_first = (agent.get("first_name") or "").strip()
    agent_last = (agent.get("last_name") or "").strip()
    agent_name = f"{agent_first} {agent_last}".strip() or None

    cat_id = doc.get("category_id")
    type_id = doc.get("type_id")
    price_type_id = doc.get("price_type_id")

    url_path = doc.get("url") or ""
    listing_url = f"{BASE}/en{url_path}" if url_path else None

    return {
        "listing_id": doc.get("reference"),
        "status": CATEGORIES.get(cat_id, cat_id),
        "property_type": PROPERTY_TYPES.get(type_id, type_id),
        "title": doc.get("title_en"),
        "price_usd": doc.get("price"),
        "price_period": PRICE_TYPES.get(price_type_id, price_type_id),
        "bedrooms": doc.get("bedroom_value"),
        "bathrooms": doc.get("bathroom_value"),
        "area_sqm": doc.get("area"),
        "furnished": doc.get("furnished"),
        "community_id": doc.get("community_id"),
        "district_id": doc.get("district_id"),
        "governorate_id": doc.get("province_id"),
        "agency": client.get("display_name"),
        "agent_name": agent_name,
        "listing_url": listing_url,
        "scraped_at_utc": int(time.time()),
    }


def scrape_all(max_pages: int) -> List[Dict]:
    """Paginate through the API and collect all listings."""
    rows: List[Dict] = []
    seen_ids = set()

    # First page to get total count
    data = fetch_page(1)
    if not data:
        log("Failed to fetch page 1", Fore.RED)
        return rows

    num_found = data.get("numFound", 0)
    per_page = len(data.get("docs", []))
    if per_page == 0:
        log("No docs on page 1", Fore.RED)
        return rows

    total_pages = min(math.ceil(num_found / per_page), max_pages)
    log(f"Total listings: {num_found} across ~{total_pages} pages ({per_page}/page)", Fore.GREEN)

    # Process first page
    for doc in data.get("docs", []) + data.get("boostedProperties", []):
        ref = doc.get("reference")
        if ref and ref not in seen_ids:
            seen_ids.add(ref)
            rows.append(parse_doc(doc))

    # Remaining pages
    for pg in range(2, total_pages + 1):
        page_data = fetch_page(pg)
        if not page_data:
            log(f"  Page {pg} failed, skipping", Fore.RED)
            continue

        for doc in page_data.get("docs", []) + page_data.get("boostedProperties", []):
            ref = doc.get("reference")
            if ref and ref not in seen_ids:
                seen_ids.add(ref)
                rows.append(parse_doc(doc))

        if pg % 25 == 0 or pg == total_pages:
            log(f"  Page {pg}/{total_pages} — {len(rows)} unique rows so far", Fore.YELLOW)

        # Be polite
        time.sleep(0.15)

    return rows


# -- CSV output ----------------------------------------------------------------

def write_csv(rows: List[Dict], output_csv: str) -> None:
    if not rows:
        log("No rows to write!", Fore.RED)
        return

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scrape realestate.com.lb listings via API into a clean CSV."
    )
    parser.add_argument(
        "--output-csv",
        default="data/regression/lebanese_zillow_like_listings.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=9999,
        help="Max API pages to fetch (default: all)",
    )
    args = parser.parse_args()

    log("=== Fetching lookup tables ===", Fore.CYAN)
    fetch_lookup_tables()

    log("=== Scraping listings from API ===", Fore.CYAN)
    rows = scrape_all(args.max_pages)
    log(f"Total unique rows: {len(rows)}", Fore.GREEN)

    log(f"=== Writing CSV to {args.output_csv} ===", Fore.CYAN)
    write_csv(rows, args.output_csv)
    log(f"Done! {len(rows)} rows -> {args.output_csv}", Fore.GREEN)


if __name__ == "__main__":
    main()
