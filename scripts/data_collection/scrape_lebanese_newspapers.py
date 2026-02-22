"""Scrape Al Joumhouria and An-Nahar articles into a logistic-regression dataset."""

import argparse
import asyncio
import csv
import random
import re
import time
from html import unescape
from pathlib import Path
from typing import Optional

import requests as http_requests
from playwright.async_api import Browser, BrowserContext, Page, async_playwright

try:
    from colorama import Fore, Style, init

    init(autoreset=True)
except ImportError:
    Fore = type("Fore", (), {k: "" for k in ("GREEN", "YELLOW", "RED", "CYAN", "RESET")})()
    Style = type("Style", (), {"RESET_ALL": ""})()


def log(msg: str, color: str = "") -> None:
    print(f"{color}{msg}{Style.RESET_ALL}", flush=True)


FIELDNAMES = ["source", "title", "body", "date", "category"]
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


async def polite_goto(page: Page, url: str, timeout: int = 30_000) -> bool:
    await asyncio.sleep(random.uniform(1.2, 2.8))
    resp = await page.goto(url, wait_until="domcontentloaded", timeout=timeout)
    if resp is None or resp.status >= 400:
        log(f"  Could not load {url[:80]}", Fore.RED)
        return False
    return True


JOUMHOURIA_BASE = "https://www.aljoumhouria.com"
JOUMHOURIA_CATEGORIES = [
    (f"{JOUMHOURIA_BASE}/ar/news/category/1/محلي", "محلي"),
    (f"{JOUMHOURIA_BASE}/ar/news/category/2/عربي-ودولي", "عربي ودولي"),
    (f"{JOUMHOURIA_BASE}/ar/news/category/4/اقتصاد", "اقتصاد"),
    (f"{JOUMHOURIA_BASE}/ar/news/category/5/رياضة", "رياضة"),
    (f"{JOUMHOURIA_BASE}/ar/news/category/6/ثقافة", "ثقافة"),
    (f"{JOUMHOURIA_BASE}/ar/news/category/7/صحة", "صحة"),
    (f"{JOUMHOURIA_BASE}/ar/news/category/8/تكنولوجيا", "تكنولوجيا"),
    (f"{JOUMHOURIA_BASE}/ar/news/category/3/رأي", "رأي"),
]
_JOUMHOURIA_ARTICLE_RE = re.compile(r"/ar/news/\d{4,}/")


async def _joumhouria_collect_links(page: Page, cat_url: str, target: int) -> list[str]:
    if not await polite_goto(page, cat_url):
        return []

    links: list[str] = []
    seen: set[str] = set()

    async def harvest() -> None:
        for sel in ["div.card.animation > a[href]", "a[href*='/ar/news/']"]:
            els = await page.query_selector_all(sel)
            for el in els:
                href = (await el.get_attribute("href")) or ""
                if _JOUMHOURIA_ARTICLE_RE.search(href) and href not in seen:
                    seen.add(href)
                    full = href if href.startswith("http") else JOUMHOURIA_BASE + href
                    links.append(full)
            if links:
                break

    await harvest()

    clicks = 0
    while len(links) < target and clicks < 25:
        btn = await page.query_selector(
            "#loadMore, button#loadMore, div.load-more, .load-more button, "
            "button:has-text('المزيد'), a:has-text('المزيد')"
        )
        if not btn:
            break
        await btn.scroll_into_view_if_needed()
        await btn.click()
        await page.wait_for_timeout(2_500)
        prev = len(links)
        await harvest()
        clicks += 1
        log(f"    load-more #{clicks}: {len(links)} links (+{len(links) - prev})", Fore.CYAN)
        if len(links) == prev:
            break

    return links


async def _joumhouria_parse_article(page: Page, url: str, category: str) -> Optional[dict]:
    if not await polite_goto(page, url):
        return None

    raw_title = await page.title()
    title = re.sub(r"^\s*الجمهورية\s*[|\-–—]\s*", "", raw_title).strip()
    if not title:
        return None

    body_parts: list[str] = []
    for sel in [
        "div.detailed-article div.description.direction-rtl p",
        "div.description.direction-rtl p",
        ".article-body p",
        "article p",
    ]:
        els = await page.query_selector_all(sel)
        if els:
            for el in els:
                t = clean(await el.inner_text())
                if t:
                    body_parts.append(t)
            break

    body = " ".join(body_parts)
    if len(body) < 80:
        return None

    date_str = ""
    for sel in [
        "div.detailed-article div.time.direction-rtl.blue",
        "div.time.direction-rtl",
        "time[datetime]",
        ".date",
    ]:
        el = await page.query_selector(sel)
        if el:
            date_str = await el.get_attribute("datetime") or clean(await el.inner_text())
            break

    cat_str = category
    for sel in [
        "div.info-feed a[href*='/news/category/'] div.text.direction-rtl.blue",
        "div.info-feed a[href*='/news/category/']",
    ]:
        el = await page.query_selector(sel)
        if el:
            t = clean(await el.inner_text())
            if t:
                cat_str = t
            break

    return {
        "source": "joumhouria",
        "title": title,
        "body": body,
        "date": date_str,
        "category": cat_str,
    }


async def scrape_joumhouria(page: Page, max_articles: int) -> list[dict]:
    log("═══ Al Joumhouria ═══", Fore.CYAN)
    rows: list[dict] = []
    seen_urls: set[str] = set()
    per_cat = max(40, max_articles // len(JOUMHOURIA_CATEGORIES) + 30)

    for cat_url, cat_name in JOUMHOURIA_CATEGORIES:
        if len(rows) >= max_articles:
            break
        log(f"  ▸ {cat_name}  ({cat_url})", Fore.YELLOW)
        links = await _joumhouria_collect_links(page, cat_url, per_cat)
        log(f"    {len(links)} links found", Fore.CYAN)

        for url in links:
            if len(rows) >= max_articles:
                break
            if url in seen_urls:
                continue
            seen_urls.add(url)

            row = await _joumhouria_parse_article(page, url, cat_name)
            if row:
                rows.append(row)
                log(f"  [{len(rows):>3}/{max_articles}] {row['title'][:65]}", Fore.GREEN)
            else:
                log(f"  [skip] {url[:70]}", Fore.YELLOW)

    log(f"Al Joumhouria → {len(rows)} articles collected", Fore.GREEN)
    return rows


NAHAR_BASE = "https://www.annahar.com"
NAHAR_HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "ar,en;q=0.5"}
_NAHAR_SECTION_MAP = {
    "lebanon": "لبنان",
    "politics": "سياسة",
    "society": "مجتمع",
    "economy": "اقتصاد",
    "arab-world": "عرب ودولي",
    "arabian-levant": "المشرق العربي",
    "arabian-gulf": "الخليج العربي",
    "international": "دولي",
    "world": "دولي",
    "culture": "ثقافة",
    "sports": "رياضة",
    "investigations": "تحقيقات",
    "articles": "مقالات",
    "voices": "آراء",
    "united-states": "أميركا",
    "europe": "أوروبا",
    "technology": "تكنولوجيا",
    "health": "صحة",
    "annahar-writers": "كتّاب النهار",
    "whispers": "أسرار الآلهة",
}


def _nahar_category_from_url(url: str) -> str:
    parts = url.replace(NAHAR_BASE, "").strip("/").split("/")
    for part in reversed(parts):
        if part in _NAHAR_SECTION_MAP:
            return _NAHAR_SECTION_MAP[part]
    return parts[0] if parts else ""


_NAHAR_SITEMAP_URL_RE = re.compile(r"<loc>(https://www\.annahar\.com/[^<]+)</loc>")
_NAHAR_ARTICLE_URL_RE = re.compile(r"https://www\.annahar\.com/[a-z\-]+(?:/[a-z\-]+)?/\d{4,}/")
NAHAR_SITEMAPS = [
    "https://www.annahar.com/sitemap/sitemap-2025-08.xml",
    "https://www.annahar.com/sitemap/sitemap-2025-07.xml",
    "https://www.annahar.com/sitemap/sitemap-2025-06.xml",
    "https://www.annahar.com/sitemap/sitemap-2025-05.xml",
]


def _nahar_collect_links(session: http_requests.Session, max_needed: int) -> list[tuple[str, str]]:
    articles: list[tuple[str, str]] = []
    seen: set[str] = set()

    for sitemap_url in NAHAR_SITEMAPS:
        if len(articles) >= max_needed * 2:
            break
        r = session.get(sitemap_url, timeout=30)
        if r.status_code != 200:
            log(f"    {sitemap_url}: status {r.status_code}", Fore.YELLOW)
            continue

        for m in _NAHAR_SITEMAP_URL_RE.finditer(r.text):
            url = m.group(1)
            if url in seen or not _NAHAR_ARTICLE_URL_RE.match(url):
                continue
            seen.add(url)
            articles.append((url, _nahar_category_from_url(url)))

        log(f"    {sitemap_url.rsplit('/', 1)[-1]}: {len(articles)} articles so far", Fore.CYAN)
        time.sleep(random.uniform(0.3, 0.6))

    random.shuffle(articles)
    return articles


_NAHAR_SKIP = {"اشترك في نشرتنا", "شكرا على الاشتراك", "نشرتنا الاخبار"}


def _nahar_parse_article(html: str, fallback_cat: str) -> Optional[dict]:
    m = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.DOTALL)
    if not m:
        title_m = re.search(r"<title>(.*?)</title>", html)
        if not title_m:
            return None
        title = unescape(re.sub(r"\s*\|\s*النهار.*$", "", title_m.group(1))).strip()
    else:
        title = unescape(clean(re.sub(r"<[^>]+>", "", m.group(1))))

    if not title:
        return None

    body_parts: list[str] = []
    for p_match in re.finditer(r"<p[^>]*>(.*?)</p>", html, re.DOTALL):
        text = clean(re.sub(r"<[^>]+>", "", unescape(p_match.group(1))))
        if len(text) > 30 and not any(s in text for s in _NAHAR_SKIP):
            body_parts.append(text)

    body = " ".join(body_parts)
    if len(body) < 80:
        return None

    date_str = ""
    for pat in [
        r'datetime="([^"]+)"',
        r"(\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}:\d{2}\s*[AP]M)",
        r"(\d{2}-\d{2}-\d{4}\s*\|\s*\d{2}:\d{2})",
    ]:
        dm = re.search(pat, html)
        if dm:
            date_str = dm.group(1).strip()
            break

    cat_str = fallback_cat
    cat_m = re.search(
        r'class="[^"]*breadcrumb[^"]*"[^>]*>.*?<a[^>]*>([^<]+)</a>',
        html,
        re.DOTALL,
    )
    if cat_m:
        t = clean(cat_m.group(1))
        if t and t not in ("النهار", "الرئيسية"):
            cat_str = t

    return {
        "source": "nahar",
        "title": title,
        "body": body,
        "date": date_str,
        "category": cat_str,
    }


def scrape_nahar(max_articles: int) -> list[dict]:
    log("═══ An-Nahar ═══", Fore.CYAN)

    session = http_requests.Session()
    session.headers.update(NAHAR_HEADERS)

    log("  Collecting article links from sitemaps…", Fore.YELLOW)
    article_links = _nahar_collect_links(session, max_articles)
    log(f"  {len(article_links)} unique article URLs found", Fore.CYAN)

    rows: list[dict] = []
    seen: set[str] = set()

    for url, cat_label in article_links:
        if len(rows) >= max_articles:
            break
        if url in seen:
            continue
        seen.add(url)

        time.sleep(random.uniform(0.8, 1.5))
        r = session.get(url, timeout=20)
        if r.status_code != 200:
            log(f"  [skip] {url.split('/')[-1][:50]} status {r.status_code}", Fore.YELLOW)
            continue

        row = _nahar_parse_article(r.text, cat_label)
        if row:
            rows.append(row)
            log(f"  [{len(rows):>3}/{max_articles}] {row['title'][:65]}", Fore.GREEN)
        else:
            log(f"  [skip] {url.split('/')[-1][:50]}", Fore.YELLOW)

    log(f"An-Nahar → {len(rows)} articles collected", Fore.GREEN)
    return rows


def write_csv(rows: list[dict], path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    log(f"\nWrote {len(rows)} rows → {output}", Fore.GREEN)


async def _run(args: argparse.Namespace) -> None:
    all_rows: list[dict] = []

    if "joumhouria" in args.sources:
        async with async_playwright() as pw:
            browser: Browser = await pw.chromium.launch(headless=args.headless)
            ctx: BrowserContext = await browser.new_context(
                locale="ar-LB",
                extra_http_headers={"Accept-Language": "ar,en-US;q=0.9,en;q=0.8"},
                user_agent=USER_AGENT,
            )
            page: Page = await ctx.new_page()
            page.set_default_timeout(30_000)
            all_rows += await scrape_joumhouria(page, args.max_articles)
            await browser.close()

    if "nahar" in args.sources:
        all_rows += scrape_nahar(args.max_articles)

    log(f"\nTotal articles collected: {len(all_rows)}", Fore.CYAN)
    write_csv(all_rows, args.output_csv)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape Al Joumhouria and An-Nahar articles into a CSV dataset."
    )
    parser.add_argument(
        "--output-csv",
        default="data/logistic_regression/lebanese_newspapers.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=500,
        help="Max articles per source",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["joumhouria", "nahar"],
        choices=["joumhouria", "nahar"],
        help="Sources to scrape",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser headless (Joumhouria only)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(_run(args))
    except Exception as e:
        log(f"Fatal: {e}", Fore.RED)
        raise


if __name__ == "__main__":
    main()
