"""
SHL Catalog Scraper
-------------------
Scrapes all Individual Test Solutions from the SHL product catalog.
Run ONCE locally before deploying:
    pip install playwright beautifulsoup4 requests
    playwright install chromium
    python scraper.py

Outputs: catalog.json
"""

import asyncio
import json
import re
import time
from pathlib import Path

from playwright.async_api import async_playwright

BASE_URL = "https://www.shl.com"
CATALOG_URL = f"{BASE_URL}/solutions/products/product-catalog/"
PAGE_SIZE = 12  # Items per catalog page

TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


async def scrape_catalog_page(page, start: int) -> list[dict]:
    """Scrape a single paginated catalog page for Individual Test Solutions."""
    url = (
        f"{CATALOG_URL}?action_doFilteringForm=Search&f=1"
        f"&start={start}&type=1"
    )
    await page.goto(url, wait_until="networkidle", timeout=30000)
    await page.wait_for_selector("table", timeout=10000)

    rows = await page.query_selector_all("table tbody tr")
    items = []
    for row in rows:
        cells = await row.query_selector_all("td")
        if len(cells) < 4:
            continue

        # Column 0: Name + URL
        link = await cells[0].query_selector("a")
        if not link:
            continue
        name = (await link.inner_text()).strip()
        href = await link.get_attribute("href")
        if not href:
            continue
        url_full = href if href.startswith("http") else BASE_URL + href

        # Column 1: Remote Testing (tick = yes)
        remote = await _has_tick(cells[1])

        # Column 2: Adaptive/IRT (tick = yes)
        adaptive = await _has_tick(cells[2])

        # Column 3: Test Type (letter badge)
        type_span = await cells[3].query_selector(".catalogue__circle, span")
        test_type = ""
        if type_span:
            test_type = (await type_span.inner_text()).strip()

        items.append({
            "name": name,
            "url": url_full,
            "remote_testing": remote,
            "adaptive_irt": adaptive,
            "test_type": test_type,
            "test_type_label": TEST_TYPE_MAP.get(test_type, test_type),
        })

    return items


async def _has_tick(cell) -> bool:
    """Return True if the cell contains a tick/checkmark icon."""
    text = (await cell.inner_text()).strip()
    html = await cell.inner_html()
    return bool(text) or "tick" in html.lower() or "✓" in text or "check" in html.lower()


async def count_total_pages(page) -> int:
    """Get total number of pages in the catalog."""
    url = f"{CATALOG_URL}?action_doFilteringForm=Search&f=1&start=0&type=1"
    await page.goto(url, wait_until="networkidle", timeout=30000)
    await page.wait_for_selector("table", timeout=10000)

    # Find last page number in pagination
    pagination = await page.query_selector_all(".pagination a, .paginationControl a")
    max_page = 1
    for link in pagination:
        text = (await link.inner_text()).strip()
        if text.isdigit():
            max_page = max(max_page, int(text))
    return max_page


async def scrape_product_detail(page, url: str) -> dict:
    """Scrape additional details from an individual product page."""
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=20000)
        details = {}

        # Description
        desc_sel = ".product-catalogue__description, .product-description, article p"
        desc_el = await page.query_selector(desc_sel)
        if desc_el:
            details["description"] = (await desc_el.inner_text()).strip()[:500]

        # Job Levels
        levels = []
        level_els = await page.query_selector_all(
            ".product-catalogue__job-level .tag, "
            "[class*='job-level'] span, "
            "[class*='level'] .catalogue__tag"
        )
        for el in level_els:
            t = (await el.inner_text()).strip()
            if t:
                levels.append(t)
        if levels:
            details["job_levels"] = levels

        # Languages
        langs = []
        lang_els = await page.query_selector_all(
            ".product-catalogue__languages .tag, "
            "[class*='language'] span"
        )
        for el in lang_els:
            t = (await el.inner_text()).strip()
            if t:
                langs.append(t)
        if langs:
            details["languages"] = langs

        return details
    except Exception as e:
        print(f"  Warning: could not fetch detail for {url}: {e}")
        return {}


async def main():
    print("Starting SHL catalog scrape...")
    all_items = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )
        page = await context.new_page()

        # Discover total pages
        print("Counting pages...")
        total_pages = await count_total_pages(page)
        print(f"Found {total_pages} pages of Individual Test Solutions")

        # Scrape listing pages
        for p_num in range(1, total_pages + 1):
            start = (p_num - 1) * PAGE_SIZE
            print(f"  Scraping listing page {p_num}/{total_pages} (start={start})...")
            items = await scrape_catalog_page(page, start)
            all_items.extend(items)
            await asyncio.sleep(0.5)  # polite delay

        print(f"Found {len(all_items)} assessments total. Fetching details...")

        # Scrape individual product pages for descriptions + metadata
        for i, item in enumerate(all_items):
            print(f"  Detail {i+1}/{len(all_items)}: {item['name']}")
            details = await scrape_product_detail(page, item["url"])
            item.update(details)
            await asyncio.sleep(0.3)

        await browser.close()

    # Deduplicate by URL
    seen = set()
    unique = []
    for item in all_items:
        if item["url"] not in seen:
            seen.add(item["url"])
            unique.append(item)

    out_path = Path("catalog.json")
    out_path.write_text(json.dumps(unique, indent=2, ensure_ascii=False))
    print(f"\nDone! Saved {len(unique)} assessments to {out_path}")
    return unique


if __name__ == "__main__":
    asyncio.run(main())
