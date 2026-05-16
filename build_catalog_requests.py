"""
build_catalog_requests.py
--------------------------
Alternative scraper using plain requests + BeautifulSoup.
Works if SHL serves the table server-side (no JS required).
Try this FIRST — if catalog.json ends up empty, use scraper.py (Playwright).

Usage:
    pip install requests beautifulsoup4
    python build_catalog_requests.py
"""

import json
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE = "https://www.shl.com"
CATALOG = f"{BASE}/solutions/products/product-catalog/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
PAGE_SIZE = 12

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


def fetch_page(session: requests.Session, start: int) -> list[dict]:
    params = {
        "action_doFilteringForm": "Search",
        "f": "1",
        "start": str(start),
        "type": "1",
    }
    r = session.get(CATALOG, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    table = soup.find("table")
    if not table:
        return []

    items = []
    for row in table.find_all("tr")[1:]:  # skip header
        cols = row.find_all("td")
        if len(cols) < 4:
            continue

        link = cols[0].find("a")
        if not link:
            continue

        name = link.get_text(strip=True)
        href = link.get("href", "")
        url = href if href.startswith("http") else BASE + href

        # Remote testing: presence of any image / tick span in col 1
        remote = bool(cols[1].find("img") or cols[1].find(class_=lambda c: c and "tick" in c.lower()))

        # Adaptive/IRT: same pattern col 2
        adaptive = bool(cols[2].find("img") or cols[2].find(class_=lambda c: c and "tick" in c.lower()))

        # Test type: text of the last span / badge in col 3
        badge = cols[3].find("span") or cols[3]
        test_type = badge.get_text(strip=True)

        items.append({
            "name": name,
            "url": url,
            "remote_testing": remote,
            "adaptive_irt": adaptive,
            "test_type": test_type,
            "test_type_label": TEST_TYPE_MAP.get(test_type, test_type),
            "description": "",
            "job_levels": [],
            "languages": [],
        })
    return items


def count_total(session: requests.Session) -> int:
    """Read pagination to find the last page number."""
    params = {"action_doFilteringForm": "Search", "f": "1", "start": "0", "type": "1"}
    r = session.get(CATALOG, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    last = 1
    for a in soup.select(".pagination a, [class*='pagination'] a"):
        txt = a.get_text(strip=True)
        if txt.isdigit():
            last = max(last, int(txt))
    return last


def fetch_detail(session: requests.Session, url: str) -> dict:
    """Fetch description + job levels from an individual product page."""
    try:
        r = session.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        details: dict = {}

        # Description: first substantial <p> in main content
        for sel in [".product-catalogue__description p", "article p", "main p"]:
            el = soup.select_one(sel)
            if el:
                txt = el.get_text(strip=True)
                if len(txt) > 30:
                    details["description"] = txt[:600]
                    break

        # Job levels
        levels = []
        for el in soup.select("[class*='job-level'] .tag, [class*='job-level'] span"):
            t = el.get_text(strip=True)
            if t:
                levels.append(t)
        if levels:
            details["job_levels"] = levels

        return details
    except Exception as e:
        print(f"  Warning: detail fetch failed for {url}: {e}")
        return {}


def main():
    print("SHL catalog scraper (requests-based)")
    all_items: list[dict] = []

    with requests.Session() as session:
        print("Discovering pages…")
        total_pages = count_total(session)
        print(f"  → {total_pages} pages")

        for page_num in range(1, total_pages + 1):
            start = (page_num - 1) * PAGE_SIZE
            print(f"  Page {page_num}/{total_pages} (start={start})")
            items = fetch_page(session, start)
            all_items.extend(items)
            time.sleep(0.4)

        print(f"Found {len(all_items)} assessments. Fetching details…")
        for i, item in enumerate(all_items):
            print(f"  [{i+1}/{len(all_items)}] {item['name']}")
            extra = fetch_detail(session, item["url"])
            item.update(extra)
            time.sleep(0.3)

    # Deduplicate
    seen: set[str] = set()
    unique = []
    for item in all_items:
        if item["url"] not in seen:
            seen.add(item["url"])
            unique.append(item)

    out = Path("catalog.json")
    out.write_text(json.dumps(unique, indent=2, ensure_ascii=False))
    print(f"\n✓ Saved {len(unique)} assessments → {out}")


if __name__ == "__main__":
    main()
