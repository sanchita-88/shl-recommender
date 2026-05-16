"""
download_catalog.py
--------------------
Downloads the official SHL catalog JSON from the provided endpoint,
normalises field names, and saves to catalog.json.

Run once before deploying:
    python download_catalog.py
"""

import json
import sys
from pathlib import Path

import requests

CATALOG_URL = (
    "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
)

# Full label → single-letter code used in API responses
KEY_TO_CODE = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Biodata & Situational Judgement": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Personality & Behaviour": "P",
    "Simulations": "S",
}


def keys_to_codes(keys: list[str]) -> str:
    """Convert a list of full key labels to comma-separated letter codes, e.g. 'K,S'."""
    codes = []
    seen = set()
    for k in keys:
        code = KEY_TO_CODE.get(k, "")
        if code and code not in seen:
            codes.append(code)
            seen.add(code)
    return ",".join(codes)


def normalise(raw: dict) -> dict:
    """Normalise a raw catalog entry into the format consumed by agent.py."""
    keys = raw.get("keys", [])
    return {
        "entity_id": raw.get("entity_id", ""),
        "name": raw.get("name", "").strip(),
        "url": raw.get("link", "").strip(),          # rename link → url
        "test_type": keys_to_codes(keys),             # e.g. "K" or "K,S"
        "test_type_labels": keys,                     # e.g. ["Knowledge & Skills"]
        "remote_testing": raw.get("remote", "no").lower() == "yes",
        "adaptive_irt": raw.get("adaptive", "no").lower() == "yes",
        "description": (raw.get("description") or "").strip(),
        "job_levels": raw.get("job_levels") or [],
        "languages": raw.get("languages") or [],
        "duration": raw.get("duration", ""),
    }


def main():
    print(f"Downloading catalog from:\n  {CATALOG_URL}\n")
    try:
        r = requests.get(CATALOG_URL, timeout=30)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # strict=False tolerates invalid control characters in the SHL catalog JSON
    try:
        raw_items = r.json()
    except Exception:
        import json as _json
        raw_items = _json.loads(r.text, strict=False)
    print(f"  → {len(raw_items)} raw entries received")

    normalised = []
    for item in raw_items:
        n = normalise(item)
        if n["name"] and n["url"]:
            normalised.append(n)

    out = Path("catalog.json")
    out.write_text(json.dumps(normalised, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"  → {len(normalised)} assessments saved to {out}")


if __name__ == "__main__":
    main()
