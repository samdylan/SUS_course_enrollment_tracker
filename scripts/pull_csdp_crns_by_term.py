#!/usr/bin/env python3
"""Pull CSDP CRNs by term and flag CPDP overlap.

Filters applied:
- Include only sections with attribute CSDP
- Exclude cancelled sections
- Exclude recitation sections
- Flag rows whose CRN also appears in CPDP for the same term
"""

from __future__ import annotations

import argparse
import json
import random
import time
from typing import Dict, Iterable, List, Set

import requests

CLASSES_API_URL = "https://classes.oregonstate.edu/api/"
SEARCH_QUERY = {"page": "fose", "route": "search"}
DEFAULT_TERMS = ["202601", "202602", "202603"]

# Conservative recitation matching; keep this explicit.
RECITATION_CODES = {"R", "REC", "RECITATION"}


def browser_headers(referer: str) -> Dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) "
            "Version/26.1 Safari/605.1.15"
        ),
        "Content-Type": "application/json",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "X-Requested-With": "XMLHttpRequest",
        "Origin": "https://classes.oregonstate.edu",
        "Referer": referer,
    }


def fetch_attr_results(session: requests.Session, srcdb: str, attr: str, max_tries: int = 6) -> List[Dict]:
    payload = {
        "other": {"srcdb": srcdb},
        "criteria": [{"field": f"attributes_{attr}", "value": "Y"}],
    }
    referer = f"https://classes.oregonstate.edu/?srcdb={srcdb}&attributes=attributes_{attr}"

    last_preview = ""
    for attempt in range(1, max_tries + 1):
        response = session.post(
            CLASSES_API_URL,
            params=SEARCH_QUERY,
            json=payload,
            headers=browser_headers(referer),
            timeout=45,
        )
        content_type = (response.headers.get("Content-Type") or "").lower()
        body = response.text or ""
        last_preview = body[:300]

        retryable = (
            response.status_code in (202, 429)
            or 500 <= response.status_code <= 599
            or "text/html" in content_type
            or not body.strip()
        )
        if retryable:
            time.sleep(min(2**attempt, 20) + random.uniform(0, 0.25))
            continue

        response.raise_for_status()
        parsed = response.json()
        return parsed.get("results", []) or []

    raise RuntimeError(f"Failed to fetch {attr} for {srcdb}. Last preview: {last_preview}")


def is_cancelled(record: Dict) -> bool:
    return str(record.get("isCancelled", "")).strip() == "1"


def is_recitation(record: Dict) -> bool:
    schd = str(record.get("schd") or "").strip().upper()
    section = str(record.get("no") or record.get("section") or "").strip().upper()
    title = str(record.get("title") or "").strip().upper()

    if schd in RECITATION_CODES:
        return True
    if section.startswith("R") and section[1:].isdigit():
        return True
    if "RECITATION" in title:
        return True
    return False


def normalize_crn(record: Dict) -> str:
    return str(record.get("crn") or "").strip()


def build_term_list(session: requests.Session, srcdb: str) -> List[Dict[str, str]]:
    csdp_results = fetch_attr_results(session, srcdb, "CSDP")
    cpdp_results = fetch_attr_results(session, srcdb, "CPDP")

    cpdp_crns: Set[str] = {normalize_crn(r) for r in cpdp_results if normalize_crn(r)}

    rows: List[Dict[str, str]] = []
    seen: Set[str] = set()

    for rec in csdp_results:
        crn = normalize_crn(rec)
        if not crn:
            continue
        if crn in seen:
            continue
        seen.add(crn)

        if is_cancelled(rec):
            continue
        if is_recitation(rec):
            continue

        rows.append(
            {
                "crn": crn,
                "cpdp": "Y" if crn in cpdp_crns else "N",
                "code": str(rec.get("code") or "").strip(),
                "section": str(rec.get("no") or rec.get("section") or "").strip(),
                "schedule_type": str(rec.get("schd") or "").strip(),
                "status": str(rec.get("stat") or "").strip(),
                "title": str(rec.get("title") or "").strip(),
            }
        )

    rows.sort(key=lambda r: (int(r["crn"]) if r["crn"].isdigit() else r["crn"]))
    return rows


def format_report(term_rows: Dict[str, List[Dict[str, str]]]) -> str:
    lines: List[str] = []
    for term, rows in term_rows.items():
        lines.append(f"{term} ({len(rows)} sections)")
        for row in rows:
            lines.append(f"- {row['crn']} | CPDP={row['cpdp']} | {row['code']}-{row['section']} | {row['schedule_type']} | {row['status']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull CSDP CRNs and flag CPDP overlap by term")
    parser.add_argument(
        "--terms",
        nargs="+",
        default=DEFAULT_TERMS,
        help="Term srcdb values (default: 202601 202602 202603)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output instead of text report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    term_rows: Dict[str, List[Dict[str, str]]] = {}

    with requests.Session() as session:
        for term in args.terms:
            term_rows[term] = build_term_list(session, term)

    if args.json:
        print(json.dumps(term_rows, indent=2))
    else:
        print(format_report(term_rows), end="")


if __name__ == "__main__":
    main()
