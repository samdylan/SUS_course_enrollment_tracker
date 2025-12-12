"""
osu_coreed_capacity_backfill_details.py

Backfill *detailed* capacity and enrollment for CoreEd courses
(CFSI, CSSS, CSDP, CFSS) in the current OSU academic year.

- Figures out the 4 term codes for the current academic year label
  (Summer, Fall, Winter, Spring), e.g. 202600–202603.
- For each term and CoreEd attribute, calls the *search* endpoint
  to get all sections.
- For each section, calls the *details* endpoint to fetch true
  `enrollment` and `max_enroll`.
- Writes rows into the existing `coreed_capacity` table with
  snapshot_timestamp, term_srcdb, coreed_attr, campus_code, code,
  section, enrolled, capacity.

You can run this any time you want to refresh CoreEd capacity+enrollment
for the *current* academic year (as determined in OSU local time).

Run:
    source .venv/bin/activate
    python osu_coreed_capacity_backfill_details.py
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo
import pathlib
import sqlite3
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import requests
import pandas as pd


# ---------- CONSTANTS / CONFIG ----------

CLASSES_API_URL = "https://classes.oregonstate.edu/api/"

SEARCH_QUERY = {
    "page": "fose",
    "route": "search",
}

DETAILS_QUERY = {
    "page": "fose",
    "route": "details",
}

# CoreEd attributes we track
CORED_ATTRS = ["CFSI", "CSSS", "CSDP", "CFSS"]

# Local DB path (same DB as the SUS dashboard)
DB_PATH = pathlib.Path("osu_enrollment_log_classes.db")

OSU_TZ = ZoneInfo("America/Los_Angeles")


# ---------- SMALL HELPERS ----------


def now_osu() -> dt.datetime:
    """Current datetime in OSU local time."""
    return dt.datetime.now(tz=OSU_TZ)


def compute_academic_year_label(today: dt.date | None = None) -> int:
    """
    Given OSU local date, compute the academic year label used in term codes.

    Convention (already used in your other scripts):

        Summer 2025  -> 2026  (202600)
        Fall   2025  -> 2026  (202601)
        Winter 2026  -> 2026  (202602)
        Spring 2026  -> 2026  (202603)

    i.e. the label is *next* calendar year for Summer/Fall,
    and current calendar year for Winter/Spring.
    """
    if today is None:
        today = now_osu().date()

    year = today.year
    month = today.month

    if month in (6, 7, 8):  # roughly summer
        return year + 1
    elif month in (9, 10, 11, 12):  # fall
        return year + 1
    else:  # Jan–May -> winter/spring
        return year


def academic_year_terms(label: int) -> List[str]:
    """
    Given an academic year label (e.g. 2026) return the 4 srcdb codes:

        Summer -> label00
        Fall   -> label01
        Winter -> label02
        Spring -> label03
    """
    base = label * 100
    return [f"{base + i:06d}" for i in range(0, 4)]


def safe_int(val):
    if val in (None, "", " "):
        return None
    try:
        return int(val)
    except Exception:
        return None


@dataclass
class SectionStub:
    term_srcdb: str
    coreed_attr: str
    campus_code: str
    code: str
    section: str
    crn: str


# ---------- API CALLS ----------


def build_coreed_search_payload(srcdb: str, coreed_attr: str) -> dict:
    """
    Build the JSON payload for the search endpoint for a CoreEd attribute,
    returning *all campuses* in that term.
    """
    field = {
        "CFSI": "attributes_CFSI",
        "CSSS": "attributes_CSSS",
        "CSDP": "attributes_CSDP",
        "CFSS": "attributes_CFSS",
    }[coreed_attr]

    payload = {
        "other": {"srcdb": srcdb},
        "criteria": [
            {"field": field, "value": "Y"},
            # No "camp" filter here -> all campuses
        ],
    }
    return payload


def fetch_coreed_stubs_for_term_attr(srcdb: str, coreed_attr: str) -> List[SectionStub]:
    """
    Use the search endpoint to get section stubs for one term & CoreEd attribute.
    """
    params = SEARCH_QUERY
    payload = build_coreed_search_payload(srcdb, coreed_attr)

    resp = requests.post(CLASSES_API_URL, params=params, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    # In normal, successful cases there should be a "data" list.
    rows = data.get("data") or data.get("results") or []

    stubs: List[SectionStub] = []
    for row in rows:
        # Rows from search have fields like "campus", "code", "crn", "section"
        campus_code = row.get("campus", "")  # e.g. "C", "E", "B", "DI", "DB", "L"
        code = (row.get("code") or "").strip()
        section = (row.get("section") or "").strip()
        crn = str(row.get("crn") or "").strip()

        if not code or not section or not crn:
            continue

        # Normalize campus codes into C/E/B buckets, as in the dashboard
        campus_bucket = normalize_campus(campus_code)

        stubs.append(
            SectionStub(
                term_srcdb=srcdb,
                coreed_attr=coreed_attr,
                campus_code=campus_bucket,
                code=code,
                section=section,
                crn=crn,
            )
        )

    return stubs


def normalize_campus(campus_code: str) -> str:
    """
    Map many campus codes to the 3 buckets you actually care about.
    """
    campus_code = (campus_code or "").upper()

    # Ecampus / Distance codes
    if campus_code in {"E", "D", "DI", "DIST", "DISTANCE"}:
        return "E"

    # Cascades / Bend codes
    if campus_code in {"B", "CASC", "DB", "L"}:
        return "B"

    # Everything else -> Corvallis
    return "C"


def build_details_payload(srcdb: str, code: str, crn: str, matched_crns: Iterable[str]) -> dict:
    """
    Build the JSON body used by the details endpoint.

    This follows the working pattern you already used for the SUS enrollment script:
        {
          "other": {"srcdb": srcdb},
          "criteria": [
              {"field": "key",     "value": f"crn:{crn}"},
              {"field": "group",   "value": f"code:{code}"},
              {"field": "matched", "value": "crn:CRN1,CRN2,..."},
          ]
        }
    """
    matched_list = sorted({str(c).strip() for c in matched_crns if str(c).strip()})
    matched_value = "crn:" + ",".join(matched_list) if matched_list else f"crn:{crn}"

    return {
        "other": {"srcdb": srcdb},
        "criteria": [
            {"field": "key", "value": f"crn:{crn}"},
            {"field": "group", "value": f"code:{code}"},
            {"field": "matched", "value": matched_value},
        ],
    }


def fetch_details_enroll_capacity(
    srcdb: str,
    code: str,
    crn: str,
    matched_crns: Iterable[str],
) -> Tuple[int | None, int | None]:
    """
    Call the details endpoint for one section and extract (capacity, enrolled).

    The OSU classes details API usually returns a single object with fields like
    "max_enroll" and "enrollment", or a top-level {"fatal": "..."} on error.
    """
    payload = build_details_payload(srcdb, code, crn, matched_crns)
    resp = requests.post(CLASSES_API_URL, params=DETAILS_QUERY, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    # If "fatal" present, log a warning and return None/None
    if isinstance(data, dict) and data.get("fatal"):
        # You can uncomment this if you want more verbose logging:
        # print(f"  [WARN] details fatal for {srcdb} {code} {crn}: {data.get('fatal')}")
        return None, None

    # Some variants might wrap section in "data" or "results"
    if isinstance(data, dict) and ("data" in data or "results" in data):
        rows = data.get("data") or data.get("results") or []
        if rows:
            data = rows[0]

    capacity = safe_int(data.get("max_enroll"))
    enrolled = safe_int(data.get("enrollment"))
    return capacity, enrolled


# ---------- MAIN BACKFILL LOGIC ----------


def main() -> None:
    osu_now = now_osu()
    today = osu_now.date()
    label = compute_academic_year_label(today)
    terms = academic_year_terms(label)

    print(
        f"Detected academic year label {label} from OSU local date {today}, "
        f"terms: {', '.join(terms)}"
    )
    print(
        f"[{osu_now.isoformat()}] Starting CoreEd DETAILS backfill for terms: "
        + ", ".join(terms)
    )

    all_rows: List[Dict] = []

    for srcdb in terms:
        print(f"\n[TERM {srcdb}] Building details backfill...")
        term_stubs: List[SectionStub] = []

        # 1) Collect stubs via search
        for attr in CORED_ATTRS:
            print(f"  Term {srcdb}: CoreEd={attr} -> searching...")
            stubs = fetch_coreed_stubs_for_term_attr(srcdb, attr)
            print(f"    -> {len(stubs)} section stubs for CoreEd={attr}.")
            term_stubs.extend(stubs)

        print(f"  [TERM {srcdb}] Total stubs across attributes: {len(term_stubs)}")

        if not term_stubs:
            continue

        # 2) Build matched-crn map per (srcdb, code)
        code_to_crns: Dict[Tuple[str, str], List[str]] = {}
        for s in term_stubs:
            key = (s.term_srcdb, s.code)
            code_to_crns.setdefault(key, []).append(s.crn)

        # 3) Fetch details for each stub
        print("  Fetching details for each stub...")
        snapshot_ts = osu_now.isoformat()

        for s in term_stubs:
            matched_crns = code_to_crns[(s.term_srcdb, s.code)]
            capacity, enrolled = fetch_details_enroll_capacity(
                s.term_srcdb, s.code, s.crn, matched_crns
            )

            all_rows.append(
                {
                    "snapshot_timestamp": snapshot_ts,
                    "term_srcdb": s.term_srcdb,
                    "coreed_attr": s.coreed_attr,
                    "campus_code": s.campus_code,
                    "code": s.code,
                    "section": s.section,
                    "enrolled": enrolled,
                    "capacity": capacity,
                }
            )

    if not all_rows:
        print("No rows built; nothing to insert.")
        return

    df = pd.DataFrame(all_rows)

    print(
        f"\n[SUMMARY] Built {len(df)} CoreEd rows with details for terms: "
        + ", ".join(sorted(df['term_srcdb'].unique()))
    )

    # 4) Insert into DB
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS coreed_capacity (
            snapshot_timestamp TEXT NOT NULL,
            term_srcdb         TEXT NOT NULL,
            coreed_attr        TEXT NOT NULL,
            campus_code        TEXT,
            code               TEXT,
            section            TEXT,
            enrolled           INTEGER,
            capacity           INTEGER
        )
        """
    )
    conn.commit()

    rows_to_insert = [
        (
            r["snapshot_timestamp"],
            r["term_srcdb"],
            r["coreed_attr"],
            r["campus_code"],
            r["code"],
            r["section"],
            int(r["enrolled"]) if pd.notna(r["enrolled"]) else None,
            int(r["capacity"]) if pd.notna(r["capacity"]) else None,
        )
        for _, r in df.iterrows()
    ]

    cur.executemany(
        """
        INSERT INTO coreed_capacity (
            snapshot_timestamp,
            term_srcdb,
            coreed_attr,
            campus_code,
            code,
            section,
            enrolled,
            capacity
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows_to_insert,
    )
    conn.commit()
    conn.close()

    print(f"  [DB] Inserted {len(rows_to_insert)} rows into coreed_capacity.\n")

    # Show a few sample rows for sanity check
    print("Sample rows from this backfill run:")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
