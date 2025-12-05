"""
osu_coreed_capacity_snapshot.py

Snapshot CoreEd (CFSI, CSSS, CSDP) section-level capacity and enrollment
for a given term, using the same details-endpoint logic as the SUS tracker.

Key points:
- Uses classes.oregonstate.edu search endpoint to find sections with a given
  CoreEd attribute (CFSI, CSSS, CSDP) across ALL campuses.
- For each (code, CRN) pair, calls the details endpoint with:
      {
        "group":   "code:{CODE}",
        "key":     "crn:{CRN}",
        "srcdb":   "{TERM_SRCDB}",
        "matched": "crn:CRN1,CRN2,..."
      }
  to obtain accurate "enrollment" and "max_enroll".
- Appends rows into the SQLite table `coreed_capacity` in
  osu_enrollment_log_classes.db.

Run manually with:
    source .venv/bin/activate
    python osu_coreed_capacity_snapshot.py
"""

import datetime as dt
import json
import pathlib
import sys
from typing import Any, Dict, List, Tuple

import sqlite3
import requests
import pandas as pd

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:  # pragma: no cover (older Python)
    from backports.zoneinfo import ZoneInfo  # type: ignore


# ---------- CONFIG ----------

CLASSES_API_URL = "https://classes.oregonstate.edu/api/"

SEARCH_QUERY = {
    "page": "fose",
    "route": "search",
}

DETAILS_QUERY = {
    "page": "fose",
    "route": "details",
}

# CoreEd attributes to track
COREED_ATTRS = ["CFSI", "CSSS", "CSDP", "CFSS"]

# Optional manual override: set to a specific srcdb string like "202602" to bypass date inference.
# Leave as None to infer term from today's date.
TERM_SRCDB: str | None = None

DB_PATH = pathlib.Path("osu_enrollment_log_classes.db")


# ---------- HELPERS ----------

def now_pst_iso() -> str:
    """Return current time in OSU local time (America/Los_Angeles) as ISO string."""
    tz = ZoneInfo("America/Los_Angeles")
    return dt.datetime.now(tz).isoformat(timespec="seconds")


def infer_term_srcdb_from_today(now: dt.datetime | None = None) -> str:
    """
    Infer OSU term srcdb (e.g., 202602) from the current date in OSU local time.

    Approximate mapping of calendar date -> term code:
      - Fall (Sep–Dec):   term code 01, year = calendar year + 1   (e.g., Oct 2025 -> 202601)
      - Winter (Jan–Mar): term code 02, year = calendar year       (e.g., Jan 2026 -> 202602)
      - Spring (Apr–May): term code 03, year = calendar year       (e.g., May 2026 -> 202603)
      - Summer (Jun–Aug): term code 00, year = calendar year + 1   (e.g., Jul 2025 -> 202600)

    These rules are designed to match the pattern used by classes.oregonstate.edu,
    where Summer+Fall precede Winter+Spring in the same academic year.
    """
    tz = ZoneInfo("America/Los_Angeles")
    if now is None:
        now = dt.datetime.now(tz)
    else:
        if now.tzinfo is None:
            now = now.replace(tzinfo=tz)
        else:
            now = now.astimezone(tz)

    y = now.year
    m = now.month

    if m in (9, 10, 11, 12):  # Fall
        src_year = y + 1
        term_code = 1
    elif m in (1, 2, 3):      # Winter
        src_year = y
        term_code = 2
    elif m in (4, 5):         # Spring
        src_year = y
        term_code = 3
    else:                     # 6, 7, 8 -> Summer
        src_year = y + 1
        term_code = 0

    return f"{src_year}{term_code:02d}"


def safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        s = str(value).strip()
        if not s:
            return None
        return int(s)
    except Exception:
        return None


def split_code(code: str) -> Tuple[str | None, str | None]:
    """Split 'SUS 102' -> ('SUS', '102')."""
    if not code:
        return None, None
    parts = code.split()
    if len(parts) >= 2:
        return parts[0], " ".join(parts[1:])
    return None, code


def build_search_payload(srcdb: str, coreed_attr: str) -> Dict[str, Any]:
    """
    Build the JSON payload for the CoreEd search.

    Matches the pattern:
      {"other":{"srcdb":"202602"},
       "criteria":[
          {"field":"attributes_CFSI","value":"Y"}
       ]}
    """
    field_name = f"attributes_{coreed_attr}"
    return {
        "other": {
            "srcdb": srcdb
        },
        "criteria": [
            {
                "field": field_name,
                "value": "Y"
            }
        ]
    }


def fetch_coreed_search(srcdb: str, coreed_attr: str) -> List[Dict[str, Any]]:
    """Call the search endpoint for a given CoreEd attribute across all campuses."""
    payload = build_search_payload(srcdb, coreed_attr)
    resp = requests.post(
        CLASSES_API_URL,
        params=SEARCH_QUERY,
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results", [])
    if not isinstance(results, list):
        results = []
    return results


def build_details_payload(srcdb: str, code: str, crn: str, matched_crns: List[str]) -> Dict[str, Any]:
    """
    Build the JSON payload for the details endpoint, matching the format
    seen in DevTools, e.g.:

      {
        "group":   "code:SUS 102",
        "key":     "crn:37717",
        "srcdb":   "202602",
        "matched": "crn:37716,36727,37717,36976,..."
      }
    """
    matched_str = ",".join(matched_crns)
    return {
        "group": f"code:{code}",
        "key": f"crn:{crn}",
        "srcdb": srcdb,
        "matched": f"crn:{matched_str}",
    }


def extract_enrollment_capacity(details_json: Any) -> Tuple[int | None, int | None]:
    """
    From a details JSON blob, try to pull out (enrollment, max_enroll).
    We handle a few possible shapes:
      - direct dict with keys 'enrollment', 'max_enroll'
      - dict with 'fatal' (error)
      - dict with 'results' or 'data' list
      - list at the top-level
    """
    if details_json is None:
        return None, None

    # If it's a list, recurse on the first element.
    if isinstance(details_json, list):
        if not details_json:
            return None, None
        return extract_enrollment_capacity(details_json[0])

    if not isinstance(details_json, dict):
        return None, None

    # If there's an error marker, bail out.
    if "fatal" in details_json:
        return None, None

    # Direct keys
    if "enrollment" in details_json or "max_enroll" in details_json:
        enrolled = safe_int(details_json.get("enrollment"))
        capacity = safe_int(details_json.get("max_enroll"))
        return enrolled, capacity

    # Sometimes nested under "results" or "data"
    for key in ("results", "data"):
        if key in details_json and isinstance(details_json[key], list) and details_json[key]:
            return extract_enrollment_capacity(details_json[key][0])

    return None, None


def fetch_details_for_section(srcdb: str, code: str, crn: str, matched_crns: List[str]) -> Tuple[int | None, int | None]:
    """Call the details endpoint once for a given (code, crn)."""
    payload = build_details_payload(srcdb, code, crn, matched_crns)
    resp = requests.post(
        CLASSES_API_URL,
        params=DETAILS_QUERY,
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return extract_enrollment_capacity(data)


def ensure_coreed_table(conn: sqlite3.Connection) -> None:
    """Create the coreed_capacity table if it does not exist."""
    conn.execute(
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


# ---------- MAIN SNAPSHOT LOGIC ----------

def build_coreed_snapshot_for_term(srcdb: str) -> pd.DataFrame:
    """
    For the given term, build a DataFrame of CoreEd sections with
    (enrolled, capacity) based on the details endpoint.
    """
    all_records: List[Dict[str, Any]] = []

    for coreed_attr in COREED_ATTRS:
        print(f"Term {srcdb}: CoreEd={coreed_attr} -> searching (all campuses)...")
        search_results = fetch_coreed_search(srcdb, coreed_attr)
        print(f"  Found {len(search_results)} raw sections for CoreEd={coreed_attr}.")

        # Group by course code so we can build the 'matched' list of CRNs.
        by_code: Dict[str, Dict[str, Any]] = {}
        for rec in search_results:
            code = rec.get("code", "").strip()
            if not code:
                continue
            crn = str(rec.get("crn", "")).strip()
            if not crn:
                continue
            group = by_code.setdefault(code, {"crns": [], "rows": []})
            if crn not in group["crns"]:
                group["crns"].append(crn)
            group["rows"].append(rec)

        # For each code, call details for each CRN
        for code, group in by_code.items():
            matched_crns = group["crns"]
            subject, course_number = split_code(code)

            for rec in group["rows"]:
                crn = str(rec.get("crn", "")).strip()
                if not crn:
                    continue

                campus_code = rec.get("camp")  # e.g. C, B, DI, DB, L
                section = rec.get("no") or rec.get("section")

                try:
                    enrolled, capacity = fetch_details_for_section(srcdb, code, crn, matched_crns)
                except Exception as e:
                    print(f"  WARNING: details fetch failed for {code} CRN={crn}: {e}")
                    enrolled, capacity = None, None

                all_records.append(
                    {
                        "term_srcdb": srcdb,
                        "coreed_attr": coreed_attr,
                        "campus_code": campus_code,
                        "subject": subject,
                        "course_number": course_number,
                        "code": code,
                        "section": section,
                        "crn": crn,
                        "enrolled": enrolled,
                        "capacity": capacity,
                    }
                )

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame.from_records(all_records)

    # For debugging / sanity-checking:
    print("Sample rows in this snapshot:")
    print(df[["coreed_attr", "campus_code", "code", "section", "enrolled", "capacity"]].head())

    return df


def append_snapshot_to_db(df: pd.DataFrame, srcdb: str, conn: sqlite3.Connection) -> None:
    """Append this snapshot into coreed_capacity with a PST timestamp."""
    if df.empty:
        print("No CoreEd rows to append; skipping.")
        return

    ensure_coreed_table(conn)

    snapshot_ts = now_pst_iso()
    df_to_insert = df.copy()
    df_to_insert["snapshot_timestamp"] = snapshot_ts

    # Only insert the columns we know are in the table:
    cols = [
        "snapshot_timestamp",
        "term_srcdb",
        "coreed_attr",
        "campus_code",
        "code",
        "section",
        "enrolled",
        "capacity",
    ]

    df_to_insert["term_srcdb"] = srcdb

    records = [
        tuple(row[col] for col in cols)
        for _, row in df_to_insert.iterrows()
    ]

    conn.executemany(
        """
        INSERT INTO coreed_capacity
            (snapshot_timestamp, term_srcdb, coreed_attr, campus_code, code, section, enrolled, capacity)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        records,
    )
    conn.commit()

    print(f"Appended {len(records)} rows into table 'coreed_capacity' in {DB_PATH}.")


def main() -> None:
    print(f"[{now_pst_iso()}] Starting CoreEd capacity snapshot...")

    # Priority for selecting term:
    #  1. Command-line argument (python script.py 202601)
    #  2. TERM_SRCDB override (if not None)
    #  3. Infer from current date in OSU local time
    if len(sys.argv) > 1:
        srcdb = sys.argv[1]
        print(f"Using term from command-line argument: {srcdb}")
    elif TERM_SRCDB is not None:
        srcdb = TERM_SRCDB
        print(f"Using TERM_SRCDB override: {srcdb}")
    else:
        srcdb = infer_term_srcdb_from_today()
        print(f"Inferred term from current date: {srcdb}")

    print(f"Building CoreEd snapshot for term {srcdb}...")

    df = build_coreed_snapshot_for_term(srcdb)
    if df.empty:
        print("No data retrieved; exiting.")
        return

    with sqlite3.connect(DB_PATH) as conn:
        append_snapshot_to_db(df, srcdb, conn)

    print("Done.")


if __name__ == "__main__":
    main()