"""
osu_enrollment_snapshot_classes_api_v2.py

Daily CoreEd sections snapshot using the same classes.oregonstate.edu
search/details pattern from osu_coreed_capacity_snapshot_v3.py.

Builds df_coreed for the current term with:
  - code (subject + course number)
  - subject, course_number
  - crn, section
  - campus_simple (Corvallis / Ecampus / Cascades)
  - coreed_cat4 (CFSI, CSSS, CSDP, CFSS)
  - enrolled (details enrollment)
  - capacity (details max_enroll)
  - term_srcdb

Writes to SQLite table coreed_daily_sections with columns:
  snapshot_date, term_srcdb, subject, course_number, crn, section,
  coreed_cat4, campus_simple, is_lab, enrolled, capacity
"""

from __future__ import annotations

import datetime as dt
import pathlib
import sqlite3
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
import requests
from zoneinfo import ZoneInfo
import sys

# Ensure we can import the CoreEd snapshot helper whether it's in the repo root
# or inside the "CoreEd Daily Tracker" subfolder.
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
POSSIBLE_ROOTS = [
    THIS_DIR,
    THIS_DIR.parent,  # repo root if this script is inside a subfolder
    THIS_DIR / "CoreEd Daily Tracker",
]
for path in POSSIBLE_ROOTS:
    helper = path / "osu_coreed_capacity_snapshot_v3.py"
    if helper.exists():
        sys.path.append(str(path))
        break

from osu_coreed_capacity_snapshot_v3 import build_coreed_snapshot_for_term

# ---------- CONFIG ----------

CLASSES_API_URL = "https://classes.oregonstate.edu/api/"

SEARCH_QUERY = {"page": "fose", "route": "search"}
DETAILS_QUERY = {"page": "fose", "route": "details"}

COREED_ATTRS = ["CFSI", "CSSS", "CSDP", "CFSS"]

DB_PATH = pathlib.Path("osu_enrollment_log_classes.db")

OSU_TZ = ZoneInfo("America/Los_Angeles")


# ---------- HELPERS ----------

def osu_now() -> dt.datetime:
    return dt.datetime.now(tz=OSU_TZ)


def osu_today() -> dt.date:
    return osu_now().date()


def infer_term_srcdb_from_today(now: dt.datetime | None = None) -> str:
    """
    Infer term using the sliding window logic from the enrollment snapshot script:
      - Fall:   last Wednesday in September
      - Winter: first Monday on/after Jan 3
      - Spring: first Monday on/after Mar 29
    We pick the term whose window (classes_begin - 50 days to classes_begin + 6 days)
    contains today; otherwise the nearest future term. This points to Winter (02)
    during the December registration window.
    """
    if now is None:
        now = osu_now()
    if now.tzinfo is None:
        now = now.replace(tzinfo=OSU_TZ)
    else:
        now = now.astimezone(OSU_TZ)
    today = now.date()

    def estimate_classes_begin(year: int, term: str) -> dt.date:
        term = term.lower()
        if term == "fall":
            # Last Wednesday in September
            next_month = dt.date(year, 10, 1)
            d = next_month - dt.timedelta(days=1)
            while d.weekday() != 2:  # Wednesday
                d -= dt.timedelta(days=1)
            return d
        if term == "winter":
            d = dt.date(year, 1, 3)
            offset = (0 - d.weekday()) % 7  # Monday
            return d + dt.timedelta(days=offset)
        if term == "spring":
            d = dt.date(year, 3, 29)
            offset = (0 - d.weekday()) % 7  # Monday
            return d + dt.timedelta(days=offset)
        raise ValueError("Unsupported term")

    term_codes = {"Fall": "01", "Winter": "02", "Spring": "03"}
    candidates = []
    for year in range(today.year - 1, today.year + 2):
        for term_name, code in term_codes.items():
            classes_begin = estimate_classes_begin(year, term_name)
            window_start = classes_begin - dt.timedelta(days=50)
            window_end = classes_begin + dt.timedelta(days=6)
            srcdb = f"{classes_begin.year}{code}"
            candidates.append(
                {
                    "srcdb": srcdb,
                    "term_name": term_name,
                    "classes_begin": classes_begin,
                    "window_start": window_start,
                    "window_end": window_end,
                }
            )

    in_window = [
        c for c in candidates if c["window_start"] <= today <= c["window_end"]
    ]
    if in_window:
        chosen = sorted(in_window, key=lambda c: c["classes_begin"])[0]
        return chosen["srcdb"]

    future_terms = [c for c in candidates if c["classes_begin"] >= today]
    if future_terms:
        chosen = sorted(future_terms, key=lambda c: c["classes_begin"])[0]
    else:
        chosen = sorted(candidates, key=lambda c: abs((c["classes_begin"] - today).days))[0]
    return chosen["srcdb"]


def safe_int(value: Any) -> int | None:
    try:
        s = str(value).strip()
        if not s:
            return None
        return int(s)
    except Exception:
        return None


def split_code(code: str) -> Tuple[str | None, str | None]:
    if not code:
        return None, None
    parts = code.split()
    if len(parts) >= 2:
        return parts[0], " ".join(parts[1:])
    return None, code


def build_search_payload(srcdb: str, coreed_attr: str) -> Dict[str, Any]:
    field_name = f"attributes_{coreed_attr}"
    return {"other": {"srcdb": srcdb}, "criteria": [{"field": field_name, "value": "Y"}]}


def fetch_coreed_search(srcdb: str, coreed_attr: str) -> List[Dict[str, Any]]:
    payload = build_search_payload(srcdb, coreed_attr)
    resp = requests.post(CLASSES_API_URL, params=SEARCH_QUERY, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results", [])
    if not isinstance(results, list):
        results = []
    return results


def build_details_payload(srcdb: str, code: str, crn: str, matched_crns: Iterable[str]) -> Dict[str, Any]:
    matched_list = ",".join(str(c).strip() for c in matched_crns if str(c).strip())
    return {
        "group": f"code:{code}",
        "key": f"crn:{crn}",
        "srcdb": srcdb,
        "matched": f"crn:{matched_list}",
    }


def extract_enrollment_capacity(details_json: Any) -> Tuple[int | None, int | None]:
    if details_json is None:
        return None, None
    if isinstance(details_json, list):
        if not details_json:
            return None, None
        return extract_enrollment_capacity(details_json[0])
    if not isinstance(details_json, dict):
        return None, None
    if "fatal" in details_json:
        return None, None
    if "enrollment" in details_json or "max_enroll" in details_json:
        return safe_int(details_json.get("enrollment")), safe_int(details_json.get("max_enroll"))
    for key in ("results", "data"):
        if key in details_json and isinstance(details_json[key], list) and details_json[key]:
            return extract_enrollment_capacity(details_json[key][0])
    return None, None


def fetch_details_for_section(srcdb: str, code: str, crn: str, matched_crns: List[str]) -> Tuple[int | None, int | None]:
    payload = build_details_payload(srcdb, code, crn, matched_crns)
    resp = requests.post(CLASSES_API_URL, params=DETAILS_QUERY, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return extract_enrollment_capacity(data)


def derive_campus_simple(campus_code: str | None, section: Any) -> str:
    """Prefer section number for campus; fall back to campus_code buckets."""
    try:
        sec_num = int(str(section).strip())
        if 400 <= sec_num <= 499:
            return "Ecampus"
        if sec_num >= 500:
            return "Cascades"
        return "Corvallis"
    except Exception:
        code = (campus_code or "").upper()
        if code in {"E", "DI", "DIST", "DISTANCE"}:
            return "Ecampus"
        if code in {"B", "DB", "L", "CASC"}:
            return "Cascades"
        return "Corvallis"


def is_lab_like_section(section: Any, campus_simple: str) -> bool:
    try:
        sec_num = int(str(section).strip())
    except Exception:
        return False
    if campus_simple == "Corvallis" and 10 <= sec_num < 400:
        return True
    return False


# ---------- COREED BUILD ----------

def build_coreed_df_for_term(srcdb: str) -> pd.DataFrame:
    """Reuse the capacity snapshot builder to ensure parity."""
    df = build_coreed_snapshot_for_term(srcdb)
    if df.empty:
        return df

    # Derive lab flag and snapshot date
    df["campus_simple"] = df.apply(
        lambda r: derive_campus_simple(r.get("campus_code"), r.get("section")), axis=1
    )
    df["is_lab"] = df.apply(
        lambda r: is_lab_like_section(r.get("section"), r.get("campus_simple")), axis=1
    )
    df["coreed_cat4"] = df["coreed_attr"]
    df["snapshot_date"] = osu_today().isoformat()
    return df


# ---------- DB WRITE ----------

def ensure_coreed_daily_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS coreed_daily_sections (
            snapshot_date TEXT NOT NULL,
            term_srcdb    TEXT NOT NULL,
            subject       TEXT,
            course_number TEXT,
            crn           TEXT,
            section       TEXT,
            coreed_cat4   TEXT,
            campus_simple TEXT,
            is_lab        INTEGER,
            enrolled      INTEGER,
            capacity      INTEGER
        )
        """
    )
    conn.commit()


def append_coreed_daily(df: pd.DataFrame, conn: sqlite3.Connection) -> None:
    if df.empty:
        print("No CoreEd rows to insert; skipping.")
        return

    ensure_coreed_daily_table(conn)
    df_clean = df.copy()
    # Coerce numerics to Python ints so SQLite stores INTEGER, not BLOB.
    for col in ("enrolled", "capacity"):
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").astype("Int64")
    df_clean["is_lab"] = df_clean["is_lab"].astype(int)

    cols = [
        "snapshot_date",
        "term_srcdb",
        "subject",
        "course_number",
        "crn",
        "section",
        "coreed_cat4",
        "campus_simple",
        "is_lab",
        "enrolled",
        "capacity",
    ]

    def to_python(val):
        if pd.isna(val):
            return None
        try:
            return int(val)
        except Exception:
            return val

    records = [
        tuple(to_python(df_clean.loc[i, c]) if c in ("enrolled", "capacity", "is_lab") else df_clean.loc[i, c]
              for c in cols)
        for i in df_clean.index
    ]
    conn.executemany(
        """
        INSERT INTO coreed_daily_sections
            (snapshot_date, term_srcdb, subject, course_number, crn, section,
             coreed_cat4, campus_simple, is_lab, enrolled, capacity)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        records,
    )
    # Ensure SUS enrollment view exists (for consistent access)
    conn.execute("CREATE VIEW IF NOT EXISTS sus_enrollment AS SELECT * FROM enrollment")
    conn.commit()
    print(f"Inserted {len(records)} rows into coreed_daily_sections.")


# ---------- MAIN ----------

def main() -> None:
    term = infer_term_srcdb_from_today()
    print(f"Building CoreEd daily snapshot for term {term}...")
    df_coreed = build_coreed_df_for_term(term)
    if df_coreed.empty:
        print("No CoreEd data retrieved; exiting.")
        return

    with sqlite3.connect(DB_PATH) as conn:
        append_coreed_daily(df_coreed, conn)

    print("Done.")


if __name__ == "__main__":
    main()
