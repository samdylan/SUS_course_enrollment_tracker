"""
osu_enrollment_snapshot_classes_api_coreed.py

Daily snapshot logger for:
  1) SUS sections (keyword search "SUS") -> writes to SQLite table: enrollment
  2) CoreEd sections (attribute search for CFSI/CSSS/CSDP/CFSS) -> writes to: coreed_daily_sections

Design goals:
- Match browser behavior (headers, X-Requested-With) to avoid 202/text/html responses.
- Robust retry/backoff for 202 Accepted / HTML / empty payloads.
- Append-only DB writes with stable schemas used by Streamlit dashboards.

Run:
  source .venv/bin/activate
  python scripts/osu_enrollment_snapshot_classes_api_coreed.py

Optional:
  python scripts/osu_enrollment_snapshot_classes_api_coreed.py 202602
    (force srcdb; e.g., 202600/202601/202602/202603)
  python scripts/osu_enrollment_snapshot_classes_api_coreed.py 202601 --coreed-backfill
    (Writes CoreEd snapshot into table: coreed_capacity_backfill (for QA/compare; dashboards unaffected))
  python scripts/osu_enrollment_snapshot_classes_api_coreed.py --coreed-capacity-refresh
    (Forces a look-ahead term refresh into coreed_capacity; runs even outside window)

Notes:
  - On the 1st and 15th (OSU local date), the script refreshes CoreEd capacity for the look-ahead term
    (current term srcdb + 1) into `coreed_capacity`.
"""

import datetime as dt
import json
import os
import pathlib
import random
import sqlite3
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from zoneinfo import ZoneInfo

# --------------------
# TIME / TZ
# --------------------
OSU_TZ = ZoneInfo("America/Los_Angeles")

def osu_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).astimezone(OSU_TZ)

def osu_today() -> dt.date:
    return osu_now().date()

# --------------------
# CONFIG
# --------------------
CLASSES_API_URL = "https://classes.oregonstate.edu/api/"
SEARCH_QUERY = {"page": "fose", "route": "search"}
DETAILS_QUERY = {"page": "fose", "route": "details"}

# Resolve DB path relative to the repo root (parent of scripts/).
# Allow override via env var for CI or alternative layouts.
DB_PATH = (
    pathlib.Path(os.environ["ENROLLMENT_DB_PATH"]).expanduser()
    if os.environ.get("ENROLLMENT_DB_PATH")
    else pathlib.Path(__file__).resolve().parent.parent / "data" / "osu_enrollment_log_classes.db"
)
DETAIL_SLEEP_SECONDS = 0.15

# SUS tracking
TRACK_SUBJECTS = {"SUS"}          # all SUS courses
TRACK_EXACT_CODES = {"AEC 230X"}  # explicit extra codes (optional)

# CoreEd tracking
COREED_ATTRS = ["CFSI", "CSSS", "CSDP", "CFSS"]

# CoreEd capacity refresh schedule (OSU local date)
COREED_CAPACITY_REFRESH_DAYS = {1, 15}

# --------------------
# TERM LOGIC (same as your SUS script)
# --------------------
TERM_CODE_MAP = {"Summer": "00", "Fall": "01", "Winter": "02", "Spring": "03"}

def estimate_classes_begin(year: int, term: str) -> dt.date:
    term = term.lower()
    if term == "fall":
        next_month = dt.date(year, 10, 1)
        last_day = next_month - dt.timedelta(days=1)
        d = last_day
        while d.weekday() != 2:  # Wed
            d -= dt.timedelta(days=1)
        return d
    if term == "winter":
        d = dt.date(year, 1, 3)
        offset = (0 - d.weekday()) % 7
        return d + dt.timedelta(days=offset)
    if term == "spring":
        d = dt.date(year, 3, 29)
        offset = (0 - d.weekday()) % 7
        return d + dt.timedelta(days=offset)

    if term == "summer":
        # First Monday on or after Jun 22 (stable approximation for OSU summer start)
        d = dt.date(year, 6, 22)
        offset = (0 - d.weekday()) % 7  # 0 = Monday
        return d + dt.timedelta(days=offset)

    raise ValueError(f"Unsupported term for classes_begin: {term}")

def determine_term_for_today(today: Optional[dt.date] = None) -> Dict[str, Any]:
    if today is None:
        today = osu_today()

    candidates: List[Dict[str, Any]] = []
    for year in range(today.year - 1, today.year + 2):
        for term_name, code in TERM_CODE_MAP.items():
            classes_begin = estimate_classes_begin(year, term_name)
            window_start = classes_begin - dt.timedelta(days=50)
            window_end = classes_begin + dt.timedelta(days=6)
            srcdb = f"{classes_begin.year}{code}"
            candidates.append(
                dict(
                    srcdb=srcdb,
                    term_name=term_name,
                    year=classes_begin.year,
                    classes_begin=classes_begin,
                    window_start=window_start,
                    window_end=window_end,
                    term_label=f"{term_name} {classes_begin.year}",
                )
            )

    in_window = [c for c in candidates if c["window_start"] <= today <= c["window_end"]]
    if in_window:
        chosen = sorted(in_window, key=lambda c: c["classes_begin"])[0]
        chosen["within_window"] = True
        return chosen

    future_terms = [c for c in candidates if c["classes_begin"] >= today]
    if future_terms:
        chosen = sorted(future_terms, key=lambda c: c["classes_begin"])[0]
    else:
        chosen = sorted(candidates, key=lambda c: abs((c["classes_begin"] - today).days))[0]
    chosen["within_window"] = False
    return chosen


def next_srcdb(srcdb: str, steps: int = 1) -> str:
    """Advance an OSU term srcdb by `steps` (00->01->02->03->00 with year increment)."""
    s = str(srcdb).strip()
    if len(s) != 6 or not s.isdigit():
        raise ValueError(f"Invalid srcdb: {srcdb!r}")
    year = int(s[:4])
    code = s[4:]
    seq = ["00", "01", "02", "03"]
    if code not in seq:
        raise ValueError(f"Unsupported term code in srcdb: {srcdb!r}")
    idx = seq.index(code)
    for _ in range(int(steps)):
        idx += 1
        if idx >= len(seq):
            idx = 0
            year += 1
    return f"{year}{seq[idx]}"

# Helper: go backwards N terms
def prev_srcdb(srcdb: str, steps: int = 1) -> str:
    """Go back an OSU term srcdb by `steps` (inverse of next_srcdb)."""
    # To go back N, go forward (4 - (N % 4)) % 4 times (modulo 4 terms/yr)
    return next_srcdb(srcdb, steps=(4 - (int(steps) % 4)) % 4)

# --------------------
# HTTP HELPERS (robust, browser-like)
# --------------------
def _browser_headers(referer: str) -> Dict[str, str]:
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

def _post_json_retry(
    session: requests.Session,
    params: Dict[str, Any],
    payload: Dict[str, Any],
    referer: str,
    label: str,
    max_tries: int = 4,
    timeout: int = 30,
) -> Dict[str, Any]:
    headers = _browser_headers(referer)
    last_preview = ""

    for attempt in range(1, max_tries + 1):
        resp = session.post(
            CLASSES_API_URL,
            params=params,
            json=payload,
            headers=headers,
            timeout=timeout,
        )

        ct = (resp.headers.get("Content-Type") or "").lower()
        body = (resp.text or "")
        last_preview = body[:800]

        # Retry triggers
        if (
            resp.status_code in (202, 429)
            or 500 <= resp.status_code <= 599
            or not body.strip()
            or "text/html" in ct
        ):
            wait = min(2 ** attempt, 20) + random.uniform(0, 0.25)
            print(f"[{label}] retry {attempt}/{max_tries} (status={resp.status_code}, ct={ct!r}); wait {wait:.2f}s")
            time.sleep(wait)
            continue

        # HTTP errors
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            raise RuntimeError(
                f"[{label}] HTTP error status={resp.status_code} ct={ct!r} url={resp.url} preview={last_preview}"
            ) from e

        # Non-JSON
        if "json" not in ct:
            raise RuntimeError(
                f"[{label}] Non-JSON response status={resp.status_code} ct={ct!r} preview={last_preview}"
            )

        return resp.json()

    raise RuntimeError(f"[{label}] Exhausted retries; last preview={last_preview}")

# --------------------
# PARSING HELPERS
# --------------------
def safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        s = str(value).strip()
        if not s:
            return None
        return int(s)
    except Exception:
        return None

def split_code(code: str) -> Tuple[Optional[str], Optional[str]]:
    if not code:
        return None, None
    parts = code.split()
    if len(parts) >= 2:
        return parts[0], " ".join(parts[1:])
    return None, code

def find_value_by_keys(obj: Any, keys: Iterable[str]) -> Optional[Any]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in keys:
                return v
            found = find_value_by_keys(v, keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = find_value_by_keys(item, keys)
            if found is not None:
                return found
    return None

def build_details_payload(srcdb: str, code: str, crn: str, matched_crns: List[str]) -> Dict[str, Any]:
    crn_list = [str(c).strip() for c in matched_crns if str(c).strip()]
    if not crn_list:
        crn_list = [crn]
    matched_str = "crn:" + ",".join(crn_list)
    return {"group": f"code:{code}", "key": f"crn:{crn}", "srcdb": srcdb, "matched": matched_str}

def fetch_details(
    session: requests.Session, srcdb: str, code: str, crn: str, matched_crns: List[str]
) -> Dict[str, Any]:
    payload = build_details_payload(srcdb, code, crn, matched_crns)
    referer = f"https://classes.oregonstate.edu/?srcdb={srcdb}"
    return _post_json_retry(session, DETAILS_QUERY, payload, referer, label="details")

# --------------------
# SUS SEARCH / NORMALIZE
# --------------------
def build_search_payload_keyword(srcdb: str, keyword: str) -> Dict[str, Any]:
    return {"other": {"srcdb": srcdb}, "criteria": [{"field": "keyword", "value": keyword}]}

def fetch_search_keyword(session: requests.Session, srcdb: str, keyword: str) -> Dict[str, Any]:
    payload = build_search_payload_keyword(srcdb, keyword)
    referer = f"https://classes.oregonstate.edu/?keyword={keyword}&srcdb={srcdb}"
    return _post_json_retry(session, SEARCH_QUERY, payload, referer, label=f"search:{keyword}")

def filter_sus_section(rec: Dict[str, Any]) -> bool:
    code = rec.get("code", "")
    subject, _ = split_code(code)
    if TRACK_SUBJECTS and subject not in TRACK_SUBJECTS and code not in TRACK_EXACT_CODES:
        return False
    if str(rec.get("isCancelled", "")).strip() == "1":
        return False
    return True

def normalize_sus_results(raw_json: Dict[str, Any], srcdb: str) -> pd.DataFrame:
    results = raw_json.get("results", []) or []
    out: List[Dict[str, Any]] = []
    for rec in results:
        if not filter_sus_section(rec):
            continue
        code = rec.get("code", "")
        subject, course_number = split_code(code)
        out.append(
            dict(
                term_srcdb=srcdb or rec.get("srcdb"),
                key=rec.get("key"),
                subject=subject,
                course_number=course_number,
                code=code,
                crn=rec.get("crn"),
                section=rec.get("no"),
                title=rec.get("title"),
                campus_code=rec.get("camp"),
                schedule_type=rec.get("schd"),
                status_code=rec.get("stat"),
                is_cancelled=str(rec.get("isCancelled", "")).strip() == "1",
                meets=rec.get("meets"),
                instructor=rec.get("instr"),
                start_date=rec.get("start_date"),
                end_date=rec.get("end_date"),
            )
        )
    return pd.DataFrame.from_records(out) if out else pd.DataFrame()

# --------------------
# COREED SEARCH / NORMALIZE
# --------------------
def build_search_payload_coreed_attr(srcdb: str, coreed_attr: str) -> Dict[str, Any]:
    field_name = f"attributes_{coreed_attr}"
    return {"other": {"srcdb": srcdb}, "criteria": [{"field": field_name, "value": "Y"}]}

def fetch_search_coreed_attr(session: requests.Session, srcdb: str, coreed_attr: str) -> Dict[str, Any]:
    payload = build_search_payload_coreed_attr(srcdb, coreed_attr)
    referer = f"https://classes.oregonstate.edu/?srcdb={srcdb}"
    return _post_json_retry(session, SEARCH_QUERY, payload, referer, label=f"coreed:{coreed_attr}")

def normalize_coreed_results(raw_json: Dict[str, Any], srcdb: str, coreed_attr: str) -> pd.DataFrame:
    results = raw_json.get("results", []) or []
    out: List[Dict[str, Any]] = []
    for rec in results:
        # respect cancellation marker if present
        if str(rec.get("isCancelled", "")).strip() == "1":
            continue
        code = (rec.get("code") or "").strip()
        if not code:
            continue
        subject, course_number = split_code(code)
        out.append(
            dict(
                term_srcdb=srcdb,
                coreed_cat4=coreed_attr,   # name used by your dashboard
                coreed_attr=coreed_attr,
                key=rec.get("key"),
                subject=subject,
                course_number=course_number,
                code=code,
                crn=str(rec.get("crn") or "").strip(),
                section=rec.get("no") or rec.get("section"),
                title=rec.get("title"),
                campus_code=rec.get("camp"),
                campus_simple=rec.get("campus_simple"),  # may be absent; dashboard derives anyway
                schedule_type=rec.get("schd"),
                status_code=rec.get("stat"),
                is_cancelled=str(rec.get("isCancelled", "")).strip() == "1",
            )
        )
    df = pd.DataFrame.from_records(out) if out else pd.DataFrame()
    if not df.empty:
        df = df[df["crn"].astype(str).str.strip().ne("")]
    return df

# --------------------
# DETAILS ENRICHMENT (shared)
# --------------------
def enrich_with_details(df: pd.DataFrame, srcdb: str, session: requests.Session) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()

    code_to_crns: Dict[str, List[str]] = (
        df.groupby("code")["crn"]
        .apply(lambda s: [str(x).strip() for x in s if str(x).strip()])
        .to_dict()
    )

    enrolled_list: List[Optional[int]] = []
    capacity_list: List[Optional[int]] = []

    for _, row in df.iterrows():
        code = row.get("code")
        crn = str(row.get("crn") or "").strip()
        if not code or not crn:
            enrolled_list.append(None)
            capacity_list.append(None)
            continue

        matched_crns = code_to_crns.get(code, [crn])

        try:
            details = fetch_details(session, srcdb, code, crn, matched_crns)
        except Exception as e:
            print(f"Warning: details fetch failed for code={code!r}, crn={crn!r}: {e}")
            enrolled_list.append(None)
            capacity_list.append(None)
            time.sleep(DETAIL_SLEEP_SECONDS)
            continue

        enrolled_val = find_value_by_keys(details, {"enrollment", "enrolled", "enrl", "enroll"})
        capacity_val = find_value_by_keys(details, {"max_enroll", "capacity", "cap", "max_enrollment"})

        enrolled_list.append(safe_int(enrolled_val))
        capacity_list.append(safe_int(capacity_val))

        time.sleep(DETAIL_SLEEP_SECONDS)

    df["enrolled"] = enrolled_list
    df["capacity"] = capacity_list
    return df

# --------------------
# DB WRITES
# --------------------
def ensure_coreed_daily_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS coreed_daily_sections (
            snapshot_date    TEXT,
            timestamp        TEXT,
            term_srcdb       TEXT,
            coreed_cat4      TEXT,
            coreed_attr      TEXT,
            subject          TEXT,
            course_number    TEXT,
            code             TEXT,
            crn              TEXT,
            section          TEXT,
            title            TEXT,
            enrolled         INTEGER,
            capacity         INTEGER,
            campus_code      TEXT,
            campus_simple    TEXT,
            schedule_type    TEXT,
            status_code      TEXT,
            is_cancelled     INTEGER,
            key              TEXT
        )
        """
    )
    conn.commit()

def append_sus_to_db(df: pd.DataFrame, db_path: pathlib.Path) -> None:
    if df.empty:
        print("No SUS rows to append.")
        return

    now = osu_now()
    df = df.copy()
    df["snapshot_date"] = now.date().isoformat()
    df["timestamp"] = now.isoformat(timespec="seconds")

    col_order = [
        "snapshot_date","timestamp","term_srcdb","subject","course_number","code","crn","section","title",
        "enrolled","capacity","campus_code","schedule_type","status_code","is_cancelled","meets","instructor",
        "start_date","end_date","key",
    ]
    for c in col_order:
        if c not in df.columns:
            df[c] = None
    df = df[col_order]

    with sqlite3.connect(db_path) as conn:
        df.to_sql("enrollment", conn, if_exists="append", index=False)
        conn.execute("CREATE VIEW IF NOT EXISTS sus_enrollment AS SELECT * FROM enrollment")

    print(f"Appended {len(df)} SUS rows to {db_path} (table enrollment).")

def append_coreed_to_db(df: pd.DataFrame, db_path: pathlib.Path) -> None:
    if df.empty:
        print("No CoreEd rows to append.")
        return

    now = osu_now()
    df = df.copy()

    # These two are new-ish relative to older DB schema
    df["snapshot_date"] = now.date().isoformat()
    df["timestamp"] = now.isoformat(timespec="seconds")

    # normalize booleans
    if "is_cancelled" in df.columns:
        df["is_cancelled"] = df["is_cancelled"].astype(int)
    else:
        df["is_cancelled"] = 0

    # Canonical output columns we want
    col_order = [
        "snapshot_date",
        "term_srcdb",
        # NOTE: older DB may NOT have timestamp/coreed_attr yet; we will add them
        "timestamp",
        "coreed_cat4",
        "coreed_attr",
        "subject",
        "course_number",
        "code",
        "crn",
        "section",
        "title",
        "enrolled",
        "capacity",
        "campus_code",
        "campus_simple",
        "schedule_type",
        "status_code",
        "is_cancelled",
        "key",
    ]

    for c in col_order:
        if c not in df.columns:
            df[c] = None
    df = df[col_order]

    def ensure_table_and_columns(conn: sqlite3.Connection) -> None:
        """Ensure coreed_daily_sections exists and has all columns needed for this script."""

        # Create a minimal base table if missing (safe no-op if it already exists)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS coreed_daily_sections (
                snapshot_date    TEXT,
                term_srcdb       TEXT,
                subject          TEXT,
                course_number    TEXT,
                crn              TEXT,
                section          TEXT,
                coreed_cat4      TEXT,
                campus_simple    TEXT,
                is_lab           INTEGER,
                enrolled         INTEGER,
                capacity         INTEGER
            )
            """
        )

        existing = {
            r[1]
            for r in conn.execute("PRAGMA table_info(coreed_daily_sections);").fetchall()
        }

        needed: Dict[str, str] = {
            "snapshot_date": "TEXT",
            "term_srcdb": "TEXT",
            "timestamp": "TEXT",
            "coreed_cat4": "TEXT",
            "coreed_attr": "TEXT",
            "subject": "TEXT",
            "course_number": "TEXT",
            "code": "TEXT",
            "crn": "TEXT",
            "section": "TEXT",
            "title": "TEXT",
            "enrolled": "INTEGER",
            "capacity": "INTEGER",
            "campus_code": "TEXT",
            "campus_simple": "TEXT",
            "schedule_type": "TEXT",
            "status_code": "TEXT",
            "is_cancelled": "INTEGER",
            "key": "TEXT",
            # legacy/optional column (won't hurt if present)
            "is_lab": "INTEGER",
        }

        for col, ctype in needed.items():
            if col in existing:
                continue
            try:
                conn.execute(f"ALTER TABLE coreed_daily_sections ADD COLUMN {col} {ctype};")
                existing.add(col)
            except sqlite3.OperationalError as e:
                # tolerate races / duplicate-column errors
                if "duplicate column name" in str(e).lower():
                    continue
                raise

        conn.commit()

    with sqlite3.connect(db_path) as conn:
        ensure_table_and_columns(conn)

        # ---- Idempotency guard ----
        # If this script is re-run the same day for the same term(s), we don't want
        # to append duplicate rows and accidentally double capacity in the dashboard.
        # We treat (snapshot_date, term_srcdb) as a "replace the day's snapshot" key.
        unique_pairs = (
            df[["snapshot_date", "term_srcdb"]]
            .dropna()
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        for snap_date, term in unique_pairs:
            conn.execute(
                "DELETE FROM coreed_daily_sections WHERE snapshot_date = ? AND term_srcdb = ?;",
                (str(snap_date), str(term)),
            )
        conn.commit()

        cols_after = [r[1] for r in conn.execute("PRAGMA table_info(coreed_daily_sections);").fetchall()]
        print(f"[coreed] coreed_daily_sections columns now: {cols_after}")

        df.to_sql("coreed_daily_sections", conn, if_exists="append", index=False)

    print(f"Appended {len(df)} CoreEd rows to {db_path} (table coreed_daily_sections).")

# --------------------
# Backfill: CoreEd results to alternate table for QA/compare
# --------------------
def append_coreed_backfill_to_db(df: pd.DataFrame, db_path: pathlib.Path) -> None:
    """Write a CoreEd snapshot into a separate table for QA/compare (does NOT touch coreed_capacity).

    Intended for backfilling a past term (e.g., 202601) while the API still serves that srcdb.
    """
    if df.empty:
        print("No CoreEd rows to backfill.")
        return

    now = osu_now()
    df = df.copy()

    # Standard snapshot stamps
    df["snapshot_date"] = now.date().isoformat()
    df["snapshot_timestamp"] = now.isoformat(timespec="seconds")

    # Canonical subset resembling coreed_capacity (plus snapshot stamps)
    col_order = [
        "snapshot_date",
        "snapshot_timestamp",
        "term_srcdb",
        "coreed_attr",
        "coreed_cat4",
        "subject",
        "course_number",
        "code",
        "crn",
        "section",
        "title",
        "campus_code",
        "campus_simple",
        "enrolled",
        "capacity",
    ]
    for c in col_order:
        if c not in df.columns:
            df[c] = None
    df = df[col_order]

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS coreed_capacity_backfill (
                snapshot_date      TEXT,
                snapshot_timestamp TEXT,
                term_srcdb         TEXT,
                coreed_attr        TEXT,
                coreed_cat4        TEXT,
                subject            TEXT,
                course_number      TEXT,
                code               TEXT,
                crn                TEXT,
                section            TEXT,
                title              TEXT,
                campus_code        TEXT,
                campus_simple      TEXT,
                enrolled           INTEGER,
                capacity           INTEGER
            )
            """
        )

        # Idempotency: replace same-day backfill for the same term
        term_vals = (
            df[["snapshot_date", "term_srcdb"]]
            .dropna()
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        for snap_date, term in term_vals:
            conn.execute(
                "DELETE FROM coreed_capacity_backfill WHERE snapshot_date = ? AND term_srcdb = ?;",
                (str(snap_date), str(term)),
            )
        conn.commit()

        df.to_sql("coreed_capacity_backfill", conn, if_exists="append", index=False)

    print(f"Backfilled {len(df)} CoreEd rows to {db_path} (table coreed_capacity_backfill).")


# --------------------
# CoreEd capacity refresh: write to coreed_capacity table (scheduled/forced)
# --------------------
def refresh_coreed_capacity_to_db(df: pd.DataFrame, db_path: pathlib.Path) -> None:
    """Refresh CoreEd capacity snapshot into `coreed_capacity`.

    Intended use:
      - twice-monthly (1st/15th) refresh for the look-ahead term (current+1)
      - may be run even when outside the daily snapshot window

    Behavior:
      - writes a point-in-time snapshot with snapshot_date + snapshot_timestamp
      - idempotent for (snapshot_date, term_srcdb): re-runs replace that day's snapshot
    """
    if df.empty:
        print("No CoreEd rows to write to coreed_capacity.")
        return

    now = osu_now()
    df = df.copy()

    df["snapshot_date"] = now.date().isoformat()
    df["snapshot_timestamp"] = now.isoformat(timespec="seconds")

    # Canonical columns (superset; safe to add missing cols to DB)
    col_order = [
        "snapshot_date",
        "snapshot_timestamp",
        "term_srcdb",
        "coreed_attr",
        "coreed_cat4",
        "subject",
        "course_number",
        "code",
        "crn",
        "section",
        "title",
        "campus_code",
        "campus_simple",
        "enrolled",
        "capacity",
    ]

    for c in col_order:
        if c not in df.columns:
            df[c] = None
    df = df[col_order]

    def ensure_table_and_columns(conn: sqlite3.Connection) -> None:
        # Create minimal base table if missing
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS coreed_capacity (
                snapshot_timestamp TEXT,
                term_srcdb         TEXT,
                coreed_attr        TEXT,
                code               TEXT,
                section            TEXT,
                enrolled           INTEGER,
                capacity           INTEGER,
                campus_code        TEXT
            )
            """
        )

        existing = {
            r[1]
            for r in conn.execute("PRAGMA table_info(coreed_capacity);").fetchall()
        }

        needed: Dict[str, str] = {
            "snapshot_date": "TEXT",
            "snapshot_timestamp": "TEXT",
            "term_srcdb": "TEXT",
            "coreed_attr": "TEXT",
            "coreed_cat4": "TEXT",
            "subject": "TEXT",
            "course_number": "TEXT",
            "code": "TEXT",
            "crn": "TEXT",
            "section": "TEXT",
            "title": "TEXT",
            "campus_code": "TEXT",
            "campus_simple": "TEXT",
            "enrolled": "INTEGER",
            "capacity": "INTEGER",
        }

        for col, ctype in needed.items():
            if col in existing:
                continue
            try:
                conn.execute(f"ALTER TABLE coreed_capacity ADD COLUMN {col} {ctype};")
                existing.add(col)
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e).lower():
                    continue
                raise

        conn.commit()

    with sqlite3.connect(db_path) as conn:
        ensure_table_and_columns(conn)

        # Idempotency: replace same-day snapshot for each term in the df
        term_vals = (
            df[["snapshot_date", "term_srcdb"]]
            .dropna()
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        for snap_date, term in term_vals:
            conn.execute(
                "DELETE FROM coreed_capacity WHERE snapshot_date = ? AND term_srcdb = ?;",
                (str(snap_date), str(term)),
            )
        conn.commit()

        df.to_sql("coreed_capacity", conn, if_exists="append", index=False)

    print(
        f"Refreshed {len(df)} CoreEd rows to {db_path} (table coreed_capacity) "
        f"for term(s): {sorted(df['term_srcdb'].dropna().unique().tolist())}"
    )
# --------------------
# MAIN
# --------------------
def main() -> None:
    today = osu_today()
    term_info = determine_term_for_today(today)

    # First day of a new enrollment window (used to refresh the *previous* term CoreEd capacity once)
    is_first_day_of_window = today == term_info["window_start"]

    # Allow manual override srcdb via CLI, plus optional backfill mode and capacity refresh
    argv = [a.strip() for a in sys.argv[1:] if str(a).strip()]
    backfill_coreed = "--coreed-backfill" in argv
    force_capacity_refresh = "--coreed-capacity-refresh" in argv

    forced_srcdb = None
    if argv:
        first = argv[0]
        if len(first) == 6 and first.isdigit():
            forced_srcdb = first

    if forced_srcdb:
        srcdb = forced_srcdb
        within_window = True  # if you force it, we run it
        term_label = f"(forced srcdb={srcdb})"
        print(f"[{today.isoformat()}] Forced srcdb from CLI: {srcdb}")
        if backfill_coreed:
            print("[mode] --coreed-backfill enabled: will also write coreed_capacity_backfill (QA/compare).")
    else:
        srcdb = term_info["srcdb"]
        term_label = term_info["term_label"]
        within_window = term_info["within_window"]
        print(
            f"[{today.isoformat()}] Tracking term: {term_label} (srcdb={srcdb}); "
            f"window {term_info['window_start']} → {term_info['window_end']} "
            f"(classes begin {term_info['classes_begin']})"
        )

    # Daily snapshots are limited to the enrollment window, but capacity refresh may still run.
    should_run_daily = within_window
    should_run_capacity_refresh = force_capacity_refresh or (today.day in COREED_CAPACITY_REFRESH_DAYS)

    # On the *first day* of the new window, do a one-time refresh of the PRIOR term into coreed_capacity.
    # This supports the CoreEd dashboard's historical look-back without needing frequent refreshes.
    should_run_prev_term_coreed_refresh = bool(within_window and is_first_day_of_window and not forced_srcdb)

    if not should_run_daily and not should_run_capacity_refresh and not should_run_prev_term_coreed_refresh:
        print("Today is outside snapshot window and not a scheduled capacity refresh day; skipping fetch and DB write.")
        return

    print(f"[api] DB_PATH = {DB_PATH.resolve()}")

    with requests.Session() as session:
        # -----------------
        # One-time prior-term refresh (CoreEd dashboard support)
        # Runs only on the first day of the new window.
        # -----------------
        if should_run_prev_term_coreed_refresh:
            prev_term_srcdb = prev_srcdb(srcdb, 1)
            print(
                "\n=== CoreEd prior-term refresh (one-time; first day of window) ===\n"
                f"New-window term srcdb={srcdb} → prior-term srcdb={prev_term_srcdb}"
            )

            prev_frames: List[pd.DataFrame] = []
            for attr in COREED_ATTRS:
                print(f"CoreEd {attr}: search…")
                raw = fetch_search_coreed_attr(session, prev_term_srcdb, attr)
                df_attr = normalize_coreed_results(raw, srcdb=prev_term_srcdb, coreed_attr=attr)
                print(f"  {attr}: {len(df_attr)} sections (pre-details)")
                if not df_attr.empty:
                    df_attr = enrich_with_details(df_attr, srcdb=prev_term_srcdb, session=session)
                    prev_frames.append(df_attr)

            df_prev = pd.concat(prev_frames, ignore_index=True) if prev_frames else pd.DataFrame()
            if not df_prev.empty:
                refresh_coreed_capacity_to_db(df_prev, DB_PATH)
            else:
                print("CoreEd prior-term refresh: no rows fetched; nothing to write.")
        else:
            print("\n[prior-term] Not first day of window (or forced srcdb); skipping prior-term CoreEd refresh.")

        # -----------------
        # Daily snapshots (current term) — only within window
        # -----------------
        if should_run_daily:
            # ---- SUS ----
            print("\n=== SUS snapshot ===")
            raw_sus = fetch_search_keyword(session, srcdb, keyword="SUS")
            df_sus = normalize_sus_results(raw_sus, srcdb=srcdb)
            print(f"SUS: {len(df_sus)} sections after filtering (pre-details)")
            if not df_sus.empty:
                df_sus = enrich_with_details(df_sus, srcdb=srcdb, session=session)
                append_sus_to_db(df_sus, DB_PATH)

            # ---- CoreEd daily ----
            print("\n=== CoreEd snapshot (daily; current term) ===")
            coreed_frames: List[pd.DataFrame] = []
            for attr in COREED_ATTRS:
                print(f"CoreEd {attr}: search…")
                raw = fetch_search_coreed_attr(session, srcdb, attr)
                df_attr = normalize_coreed_results(raw, srcdb=srcdb, coreed_attr=attr)
                print(f"  {attr}: {len(df_attr)} sections (pre-details)")
                if not df_attr.empty:
                    df_attr = enrich_with_details(df_attr, srcdb=srcdb, session=session)
                    coreed_frames.append(df_attr)

            df_coreed = pd.concat(coreed_frames, ignore_index=True) if coreed_frames else pd.DataFrame()
            if not df_coreed.empty:
                append_coreed_to_db(df_coreed, DB_PATH)
                if backfill_coreed:
                    append_coreed_backfill_to_db(df_coreed, DB_PATH)
            else:
                print("CoreEd (daily): no rows fetched; nothing to append.")
        else:
            print("\n[daily] Outside window: skipping daily SUS/CoreEd snapshots.")

        # -----------------
        # Capacity refresh (look-ahead term) — runs on 1st/15th (OSU date) or forced
        # -----------------
        if should_run_capacity_refresh:
            base_srcdb = srcdb
            lookahead_srcdb = next_srcdb(base_srcdb, 1)
            print(
                "\n=== CoreEd capacity refresh (scheduled/forced; look-ahead term) ===\n"
                f"Base term srcdb={base_srcdb} → look-ahead srcdb={lookahead_srcdb}"
            )

            cap_frames: List[pd.DataFrame] = []
            for attr in COREED_ATTRS:
                print(f"CoreEd {attr}: search…")
                raw = fetch_search_coreed_attr(session, lookahead_srcdb, attr)
                df_attr = normalize_coreed_results(raw, srcdb=lookahead_srcdb, coreed_attr=attr)
                print(f"  {attr}: {len(df_attr)} sections (pre-details)")
                if not df_attr.empty:
                    df_attr = enrich_with_details(df_attr, srcdb=lookahead_srcdb, session=session)
                    cap_frames.append(df_attr)

            df_cap = pd.concat(cap_frames, ignore_index=True) if cap_frames else pd.DataFrame()
            if not df_cap.empty:
                refresh_coreed_capacity_to_db(df_cap, DB_PATH)
            else:
                print("CoreEd capacity refresh: no rows fetched; nothing to write.")
        else:
            print("\n[capacity] Not a scheduled capacity refresh day; skipping look-ahead refresh.")

    print("\nDone.")

if __name__ == "__main__":
    main()