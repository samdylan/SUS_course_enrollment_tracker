"""
Shared data loading and census-measure logic for the SIA supply/demand analysis.

CENSUS TARGET: day 20 after classes_begin (early 4th-week census, after
add/drop stabilizes).

DATA LIMITATION the user should know: the daily snapshot script window
ends at classes_begin + 6 for most terms, so for Winter 2026 onward the
best available census day is ~day 6 (end of our tracking window). For
Fall 2025 (one-off longer tracking) and the SUS legacy table (2024-25),
later snapshots exist. For Fall 2026 (current term today, 2026-10-07),
day 7 is the latest.

PER-SECTION PICK RULE: for each (term, section), choose the snapshot
whose day_from_start is CLOSEST to the target day (20). Record both the
chosen day and a quality label so cross-term comparability can be judged:
  - "on-day20"      : exactly day 20
  - "near-day20"    : within +/- 3 days of day 20
  - "day20-range"   : within +/- 7 days (lower confidence)
  - "early-term"    : day < 13 (first two weeks only; recent terms stop here)
  - "late-term"     : day > 27 (end of term; older terms)
  - "early-current" : future day 20 (term in session but not yet at day 20)
"""
from __future__ import annotations

import datetime as dt
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = BASE_DIR / "data" / "osu_enrollment_log_classes.db"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TODAY = dt.date(2026, 10, 7)  # analysis freeze date; see summary.md

# ---------- Term calendar ----------
# OSU srcdb convention: Summer and Fall of calendar year Y belong to AY Y+1.
# Winter and Spring of year Y belong to AY Y.
# TERM_CODE: 00=Summer, 01=Fall, 02=Winter, 03=Spring

TERM_META = {
    # srcdb: (season, calendar_year_of_term, classes_begin_estimate)
    "202500": ("Summer", 2024, dt.date(2024, 6, 24)),
    "202501": ("Fall",   2024, dt.date(2024, 9, 25)),   # last Wed of Sep
    "202502": ("Winter", 2025, dt.date(2025, 1, 6)),
    "202503": ("Spring", 2025, dt.date(2025, 3, 31)),
    "202600": ("Summer", 2025, dt.date(2025, 6, 23)),
    "202601": ("Fall",   2025, dt.date(2025, 9, 24)),
    "202602": ("Winter", 2026, dt.date(2026, 1, 5)),
    "202603": ("Spring", 2026, dt.date(2026, 3, 30)),
    "202700": ("Summer", 2026, dt.date(2026, 6, 22)),
    "202701": ("Fall",   2026, dt.date(2026, 9, 30)),
    "202702": ("Winter", 2027, dt.date(2027, 1, 4)),
}

def term_label(srcdb: str) -> str:
    season, year, _ = TERM_META[str(srcdb)]
    return f"{season[:2].upper()}{str(year)[-2:]}"  # e.g. FA25, WI26

def term_sort_key(srcdb: str) -> tuple:
    """Chronological: sort by actual classes_begin date."""
    _, _, cb = TERM_META[str(srcdb)]
    return (cb.year, cb.month, cb.day)

def classes_begin(srcdb: str) -> dt.date:
    return TERM_META[str(srcdb)][2]

def is_post_coreed(srcdb: str) -> bool:
    """True for terms Summer 2025 (202600) and later (Core Ed era)."""
    return int(str(srcdb)) >= 202600

def season_of(srcdb: str) -> str:
    return TERM_META[str(srcdb)][0]


# ---------- Section-level census builder ----------

@dataclass
class CensusRow:
    term_srcdb: str
    subject: str
    course_number: str
    section: str
    campus_simple: str
    enrolled: Optional[float]
    capacity: Optional[float]
    snapshot_date: Optional[dt.date]
    day_from_start: Optional[int]
    quality: str  # "on-day20", "near-day20", "early", "late", "latest-available"


def _derive_campus(campus_code: Optional[str], section: Optional[str]) -> str:
    """Map campus_code and section number to Corvallis/Ecampus/Cascades/Other."""
    try:
        sec_num = int(str(section).strip())
        if 400 <= sec_num <= 499:
            return "Ecampus"
        if sec_num >= 500:
            return "Cascades"
        return "Corvallis"
    except (ValueError, TypeError, AttributeError):
        pass
    c = (campus_code or "").upper()
    if c in {"E", "DI", "DB"}:
        return "Ecampus"
    if c in {"B"}:
        return "Cascades"
    return "Corvallis"


def _pick_closest_snapshot(df: pd.DataFrame, target_day: int = 20) -> tuple[pd.Series, str]:
    """From a per-section DataFrame of snapshots, pick the row closest to target_day.
    Returns (row, quality_label) describing how good a day-20 census it is."""
    df = df.copy()
    df["diff"] = (df["day_from_start"] - target_day).abs()
    best = df.loc[df["diff"].idxmin()]
    d = int(best["diff"])
    day = int(best["day_from_start"])

    if d == 0:
        return best, "on-day20"
    if d <= 3:
        return best, "near-day20"
    if d <= 7:
        return best, "day20-range"
    # Beyond +/- 7 days from target: label by direction
    is_current = best["snapshot_date"] >= (TODAY - dt.timedelta(days=3))
    if day < target_day:
        if is_current:
            return best, "early-current"
        if day < 13:
            return best, "early-term"
        return best, "day20-range"
    # day > target_day + 7
    return best, "late-term"


def load_sus_census(target_day: int = 20) -> pd.DataFrame:
    """Return one CensusRow per (term, section) for ALL SUS courses, combining
    the legacy table (pre-tracker) and the live enrollment table.

    Capacity is only available for sections sourced from the live enrollment table.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        # Legacy table: no capacity
        legacy = pd.read_sql_query(
            """
            SELECT snapshot_date, term_srcdb, subject, course_number, section, enrolled
            FROM sus_daily_registrations_2025
            """,
            conn,
        )
        # Live enrollment table: SUS courses only
        live = pd.read_sql_query(
            """
            SELECT snapshot_date, term_srcdb, subject, course_number, section,
                   enrolled, capacity, campus_code
            FROM enrollment
            WHERE subject = 'SUS'
            """,
            conn,
        )
    finally:
        conn.close()

    legacy["snapshot_date"] = pd.to_datetime(legacy["snapshot_date"], errors="coerce").dt.date
    live["snapshot_date"] = pd.to_datetime(live["snapshot_date"], errors="coerce").dt.date
    legacy["enrolled"] = pd.to_numeric(legacy["enrolled"], errors="coerce")
    live["enrolled"] = pd.to_numeric(live["enrolled"], errors="coerce")
    live["capacity"] = pd.to_numeric(live["capacity"], errors="coerce")
    legacy["source"] = "legacy"
    live["source"] = "live"
    legacy["capacity"] = pd.NA
    legacy["campus_code"] = pd.NA

    # Prefer live table when both have the same (term, section) and date
    df = pd.concat([legacy, live], ignore_index=True)
    df = df.dropna(subset=["snapshot_date", "term_srcdb"])
    df["term_srcdb"] = df["term_srcdb"].astype(str)
    df["section"] = df["section"].astype(str)
    df["campus_simple"] = df.apply(lambda r: _derive_campus(r.get("campus_code"), r["section"]), axis=1)

    # Compute day_from_start
    df = df[df["term_srcdb"].isin(TERM_META.keys())]
    df["classes_begin"] = df["term_srcdb"].map(classes_begin)
    df["day_from_start"] = df.apply(
        lambda r: (r["snapshot_date"] - r["classes_begin"]).days, axis=1
    )

    # Dedupe (term, course, section, date): prefer live rows (they have capacity)
    df = df.sort_values(["term_srcdb", "course_number", "section", "snapshot_date", "source"])
    df = df.drop_duplicates(
        subset=["term_srcdb", "course_number", "section", "snapshot_date"], keep="last"
    )

    # For each (term, course, section), pick the snapshot closest to target_day
    records = []
    for (term, course, section), group in df.groupby(["term_srcdb", "course_number", "section"]):
        best, quality = _pick_closest_snapshot(group, target_day)
        records.append({
            "term_srcdb": term,
            "subject": "SUS",
            "course_number": str(course),
            "section": section,
            "campus_simple": best["campus_simple"],
            "enrolled": best["enrolled"],
            "capacity": best["capacity"] if pd.notna(best.get("capacity")) else pd.NA,
            "snapshot_date": best["snapshot_date"],
            "day_from_start": int(best["day_from_start"]),
            "quality": quality,
            "source": best["source"],
        })

    out = pd.DataFrame(records)
    # Attach season/year for pre/post comparison and term-of-year matching
    out["season"] = out["term_srcdb"].map(season_of)
    out["calendar_year"] = out["term_srcdb"].map(lambda t: TERM_META[t][1])
    out["term_label"] = out["term_srcdb"].map(term_label)
    out["coreed_era"] = out["term_srcdb"].map(lambda t: "post" if is_post_coreed(t) else "pre")
    return out


def load_sia_census(target_day: int = 20) -> pd.DataFrame:
    """Return one CensusRow per (term, crn) for ALL SIA (CFSI) sections across
    all subject codes. Uses coreed_daily_sections (daily) + coreed_capacity
    (periodic) with daily preferred when both are present.

    Returns empty frame for pre-202600 terms (no SIA coverage before then).
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        daily = pd.read_sql_query(
            """
            SELECT snapshot_date, term_srcdb, subject, course_number, crn, section,
                   coreed_cat4, campus_simple, enrolled, capacity, code, title,
                   is_lab, campus_code
            FROM coreed_daily_sections
            WHERE coreed_cat4 = 'CFSI'
            """,
            conn,
        )
        cap = pd.read_sql_query(
            """
            SELECT snapshot_date, term_srcdb, subject, course_number, crn, section,
                   coreed_cat4, campus_simple, enrolled, capacity, code, title,
                   campus_code
            FROM coreed_capacity
            WHERE coreed_cat4 = 'CFSI' OR coreed_attr = 'CFSI'
            """,
            conn,
        )
    finally:
        conn.close()

    for df in (daily, cap):
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce").dt.date
        df["enrolled"] = pd.to_numeric(df["enrolled"], errors="coerce")
        df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce")
        df["term_srcdb"] = df["term_srcdb"].astype(str)
        df["crn"] = df["crn"].astype(str)

    daily["src"] = "daily"
    cap["src"] = "capacity"
    cap["is_lab"] = pd.NA

    df = pd.concat([daily, cap], ignore_index=True)
    df = df.dropna(subset=["snapshot_date", "term_srcdb", "crn"])
    df = df[df["term_srcdb"].isin(TERM_META.keys())]

    # Clean campus
    df["campus_simple"] = df.apply(
        lambda r: _derive_campus(r.get("campus_code"), r.get("section")) if (
            pd.isna(r.get("campus_simple")) or not r.get("campus_simple")
        ) else r["campus_simple"],
        axis=1,
    )
    df["classes_begin"] = df["term_srcdb"].map(classes_begin)
    df["day_from_start"] = df.apply(
        lambda r: (r["snapshot_date"] - r["classes_begin"]).days, axis=1
    )

    # Prefer daily rows over capacity rows when both exist for same section/date
    df = df.sort_values(["term_srcdb", "crn", "snapshot_date", "src"])
    df = df.drop_duplicates(subset=["term_srcdb", "crn", "snapshot_date"], keep="first")

    # For each (term, crn), pick snapshot closest to target_day
    records = []
    for (term, crn), group in df.groupby(["term_srcdb", "crn"]):
        best, quality = _pick_closest_snapshot(group, target_day)
        records.append({
            "term_srcdb": term,
            "subject": best["subject"],
            "course_number": str(best["course_number"]),
            "course_code": f"{best['subject']} {best['course_number']}",
            "crn": crn,
            "section": str(best["section"]),
            "campus_simple": best["campus_simple"] or "Unknown",
            "enrolled": best["enrolled"],
            "capacity": best["capacity"],
            "snapshot_date": best["snapshot_date"],
            "day_from_start": int(best["day_from_start"]),
            "quality": quality,
            "source": best["src"],
        })

    out = pd.DataFrame(records)
    out["season"] = out["term_srcdb"].map(season_of)
    out["calendar_year"] = out["term_srcdb"].map(lambda t: TERM_META[t][1])
    out["term_label"] = out["term_srcdb"].map(term_label)
    return out


if __name__ == "__main__":
    # Smoke test
    sus = load_sus_census()
    sia = load_sia_census()
    print(f"SUS census: {len(sus)} section-terms")
    print(sus.groupby(["term_srcdb", "course_number"]).size().unstack(fill_value=0))
    print(f"\nSIA census: {len(sia)} section-terms")
    print(sia.groupby(["term_srcdb", "subject"]).size().unstack(fill_value=0))
