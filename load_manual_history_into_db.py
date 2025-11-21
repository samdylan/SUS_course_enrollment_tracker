"""
load_manual_history_into_db.py

Read manually collected enrollment history from an Excel file
and insert it into the existing osu_enrollment_log_classes.db
`enrollment` table.

Behavior:
- parses snapshot dates and section headers like "SUS102H-001", "SUS331_400", "SUS230X-001"
- normalizes hybrid shorthand (SUS102H -> SUS 102) while keeping real "X" suffix (SUS230X -> SUS 230X)
- treats 400+ sections as Ecampus / Distance (campus_code="DI"), others as Corvallis ("C")
- SKIPS columns that do not contain a section delimiter ("-" or "_")
- deletes any existing rows in `enrollment` for the same dates & term
  before inserting the new manual rows.
"""

import pathlib
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional

import pandas as pd

# --- CONFIG ---

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "osu_enrollment_log_classes.db"

# Your Excel file with manual history
EXCEL_PATH = PROJECT_ROOT / "Enrollment_tracker_SUScourses.xlsx"

# Manual sheet is for Winter 2026
TERM_SRCDB = "202602"


def parse_header(header: str) -> Optional[Dict[str, str]]:
    """
    Parse Excel header like:
      - 'SUS102H-001'  (manual shorthand for SUS 102, Corvallis hybrid)
      - 'SUS102-400'   (Ecampus)
      - 'SUS331_400'
      - 'SUS230X-001'  (SUS 230X)

    into normalized pieces:
      subject='SUS',
      course_number like '102' or '230X',
      code like 'SUS 102' or 'SUS 230X',
      section='001',
      campus_code='C'/'DI'.

    If the header does NOT contain a section delimiter ('-' or '_')
    (e.g. 'SUS230X' total column), return None and let the caller skip it.
    """
    if not isinstance(header, str):
        return None

    h = header.strip()
    if not h:
        return None

    # Require a delimiter that separates course-ish from section
    if "-" in h:
        left, section = h.split("-", 1)
    elif "_" in h:
        left, section = h.split("_", 1)
    else:
        # No explicit section -> skip (likely a total column)
        return None

    left = left.strip()
    section = section.strip()

    if not left or not section:
        return None

    # Expect something like "SUS102H" or "SUS230X"
    if len(left) < 4:
        return None

    subject = left[:3].upper()
    suffix = left[3:]  # e.g. "102H", "230X", "350"

    # Drop a trailing 'H' used in the manual sheet for "hybrid" notation,
    # so SUS102H -> SUS 102, but SUS230X keeps the X.
    if suffix.endswith("H"):
        base_suffix = suffix[:-1]
    else:
        base_suffix = suffix

    if not base_suffix:
        return None

    course_number = base_suffix
    code = f"{subject} {course_number}"

    # campus: treat 4xx sections as Ecampus / Distance (DI), others as Corvallis (C)
    campus_code = "DI" if section.startswith("4") else "C"

    return {
        "subject": subject,
        "course_number": course_number,
        "code": code,
        "section": section,
        "campus_code": campus_code,
    }


def build_manual_records() -> pd.DataFrame:
    """
    Read the Excel and return a long-form DataFrame with columns:
      snapshot_date (date), term_srcdb, subject, course_number,
      code, section, campus_code, enrolled, timestamp
    """
    raw = pd.read_excel(EXCEL_PATH)

    if raw.shape[0] < 2:
        raise RuntimeError("Excel file does not appear to have any data rows.")

    # row 0 holds the column labels like 'SUS102H-001', 'SUS331_400', etc.
    headers_row = raw.iloc[0]

    records: List[Dict] = []

    # Data rows start at index 1 (each row is a snapshot date)
    for i in range(1, raw.shape[0]):
        snap_date = raw.iloc[i, 3]  # 4th column ('Unnamed: 3') is the date
        if pd.isna(snap_date):
            continue
        snap_date = pd.to_datetime(snap_date).date()

        # From 5th column onward are section headers + counts
        for col_idx in range(4, raw.shape[1]):
            header = headers_row[col_idx]

            course_info = parse_header(str(header))
            if course_info is None:
                # Skip non-section or malformed headers (e.g. 'SUS230X' total)
                continue

            val = raw.iloc[i, col_idx]

            # Skip blanks and '-' placeholders
            if pd.isna(val):
                continue
            if isinstance(val, str) and val.strip() == "-":
                continue

            try:
                enrolled = int(val)
            except Exception:
                # Non-numeric garbage; skip
                continue

            records.append(
                {
                    "snapshot_date": snap_date,
                    "term_srcdb": TERM_SRCDB,
                    **course_info,
                    "enrolled": enrolled,
                }
            )

    df = pd.DataFrame.from_records(records)

    # Add a fixed time-of-day so it looks like a timestamp string
    df["timestamp"] = df["snapshot_date"].apply(
        lambda d: datetime(d.year, d.month, d.day, 12, 0, 0).isoformat(
            timespec="seconds"
        )
    )
    return df


def upsert_manual_history():
    df = build_manual_records()

    if df.empty:
        print("No manual records found in Excel; nothing to load.")
        return

    unique_dates = sorted({d.isoformat() for d in df["snapshot_date"].unique()})
    print(f"Manual snapshot dates found in Excel: {', '.join(unique_dates)}")
    print(f"Total manual records parsed: {len(df)}")

    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()

        # 1) Delete any existing enrollment rows for these dates & this term
        placeholders = ",".join("?" for _ in unique_dates)
        delete_sql = f"""
            DELETE FROM enrollment
            WHERE term_srcdb = ?
              AND date(timestamp) IN ({placeholders})
        """
        cur.execute(delete_sql, [TERM_SRCDB, *unique_dates])
        deleted = cur.rowcount
        print(f"Deleted {deleted} existing rows for these dates in term {TERM_SRCDB}.")

        # 2) Insert new rows (partial column insert; other fields will be NULL)
        insert_sql = """
            INSERT INTO enrollment (
                timestamp,
                term_srcdb,
                subject,
                course_number,
                code,
                section,
                campus_code,
                enrolled
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """

        records = [
            (
                row["timestamp"],
                row["term_srcdb"],
                row["subject"],
                row["course_number"],
                row["code"],
                row["section"],
                row["campus_code"],
                row["enrolled"],
            )
            for _, row in df.iterrows()
        ]

        cur.executemany(insert_sql, records)
        conn.commit()
        print(f"Inserted {len(records)} manual rows into enrollment.")

        total = cur.execute("SELECT COUNT(*) FROM enrollment").fetchone()[0]
        print(f"New total rows in enrollment: {total}")

    finally:
        conn.close()


def main():
    print(f"Using DB: {DB_PATH}")
    print(f"Using Excel: {EXCEL_PATH}")
    upsert_manual_history()


if __name__ == "__main__":
    main()
