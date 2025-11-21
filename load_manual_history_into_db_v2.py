"""
load_manual_history_into_db_v2.py

One-off helper to load manual historical enrollment snapshots from
Enrollment_tracker_SUScourses.xlsx into osu_enrollment_log_classes.db.

It does the following:
1. Connects to osu_enrollment_log_classes.db and reads the existing
   `enrollment` table.
2. Removes previous "manual" rows (identified as rows where CRN IS NULL).
3. Parses the Excel file and builds a manual_df with:
     - term_srcdb="202602" (Winter 2026)
     - timestamp set to the snapshot date at 12:00:00
     - code like "SUS 102", "SUS 103", "SUS 331", "SUS 350", etc
       (note: SUS102H maps to SUS 102; SUS103H -> SUS 103, etc.)
4. Appends manual_df to the remaining rows and overwrites the
   `enrollment` table.
"""

import sqlite3
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd


# --- PATHS ---

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = BASE_DIR / "osu_enrollment_log_classes.db"
EXCEL_PATH = BASE_DIR / "Enrollment_tracker_SUScourses.xlsx"

TERM_SRCDB = "202602"  # Winter 2026


# --- HELPER: parse header like "SUS102-001", "SUS331_400", "USS450-400", "SUS102H-002" ---

def parse_section_header(header: str) -> Optional[Tuple[str, str, str]]:
    """
    Parse a header label into (subject, course_number, section_str).

    Examples:
      "SUS102-001"   -> ("SUS", "102", "001")
      "SUS102H-001"  -> ("SUS", "102", "001")   # NOTE: drop trailing H
      "SUS331_400"   -> ("SUS", "331", "400")
      "USS450-400"   -> ("SUS", "450", "400")   # fix USS -> SUS
    """
    if not isinstance(header, str):
        return None

    h = header.strip()
    if not h:
        return None

    # Fix known typos / variants
    if h.startswith("USS"):
        h = "SUS" + h[3:]

    # Normalize underscore to dash between course and section
    h = h.replace("_", "-")

    # Expect something like SUS102-001
    parts = h.split("-")
    if len(parts) != 2:
        return None

    course_token, section = parts[0], parts[1]
    if not course_token.startswith("SUS"):
        return None

    subject = "SUS"
    raw_number = course_token[len(subject):]  # e.g. "102", "102H", "331", "230X"

    # Treat trailing "H" as hybrid modality and drop it, so SUS102H -> SUS 102
    if raw_number.endswith("H"):
        course_number = raw_number[:-1]
    else:
        course_number = raw_number

    section_str = section

    if not course_number or not section_str:
        return None

    return subject, course_number, section_str


def infer_campus_from_section(section_str: str) -> Tuple[str, str]:
    """
    Very simple rule:
      - section >= "400" => Ecampus (DI)
      - otherwise       => Corvallis (C)
    """
    try:
        sec_int = int(section_str)
    except ValueError:
        sec_int = 0

    if sec_int >= 400:
        return "DI", "Ecampus"
    else:
        return "C", "Corvallis"


def build_manual_dataframe() -> pd.DataFrame:
    """Read the Excel file and build a manual_df compatible with `enrollment` table."""
    print(f"Reading Excel file: {EXCEL_PATH}")
    raw = pd.read_excel(EXCEL_PATH, header=None)

    # Header row is the one with the SUS labels.
    # From the sample, that's row index 1.
    header_row_idx = 1

    # Data rows are below that (row 2, 3, ...)
    data_start_idx = header_row_idx + 1

    records: List[dict] = []

    # Loop over columns that have a SUS* header
    for col_idx in range(raw.shape[1]):
        header = raw.iloc[header_row_idx, col_idx]
        parsed = parse_section_header(header)
        if parsed is None:
            continue

        subject, course_number, section_str = parsed
        code = f"{subject} {course_number}"
        campus_code, campus_name = infer_campus_from_section(section_str)

        # For each data row for this column, pick up enrollment if present
        for row_idx in range(data_start_idx, raw.shape[0]):
            snapshot_date = raw.iloc[row_idx, 1]  # second column holds the date
            value = raw.iloc[row_idx, col_idx]

            # Skip rows without a date or without a numeric value
            if pd.isna(snapshot_date):
                continue
            if pd.isna(value) or value == "-":
                continue

            try:
                enrolled = int(value)
            except Exception:
                continue

            # timestamp as ISO string
            ts = pd.to_datetime(snapshot_date).strftime("%Y-%m-%dT12:00:00")

            record = {
                "timestamp": ts,
                "term_srcdb": TERM_SRCDB,
                "subject": subject,
                "course_number": course_number,
                "code": code,
                "crn": None,  # unknown in manual file
                "section": section_str,
                "title": None,
                "enrolled": enrolled,
                "campus_code": campus_code,
                "campus_name": campus_name,
                "status_code": None,
                "is_cancelled": 0,
                "meets": None,
                "instructor": None,
                "start_date": None,
                "end_date": None,
                "meetingTimes": None,
                "capacity": None,
            }
            records.append(record)

    manual_df = pd.DataFrame.from_records(records)
    print(f"Built manual_df with {len(manual_df)} rows.")
    return manual_df


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at: {DB_PATH}")

    print(f"Loading existing enrollment table from: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    existing = pd.read_sql_query("SELECT * FROM enrollment", conn)
    print(f"Existing rows: {len(existing)}")

    # Identify previous manual rows (where CRN is null)
    # API-sourced rows always have a CRN string.
    manual_mask = existing["crn"].isna()
    n_manual_old = manual_mask.sum()
    if n_manual_old > 0:
        print(f"Removing {n_manual_old} old manual rows (crn IS NULL).")
        existing = existing[~manual_mask].reset_index(drop=True)
    else:
        print("No existing manual rows found to remove.")

    manual_df = build_manual_dataframe()

    # Align columns: ensure manual_df has all columns from existing
    for col in existing.columns:
        if col not in manual_df.columns:
            manual_df[col] = None

    # Ensure ordering of columns
    manual_df = manual_df[existing.columns]

    combined = pd.concat([existing, manual_df], ignore_index=True)
    print(f"New total rows in enrollment: {len(combined)}")

    # Write back to the same table
    combined.to_sql("enrollment", conn, if_exists="replace", index=False)
    conn.close()

    print("Done. You can now re-run the Streamlit dashboard to see the new historical snapshots.")


if __name__ == "__main__":
    main()
