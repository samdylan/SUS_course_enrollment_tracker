"""
import_sus_daily_registrations_2025.py

One-off importer for historic SUS daily registrations.

Expects a UTF-16LE, tab-delimited file with columns:

    Seats Used    Activity Date    Term    Section Number    Course Number

Writes an aggregated daily table to SQLite table:
    sus_daily_registrations_2025

Columns in the DB table:
    snapshot_date  (datetime64[ns])
    term_srcdb     (TEXT)
    subject        (TEXT, always 'SUS')
    course_number  (TEXT)
    section        (TEXT, zero-padded 3 digits)
    enrolled       (INT, daily max of Seats Used)
"""

from pathlib import Path
import sqlite3
import pandas as pd

# --- Paths -----------------------------------------------------------------

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent         # repo root (father)
DB_PATH = BASE_DIR / "osu_enrollment_log_classes.db"

SRC_FILE = BASE_DIR / "SUS_DailyRegistrations_2025.txt"

TABLE_NAME = "sus_daily_registrations_2025"


# --- Load raw file ---------------------------------------------------------

def load_raw_file(path: Path) -> pd.DataFrame:
    """
    Load the UTF-16LE, tab-delimited historic SUS registrations file.
    """
    df = pd.read_csv(
        path,
        sep="\t",
        encoding="utf-16-le",
    )
    # Strip any stray whitespace from column names
    df.columns = [c.strip() for c in df.columns]
    return df


# --- Clean & aggregate -----------------------------------------------------

def clean_and_aggregate(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw historic rows into a daily-timeseries table compatible with
    load_sus_historic_daily() in the dashboard.

    - Enforces required columns.
    - Parses Activity Date.
    - Normalizes types.
    - Aggregates to one row per (term, course, section, snapshot_date)
      using the max Seats Used as the daily "enrolled" value.
    """
    required = ["Seats Used", "Activity Date", "Term", "Section Number", "Course Number"]
    missing = [c for c in required if c not in df_raw.columns]
    if missing:
        raise ValueError(
            "Historic file is missing required columns.\n"
            f"Missing: {missing}\n"
            f"Available columns: {list(df_raw.columns)}\n\n"
            "Expected header row:\n"
            "Seats Used\tActivity Date\tTerm\tSection Number\tCourse Number"
        )

    df = df_raw.copy()

    # Parse datetime
    df["Activity Date"] = pd.to_datetime(df["Activity Date"], errors="coerce")
    df = df.dropna(subset=["Activity Date"])

    # Rename to internal names
    df = df.rename(
        columns={
            "Seats Used": "enrolled",
            "Activity Date": "snapshot_date",
            "Term": "term_srcdb",
            "Section Number": "section",
            "Course Number": "course_number",
        }
    )

    # Normalize types
    df["term_srcdb"] = df["term_srcdb"].astype(str)
    df["section"] = df["section"].astype(str).str.zfill(3)
    df["course_number"] = df["course_number"].astype(str)
    df["enrolled"] = pd.to_numeric(df["enrolled"], errors="coerce").fillna(0).astype(int)

    # Aggregate: for each day & section, take the max seats used
    df = (
        df.groupby(
            ["term_srcdb", "course_number", "section", "snapshot_date"],
            as_index=False,
        )["enrolled"]
        .max()
    )

    # Add subject so the dashboard can default to it
    df["subject"] = "SUS"

    # Reorder columns to something tidy
    df = df[
        [
            "snapshot_date",
            "term_srcdb",
            "subject",
            "course_number",
            "section",
            "enrolled",
        ]
    ]

    return df


# --- Write to SQLite -------------------------------------------------------

def write_to_db(df: pd.DataFrame, db_path: Path, table_name: str) -> None:
    """
    Replace the target table with the cleaned historic data.
    """
    if df.empty:
        raise ValueError("Cleaned historic dataframe is empty; refusing to write to DB.")

    with sqlite3.connect(db_path) as conn:
        # Replace existing table with the new data
        df.to_sql(table_name, conn, if_exists="replace", index=False)


# --- Main ------------------------------------------------------------------

def main() -> None:
    print(f"[importer] DB_PATH = {DB_PATH.resolve()}")
    print(f"Loading historic SUS registrations from: {SRC_FILE}")
    if not SRC_FILE.exists():
        raise FileNotFoundError(f"Source file not found: {SRC_FILE}")

    df_raw = load_raw_file(SRC_FILE)
    print(f"Raw rows loaded: {len(df_raw)}")
    print(f"Raw columns: {list(df_raw.columns)}")

    df_clean = clean_and_aggregate(df_raw)
    print(f"Cleaned & aggregated rows: {len(df_clean)}")

    print(f"Writing to DB: {DB_PATH} (table '{TABLE_NAME}')")
    write_to_db(df_clean, DB_PATH, TABLE_NAME)

    # Small sanity check / preview
    print("\n=== SAMPLE CLEANED ROWS ===")
    print(df_clean.head(20).to_string(index=False))

    print("\nDone.")

    print("Importer writing to DB:", DB_PATH.resolve())

if __name__ == "__main__":
    main()