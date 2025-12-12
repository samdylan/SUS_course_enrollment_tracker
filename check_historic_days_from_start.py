import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path("osu_enrollment_log_classes.db")

def estimate_classes_begin_from_srcdb(term_srcdb: str) -> pd.Timestamp | None:
    """
    Same logic as in the dashboard – OSU-style term mapping:
      XX00 = Summer of (label_year - 1)
      XX01 = Fall   of (label_year - 1)
      XX02 = Winter of (label_year)
      XX03 = Spring of (label_year)
    """
    if not term_srcdb or len(str(term_srcdb)) < 6:
        return None

    s = str(term_srcdb)
    try:
        label_year = int(s[:4])
        term_code = s[-2:]
    except Exception:
        return None

    if term_code == "00":  # Summer
        year = label_year - 1
        d = pd.Timestamp(year=year, month=6, day=24)
        offset = (0 - d.weekday()) % 7  # Monday
        return (d + pd.Timedelta(days=offset)).normalize()

    if term_code == "01":  # Fall
        year = label_year - 1
        d = pd.Timestamp(year=year, month=9, day=30)
        while d.weekday() != 2:  # Wednesday
            d -= pd.Timedelta(days=1)
        return d.normalize()

    if term_code == "02":  # Winter
        year = label_year
        d = pd.Timestamp(year=year, month=1, day=3)
        offset = (0 - d.weekday()) % 7
        return (d + pd.Timedelta(days=offset)).normalize()

    if term_code == "03":  # Spring
        year = label_year
        d = pd.Timestamp(year=year, month=3, day=29)
        offset = (0 - d.weekday()) % 7
        return (d + pd.Timedelta(days=offset)).normalize()

    return None


def main():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT snapshot_date, term_srcdb, subject, course_number, section, enrolled "
        "FROM sus_daily_registrations_2025 "
        "ORDER BY snapshot_date "
        "LIMIT 50",
        conn,
    )
    conn.close()

    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
    df["classes_begin"] = df["term_srcdb"].apply(estimate_classes_begin_from_srcdb)
    df["days_from_start"] = (df["snapshot_date"] - df["classes_begin"]).dt.days

    print("\n=== SAMPLE HISTORIC ROWS WITH DAYS_FROM_START ===")
    print(df[["term_srcdb", "snapshot_date", "classes_begin", "days_from_start",
              "subject", "course_number", "section", "enrolled"]])

    print("\n=== DAYS_FROM_START SUMMARY (HISTORIC) ===")
    print(df["days_from_start"].describe())


if __name__ == "__main__":
    main()