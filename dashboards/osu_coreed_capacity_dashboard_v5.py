import sqlite3
import re

import altair as alt
import pandas as pd
import streamlit as st

# ---------- CONFIG ----------
from pathlib import Path
import os
from datetime import datetime

# Allow override (useful for Streamlit Cloud / servers / different repo layouts)
# Example: export COREED_DB_PATH="/path/to/osu_enrollment_log_classes.db"
DB_PATH = Path(os.environ.get("COREED_DB_PATH", "")).expanduser() if os.environ.get("COREED_DB_PATH") else (
    Path(__file__).resolve().parent.parent / "data" / "osu_enrollment_log_classes.db"
)

# (DB path and modification time are shown in the sidebar diagnostics section in main())
# Optional backfill table: cleaned/curated capacity snapshots (used to repair stale/buggy terms)
TABLE_NAME = "coreed_capacity"
COREED_BACKFILL_TABLE = "coreed_capacity_backfill"
# If present, prefer backfill rows for these historical terms (comma-separated via env var, e.g. "202601")
BACKFILL_TERMS_ENV = os.environ.get("COREED_BACKFILL_TERMS", "202601")
BACKFILL_TERMS = [t.strip() for t in BACKFILL_TERMS_ENV.split(",") if t.strip()]
# If coreed_capacity is stale/missing, we can derive an equivalent feed from coreed_daily_sections
FALLBACK_TO_DAILY_IF_CAPACITY_MISSING = True
STALE_DAYS_THRESHOLD = 7  # if coreed_capacity is older than this, prefer daily-derived
COREED_DAILY_TABLE = "coreed_daily_sections"  # daily enrolled/capacity (interim, e.g., 202602)

COREED_LABELS = {
    "CFSI": "Scientific Inquiry & Analysis",
    "CSSS": "Seeking Solutions",
    "CSDP": "DPO Advanced",
    "CFSS": "Social Science",
}

CAMPUS_ORDER = ["Corvallis", "Ecampus", "Cascades"]
CAMPUS_COLORS = ["#1f77b4", "#d62728", "#2ca02c"]  # consistent mapping for Corvallis/Ecampus/Cascades

CAS_SUBJECT_CODES = {
    "AED",
    "AEC",
    "AG",
    "AGRI",
    "ANS",
    "BDS",
    "BOT",
    "BRR",
    "CROP",
    "CSS",
    "ENT",
    "FST",
    "FW",
    "HORT",
    "LEAD",
    "PBG",
    "RNG",
    "SOIL",
    "SUS",
    "TOX",
}


def load_data() -> tuple[pd.DataFrame, list[str]]:
    """
    Load the CoreEd capacity backbone for the dashboard.

    Rules (IMPORTANT):
    - coreed_capacity is the authoritative multi-term backbone for historical + future/lookahead terms.
    - coreed_daily_sections is the authoritative source for the CURRENT term (capacity + enrolled), because it reflects the daily API pulls.
    - Daily data may also supplement terms missing from capacity (rare), but it should not overwrite historical/future terms.

    Returns (df, load_messages) where load_messages are diagnostic strings for the sidebar.
    """
    load_messages: list[str] = []

    if not DB_PATH.exists():
        st.error(f"Database file not found at {DB_PATH}")
        st.stop()

    with sqlite3.connect(DB_PATH) as conn:
        tables = set(
            pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type IN ('table','view');",
                conn,
            )["name"]
        )

        # --- load capacity backbone ---
        if TABLE_NAME not in tables:
            st.error("Expected table/view 'coreed_capacity' not found in database.")
            st.stop()

        df_capacity = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)

        # --- optional backfill table (cleaned capacity snapshots) ---
        df_backfill = (
            pd.read_sql_query(f"SELECT * FROM {COREED_BACKFILL_TABLE}", conn)
            if COREED_BACKFILL_TABLE in tables
            else pd.DataFrame()
        )

        # --- load daily table if present (optional) ---
        df_daily = (
            pd.read_sql_query(f"SELECT * FROM {COREED_DAILY_TABLE}", conn)
            if COREED_DAILY_TABLE in tables
            else pd.DataFrame()
        )

    # --- normalize capacity ---
    df_capacity["snapshot_timestamp"] = pd.to_datetime(
        df_capacity["snapshot_timestamp"], errors="coerce"
    )
    df_capacity["term_srcdb"] = df_capacity["term_srcdb"].astype(str)
    df_capacity["capacity"] = pd.to_numeric(df_capacity["capacity"], errors="coerce")
    df_capacity["enrolled"] = pd.to_numeric(df_capacity["enrolled"], errors="coerce")
    df_capacity["coreed_attr"] = df_capacity["coreed_attr"].astype(str)
    df_capacity["code"] = df_capacity["code"].astype(str)
    df_capacity["section"] = df_capacity["section"].astype(str)

    # --- normalize backfill (if present) ---
    if 'df_backfill' not in locals():
        df_backfill = pd.DataFrame()
    if not df_backfill.empty:
        # Allow either snapshot_timestamp or snapshot_date to exist
        if "snapshot_timestamp" in df_backfill.columns:
            df_backfill["snapshot_timestamp"] = pd.to_datetime(df_backfill["snapshot_timestamp"], errors="coerce")
        else:
            df_backfill["snapshot_timestamp"] = pd.to_datetime(df_backfill.get("snapshot_date"), errors="coerce")

        df_backfill["term_srcdb"] = df_backfill["term_srcdb"].astype(str)
        df_backfill["capacity"] = pd.to_numeric(df_backfill.get("capacity"), errors="coerce")
        df_backfill["enrolled"] = pd.to_numeric(df_backfill.get("enrolled"), errors="coerce")

        # coreed_attr may be stored as coreed_attr or coreed_cat4 in some older backfills
        if "coreed_attr" in df_backfill.columns:
            df_backfill["coreed_attr"] = df_backfill["coreed_attr"].astype(str)
        else:
            df_backfill["coreed_attr"] = df_backfill.get("coreed_cat4").astype(str)

        # code/section are required keys in this dashboard
        if "code" not in df_backfill.columns or df_backfill["code"].isna().all():
            # try to build from subject/course_number if present
            if "subject" in df_backfill.columns and "course_number" in df_backfill.columns:
                df_backfill["code"] = (
                    df_backfill["subject"].astype(str).str.strip()
                    + " "
                    + df_backfill["course_number"].astype(str).str.strip()
                )
            else:
                df_backfill["code"] = ""
        else:
            df_backfill["code"] = df_backfill["code"].astype(str)

        df_backfill["section"] = df_backfill["section"].astype(str)

    # --- optionally repair specific historical terms using backfill table ---
    # This is useful when coreed_capacity contains partial/NULL rows for a term (e.g., 202601).
    if not df_backfill.empty and BACKFILL_TERMS:
        bf = df_backfill.copy()

        # Keep only rows for requested terms
        bf = bf[bf["term_srcdb"].isin(BACKFILL_TERMS)].copy()

        if not bf.empty:
            # Collapse to latest snapshot per section identity (same keying as capacity)
            bf = (
                bf.dropna(subset=["snapshot_timestamp", "term_srcdb", "coreed_attr", "code", "section"])
                  .sort_values("snapshot_timestamp")
                  .groupby(["term_srcdb", "coreed_attr", "code", "section"], as_index=False)
                  .tail(1)
            )

            # Replace those terms in df_capacity
            before_terms = sorted(set(df_capacity["term_srcdb"]) & set(BACKFILL_TERMS))
            if before_terms:
                df_capacity = df_capacity[~df_capacity["term_srcdb"].isin(BACKFILL_TERMS)].copy()
                df_capacity = pd.concat([df_capacity, bf[df_capacity.columns.intersection(bf.columns)]], ignore_index=True)

                load_messages.append(
                    f"Capacity repaired via {COREED_BACKFILL_TABLE} for terms: {', '.join(before_terms)}"
                )

    # --- use daily as authoritative for the CURRENT term (and optionally supplement missing terms) ---
    if not df_daily.empty:
        d = df_daily.copy()

        d["snapshot_timestamp"] = pd.to_datetime(
            d.get("timestamp", d.get("snapshot_date")), errors="coerce"
        )
        d["term_srcdb"] = d["term_srcdb"].astype(str)

        # Prefer explicit coreed_attr if present; otherwise fall back to coreed_cat4
        d["coreed_attr"] = d.get("coreed_attr", d.get("coreed_cat4")).astype(str)

        d["capacity"] = pd.to_numeric(d.get("capacity"), errors="coerce")
        d["enrolled"] = pd.to_numeric(d.get("enrolled"), errors="coerce")
        d["section"] = d["section"].astype(str)

        if "code" not in d.columns or d["code"].isna().all():
            d["code"] = (
                d["subject"].astype(str).str.strip()
                + " "
                + d["course_number"].astype(str).str.strip()
            )
        else:
            d["code"] = d["code"].astype(str)

        # collapse to latest snapshot per section
        d = (
            d.dropna(subset=["snapshot_timestamp", "term_srcdb", "coreed_attr", "code", "section"])
             .sort_values("snapshot_timestamp")
             .groupby(["term_srcdb", "coreed_attr", "code", "section"], as_index=False)
             .tail(1)
        )

        daily_equiv = d[[
            "snapshot_timestamp",
            "term_srcdb",
            "coreed_attr",
            "code",
            "section",
            "capacity",
            "enrolled",
        ]].copy()

        cap_terms = set(df_capacity["term_srcdb"].unique())
        daily_terms = set(daily_equiv["term_srcdb"].unique())

        # Heuristic: daily table typically only contains the current term.
        # Use the max term present in daily as "current".
        current_term = max(daily_terms) if daily_terms else None

        # 1) Replace capacity rows for the current term with daily-derived rows (authoritative)
        if current_term and current_term in cap_terms:
            df = df_capacity[df_capacity["term_srcdb"] != current_term].copy()
            df = pd.concat(
                [df, daily_equiv[daily_equiv["term_srcdb"] == current_term]],
                ignore_index=True,
            )
            load_messages.append(
                f"Source: coreed_capacity + coreed_daily_sections (current term {current_term})"
            )
        else:
            df = df_capacity.copy()

        # 2) Also supplement any terms that exist in daily but not in capacity (rare)
        missing_terms = daily_terms - cap_terms
        if missing_terms:
            df = pd.concat(
                [df, daily_equiv[daily_equiv["term_srcdb"].isin(missing_terms)]],
                ignore_index=True,
            )
            load_messages.append(
                f"Source: coreed_capacity + daily (current term {current_term}) + supplement for {sorted(missing_terms)}"
            )

    else:
        df = df_capacity.copy()
        load_messages.append("Source: coreed_capacity")

    # --- labels & campus grouping ---
    df["coreed_label"] = df["coreed_attr"].map(COREED_LABELS).fillna(df["coreed_attr"])

    def derive_campus_group(row):
        """Derive a simplified campus group.

        Priority rule (authoritative): section number.
          - 400–499  => Ecampus
          - 500–699  => Cascades
          - >=700    => Other (La Grande / special programs; exclude from dashboard)
          - 000–399  => Corvallis

        Fallback (only if section is non-numeric): campus_code mapping.
        """
        sec = str(row.get("section", "")).strip()
        code = str(row.get("campus_code", "")).strip().upper()

        try:
            sec_num = int(sec)
            if 400 <= sec_num <= 499:
                return "Ecampus"
            elif 500 <= sec_num <= 699:
                return "Cascades"
            elif sec_num >= 700:
                return "Other"
            else:
                return "Corvallis"
        except ValueError:
            # Fallback to code only when section isn't numeric
            if code in {"E", "DI"}:
                return "Ecampus"
            elif code in {"B", "DB"}:
                return "Cascades"
            elif code in {"L"}:
                return "Other"
            return "Corvallis"

    df["campus_group"] = df.apply(derive_campus_group, axis=1)
    # Exclude non-target campuses/programs (e.g., La Grande / 700+ sections)
    df = df[df["campus_group"].isin(CAMPUS_ORDER)].copy()

    # Drop terms that have no usable capacity data (e.g., 202600 placeholder rows)
    cap_by_term = (
        df.groupby("term_srcdb", as_index=False)["capacity"]
        .sum(min_count=1)
        .rename(columns={"capacity": "total_cap"})
    )
    empty_terms = cap_by_term.loc[
        cap_by_term["total_cap"].isna() | (cap_by_term["total_cap"] <= 0),
        "term_srcdb",
    ].tolist()
    if empty_terms:
        df = df[~df["term_srcdb"].isin(empty_terms)].copy()

    return df, load_messages

def load_coreed_daily_latest() -> pd.DataFrame:
    """
    Load coreed_daily_sections and collapse to the latest snapshot_date per
    (term, category, campus, code, section). Used for enrollment overlay.

    IMPORTANT:
    - Do NOT require CRN. Capacity backbone often doesn't have CRN consistently,
      and CRN is not needed to align by course+section.
    """
    if not DB_PATH.exists():
        return pd.DataFrame()

    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(f"SELECT * FROM {COREED_DAILY_TABLE}", conn)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()

    if df.empty:
        return df

    # Normalize types we need
    df["snapshot_date"] = pd.to_datetime(df.get("snapshot_date"), errors="coerce")
    df["term_srcdb"] = df.get("term_srcdb").astype(str)

    # Align naming with coreed_capacity (prefer explicit coreed_attr if present)
    if "coreed_attr" in df.columns:
        df["coreed_attr"] = df.get("coreed_attr").astype(str)
    else:
        df["coreed_attr"] = df.get("coreed_cat4").astype(str)

    # Campus naming: section number is authoritative
    df["campus_group"] = "Corvallis"
    sec_num = pd.to_numeric(df.get("section"), errors="coerce")
    df.loc[(sec_num >= 400) & (sec_num <= 499), "campus_group"] = "Ecampus"
    df.loc[(sec_num >= 500) & (sec_num <= 699), "campus_group"] = "Cascades"
    df.loc[sec_num >= 700, "campus_group"] = "Other"

    # If section is non-numeric, fall back to campus_simple mapping
    nonnum = sec_num.isna()
    if "campus_simple" in df.columns:
        fallback = df.loc[nonnum, "campus_simple"].astype(str).replace(
            {
                "Corvallis Campus": "Corvallis",
                "Ecampus Campus": "Ecampus",
                "Cascades Campus": "Cascades",
                "La Grande Campus": "Other",
            }
        )
        df.loc[nonnum, "campus_group"] = fallback

    # Exclude non-target campuses/programs (e.g., La Grande / 700+ sections)
    df = df[df["campus_group"].isin(CAMPUS_ORDER)]

    # Ensure numeric
    df["enrolled"] = pd.to_numeric(df.get("enrolled"), errors="coerce")
    df["capacity"] = pd.to_numeric(df.get("capacity"), errors="coerce")

    # Section identifier
    df["section"] = df.get("section").astype(str)

    # Build code to match coreed_capacity ("SUBJ NNN")
    if "code" in df.columns and df["code"].notna().any():
        df["code"] = df["code"].astype(str)
    else:
        df["subject"] = df.get("subject").astype(str)
        df["course_number"] = df.get("course_number").astype(str)
        df["code"] = df["subject"].str.strip() + " " + df["course_number"].str.strip()

    # Keep only rows with a valid snapshot_date and keys needed for overlay matching
    df = df.dropna(subset=["snapshot_date", "term_srcdb", "coreed_attr", "campus_group", "code", "section"])

    # Collapse to latest snapshot_date per section identity (NO CRN)
    df = df.sort_values("snapshot_date")
    df = (
        df.groupby(
            ["term_srcdb", "coreed_attr", "campus_group", "code", "section"],
            as_index=False,
        )
        .tail(1)
        .reset_index(drop=True)
    )

    # Friendly labels
    df["coreed_label"] = df["coreed_attr"].map(COREED_LABELS).fillna(df["coreed_attr"])

    return df

def extract_subject_code(course_code: str | None) -> str | None:
    """Return the alphabetic subject prefix from a course code like 'AEC 121'."""
    if not course_code:
        return None
    match = re.match(r"([A-Za-z]+)", str(course_code).strip())
    return match.group(1).upper() if match else None


def is_lab_like_section(section, campus_group: str) -> bool:
    """Heuristic to flag lab/recitation-style sections so we can optionally exclude them.

    Rules (subject to refinement):
      - For Corvallis, treat numeric section numbers 010–399 as lab/recitation-like.
        Regular Corvallis lecture sections are typically 000–009.
      - Ecampus (400–499) and Cascades (>=500) are not treated as labs by this rule.
    """
    try:
        sec_num = int(str(section).strip())
    except (TypeError, ValueError):
        return False

    if campus_group == "Corvallis" and 10 <= sec_num < 400:
        return True

    return False


def dedupe_latest(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the latest *non-null* snapshot per (term, category, campus, course, section).

    IMPORTANT: coreed_capacity often has crn NULL. If we groupby crn, pandas drops NA groups
    and you can end up with an empty df_latest. So we dedupe WITHOUT crn here.
    """
    df_sorted = df.sort_values("snapshot_timestamp")

    keep_cols = [
        "snapshot_timestamp",
        "term_srcdb",
        "coreed_attr",
        "coreed_label",
        "campus_group",
        "code",
        # NOTE: do NOT require crn here (it is frequently NULL in coreed_capacity)
        "section",
        "capacity",
        "enrolled",
    ]
    df_sorted = df_sorted[keep_cols]

    def pick_latest_non_null(group: pd.DataFrame) -> pd.Series:
        non_null_mask = group[["capacity", "enrolled"]].notna().any(axis=1)
        if non_null_mask.any():
            return group.loc[non_null_mask].iloc[-1]
        return group.iloc[-1]

    deduped = (
        df_sorted.groupby(
            ["term_srcdb", "coreed_attr", "campus_group", "code", "section"],
            as_index=False,
            group_keys=False,
        )
        .apply(pick_latest_non_null)
        .reset_index(drop=True)
    )
    return deduped

def build_category_aggregate_chart(
    cat_df: pd.DataFrame,
    cat_label: str,
    term_order: list[str] | None = None,
    daily_latest: pd.DataFrame | None = None,
) -> alt.Chart:
    """
    Aggregate chart for one CoreEd category: term on X, clustered by campus.

    Bars = total capacity (from coreed_capacity).
    Enrollment overlay (black horizontal tick):
      - Default: use enrolled from coreed_capacity (final snapshot).
      - If daily data exists for a term: override enrollment using coreed_daily_sections (latest daily snapshot).
        Active registration terms (with daily updates) are marked with * on the x-axis.
    """
    working = cat_df.copy()

    # --- numeric fields ---
    working["capacity_num"] = pd.to_numeric(working.get("capacity"), errors="coerce").fillna(0)
    working["enrolled_num"] = pd.to_numeric(working.get("enrolled"), errors="coerce").fillna(0)

    # --- capacity + (table) enrolled rollups from coreed_capacity ---
    agg_cap = (
        working.groupby(["campus_group", "term_srcdb"], as_index=False)
        .agg(
            capacity_total=("capacity_num", "sum"),
            enrolled_total_capacity_table=("enrolled_num", "sum"),
        )
    )

    # --- ensure consistent term + campus grids (so bars don't disappear) ---
    all_terms = term_order or sorted(working["term_srcdb"].dropna().unique().tolist())
    all_terms = [str(t) for t in all_terms]

    campus_vals = working["campus_group"].dropna().unique().tolist() or CAMPUS_ORDER
    campus_vals = sorted(
        campus_vals, key=lambda x: CAMPUS_ORDER.index(x) if x in CAMPUS_ORDER else 99
    )

    filler = pd.DataFrame(
        [(camp, term) for camp in campus_vals for term in all_terms],
        columns=["campus_group", "term_srcdb"],
    )

    agg = filler.merge(agg_cap, on=["campus_group", "term_srcdb"], how="left").fillna(
        {"capacity_total": 0, "enrolled_total_capacity_table": 0}
    )

    if agg.empty:
        return alt.Chart(pd.DataFrame({"msg": ["No data"]})).mark_text().encode(text="msg")

    # --- enrolled overlay default: from coreed_capacity ---
    agg["enrolled_overlay"] = agg["enrolled_total_capacity_table"]
    agg["enrolled_source"] = "coreed_capacity (snapshot)"
    agg["daily_snapshot_date"] = pd.NaT

    # --- override enrollment from coreed_daily_sections where possible ---
    if daily_latest is not None and not daily_latest.empty:
        # Join on fields we actually have in coreed_capacity-derived df:
        # term, category, campus, course code, section
        keys = ["term_srcdb", "coreed_attr", "campus_group", "code", "section"]

        if all(k in working.columns for k in keys) and all(k in daily_latest.columns for k in keys):
            wanted = working[keys].drop_duplicates()
            daily_match = daily_latest.merge(wanted, on=keys, how="inner")

            if not daily_match.empty:
                daily_match["enrolled"] = pd.to_numeric(daily_match["enrolled"], errors="coerce").fillna(0)

                daily_enr = (
                    daily_match.groupby(["campus_group", "term_srcdb"], as_index=False)
                    .agg(
                        enrolled_daily=("enrolled", "sum"),
                        snapshot_date=("snapshot_date", "max"),
                    )
                )

                # Merge daily enrolled back into agg for any terms present in daily_enr
                agg = agg.merge(
                    daily_enr[["campus_group", "term_srcdb", "enrolled_daily", "snapshot_date"]],
                    on=["campus_group", "term_srcdb"],
                    how="left",
                )

                use_mask = agg["enrolled_daily"].notna()
                agg.loc[use_mask, "enrolled_overlay"] = agg.loc[use_mask, "enrolled_daily"]
                agg.loc[use_mask, "enrolled_source"] = "coreed_daily_sections (latest)"
                agg.loc[use_mask, "daily_snapshot_date"] = agg.loc[use_mask, "snapshot_date"]

    # --- term axis labels: append * to terms with recent daily enrollment updates ---
    # Only mark a term as "active" if its daily snapshot is within the last 7 days,
    # so stale historical terms don't get the asterisk.
    today = pd.Timestamp.today().normalize()
    daily_mask = agg["enrolled_source"] == "coreed_daily_sections (latest)"
    recent_mask = daily_mask & (
        pd.to_datetime(agg["daily_snapshot_date"], errors="coerce")
        >= today - pd.Timedelta(days=7)
    )
    active_terms = (
        agg.loc[recent_mask, "term_srcdb"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    term_label_map = {
        t: f"{t}*" if t in active_terms else t
        for t in all_terms
    }
    agg["term_label"] = agg["term_srcdb"].astype(str).map(term_label_map)
    term_label_order = [term_label_map[t] for t in all_terms]

    # --- lock campus order for consistent xOffset positioning ---
    campus_cat = pd.CategoricalDtype(CAMPUS_ORDER, ordered=True)
    agg["campus_group"] = agg["campus_group"].astype(campus_cat)

    # ✅ IMPORTANT: define ONE shared X encoding and reuse it in BOTH layers
    x_term = alt.X(
        "term_label:N",
        title="Term",
        sort=term_label_order,
        axis=alt.Axis(labelAngle=-45),
        scale=alt.Scale(domain=term_label_order),
    )

    # --- bars (capacity) ---
    bars = (
        alt.Chart(agg, title=cat_label)
        .mark_bar()
        .encode(
            x=x_term,
            xOffset=alt.X("campus_group:N", title=None),
            y=alt.Y("capacity_total:Q", title="Total Capacity"),
            color=alt.Color(
                "campus_group:N",
                title=None,
                scale=alt.Scale(domain=CAMPUS_ORDER, range=CAMPUS_COLORS),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("term_srcdb:N", title="Term code"),
                alt.Tooltip("term_label:N", title="Term label"),
                alt.Tooltip("campus_group:N", title="Campus"),
                alt.Tooltip("capacity_total:Q", title="Capacity (bar)", format=",.0f"),
                alt.Tooltip("enrolled_overlay:Q", title="Enrolled (tick)", format=",.0f"),
                alt.Tooltip("enrolled_source:N", title="Enrolled source"),
                alt.Tooltip("daily_snapshot_date:T", title="Daily snapshot date"),
            ],
        )
        .properties(height=300)
    )

    # --- enrollment overlay as BLACK horizontal tick at the enrolled level ---
    enroll_tick = (
        alt.Chart(agg)
        .mark_tick(orient="horizontal", size=18, thickness=2)
        .encode(
            x=x_term,  # ✅ reuse the SAME x encoding so it aligns perfectly
            xOffset=alt.X("campus_group:N"),
            y=alt.Y("enrolled_overlay:Q"),
            color=alt.value("black"),
        )
    )

    return bars + enroll_tick

def build_course_chart(
    df: pd.DataFrame,
    max_labels: int = 10,
    y_title: str | None = "Total capacity (latest snapshot per section)",
    show_legend: bool = True,
    bar_width: int | None = None,
) -> alt.Chart:
    if df.empty:
        return alt.Chart(pd.DataFrame({"msg": ["No data"]})).mark_text().encode(text="msg")

    # Aggregate capacity by course & campus
    course_ag = (
        df.groupby(["code", "campus_group", "coreed_label"], as_index=False)["capacity"]
        .sum(min_count=1)
    )
    course_ag = course_ag.dropna(subset=["capacity"])
    if course_ag.empty:
        return alt.Chart(pd.DataFrame({"msg": ["No data"]})).mark_text().encode(text="msg")

    # Compute total per course (across campuses) for ordering & label selection
    totals = (
        course_ag.groupby("code", as_index=False)["capacity"]
        .sum(min_count=1)
        .rename(columns={"capacity": "total_capacity"})
    )
    course_ag = course_ag.merge(totals, on="code", how="left")

    # Sort courses by total_capacity desc
    course_order = (
        totals.sort_values("total_capacity", ascending=False)["code"].tolist()
    )
    course_ag["code"] = pd.Categorical(course_ag["code"], course_order, ordered=True)
    course_ag["campus_group"] = pd.Categorical(course_ag["campus_group"], CAMPUS_ORDER, ordered=True)

    base_chart = alt.Chart(course_ag)
    base = (
        base_chart.mark_bar(size=bar_width) if bar_width else base_chart.mark_bar()
    ).encode(
        x=alt.X(
            "code:N",
            title="Course",
            sort=course_order,
            axis=alt.Axis(labelAngle=-90),
        ),
        y=alt.Y("capacity:Q", title=y_title),
        color=alt.Color(
            "campus_group:N",
            title="Campus" if show_legend else None,
            scale=alt.Scale(domain=CAMPUS_ORDER, range=CAMPUS_COLORS),
            legend=alt.Legend(title="Campus") if show_legend else None,
        ),
        tooltip=[
            alt.Tooltip("code:N", title="Course"),
            alt.Tooltip("coreed_label:N", title="Category"),
            alt.Tooltip("campus_group:N", title="Campus"),
            alt.Tooltip("capacity:Q", title="Capacity", format=",.0f"),
        ],
    )

    # Select top-N courses by total capacity for labels (one label per course, not per campus)
    totals["total_capacity"] = totals["total_capacity"].fillna(0)
    top_label_codes = (
        totals.sort_values("total_capacity", ascending=False)["code"].head(max_labels).tolist()
    )
    label_df = totals[totals["code"].isin(top_label_codes)].copy()

    labels = (
        alt.Chart(label_df)
        .mark_text(
            align="center",
            baseline="bottom",
            dy=-10,  # lift labels above bars without obscuring them
            fontSize=10,
            color="black",
        )
        .encode(
            x=alt.X("code:N", sort=course_order),
            y=alt.Y("total_capacity:Q"),
            text=alt.Text("code:N"),
        )
    )

    return (base + labels).properties(height=450)


def main():
    st.title("OSU CoreEd Capacity Dashboard")

    df_raw, load_messages = load_data()
    if df_raw.empty:
        st.warning("No rows found in CoreEd capacity data source.")
        return
    latest_snapshot = df_raw["snapshot_timestamp"].max()
    latest_snapshot_str = (
        latest_snapshot.strftime("%Y-%m-%d %H:%M %Z")
        if pd.notna(latest_snapshot)
        else "Unknown"
    )

    df_latest = dedupe_latest(df_raw)

    # Compute diagnostics now (rendered in sidebar after filters below)
    diag_messages = []
    diag_warnings = []
    try:
        if not df_latest.empty:
            n = len(df_latest)
            null_cap = int(df_latest["capacity"].isna().sum())
            null_enr = int(df_latest["enrolled"].isna().sum())
            diag_messages.append(
                f"NULLs in df_latest: capacity {null_cap}/{n}, enrolled {null_enr}/{n}"
            )
            if n > 0 and (null_cap / n > 0.25 or null_enr / n > 0.25):
                term_diag = (
                    df_latest.assign(
                        cap_null=df_latest["capacity"].isna(),
                        enr_null=df_latest["enrolled"].isna(),
                    )
                    .groupby("term_srcdb", as_index=False)
                    .agg(
                        rows=("term_srcdb", "size"),
                        cap_null=("cap_null", "sum"),
                        enr_null=("enr_null", "sum"),
                    )
                )
                term_diag["cap_null_share"] = term_diag["cap_null"] / term_diag["rows"].where(term_diag["rows"] != 0, 1)
                worst = term_diag.sort_values("cap_null_share", ascending=False).head(3)
                worst_str = ", ".join(
                    f"{r.term_srcdb} (cap NULL {int(r.cap_null)}/{int(r.rows)}, enr NULL {int(r.enr_null)}/{int(r.rows)})"
                    for r in worst.itertuples(index=False)
                )
                diag_warnings.append(
                    "High NULL share in capacity/enrolled after dedupe. "
                    f"Worst term(s): {worst_str}"
                )
    except Exception:
        pass

    # Load daily CoreEd data (for enrolled overlay)
    df_coreed_daily_latest = load_coreed_daily_latest()

    daily_max_sql = None
    try:
        with sqlite3.connect(DB_PATH) as conn:
            daily_max_sql = conn.execute(
                f"SELECT MAX(snapshot_date) FROM {COREED_DAILY_TABLE};"
            ).fetchone()[0]
    except Exception:
        daily_max_sql = None

    # Flag lab-like sections so we can optionally exclude them (to avoid double-counting capacity).
    df_latest["is_lab_like"] = df_latest.apply(
        lambda r: is_lab_like_section(r["section"], r["campus_group"]), axis=1
    )
    df_latest["subject_code"] = df_latest["code"].apply(extract_subject_code)
    df_latest["is_cas"] = df_latest["subject_code"].isin(CAS_SUBJECT_CODES)
    # Unified metric for registrations: use capacity when available, otherwise enrolled.
    df_latest["registrations"] = pd.to_numeric(df_latest["capacity"], errors="coerce").fillna(
        pd.to_numeric(df_latest["enrolled"], errors="coerce")
    )

    # Sidebar filters
    st.sidebar.header("Filters")

    if st.sidebar.button("Reset filters"):
        st.session_state.pop("term_choice", None)

    term_codes = sorted(df_latest["term_srcdb"].dropna().unique().tolist(), reverse=True)
    # Default to the 5 most recent terms; oldest drops off as new terms appear
    default_terms = term_codes[:5]
    term_choice = st.sidebar.multiselect(
        "Terms",
        options=term_codes,
        default=default_terms,
        key="term_choice",
    )

    if term_choice:
        df_latest = df_latest[df_latest["term_srcdb"].isin(term_choice)]

    # CAS-only toggle
    cas_only = st.sidebar.checkbox(
        "Show only CAS subjects",
        value=False,
        help="Filters to College of Agricultural Sciences subjects (e.g., AEC, ANS, HORT, SUS).",
    )
    if cas_only:
        df_latest = df_latest[df_latest["is_cas"]]
        if df_latest.empty:
            st.warning("No CAS courses found after applying filters.")
            return

    # Campus filter
    available_campuses = sorted(
        df_latest["campus_group"].dropna().unique().tolist(),
        key=lambda x: CAMPUS_ORDER.index(x) if x in CAMPUS_ORDER else 99,
    )
    default_campuses = [c for c in CAMPUS_ORDER if c in available_campuses] or available_campuses
    campus_choice = st.sidebar.multiselect(
        "Campus",
        options=available_campuses,
        default=default_campuses,
    )
    if campus_choice:
        df_latest = df_latest[df_latest["campus_group"].isin(campus_choice)]

    # Course filter (based on remaining data)
    available_courses = sorted(df_latest["code"].dropna().unique().tolist())
    course_choice = st.sidebar.multiselect(
        "Courses",
        options=available_courses,
        default=available_courses,
    )
    if course_choice:
        df_latest = df_latest[df_latest["code"].isin(course_choice)]

    # Lab/recitation toggle (moved lower in the filter stack)
    include_labs = st.sidebar.checkbox(
        "Include lab/recitation sections",
        value=False,
        help=(
            "When unchecked, capacity for lab-like Corvallis sections (e.g., section numbers 010–399) "
            "is excluded so that lecture capacity is not double-counted."
        ),
    )
    if not include_labs:
        df_latest = df_latest[~df_latest["is_lab_like"]].copy()

    # --- Sidebar diagnostics (below filters) ---
    st.sidebar.markdown("---")
    st.sidebar.header("Diagnostics")
    st.sidebar.caption(f"Using DB: {DB_PATH}")
    if DB_PATH.exists():
        st.sidebar.caption(f"DB last modified: {datetime.fromtimestamp(DB_PATH.stat().st_mtime)}")
    st.sidebar.caption(f"Latest snapshot_timestamp: {latest_snapshot}")
    if daily_max_sql:
        st.sidebar.caption(f"Daily data date: {daily_max_sql}")
    else:
        st.sidebar.caption("Daily data date: (none found)")
    if not df_coreed_daily_latest.empty:
        st.sidebar.caption("Enrollment overlay uses coreed_daily_sections where available.")
    else:
        st.sidebar.caption("No daily overlay rows matched (overlay will use coreed_capacity enrolled).")
    for msg in load_messages:
        st.sidebar.caption(msg)
    for msg in diag_messages:
        st.sidebar.caption(msg)
    for warn in diag_warnings:
        st.sidebar.warning(warn)

    if df_latest.empty:
        st.warning("No data after applying filters.")
        return

    # Category panels: per-category term/campus aggregate and course chart (side by side)
    st.subheader("Section capacity by category (term × campus) with per-category course rollups")
    st.caption("* = active registration term (enrollment updated daily)")

    term_order = sorted(df_latest["term_srcdb"].dropna().unique().tolist())
    cat_order = [c for c in ["CFSI", "CSSS", "CSDP", "CFSS"] if c in df_latest["coreed_attr"].unique()]
    remaining = [c for c in df_latest["coreed_attr"].unique() if c not in cat_order]
    cat_order.extend(sorted(remaining))

    for cat in cat_order:
        cat_df = df_latest[df_latest["coreed_attr"] == cat]
        if cat_df.empty:
            continue
        cat_label = cat_df["coreed_label"].iloc[0]
        left = build_category_aggregate_chart(cat_df, cat_label, term_order, daily_latest=df_coreed_daily_latest)
        right = build_course_chart(
            cat_df,
            max_labels=8,
            y_title=None,
            show_legend=False,
            bar_width=None,
        ).properties(title=f"{cat_label} course capacities", height=300)
        col1, col2 = st.columns((1, 1))
        col1.altair_chart(left, width="stretch")
        col2.altair_chart(right, width="stretch")
        st.markdown("---")

    # Course chart (stacked, below aggregate)
    st.subheader("Course-level capacity by campus (latest snapshot per section)")
    course_chart = build_course_chart(df_latest, max_labels=10, show_legend=True, bar_width=None)
    st.altair_chart(course_chart, width="stretch")

    # Data preview
    with st.expander("Show raw data (latest snapshot per section)"):
        st.dataframe(
            df_latest.sort_values(["term_srcdb", "coreed_attr", "campus_group", "code", "section"])
        )

    st.caption(f"Latest snapshot: {latest_snapshot_str}")


if __name__ == "__main__":
    main()
