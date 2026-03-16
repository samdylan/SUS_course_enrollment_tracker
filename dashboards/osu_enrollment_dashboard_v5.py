"""
osu_enrollment_dashboard_v3.py

SUS enrollment dashboard + CoreEd daily snapshot charts.
Keeps the SUS views from v2 and appends CoreEd timeseries (capacity/enrollment)
from the coreed_daily_sections table.
"""

import os
import sqlite3
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

# Allow override (useful for Streamlit Cloud / servers / different repo layouts)
# Example: export ENROLLMENT_DB_PATH="/path/to/osu_enrollment_log_classes.db"
DB_PATH = Path(os.environ.get("ENROLLMENT_DB_PATH", "")).expanduser() if os.environ.get("ENROLLMENT_DB_PATH") else (
    Path(__file__).resolve().parent.parent / "data" / "osu_enrollment_log_classes.db"
)
COREED_DAILY_TABLE = "coreed_daily_sections"

CAMPUS_ORDER = ["Corvallis", "Ecampus", "Cascades", "Other"]
CAS_SUBJECT_CODES = {
    "AED", "AEC", "AG", "AGRI", "ANS", "BDS", "BOT", "BRR",
    "CROP", "CSS", "ENT", "FST", "FW", "HORT", "LEAD", "PBG",
    "RNG", "SOIL", "SUS", "TOX",
}
COREED_LABELS = {
    "CFSI": "Scientific Inquiry & Analysis",
    "CSSS": "Seeking Solutions",
    "CSDP": "DPO Advanced",
    "CFSS": "Social Science",
}


# ---------- SUS data loaders ----------

def load_sus_data() -> pd.DataFrame:
    if not DB_PATH.exists():
        st.error(f"Database not found at {DB_PATH}")
        return pd.DataFrame()

    conn = sqlite3.connect(DB_PATH)
    try:
        try:
            df = pd.read_sql_query("SELECT * FROM enrollment", conn)
        except Exception as exc:
            # Most likely: no such table: enrollment
            st.sidebar.warning(
                f"No live SUS enrollment table ('enrollment') found in DB {DB_PATH.name}; "
                "showing only historic SUS + CoreEd data."
            )
            return pd.DataFrame()
    finally:
        conn.close()

    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    snap_db = (
        pd.to_datetime(df["snapshot_date"], errors="coerce")
        if "snapshot_date" in df.columns
        else pd.Series(pd.NaT, index=df.index)
    )
    snap_from_ts = df["timestamp"].dt.normalize()
    df["snapshot_date"] = snap_db.fillna(snap_from_ts)

    df["code"] = df["code"].astype(str)
    df["subject"] = df["subject"].fillna(df["code"].str.split().str[0])
    df["course_number"] = df["course_number"].fillna(df["code"].str.split().str[1])
    df["course_code"] = df["subject"].str.cat(df["course_number"], sep=" ")

    df["section"] = df["section"].astype(str)
    df["section_label"] = (
        df["subject"].astype(str)
        + df["course_number"].astype(str)
        + "_"
        + df["section"].astype(str)
    )

    df["campus_code"] = df["campus_code"].astype(str)

    def map_campus(c: str) -> str:
        c = (c or "").upper()
        if c in ("C", ""):
            return "Corvallis"
        if c in ("B",):
            return "Cascades"
        if c in ("DI", "DB", "E"):
            return "Ecampus"
        return "Other"

    df["campus_simple"] = df["campus_code"].apply(map_campus)

    def is_lab(section: str) -> bool:
        s = str(section).strip()
        if len(s) != 3 or not s.isdigit():
            return False
        n = int(s)
        return 10 <= n < 30

    df["is_lab"] = df["section"].apply(is_lab)
    df["enrolled"] = pd.to_numeric(df["enrolled"], errors="coerce")
    df["capacity"] = pd.to_numeric(df.get("capacity"), errors="coerce")
    return df


def estimate_classes_begin_from_srcdb(term_srcdb: str) -> pd.Timestamp | None:
    """
    Approximate classes-begin date from term code (srcdb like 202602),
    using OSU's convention that:

      label_year = academic year, so:
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

    # Summer: late June of (label_year - 1), approximate Monday in the last week of June
    if term_code == "00":
        year = label_year - 1
        d = pd.Timestamp(year=year, month=6, day=24)
        offset = (0 - d.weekday()) % 7  # move to Monday
        return (d + pd.Timedelta(days=offset)).normalize()

    # Fall: last Wednesday of September of (label_year - 1)
    if term_code == "01":
        year = label_year - 1
        d = pd.Timestamp(year=year, month=9, day=30)
        while d.weekday() != 2:  # Wednesday
            d -= pd.Timedelta(days=1)
        return d.normalize()

    # Winter: first Monday in early January of label_year
    if term_code == "02":
        year = label_year
        d = pd.Timestamp(year=year, month=1, day=3)
        offset = (0 - d.weekday()) % 7  # Monday
        return (d + pd.Timedelta(days=offset)).normalize()

    # Spring: first Monday in late March of label_year
    if term_code == "03":
        year = label_year
        d = pd.Timestamp(year=year, month=3, day=29)
        offset = (0 - d.weekday()) % 7  # Monday
        return (d + pd.Timedelta(days=offset)).normalize()

    return None

def load_sus_historic_daily() -> pd.DataFrame:
    """
    Load historical SUS daily snapshots from sus_daily_registrations_2025
    and shape them to look like the live SUS data after load_sus_data().

    We explicitly CAST(section AS INTEGER) in SQLite so that 400/401/501
    style sections are preserved cleanly for campus derivation.
    """
    if not DB_PATH.exists():
        return pd.DataFrame()

    conn = sqlite3.connect(DB_PATH)
    try:
        # Let SQLite parse the numeric section so we don't fight with
        # any odd formatting in the stored column.
        query = """
        SELECT
            snapshot_date,
            term_srcdb,
            COALESCE(subject, 'SUS') AS subject,
            course_number,
            section,
            CAST(section AS INTEGER) AS section_num,
            enrolled
        FROM sus_daily_registrations_2025
        """
        df = pd.read_sql_query(query, conn)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()

    if df.empty:
        return df

    # ---------- Clean & normalize ----------
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
    df = df.dropna(subset=["snapshot_date"])

    df["term_srcdb"] = df["term_srcdb"].astype(str)
    df["subject"] = df["subject"].fillna("SUS")
    df["course_number"] = df["course_number"].astype(str)

    # Ensure section_num is numeric
    df["section_num"] = pd.to_numeric(df["section_num"], errors="coerce")

    # Zero-padded string version used elsewhere in the dashboard
    df["section"] = (
        df["section_num"]
        .fillna(0)          # anything non-numeric becomes 000 → Corvallis
        .astype(int)
        .astype(str)
        .str.zfill(3)
    )

    df["enrolled"] = pd.to_numeric(df["enrolled"], errors="coerce")

    # Construct fields to match load_sus_data() output
    df["course_code"] = df["subject"].astype(str) + " " + df["course_number"]
    df["code"] = df["course_code"]
    df["section_label"] = (
        df["subject"].astype(str)
        + df["course_number"].astype(str)
        + "_"
        + df["section"].astype(str)
    )

    # ---------- Campus from section_num ----------
    def campus_from_section_num(n: float | int | None) -> str:
        if pd.isna(n):
            # Default to Corvallis if we can't parse
            return "Corvallis"
        n_int = int(n)
        if 400 <= n_int <= 499:
            return "Ecampus"
        if n_int >= 500:
            return "Cascades"
        return "Corvallis"

    df["campus_simple"] = df["section_num"].apply(campus_from_section_num)
    df["campus_code"] = df["campus_simple"].map(
        {
            "Corvallis": "C",
            "Ecampus": "E",
            "Cascades": "B",
        }
    ).fillna("C")

    # Lab flag: use the same rule as live SUS data (010–030 are labs/recitations).
    def is_lab(section: str) -> bool:
        s = str(section).strip()
        if len(s) != 3 or not s.isdigit():
            return False
        n = int(s)
        return 10 <= n < 30

    df["is_lab"] = df["section"].apply(is_lab)

    # Timestamp column for consistency with live data
    df["timestamp"] = df["snapshot_date"]

    # Mark as historic
    df["is_historic"] = True

    # Keep only columns the rest of the code expects (+ section_num for debug)
    cols = [
        "timestamp",
        "snapshot_date",
        "term_srcdb",
        "subject",
        "course_number",
        "course_code",
        "section",
        "section_num",
        "section_label",
        "campus_code",
        "campus_simple",
        "is_lab",
        "enrolled",
        "is_historic",
    ]
    df = df[cols]

    return df
# ---------- CoreEd data loaders ----------

def load_coreed_daily() -> pd.DataFrame:
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

    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce").dt.date
    df["enrolled"] = pd.to_numeric(df["enrolled"], errors="coerce")
    df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce")

    def derive_campus(campus_simple: str | None, section: Any) -> str:
        try:
            sec_num = int(str(section).strip())
            if 400 <= sec_num <= 499:
                return "Ecampus"
            if sec_num >= 500:
                return "Cascades"
            return "Corvallis"
        except Exception:
            pass
        c = (campus_simple or "").upper()
        if c in {"ECAMPUS", "E", "DI", "DIST", "DISTANCE"}:
            return "Ecampus"
        if c in {"CASCADES", "B", "DB", "L", "CASC"}:
            return "Cascades"
        if c in {"CORVALLIS", "C"}:
            return "Corvallis"
        return "Other"

    df["campus_simple"] = df.apply(lambda r: derive_campus(r.get("campus_simple"), r.get("section")), axis=1)
    df["campus_simple"] = pd.Categorical(df["campus_simple"], CAMPUS_ORDER, ordered=True)

    def is_lab_like(section: Any, campus_simple: str) -> bool:
        try:
            sec_num = int(str(section).strip())
        except Exception:
            return False
        return campus_simple == "Corvallis" and 10 <= sec_num < 400

    df["is_lab"] = df.apply(lambda r: is_lab_like(r.get("section"), r.get("campus_simple")), axis=1)
    return df


# ---------- CoreEd chart helper ----------

def category_timeseries_chart(
    df: pd.DataFrame,
    category: str,
    campus_domain: list[str] | None = None,
    days_domain: tuple[int, int] | None = None,
    data_scope: str = "OSU",
) -> alt.Chart:
    df_cat = df[df["coreed_cat4"] == category]
    if df_cat.empty:
        return alt.Chart(pd.DataFrame({"msg": ["No data"]})).mark_text().encode(text="msg")
    agg = (
        df_cat.groupby(
            ["snapshot_date", "days_from_start", "campus_simple"],
            as_index=False,
            observed=True,
        )[["enrolled", "capacity"]].sum(min_count=1)
    )

    # Drop rows where capacity is zero (likely upstream data issues).
    # Allow enrolled == 0 so that pre-enrollment terms still display capacity lines.
    agg = agg[agg["capacity"] > 0]

    melted = agg.melt(
        id_vars=["snapshot_date", "days_from_start", "campus_simple"],
        value_vars=["enrolled", "capacity"],
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])

    if melted.empty:
        return alt.Chart(pd.DataFrame({"msg": ["No data"]})).mark_text().encode(text="msg")

    # Map metric labels for nicer legend and to match strokeDash domain
    melted["metric"] = melted["metric"].map({"enrolled": "Enrolled", "capacity": "Capacity"})
    # Separate field for dash styling so we don't kill the Metric legend
    melted["metric_dash"] = melted["metric"]

    domain = campus_domain or [c for c in CAMPUS_ORDER if c in melted["campus_simple"].unique()]
    melted["campus_simple"] = pd.Categorical(melted["campus_simple"], domain, ordered=True)
    melted["series_count"] = melted.groupby(["campus_simple", "metric"])["value"].transform("count")
    melted = melted.sort_values("snapshot_date")

    # 👉 use SUS-provided days_domain if available; otherwise fall back to CoreEd's own min/max
    if days_domain is not None:
        min_days, max_days = days_domain
    else:
        min_days = int(melted["days_from_start"].min())
        max_days = int(melted["days_from_start"].max())

    base = alt.Chart(melted).encode(
        x=alt.X(
            "days_from_start:Q",
            title="Days from start of term",
            scale=alt.Scale(domain=[min_days, max_days]),
            axis=alt.Axis(format="d"),
        ),
        y=alt.Y("value:Q", title="Headcount"),
        color=alt.Color(
            "campus_simple:N",
            title="Campus",
            scale=alt.Scale(
                domain=["Corvallis", "Ecampus", "Cascades", "Other"],
                range=["#4c78a8", "#e45756", "#72b7b2", "#999999"],
            ),
            sort=["Corvallis", "Ecampus", "Cascades", "Other"],
        ),
        shape=alt.Shape(
            "metric:N",
            title="Metric",
            legend=alt.Legend(title="Metric"),
            scale=alt.Scale(
                domain=["Enrolled", "Capacity"],
                range=["triangle-up", "circle"],  # Enrolled = triangle, Capacity = circle
            ),
        ),
        strokeDash=alt.StrokeDash(
            "metric_dash:N",
            legend=None,
            scale=alt.Scale(
                domain=["Enrolled", "Capacity"],
                range=[[4, 3], [1, 0]],          # Enrolled dashed, Capacity solid
            ),
        ),
        tooltip=[
            alt.Tooltip("snapshot_date:T", title="Snapshot date"),
            alt.Tooltip("days_from_start:Q", title="Days from start"),
            alt.Tooltip("campus_simple:N", title="Campus"),
            alt.Tooltip("metric:N", title="Metric"),
            alt.Tooltip("value:Q", title="Value", format=",.0f"),
        ],
    )

    # Lines: dashed vs solid by metric
    line_layer = base.mark_line()

    # Points: inherit shape/color from base, just make them visible markers
    point_layer = base.mark_point(size=60, filled=True)

    chart = line_layer + point_layer
    label = COREED_LABELS.get(category, category)
    return chart.properties(height=260, title=f"{label} ({data_scope})")


# ---------- Main ----------
# ---------- Main ----------

def main():
    st.set_page_config(page_title="SUS Enrollment Dashboard", layout="wide")
    st.title("SUS Enrollment Dashboard")

    df_raw = load_sus_data()
    df_coreed = load_coreed_daily()
    df_hist = load_sus_historic_daily()  # historic SUS snapshots

    # Tag live vs historic rows so we can style them separately
    if not df_raw.empty:
        df_raw = df_raw.copy()
        df_raw["is_historic"] = False
    if not df_hist.empty:
        df_hist = df_hist.copy()
        # load_sus_historic_daily already sets is_historic=True, but ensure it:
        df_hist["is_historic"] = True

    if df_raw.empty and df_hist.empty:
        st.warning("No SUS data available.")
        return

    # For filters and later plotting, build a combined SUS dataframe
    if not df_hist.empty:
        df_all = pd.concat([df_raw, df_hist], ignore_index=True)
    else:
        df_all = df_raw.copy()

    # Use df_all for sidebar option ranges
    df_filters = df_all.copy()

    # Sidebar filters (SUS)
    st.sidebar.header("Filters")

    # Terms ordered reverse-chronologically (latest term first)
    raw_terms = df_filters["term_srcdb"].dropna().astype(str).unique().tolist()
    term_values = sorted(raw_terms, key=int, reverse=True)

    # Default to the most recent term that has non-zero enrollment.
    # This avoids defaulting to a future term where registration hasn't started yet.
    default_terms = []
    if term_values:
        for t in term_values:  # already sorted newest-first
            term_df = df_filters[df_filters["term_srcdb"].astype(str) == t]
            if (pd.to_numeric(term_df["enrolled"], errors="coerce") > 0).any():
                default_terms = [t]
                break
        if not default_terms:
            default_terms = [term_values[0]]

    term_choice = st.sidebar.multiselect(
        "Terms (srcdb)",
        options=term_values,
        default=default_terms,
    )

    # SUS campus filter: limit to the three main campuses
    available_sus_campuses = [
        c
        for c in CAMPUS_ORDER
        if c in df_filters["campus_simple"].unique()
        and c in ("Corvallis", "Ecampus", "Cascades")
    ]
    campus_selected = st.sidebar.multiselect(
        "Campus",
        options=available_sus_campuses,
        default=available_sus_campuses,
    )

    course_values = sorted(df_filters["course_code"].dropna().unique().tolist())
    course_selected = st.sidebar.multiselect(
        "Courses",
        options=course_values,
        default=course_values,
    )

    include_labs = st.sidebar.checkbox("Include labs (010–029)", value=False)

    agg_options = ["Section", "Course", "Campus"]
    agg_choice = st.sidebar.selectbox("Aggregation level", agg_options, index=0)

    show_labels = st.sidebar.checkbox("Show section labels on chart", value=False)

    # Start from full combined SUS data (live + historic). Term selection below
    # decides which terms appear; there is no separate historic toggle anymore.
    df = df_all.copy()

    # Term filter: keep only explicitly selected terms
    if term_choice:
        chosen_terms = [str(t) for t in term_choice]
        df = df[df["term_srcdb"].astype(str).isin(chosen_terms)]
    # If no terms are selected, df remains unfiltered by term (not expected in normal use).

    # Other filters
    if campus_selected:
        df = df[df["campus_simple"].isin(campus_selected)]
    if course_selected:
        df = df[df["course_code"].isin(course_selected)]
    if not include_labs:
        df = df[~df["is_lab"]]

    # ---- SUS chart + summary ---------------------------------------------
    df = df.dropna(subset=["enrolled"])

    if df.empty:
        # No SUS data, but we still want CoreEd to render below.
        st.warning("No SUS data after applying filters.")
        df_plot = pd.DataFrame()
        sus_days_domain = None
    else:
        # Compute days_from_start for SUS rows
        df = df.copy()
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
        df["classes_begin"] = df["term_srcdb"].apply(estimate_classes_begin_from_srcdb)
        df["days_from_start"] = (df["snapshot_date"] - df["classes_begin"]).dt.days

        # Term grouping based on *recency* among the terms present in df.
        # - If only one term is present → everything is "Current term".
        # - If 2+ terms are present → numerically largest term_srcdb is "Current term",
        #   all others are "Prior term" (styled in grey).
        df["term_num"] = pd.to_numeric(df["term_srcdb"], errors="coerce")
        unique_terms = df["term_num"].dropna().unique()

        if len(unique_terms) <= 1:
            df["term_group"] = "Current term"
        else:
            most_recent = unique_terms.max()
            df["term_group"] = df["term_num"].apply(
                lambda v: "Current term" if v == most_recent else "Prior term"
            )

        # SUS aggregation logic
        if agg_choice == "Section":
            df_plot = (
                df.groupby(
                    [
                        "snapshot_date",
                        "days_from_start",
                        "course_code",
                        "section",
                        "section_label",
                        "campus_simple",
                        "term_group",   # keep current vs prior separated
                    ],
                    as_index=False,
                )[["enrolled", "capacity"]]
                .max()
            )
            color_field = "section_label"
            y_title = "Enrollment (per section)"
        elif agg_choice == "Course":
            df_plot = df.groupby(
                ["snapshot_date", "days_from_start", "course_code"],
                as_index=False,
            )[["enrolled", "capacity"]].sum()
            df_plot["series_label"] = df_plot["course_code"]
            color_field = "series_label"
            y_title = "Enrollment (sum across sections)"
        else:
            df_plot = df.groupby(
                ["snapshot_date", "days_from_start", "campus_simple"],
                as_index=False,
            )[["enrolled", "capacity"]].sum()
            df_plot["series_label"] = df_plot["campus_simple"]
            color_field = "series_label"
            y_title = "Enrollment (sum across all courses)"

        # SUS chart: use days_from_start for alignment across terms,
        # capture the domain so CoreEd charts can use the same x-range,
        # and let the user override via a sidebar slider.
        sus_days_domain = None
        if not df_plot.empty:
            raw_min = int(df_plot["days_from_start"].min())
            raw_max = int(df_plot["days_from_start"].max())

            # Default window: start 2 days before the earliest day with
            # non-zero enrollment (not just the earliest snapshot).
            # Clamp so default never falls outside the slider's [raw_min, raw_max].
            enrolled_days = df_plot.loc[
                pd.to_numeric(df_plot["enrolled"], errors="coerce") > 0,
                "days_from_start",
            ]
            first_enrolled = int(enrolled_days.min()) if not enrolled_days.empty else raw_min
            default_min = max(first_enrolled - 2, raw_min)
            default_max = min(raw_max, 20)

            sus_min_days, sus_max_days = st.sidebar.slider(
                "SUS: X-axis range (days from start)",
                min_value=raw_min,
                max_value=raw_max,
                value=(default_min, default_max),
                step=1,
            )

            sus_days_domain = (sus_min_days, sus_max_days)
            x_scale = alt.Scale(domain=[sus_min_days, sus_max_days])
        else:
            x_scale = alt.Scale()

        x_enc = alt.X(
            "days_from_start:Q",
            title="Days from start of term",
            scale=x_scale,
            axis=alt.Axis(format="d"),
        )

        # ---------- SUS chart construction ----------
        if agg_choice == "Section":
            # Split current vs prior term for layering
            df_hist_plot = df_plot[df_plot["term_group"] == "Prior term"]
            df_cur_plot = df_plot[df_plot["term_group"] == "Current term"]

            # Campus = shape + dash; section = color.
            stroke_dash_scale = alt.Scale(
                domain=["Corvallis", "Ecampus", "Cascades", "Other"],
                # Corvallis/Cascades/Other solid, Ecampus dashed
                range=[[1, 0], [4, 3], [1, 0], [1, 0]],
            )
            shape_scale = alt.Scale(
                domain=["Corvallis", "Ecampus", "Cascades", "Other"],
                range=["circle", "triangle-up", "square", "diamond"],
            )

            sus_tooltips = [
                alt.Tooltip("snapshot_date:T", title="Snapshot date"),
                alt.Tooltip("days_from_start:Q", title="Days from start"),
                alt.Tooltip("section_label:N", title="Section"),
                alt.Tooltip("campus_simple:N", title="Campus"),
                alt.Tooltip("enrolled:Q", title="Enrolled", format=",.0f"),
                alt.Tooltip("capacity:Q", title="Capacity", format=",.0f"),
            ]

            # Prior-term layer – black/grey, drawn FIRST
            hist_layer = alt.Chart(df_hist_plot).mark_line(
                color="black",
                opacity=0.45,
            ).encode(
                x=x_enc,
                y=alt.Y("enrolled:Q", title=y_title),
                strokeDash=alt.StrokeDash(
                    "campus_simple:N",
                    scale=stroke_dash_scale,
                    legend=None,
                ),
                detail="section_label:N",
                tooltip=sus_tooltips,
            )

            # Current term – colored by section, on top
            cur_base = alt.Chart(df_cur_plot).encode(
                x=x_enc,
                y=alt.Y("enrolled:Q", title=y_title),
                color=alt.Color(
                    f"{color_field}:N",
                    title="Section",
                    legend=alt.Legend(orient="right"),
                ),
                strokeDash=alt.StrokeDash(
                    "campus_simple:N",
                    scale=stroke_dash_scale,
                    legend=None,
                ),
                tooltip=sus_tooltips,
            )

            cur_line = cur_base.mark_line()
            cur_points = cur_base.mark_point(filled=True, size=60).encode(
                shape=alt.Shape(
                    "campus_simple:N",
                    scale=shape_scale,
                    legend=alt.Legend(title="Campus", orient="right"),
                )
            )

            chart = hist_layer + cur_line + cur_points

            if show_labels:
                text_layer = cur_base.mark_text(
                    align="left",
                    dx=4,
                    dy=-2,
                    fontSize=10,
                ).encode(text=alt.Text("section_label:N"))
                chart = chart + text_layer

        else:
            agg_tooltips = [
                alt.Tooltip("snapshot_date:T", title="Snapshot date"),
                alt.Tooltip("days_from_start:Q", title="Days from start"),
                alt.Tooltip(f"{color_field}:N", title=agg_choice),
                alt.Tooltip("enrolled:Q", title="Enrolled", format=",.0f"),
                alt.Tooltip("capacity:Q", title="Capacity", format=",.0f"),
            ]
            base = alt.Chart(df_plot).encode(
                x=x_enc,
                y=alt.Y("enrolled:Q", title=y_title),
                color=alt.Color(
                    f"{color_field}:N",
                    title=agg_choice,
                    legend=alt.Legend(orient="right"),
                ),
                tooltip=agg_tooltips,
            )
            chart = base.mark_line() + base.mark_point(filled=True, size=60)

        # Reference lines: cancellation notice deadlines
        ref_lines_df = pd.DataFrame([
            {"days_from_start": -18, "label": "Heads-up (day −18)"},
            {"days_from_start": -14, "label": "Cancel notice (day −14)"},
        ])
        heads_up_rule = alt.Chart(ref_lines_df[ref_lines_df["days_from_start"] == -18]).mark_rule(
            strokeDash=[6, 4], color="grey", strokeWidth=1,
        ).encode(
            x=alt.X("days_from_start:Q", scale=x_scale),
            y=alt.value(0), y2=alt.value(350),
        )
        cancel_rule = alt.Chart(ref_lines_df[ref_lines_df["days_from_start"] == -14]).mark_rule(
            color="grey", strokeWidth=1.5,
        ).encode(
            x=alt.X("days_from_start:Q", scale=x_scale),
            y=alt.value(0), y2=alt.value(350),
        )
        # Labels at the top of the reference lines
        ref_labels = alt.Chart(ref_lines_df).mark_text(
            align="left", dx=3, dy=0, fontSize=9, color="grey",
        ).encode(
            x=alt.X("days_from_start:Q", scale=x_scale),
            y=alt.value(8),
            text="label:N",
        )
        chart = chart + heads_up_rule + cancel_rule + ref_labels

        chart = chart.properties(
            height=500,
            title=f"Enrollment trajectories ({agg_choice} view)",
        ).interactive()

        # ---------- SUS snapshot summary ----------
        left, right = st.columns([3, 1])
        with left:
            st.altair_chart(chart, width="stretch")
        with right:
            st.subheader("Snapshot summary")
            latest_date = df_plot["snapshot_date"].max()
            st.write(f"Latest snapshot date: {latest_date.date()}")
            if agg_choice == "Section":
                latest = df_plot[df_plot["snapshot_date"] == latest_date][
                    ["section_label", "campus_simple", "enrolled", "capacity"]
                ].sort_values("section_label")
                latest = latest.rename(
                    columns={
                        "section_label": "Section",
                        "campus_simple": "Campus",
                        "enrolled": "Enrolled",
                        "capacity": "Capacity",
                    }
                )
            elif agg_choice == "Course":
                latest = df_plot[df_plot["snapshot_date"] == latest_date][
                    ["series_label", "enrolled", "capacity"]
                ].sort_values("series_label")
                latest = latest.rename(
                    columns={"series_label": "Course", "enrolled": "Enrolled", "capacity": "Capacity"}
                )
            else:
                latest = df_plot[df_plot["snapshot_date"] == latest_date][
                    ["series_label", "enrolled", "capacity"]
                ].sort_values("series_label")
                latest = latest.rename(
                    columns={"series_label": "Campus", "enrolled": "Enrolled", "capacity": "Capacity"}
                )
            st.dataframe(latest, width="stretch")

    # ---- CoreEd daily section ----
    st.markdown("---")
    st.header("CoreEd Daily Snapshot")

    if df_coreed.empty:
        st.info(
            "No CoreEd daily data available. "
            "Run osu_enrollment_snapshot_classes_api.py to populate."
        )
        return

    # Precompute classes_begin and days_from_start for CoreEd
    df_coreed = df_coreed.copy()
    df_coreed["classes_begin"] = df_coreed["term_srcdb"].apply(
        estimate_classes_begin_from_srcdb
    )
    df_coreed["snapshot_date"] = pd.to_datetime(
        df_coreed["snapshot_date"], errors="coerce"
    )
    df_coreed["days_from_start"] = (
        df_coreed["snapshot_date"] - df_coreed["classes_begin"]
    ).dt.days

    coreed_terms = sorted(
        df_coreed["term_srcdb"].dropna().unique().tolist(), reverse=True
    )
    # Default to the most recent term with non-zero enrollment
    coreed_default_idx = 0
    for i, t in enumerate(coreed_terms):
        term_enr = df_coreed.loc[df_coreed["term_srcdb"] == t, "enrolled"]
        if (term_enr > 0).any():
            coreed_default_idx = i
            break
    coreed_term_choice = st.sidebar.selectbox(
        "CoreEd term (srcdb)",
        coreed_terms,
        index=coreed_default_idx if coreed_terms else None,
    )
    include_labs_coreed = st.sidebar.checkbox(
        "CoreEd: Include lab/recitation sections", value=False
    )
    cas_only_coreed = st.sidebar.checkbox(
        "CoreEd: Show only CAS subjects",
        value=False,
        help="Filters to College of Agricultural Sciences subjects (e.g., AEC, ANS, HORT, SUS).",
    )

    # Limit CoreEd campus options to the main three campuses; treat "Other" as out-of-scope for charts
    available_coreed_campuses = [
        c
        for c in CAMPUS_ORDER
        if c in df_coreed["campus_simple"].unique()
        and c in ("Corvallis", "Ecampus", "Cascades")
    ]
    campus_choice_coreed = st.sidebar.multiselect(
        "CoreEd: Campus",
        options=available_coreed_campuses,
        default=available_coreed_campuses,
    )

    df_coreed_filt = df_coreed.copy()
    if coreed_term_choice:
        df_coreed_filt = df_coreed_filt[
            df_coreed_filt["term_srcdb"] == coreed_term_choice
        ]
    if not include_labs_coreed:
        df_coreed_filt = df_coreed_filt[~df_coreed_filt["is_lab"]]
    if cas_only_coreed:
        df_coreed_filt = df_coreed_filt[
            df_coreed_filt["subject"].isin(CAS_SUBJECT_CODES)
        ]
    if campus_choice_coreed:
        df_coreed_filt = df_coreed_filt[
            df_coreed_filt["campus_simple"].isin(campus_choice_coreed)
        ]
        campus_domain = campus_choice_coreed
    else:
        campus_domain = [
            c
            for c in CAMPUS_ORDER
            if c in df_coreed_filt["campus_simple"].unique()
        ]

    st.caption(
        f"Latest snapshot: {df_coreed_filt['snapshot_date'].max()} · source: coreed_daily_sections"
    )

    # Build a shared x-axis domain that covers both the SUS slider range
    # and the CoreEd data range, so CoreEd charts aren't clipped.
    coreed_days_domain = sus_days_domain
    if not df_coreed_filt.empty and sus_days_domain is not None:
        coreed_min = int(df_coreed_filt["days_from_start"].min())
        coreed_max = int(df_coreed_filt["days_from_start"].max())
        shared_min = min(sus_days_domain[0], coreed_min)
        shared_max = max(sus_days_domain[1], coreed_max)
        coreed_days_domain = (shared_min, shared_max)

    coreed_scope = "CAS" if cas_only_coreed else "OSU"
    coreed_left, coreed_right = st.columns([3, 1])
    with coreed_left:
        for cat in ["CFSI", "CSSS", "CSDP", "CFSS"]:
            if cat in df_coreed_filt["coreed_cat4"].unique():
                st.altair_chart(
                    category_timeseries_chart(
                        df_coreed_filt,
                        cat,
                        campus_domain=campus_domain,
                        days_domain=coreed_days_domain,
                        data_scope=coreed_scope,
                    ),
                    width="stretch",
                )
    with coreed_right:
        # leave empty for now, or you could put a small note/legend here later
        st.empty()

    # Summary table: total sections and full sections by category/campus (latest snapshot only).
    # For this table, we always include labs/recitations, regardless of the checkbox above.
    df_coreed_summary = df_coreed.copy()
    if coreed_term_choice:
        df_coreed_summary = df_coreed_summary[
            df_coreed_summary["term_srcdb"] == coreed_term_choice
        ]
    if cas_only_coreed:
        df_coreed_summary = df_coreed_summary[
            df_coreed_summary["subject"].isin(CAS_SUBJECT_CODES)
        ]
    if campus_choice_coreed:
        df_coreed_summary = df_coreed_summary[
            df_coreed_summary["campus_simple"].isin(campus_choice_coreed)
        ]

    if not df_coreed_summary.empty:
        df_coreed_summary = df_coreed_summary.copy()
        latest_coreed_date = df_coreed_summary["snapshot_date"].max()
        df_coreed_summary = df_coreed_summary[
            df_coreed_summary["snapshot_date"] == latest_coreed_date
        ]

        # Deduplicate sections by crn/campus/category; keep latest row with non-null capacity/enrolled
        df_coreed_summary = (
            df_coreed_summary.sort_values(["snapshot_date"])
            .drop_duplicates(
                subset=["coreed_cat4", "campus_simple", "crn"], keep="last"
            )
        )
        df_coreed_summary["campus_simple"] = df_coreed_summary[
            "campus_simple"
        ].astype(str)

        # Keep only rows with valid enrollment and capacity; compute is_full for summary + debug
        df_coreed_summary = (
            df_coreed_summary.dropna(subset=["enrolled", "capacity"])
            .query("capacity > 0")
            .copy()
        )
        df_coreed_summary["is_full"] = (
            df_coreed_summary["enrolled"] >= df_coreed_summary["capacity"]
        )

        # Aggregate to category/campus level
        summary = (
            df_coreed_summary.groupby(
                ["coreed_cat4", "campus_simple"], as_index=False
            ).agg(
                total_sections=("crn", "nunique"),
                full_sections=("is_full", "sum"),
            )
        )
        summary["percent_full"] = (
            summary["full_sections"] / summary["total_sections"] * 100
        ).round(1)
        summary["coreed_label"] = summary["coreed_cat4"].map(
            COREED_LABELS
        ).fillna(summary["coreed_cat4"])
        summary = summary[
            [
                "coreed_label",
                "campus_simple",
                "total_sections",
                "full_sections",
                "percent_full",
            ]
        ]

        # Optional debug: show section-level rows for ALL categories
        if st.sidebar.checkbox(
            "Show CoreEd full-section debug (all categories)", value=False
        ):
            st.subheader(
                "CoreEd raw rows used for full-section summary (latest snapshot)"
            )
            st.dataframe(
                df_coreed_summary[
                    [
                        "coreed_cat4",
                        "campus_simple",
                        "crn",
                        "enrolled",
                        "capacity",
                        "is_full",
                    ]
                ].sort_values(["coreed_cat4", "campus_simple", "crn"]),
                width="stretch",
            )

        # Optional debug: focus just on Seeking Solutions (CSSS)
        if st.sidebar.checkbox("Show Seeking Solutions (CSSS) detail", value=False):
            csss_detail = (
                df_coreed_summary[df_coreed_summary["coreed_cat4"] == "CSSS"][
                    [
                        "coreed_cat4",
                        "campus_simple",
                        "crn",
                        "enrolled",
                        "capacity",
                        "is_full",
                    ]
                ]
                .sort_values(["campus_simple", "crn"])
            )
            st.subheader("Seeking Solutions (CSSS) sections – latest snapshot")
            st.dataframe(csss_detail, width="stretch")

        def color_row(row):
            val = row.get("percent_full")
            if pd.isna(val):
                color = ""
            elif val >= 90:
                color = "background-color: #f8d7da"  # light red
            elif val >= 70:
                color = "background-color: #fff3cd"  # light orange/yellow
            else:
                color = "background-color: #d4edda"  # light green
            return [color] * len(row)

        styled = (
            summary.sort_values(["coreed_label", "campus_simple"])
            .style.apply(color_row, axis=1)
            .format({"percent_full": "{:.1f}"})
        )

        st.subheader("CoreEd sections by category and campus")
        st.dataframe(
            styled.set_properties(
                subset=["coreed_label"], **{"font-weight": "bold"}
            ),
            width="stretch",
        )

    # ---------- Debug: historic SUS features (bottom of page) ----------
    st.markdown("---")
    if not df_hist.empty:
        with st.expander("Debug: historic SUS rows by term & campus"):
            debug_counts = (
                df_hist
                .groupby(["term_srcdb", "campus_simple"], as_index=False)
                .size()
                .pivot(index="term_srcdb", columns="campus_simple", values="size")
                .fillna(0)
                .astype(int)
            )
            st.dataframe(debug_counts)

            st.caption(
                "Historic SUS table: "
                f"{len(df_hist)} rows, terms = "
                + ", ".join(sorted(map(str, df_hist["term_srcdb"].dropna().unique())))
            )

        with st.expander("Debug: historic SUS loader (section_num & campus)"):
            debug_cols = ["term_srcdb", "course_number", "section", "campus_simple", "enrolled"]
            if "section_num" in df_hist.columns:
                debug_cols.insert(3, "section_num")
            st.write(
                df_hist[debug_cols]
                .sort_values(["term_srcdb", "course_number", "section"])
                .head(40)
            )
            dbg = (
                df_hist.groupby(["term_srcdb", "campus_simple"], as_index=False)
                .size()
                .pivot(index="term_srcdb", columns="campus_simple", values="size")
                .fillna(0)
                .astype(int)
            )
            st.write("Row counts by term & campus (from loader):")
            st.dataframe(dbg)
    else:
        with st.expander("Debug: historic SUS data"):
            st.caption("No rows loaded from sus_daily_registrations_2025")


if __name__ == "__main__":
    main()