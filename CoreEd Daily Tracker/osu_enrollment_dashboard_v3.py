"""
osu_enrollment_dashboard_v3.py

SUS enrollment dashboard + CoreEd daily snapshot charts.
Keeps the SUS views from v2 and appends CoreEd timeseries (capacity/enrollment)
from the coreed_daily_sections table.
"""

import sqlite3
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

DB_PATH = Path("osu_enrollment_log_classes.db")
COREED_DAILY_TABLE = "coreed_daily_sections"

CAMPUS_ORDER = ["Corvallis", "Ecampus", "Cascades", "Other"]
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
        df = pd.read_sql_query("SELECT * FROM enrollment", conn)
    finally:
        conn.close()

    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    snap_db = pd.to_datetime(df["snapshot_date"], errors="coerce") if "snapshot_date" in df.columns else pd.Series(pd.NaT, index=df.index)
    snap_from_ts = df["timestamp"].dt.normalize()
    df["snapshot_date"] = snap_db.fillna(snap_from_ts)

    df["code"] = df["code"].astype(str)
    df["subject"] = df["subject"].fillna(df["code"].str.split().str[0])
    df["course_number"] = df["course_number"].fillna(df["code"].str.split().str[1])
    df["course_code"] = df["subject"].str.cat(df["course_number"], sep=" ")

    df["section"] = df["section"].astype(str)
    df["section_label"] = df["subject"].astype(str) + df["course_number"].astype(str) + "_" + df["section"].astype(str)

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
        return 10 <= n < 20

    df["is_lab"] = df["section"].apply(is_lab)
    df["enrolled"] = pd.to_numeric(df["enrolled"], errors="coerce")
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
    df: pd.DataFrame, category: str, campus_domain: list[str] | None = None
) -> alt.Chart:
    df_cat = df[df["coreed_cat4"] == category]
    if df_cat.empty:
        return alt.Chart(pd.DataFrame({"msg": ["No data"]})).mark_text().encode(text="msg")
    agg = (
        df_cat.groupby(["snapshot_date", "campus_simple"], as_index=False)[["enrolled", "capacity"]]
        .sum(min_count=1)
    )
    melted = agg.melt(
        id_vars=["snapshot_date", "campus_simple"],
        value_vars=["enrolled", "capacity"],
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])
    if melted.empty:
        return alt.Chart(pd.DataFrame({"msg": ["No data"]})).mark_text().encode(text="msg")
    melted["metric"] = melted["metric"].map({"enrolled": "Enrolled", "capacity": "Capacity"})
    domain = campus_domain or [c for c in CAMPUS_ORDER if c in melted["campus_simple"].unique()]
    melted["campus_simple"] = pd.Categorical(melted["campus_simple"], domain, ordered=True)
    return (
        alt.Chart(melted, title=COREED_LABELS.get(category, category))
        .mark_line(point=True)
        .encode(
            x=alt.X("snapshot_date:T", title="Snapshot date"),
            y=alt.Y("value:Q", title="Headcount"),
            color=alt.Color(
                "campus_simple:N",
                title="Campus",
                scale=alt.Scale(domain=domain, range=["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]),
            ),
            strokeDash=alt.StrokeDash("metric:N", title="Metric", sort=["Capacity", "Enrolled"]),
            tooltip=[
                alt.Tooltip("snapshot_date:T", title="Snapshot"),
                alt.Tooltip("campus_simple:N", title="Campus"),
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("value:Q", title="Value", format=",.0f"),
            ],
        )
        .properties(height=260)
    )


# ---------- Main ----------

def main():
    st.set_page_config(page_title="SUS Enrollment Dashboard", layout="wide")
    st.title("SUS Enrollment Dashboard")

    df_raw = load_sus_data()
    df_coreed = load_coreed_daily()
    if df_raw.empty:
        st.warning("No data available.")
        return

    # Sidebar filters (SUS)
    st.sidebar.header("Filters")
    term_values = sorted(df_raw["term_srcdb"].dropna().unique().tolist())
    term_options = ["All terms"] + term_values
    term_choice = st.sidebar.selectbox("Term (srcdb)", term_options, index=0)

    campus_values = sorted(df_raw["campus_simple"].dropna().unique().tolist())
    campus_selected = st.sidebar.multiselect("Campus", options=campus_values, default=campus_values)

    course_values = sorted(df_raw["course_code"].dropna().unique().tolist())
    course_selected = st.sidebar.multiselect("Courses", options=course_values, default=course_values)

    include_labs = st.sidebar.checkbox("Include labs (010, 011, ...)", value=False)
    agg_options = ["Section", "Course", "Campus"]
    agg_choice = st.sidebar.selectbox("Aggregation level", agg_options, index=0)
    show_labels = st.sidebar.checkbox("Show section labels on chart", value=False)

    # Apply SUS filters
    df = df_raw.copy()
    if term_choice != "All terms":
        df = df[df["term_srcdb"] == term_choice]
    if campus_selected:
        df = df[df["campus_simple"].isin(campus_selected)]
    if course_selected:
        df = df[df["course_code"].isin(course_selected)]
    if not include_labs:
        df = df[~df["is_lab"]]
    df = df.dropna(subset=["enrolled"])
    if df.empty:
        st.warning("No data after applying filters.")
        return

    # SUS aggregation logic
    if agg_choice == "Section":
        df_plot = (
            df.groupby(
                ["snapshot_date", "course_code", "section", "section_label", "campus_simple"],
                as_index=False,
            )["enrolled"]
            .max()
        )
        color_field = "section_label"
        y_title = "Enrollment (per section)"
    elif agg_choice == "Course":
        df_plot = df.groupby(["snapshot_date", "course_code"], as_index=False)["enrolled"].sum()
        df_plot["series_label"] = df_plot["course_code"]
        color_field = "series_label"
        y_title = "Enrollment (sum across sections)"
    else:
        df_plot = df.groupby(["snapshot_date", "campus_simple"], as_index=False)["enrolled"].sum()
        df_plot["series_label"] = df_plot["campus_simple"]
        color_field = "series_label"
        y_title = "Enrollment (sum across all courses)"

    # SUS chart
    if agg_choice == "Section":
        stroke_dash_scale = alt.Scale(
            domain=["Corvallis", "Ecampus", "Cascades", "Other"],
            range=[[1, 0], [4, 3], [2, 2], [1, 1]],
        )
        shape_scale = alt.Scale(
            domain=["Corvallis", "Ecampus", "Cascades", "Other"],
            range=["circle", "triangle-up", "square", "diamond"],
        )
        base = alt.Chart(df_plot).encode(
            x=alt.X("snapshot_date:T", title="Snapshot date", scale=alt.Scale(nice="day")),
            y=alt.Y("enrolled:Q", title=y_title),
            color=alt.Color(f"{color_field}:N", title="Section", legend=alt.Legend(orient="right")),
            strokeDash=alt.StrokeDash("campus_simple:N", scale=stroke_dash_scale, legend=None),
        )
        line_layer = base.mark_line()
        point_layer = base.mark_point(filled=True, size=60).encode(
            shape=alt.Shape("campus_simple:N", scale=shape_scale, legend=alt.Legend(title="Campus", orient="right"))
        )
        chart = line_layer + point_layer
        if show_labels:
            text_layer = base.mark_text(align="left", dx=4, dy=-2, fontSize=10).encode(text=alt.Text("section_label:N"))
            chart = chart + text_layer
    else:
        base = alt.Chart(df_plot).encode(
            x=alt.X("snapshot_date:T", title="Snapshot date", scale=alt.Scale(nice="day")),
            y=alt.Y("enrolled:Q", title=y_title),
            color=alt.Color(f"{color_field}:N", title=agg_choice, legend=alt.Legend(orient="right")),
        )
        chart = base.mark_line() + base.mark_point(filled=True, size=60)

    chart = chart.properties(height=500, title=f"Enrollment trajectories ({agg_choice} view)").interactive()

    left, right = st.columns([3, 1])
    with left:
        st.altair_chart(chart, width="stretch")
    with right:
        st.subheader("Snapshot summary")
        latest_date = df_plot["snapshot_date"].max()
        st.write(f"Latest snapshot date: {latest_date.date()}")
        if agg_choice == "Section":
            latest = df_plot[df_plot["snapshot_date"] == latest_date][
                ["section_label", "campus_simple", "enrolled"]
            ].sort_values("section_label")
            latest = latest.rename(columns={"section_label": "Section", "campus_simple": "Campus", "enrolled": "Enrolled"})
        elif agg_choice == "Course":
            latest = df_plot[df_plot["snapshot_date"] == latest_date][["series_label", "enrolled"]].sort_values("series_label")
            latest = latest.rename(columns={"series_label": "Course", "enrolled": "Enrolled"})
        else:
            latest = df_plot[df_plot["snapshot_date"] == latest_date][["series_label", "enrolled"]].sort_values("series_label")
            latest = latest.rename(columns={"series_label": "Campus", "enrolled": "Enrolled"})
        st.dataframe(latest, use_container_width=True)

    # ---- CoreEd daily section ----
    st.markdown("---")
    st.header("CoreEd Daily Snapshot (from coreed_daily_sections)")

    if df_coreed.empty:
        st.info("No CoreEd daily data available. Run osu_enrollment_snapshot_classes_api_v2.py to populate.")
        return

    coreed_terms = sorted(df_coreed["term_srcdb"].dropna().unique().tolist(), reverse=True)
    coreed_term_choice = st.sidebar.selectbox(
        "CoreEd term (srcdb)",
        coreed_terms,
        index=0 if coreed_terms else None,
    )
    include_labs_coreed = st.sidebar.checkbox("CoreEd: Include lab/recitation sections", value=False)
    available_coreed_campuses = [c for c in CAMPUS_ORDER if c in df_coreed["campus_simple"].unique()]
    campus_choice_coreed = st.sidebar.multiselect(
        "CoreEd: Campus",
        options=available_coreed_campuses,
        default=available_coreed_campuses,
    )

    df_coreed_filt = df_coreed.copy()
    if coreed_term_choice:
        df_coreed_filt = df_coreed_filt[df_coreed_filt["term_srcdb"] == coreed_term_choice]
    if not include_labs_coreed:
        df_coreed_filt = df_coreed_filt[~df_coreed_filt["is_lab"]]
    if campus_choice_coreed:
        df_coreed_filt = df_coreed_filt[df_coreed_filt["campus_simple"].isin(campus_choice_coreed)]
        campus_domain = campus_choice_coreed
    else:
        campus_domain = [c for c in CAMPUS_ORDER if c in df_coreed_filt["campus_simple"].unique()]

    st.caption(f"CoreEd latest snapshot date: {df_coreed_filt['snapshot_date'].max()}")

    for cat in ["CFSI", "CSSS", "CSDP", "CFSS"]:
        if cat in df_coreed_filt["coreed_cat4"].unique():
            st.altair_chart(
                category_timeseries_chart(df_coreed_filt, cat, campus_domain=campus_domain),
                use_container_width=True,
            )

    # Summary table: total sections and full sections by category/campus
    if not df_coreed_filt.empty:
        df_coreed_filt = df_coreed_filt.copy()
        df_coreed_filt["campus_simple"] = df_coreed_filt["campus_simple"].astype(str)
        summary = (
            df_coreed_filt.dropna(subset=["enrolled", "capacity"])
            .assign(is_full=lambda d: d["enrolled"] >= d["capacity"])
            .groupby(["coreed_cat4", "campus_simple"], as_index=False)
            .agg(
                total_sections=("crn", "nunique"),
                full_sections=("is_full", "sum"),
            )
        )
        summary["percent_full"] = (summary["full_sections"] / summary["total_sections"] * 100).round(1)

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
            summary.sort_values(["coreed_cat4", "campus_simple"])
            .style.apply(color_row, axis=1)
            .format({"percent_full": "{:.1f}"})
        )

        st.subheader("CoreEd sections by category and campus")
        st.dataframe(styled, use_container_width=True)


if __name__ == "__main__":
    main()
