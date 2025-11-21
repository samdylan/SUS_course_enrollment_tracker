"""
osu_enrollment_dashboard.py

Streamlit dashboard for visualizing OSU SUS course enrollments over time.

Key features:
- Reads from osu_enrollment_log_classes.db (table 'enrollment')
- Filters by term, campus, course, lab vs non-lab, and aggregation level
- Aggregation options: Section, Course, Campus
- Line chart:
    * color by section or aggregate key
    * line style (strokeDash) and point shape by campus (for Section view)
- Optional short section labels on the chart (e.g. 'SUS102_001')
"""

import sqlite3
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


DB_PATH = Path("osu_enrollment_log_classes.db")


def load_data() -> pd.DataFrame:
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

    # Parse timestamp to datetime and derive snapshot_date
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["snapshot_date"] = df["timestamp"].dt.normalize()

    # Normalized course fields from code, which is like 'SUS 102' or 'SUS 230X'
    df["code"] = df["code"].astype(str)
    df["subject"] = df["subject"].fillna(df["code"].str.split().str[0])
    df["course_number"] = df["course_number"].fillna(df["code"].str.split().str[1])

    df["course_code"] = df["subject"].str.cat(df["course_number"], sep=" ")

    # Short section label: SUS102_001, SUS230X_400, etc.
    df["section"] = df["section"].astype(str)
    df["section_label"] = (
        df["subject"].astype(str)
        + df["course_number"].astype(str)
        + "_"
        + df["section"].astype(str)
    )

    # Campus simplification
    df["campus_code"] = df["campus_code"].astype(str)

    def map_campus(c):
        c = (c or "").upper()
        if c in ("C", ""):
            return "Corvallis"
        if c in ("B",):
            return "Cascades"
        if c in ("DI", "DB", "E"):  # Distance/online
            return "Ecampus"
        return "Other"

    df["campus_simple"] = df["campus_code"].apply(map_campus)

    # labs: sections starting with 01x
    def is_lab(section: str) -> bool:
        s = str(section).strip()
        if len(s) != 3:
            return False
        if not s.isdigit():
            return False
        n = int(s)
        return 10 <= n < 20

    df["is_lab"] = df["section"].apply(is_lab)

    # Ensure enrolled is numeric
    df["enrolled"] = pd.to_numeric(df["enrolled"], errors="coerce")

    return df


def main():
    st.set_page_config(page_title="SUS Enrollment Dashboard", layout="wide")
    st.title("SUS Enrollment Dashboard")

    df_raw = load_data()
    if df_raw.empty:
        st.warning("No data available.")
        return

    # Sidebar filters
    st.sidebar.header("Filters")

    # Term filter
    term_values = sorted(df_raw["term_srcdb"].dropna().unique().tolist())
    term_options = ["All terms"] + term_values
    term_choice = st.sidebar.selectbox("Term (srcdb)", term_options, index=0)

    # Campus filter
    campus_values = sorted(df_raw["campus_simple"].dropna().unique().tolist())
    campus_selected = st.sidebar.multiselect(
        "Campus",
        options=campus_values,
        default=campus_values,
    )

    # Course filter
    course_values = sorted(df_raw["course_code"].dropna().unique().tolist())
    course_selected = st.sidebar.multiselect(
        "Courses",
        options=course_values,
        default=course_values,
    )

    # Lab filter
    include_labs = st.sidebar.checkbox("Include labs (010, 011, ...)", value=False)

    # Aggregation level
    agg_options = ["Section", "Course", "Campus"]
    agg_choice = st.sidebar.selectbox("Aggregation level", agg_options, index=0)

    # Show section labels toggle (only relevant in Section view)
    show_labels = st.sidebar.checkbox("Show section labels on chart", value=False)

    # --- Apply filters ---
    df = df_raw.copy()

    if term_choice != "All terms":
        df = df[df["term_srcdb"] == term_choice]

    if campus_selected:
        df = df[df["campus_simple"].isin(campus_selected)]

    if course_selected:
        df = df[df["course_code"].isin(course_selected)]

    if not include_labs:
        df = df[~df["is_lab"]]

    # Drop rows with missing enrolled
    df = df.dropna(subset=["enrolled"])

    if df.empty:
        st.warning("No data after applying filters.")
        return

    # --- Aggregation logic ---
    if agg_choice == "Section":
        # One line per section
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
        # Sum across sections; one line per course_code (ignoring campus)
        df_plot = (
            df.groupby(
                ["snapshot_date", "course_code"],
                as_index=False,
            )["enrolled"]
            .sum()
        )
        df_plot["series_label"] = df_plot["course_code"]
        color_field = "series_label"
        y_title = "Enrollment (sum across sections)"

    else:  # "Campus"
        df_plot = (
            df.groupby(
                ["snapshot_date", "campus_simple"],
                as_index=False,
            )["enrolled"]
            .sum()
        )
        df_plot["series_label"] = df_plot["campus_simple"]
        color_field = "series_label"
        y_title = "Enrollment (sum across all courses)"

    # --- Build chart ---
    if agg_choice == "Section":
        # Section view: color by section_label, line style + shape by campus
        stroke_dash_scale = alt.Scale(
            domain=["Corvallis", "Ecampus", "Cascades", "Other"],
            range=[[1, 0], [4, 3], [2, 2], [1, 1]],
        )

        shape_scale = alt.Scale(
            domain=["Corvallis", "Ecampus", "Cascades", "Other"],
            range=["circle", "triangle-up", "square", "diamond"],
        )

        base = alt.Chart(df_plot).encode(
            x=alt.X(
                "snapshot_date:T",
                title="Snapshot date",
                scale=alt.Scale(nice="day"),
            ),
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
        )

        line_layer = base.mark_line()

        point_layer = base.mark_point(filled=True, size=60).encode(
            shape=alt.Shape(
                "campus_simple:N",
                scale=shape_scale,
                legend=alt.Legend(title="Campus", orient="right"),
            )
        )

        chart = line_layer + point_layer

        if show_labels:
            text_layer = base.mark_text(
                align="left",
                dx=4,
                dy=-2,
                fontSize=10,
            ).encode(
                text=alt.Text("section_label:N")
            )
            chart = chart + text_layer

    else:
        # Aggregated views: one line per series_label, solid lines, circles
        base = alt.Chart(df_plot).encode(
            x=alt.X(
                "snapshot_date:T",
                title="Snapshot date",
                scale=alt.Scale(nice="day"),
            ),
            y=alt.Y("enrolled:Q", title=y_title),
            color=alt.Color(
                f"{color_field}:N",
                title=agg_choice,
                legend=alt.Legend(orient="right"),
            ),
        )

        line_layer = base.mark_line()
        point_layer = base.mark_point(filled=True, size=60)

        chart = line_layer + point_layer

    chart = chart.properties(
        height=500,
        title=f"Enrollment trajectories ({agg_choice} view)",
    ).interactive()

    # Layout
    left, right = st.columns([3, 1])

    with left:
        st.altair_chart(chart, width="stretch")

    with right:
        st.subheader("Snapshot summary")
        # Small summary table for the latest snapshot per section / series
        latest_date = df_plot["snapshot_date"].max()
        st.write(f"Latest snapshot date: {latest_date.date()}")

        if agg_choice == "Section":
            latest = df_plot[df_plot["snapshot_date"] == latest_date][
                ["section_label", "campus_simple", "enrolled"]
            ].sort_values("section_label")
            latest = latest.rename(
                columns={
                    "section_label": "Section",
                    "campus_simple": "Campus",
                    "enrolled": "Enrolled",
                }
            )
        elif agg_choice == "Course":
            latest = df_plot[df_plot["snapshot_date"] == latest_date][
                ["series_label", "enrolled"]
            ].sort_values("series_label")
            latest = latest.rename(
                columns={
                    "series_label": "Course",
                    "enrolled": "Enrolled",
                }
            )
        else:
            latest = df_plot[df_plot["snapshot_date"] == latest_date][
                ["series_label", "enrolled"]
            ].sort_values("series_label")
            latest = latest.rename(
                columns={
                    "series_label": "Campus",
                    "enrolled": "Enrolled",
                }
            )

        st.dataframe(latest, use_container_width=True)


if __name__ == "__main__":
    main()
