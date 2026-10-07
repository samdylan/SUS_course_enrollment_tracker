"""
Analysis A — SUS 102 and 103 demand.

Enrollment by term and campus for 102, 103 (incl. 103H), and 102+103 combined,
with pre/post 202600 comparison by matching term of year.

Key question: is the combined 102+103 total stable (substitution within SUS,
expected under Core Ed's same-designator rule) or falling (students leaving SUS)?
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_load import (
    OUT_DIR, TERM_META, load_sus_census, season_of, term_label, term_sort_key,
    is_post_coreed,
)


def run():
    sus = load_sus_census()

    # Focus on 102, 103, 103H
    courses = sus[sus["course_number"].isin(["102", "103", "103H"])].copy()
    # Group 103 and 103H together as "103 (incl. H)"
    courses["course_group"] = courses["course_number"].replace({"103H": "103"})

    # ---------- Table A1: enrollment by term x course_group x campus ----------
    tbl = (
        courses.groupby(["term_srcdb", "course_group", "campus_simple"], as_index=False)
        .agg(enrolled=("enrolled", "sum"), sections=("section", "nunique"))
    )
    tbl["term_label"] = tbl["term_srcdb"].map(term_label)
    tbl["season"] = tbl["term_srcdb"].map(season_of)
    tbl["coreed_era"] = tbl["term_srcdb"].map(lambda t: "post" if is_post_coreed(t) else "pre")
    tbl = tbl.sort_values(by=["term_srcdb", "course_group", "campus_simple"])
    tbl.to_csv(OUT_DIR / "A1_sus_102_103_by_term_campus.csv", index=False)

    # ---------- Table A2: 102+103 combined totals (across campuses) ----------
    totals = (
        courses.groupby(["term_srcdb", "course_group"], as_index=False)
        .agg(enrolled=("enrolled", "sum"), sections=("section", "nunique"))
    )
    pivot = totals.pivot(index="term_srcdb", columns="course_group", values="enrolled").fillna(0)
    pivot["combined_102_103"] = pivot.get("102", 0) + pivot.get("103", 0)
    pivot["term_label"] = pd.Series(pivot.index).map(term_label).values
    pivot["season"] = pd.Series(pivot.index).map(season_of).values
    pivot["coreed_era"] = pd.Series(pivot.index).map(lambda t: "post" if is_post_coreed(t) else "pre").values
    # Also attach sections
    sec_pivot = totals.pivot(index="term_srcdb", columns="course_group", values="sections").fillna(0)
    sec_pivot.columns = [f"{c}_sections" for c in sec_pivot.columns]
    pivot = pivot.join(sec_pivot)
    pivot = pivot.reset_index()
    pivot = pivot.sort_values("term_srcdb")
    pivot.to_csv(OUT_DIR / "A2_sus_102_103_combined_by_term.csv", index=False)

    # ---------- Table A3: pre/post 202600 by season (term-of-year matched) ----------
    # For each season, aggregate pre- and post-cutover terms and compute change.
    rows = []
    for season in ["Fall", "Winter", "Spring", "Summer"]:
        season_df = totals[totals["term_srcdb"].map(season_of) == season].copy()
        season_df["coreed_era"] = season_df["term_srcdb"].map(
            lambda t: "post" if is_post_coreed(t) else "pre"
        )
        for course in ["102", "103"]:
            cdf = season_df[season_df["course_group"] == course]
            pre_terms = cdf[cdf["coreed_era"] == "pre"]
            post_terms = cdf[cdf["coreed_era"] == "post"]
            rows.append({
                "season": season,
                "course": course,
                "pre_terms": ", ".join(sorted(pre_terms["term_srcdb"].astype(str).tolist())),
                "post_terms": ", ".join(sorted(post_terms["term_srcdb"].astype(str).tolist())),
                "pre_mean_enrolled": pre_terms["enrolled"].mean(),
                "post_mean_enrolled": post_terms["enrolled"].mean(),
                "pre_mean_sections": pre_terms["sections"].mean(),
                "post_mean_sections": post_terms["sections"].mean(),
            })
        # Combined row
        combo_pre = pre_terms["enrolled"].sum() if False else None  # placeholder
    # Combined (102+103) by season/era
    combo_rows = []
    for season in ["Fall", "Winter", "Spring", "Summer"]:
        pivot_season = pivot[pivot["season"] == season]
        pre = pivot_season[pivot_season["coreed_era"] == "pre"]
        post = pivot_season[pivot_season["coreed_era"] == "post"]
        combo_rows.append({
            "season": season,
            "pre_terms": ", ".join(pre["term_srcdb"].astype(str).tolist()),
            "post_terms": ", ".join(post["term_srcdb"].astype(str).tolist()),
            "pre_mean_102": pre.get("102", pd.Series([0])).mean() if "102" in pre.columns else 0,
            "post_mean_102": post.get("102", pd.Series([0])).mean() if "102" in post.columns else 0,
            "pre_mean_103": pre.get("103", pd.Series([0])).mean() if "103" in pre.columns else 0,
            "post_mean_103": post.get("103", pd.Series([0])).mean() if "103" in post.columns else 0,
            "pre_mean_combined": pre["combined_102_103"].mean(),
            "post_mean_combined": post["combined_102_103"].mean(),
        })
    pd.DataFrame(rows).to_csv(OUT_DIR / "A3_pre_post_by_course.csv", index=False)
    pd.DataFrame(combo_rows).to_csv(OUT_DIR / "A3_pre_post_combined.csv", index=False)

    # ---------- Table A4: direct Fall-to-Fall comparison (three Falls available) ----------
    fall_terms = [t for t in TERM_META if season_of(t) == "Fall" and t in pivot["term_srcdb"].values]
    fall_df = pivot[pivot["term_srcdb"].isin(fall_terms)].copy()
    fall_df = fall_df.sort_values("term_srcdb")
    fall_df.to_csv(OUT_DIR / "A4_fall_year_over_year.csv", index=False)

    # ---------- Chart A1: 102, 103, combined trajectory over terms ----------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ordered = pivot.sort_values("term_srcdb")
    x = np.arange(len(ordered))
    ax.plot(x, ordered["102"].values, "o-", label="SUS 102", color="#4c78a8", linewidth=2)
    ax.plot(x, ordered["103"].values, "s-", label="SUS 103 (incl. 103H)", color="#e45756", linewidth=2)
    ax.plot(x, ordered["combined_102_103"].values, "D--", label="102 + 103 combined",
            color="#333333", linewidth=2.5)
    # Shade post-202600
    post_mask = ordered["coreed_era"] == "post"
    first_post = next((i for i, v in enumerate(post_mask) if v), None)
    if first_post is not None:
        ax.axvspan(first_post - 0.5, len(ordered) - 0.5, alpha=0.08, color="green",
                   label="Core Ed era (202600+)")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["term_label"].values, rotation=45, ha="right")
    ax.set_ylabel("Enrolled (approx day-20 census)")
    ax.set_title("SUS 102 and 103 enrollment by term\n(sum across campuses and sections)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "A1_chart_sus_102_103_trajectory.png", dpi=150)
    plt.close(fig)

    # ---------- Chart A2: Fall-to-Fall comparison stacked bars ----------
    fig, ax = plt.subplots(figsize=(8, 5))
    fall_df_sorted = fall_df.sort_values("term_srcdb")
    x = np.arange(len(fall_df_sorted))
    bar_width = 0.6
    bar1 = ax.bar(x, fall_df_sorted["102"].values, width=bar_width,
                  label="SUS 102", color="#4c78a8")
    bar2 = ax.bar(x, fall_df_sorted["103"].values, width=bar_width,
                  bottom=fall_df_sorted["102"].values,
                  label="SUS 103 (incl. 103H)", color="#e45756")
    totals_vals = fall_df_sorted["combined_102_103"].values
    for i, (val, tv) in enumerate(zip(totals_vals, totals_vals)):
        ax.text(i, tv + max(totals_vals) * 0.02, f"{int(tv)}",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(fall_df_sorted["term_label"].values)
    ax.set_ylabel("Enrolled (approx day-20)")
    ax.set_title("Fall-to-Fall: SUS 102 and 103 (stacked = combined)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "A2_chart_fall_year_over_year.png", dpi=150)
    plt.close(fig)

    # ---------- Chart A3: campus breakdown by term for 102 and 103 ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    campus_order = ["Corvallis", "Ecampus", "Cascades"]
    campus_colors = {"Corvallis": "#4c78a8", "Ecampus": "#e45756", "Cascades": "#72b7b2"}

    for ax, course in zip(axes, ["102", "103"]):
        cdf = tbl[tbl["course_group"] == course].copy()
        all_terms = sorted(cdf["term_srcdb"].unique(), key=term_sort_key)
        wide = cdf.pivot_table(index="term_srcdb", columns="campus_simple",
                               values="enrolled", aggfunc="sum").reindex(all_terms).fillna(0)
        x = np.arange(len(all_terms))
        bottom = np.zeros(len(all_terms))
        for campus in campus_order:
            if campus in wide.columns:
                vals = wide[campus].values
                ax.bar(x, vals, bottom=bottom, label=campus, color=campus_colors[campus])
                bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels([term_label(t) for t in all_terms], rotation=45, ha="right")
        ax.set_title(f"SUS {course} by campus")
        ax.set_ylabel("Enrolled")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("SUS 102 & 103 enrollment by campus and term")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "A3_chart_campus_breakdown.png", dpi=150)
    plt.close(fig)

    # Return summary stats for later use
    print("Analysis A complete.")
    print(pivot[["term_label", "102", "103", "combined_102_103", "coreed_era"]].to_string(index=False))
    return pivot


if __name__ == "__main__":
    run()
