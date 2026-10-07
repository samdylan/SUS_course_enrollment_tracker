"""
Analysis E — SIA supply diversity and the long tail.

The headline SIA numbers (seats, enrollment) are dominated by a handful of
giant courses (BI 221Z, CH 121, PH 201). This analysis asks what the
remaining supply looks like once those are removed:
  - How many distinct courses and subjects are offered each term?
  - What is the distribution of per-course capacity?
  - Head (top N by seats) vs tail: how much supply, how many choices, and
    what is the average/median seat count in the tail?
  - Is the tail growing (more variety) or shrinking (consolidation into
    the giants)?
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_load import OUT_DIR, load_sia_census, term_label, term_sort_key, season_of

HEAD_TOP_N = 5
TAIL_LARGE_SEAT_THRESHOLD = 300  # alternative definition of "head": seats >= this


def _course_level_table(sia: pd.DataFrame) -> pd.DataFrame:
    """Collapse section-level census to course-level per term."""
    df = (
        sia.groupby(["term_srcdb", "subject", "course_code"], as_index=False)
        .agg(
            n_sections=("crn", "nunique"),
            seats=("capacity", "sum"),
            enrolled=("enrolled", "sum"),
        )
    )
    df["term_label"] = df["term_srcdb"].map(term_label)
    return df


def run():
    sia = load_sia_census()
    if sia.empty:
        print("No SIA data available.")
        return

    courses = _course_level_table(sia)

    # Drop the dummy / unknown rows if any
    courses = courses[courses["course_code"].notna()]
    courses = courses[~courses["course_code"].str.startswith("None")]

    # ---------- E1: Diversity by term ----------
    diversity_rows = []
    for term, g in courses.groupby("term_srcdb"):
        diversity_rows.append({
            "term_srcdb": term,
            "term_label": term_label(term),
            "n_courses": g["course_code"].nunique(),
            "n_subjects": g["subject"].nunique(),
            "n_sections_total": int(g["n_sections"].sum()),
            "seats_total": int(g["seats"].sum()),
            "enrolled_total": int(g["enrolled"].sum()),
            # Herfindahl of course concentration (0=even, 1=all one course)
            "course_concentration_hhi": _hhi(g["seats"].values),
            "subject_concentration_hhi": _hhi(
                g.groupby("subject")["seats"].sum().values
            ),
        })
    diversity = pd.DataFrame(diversity_rows).sort_values("term_srcdb")
    diversity.to_csv(OUT_DIR / "E1_diversity_by_term.csv", index=False)

    # ---------- E2: Per-course seat distribution by term (percentiles) ----------
    dist_rows = []
    for term, g in courses.groupby("term_srcdb"):
        seats = g["seats"].dropna().values
        if len(seats) == 0:
            continue
        dist_rows.append({
            "term_srcdb": term,
            "term_label": term_label(term),
            "n_courses": len(seats),
            "mean_seats_per_course": seats.mean(),
            "median_seats_per_course": np.median(seats),
            "p25_seats": np.percentile(seats, 25),
            "p75_seats": np.percentile(seats, 75),
            "max_seats": seats.max(),
            "max_course": g.loc[g["seats"] == seats.max(), "course_code"].iloc[0],
        })
    dist = pd.DataFrame(dist_rows).sort_values("term_srcdb")
    dist.to_csv(OUT_DIR / "E2_capacity_distribution_by_term.csv", index=False)

    # ---------- E3: Head vs tail (top-N by seats) ----------
    headtail_rows = []
    tail_detail = []
    for term, g in courses.groupby("term_srcdb"):
        g_sorted = g.sort_values("seats", ascending=False).reset_index(drop=True)
        total_seats = g_sorted["seats"].sum()
        total_enr = g_sorted["enrolled"].sum()
        head = g_sorted.head(HEAD_TOP_N)
        tail = g_sorted.iloc[HEAD_TOP_N:]
        headtail_rows.append({
            "term_srcdb": term,
            "term_label": term_label(term),
            "head_n_courses": len(head),
            "head_courses": "; ".join(head["course_code"].tolist()),
            "head_seats": int(head["seats"].sum()),
            "head_enrolled": int(head["enrolled"].sum()),
            "head_share_seats": 100.0 * head["seats"].sum() / total_seats if total_seats else np.nan,
            "head_share_enrolled": 100.0 * head["enrolled"].sum() / total_enr if total_enr else np.nan,
            "tail_n_courses": len(tail),
            "tail_n_subjects": tail["subject"].nunique(),
            "tail_seats": int(tail["seats"].sum()),
            "tail_enrolled": int(tail["enrolled"].sum()),
            "tail_mean_seats_per_course": tail["seats"].mean() if len(tail) else np.nan,
            "tail_median_seats_per_course": np.median(tail["seats"]) if len(tail) else np.nan,
            "tail_mean_enrolled_per_course": tail["enrolled"].mean() if len(tail) else np.nan,
        })
        for _, row in tail.iterrows():
            tail_detail.append({
                "term_srcdb": term,
                "term_label": term_label(term),
                "course_code": row["course_code"],
                "subject": row["subject"],
                "n_sections": row["n_sections"],
                "seats": row["seats"],
                "enrolled": row["enrolled"],
            })
    headtail = pd.DataFrame(headtail_rows).sort_values("term_srcdb")
    headtail.to_csv(OUT_DIR / "E3_head_vs_tail.csv", index=False)
    pd.DataFrame(tail_detail).to_csv(OUT_DIR / "E3_tail_courses_detail.csv", index=False)

    # ---------- E4: Diversity trend chart ----------
    terms_ordered = sorted(diversity["term_srcdb"].unique(), key=term_sort_key)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(terms_ordered))
    nc = [diversity.loc[diversity["term_srcdb"] == t, "n_courses"].iloc[0] for t in terms_ordered]
    ns = [diversity.loc[diversity["term_srcdb"] == t, "n_subjects"].iloc[0] for t in terms_ordered]
    bar1 = ax1.bar(x - 0.2, nc, width=0.4, label="# distinct SIA courses", color="#4c78a8")
    bar2 = ax1.bar(x + 0.2, ns, width=0.4, label="# distinct SIA subject codes", color="#72b7b2")
    for b, val in list(zip(bar1, nc)) + list(zip(bar2, ns)):
        ax1.text(b.get_x() + b.get_width() / 2, val + 0.5, str(int(val)),
                 ha="center", fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels([term_label(t) for t in terms_ordered], rotation=45, ha="right")
    ax1.set_ylabel("Count")
    ax1.set_title("SIA diversity by term: courses and subject codes offered")
    ax1.legend(loc="upper left")
    ax1.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "E4_chart_diversity_trend.png", dpi=150)
    plt.close(fig)

    # ---------- E5: Capacity distribution box plot by term ----------
    data_by_term = []
    labels = []
    for t in terms_ordered:
        if t == "202702":  # no enrollment yet, skip boxplot
            continue
        vals = courses.loc[courses["term_srcdb"] == t, "seats"].values
        if len(vals):
            data_by_term.append(vals)
            labels.append(term_label(t))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bp = ax.boxplot(data_by_term, labels=labels, showfliers=True, patch_artist=True,
                    boxprops=dict(facecolor="#b0c4de", edgecolor="black"),
                    medianprops=dict(color="black", linewidth=2))
    # Overlay head labels for the top course per term
    for i, t in enumerate([t for t in terms_ordered if t != "202702"]):
        g = courses[courses["term_srcdb"] == t].sort_values("seats", ascending=False)
        if len(g):
            top = g.iloc[0]
            ax.annotate(top["course_code"], xy=(i + 1, top["seats"]),
                        xytext=(5, 0), textcoords="offset points",
                        fontsize=7, color="darkred", alpha=0.8)
    ax.set_ylabel("Seats per course (capacity)")
    ax.set_title("Per-course capacity distribution by term\n"
                 "(each box is one term; outliers at top are the giants: BI 221Z, PH 201, CH 121)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "E5_chart_capacity_boxplot.png", dpi=150)
    plt.close(fig)

    # ---------- E6: Head vs tail stacked share bar ----------
    terms_ht = sorted(headtail["term_srcdb"].unique(), key=term_sort_key)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(terms_ht))
    head_sh = [headtail.loc[headtail["term_srcdb"] == t, "head_share_seats"].iloc[0]
               for t in terms_ht]
    tail_sh = [100 - h for h in head_sh]
    ax.bar(x, head_sh, color="#4c78a8", label=f"Top {HEAD_TOP_N} courses (seats share)")
    ax.bar(x, tail_sh, bottom=head_sh, color="#f5c57a",
           label="All other SIA courses (tail)")
    for i, h in enumerate(head_sh):
        ax.text(i, h / 2, f"{h:.0f}%", ha="center", color="white", fontweight="bold")
        ax.text(i, h + (100 - h) / 2, f"{100 - h:.0f}%", ha="center",
                color="black", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([term_label(t) for t in terms_ht], rotation=45, ha="right")
    ax.set_ylabel("Share of SIA seats (%)")
    ax.set_title(f"SIA capacity concentration: top {HEAD_TOP_N} courses vs the rest")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "E6_chart_head_vs_tail.png", dpi=150)
    plt.close(fig)

    # ---------- Console summary ----------
    print("E1 — Diversity by term:")
    print(diversity[["term_label", "n_courses", "n_subjects", "n_sections_total",
                     "seats_total", "course_concentration_hhi"]].to_string(index=False))
    print("\nE2 — Capacity distribution (seats per course):")
    print(dist[["term_label", "n_courses", "mean_seats_per_course",
                "median_seats_per_course", "p25_seats", "p75_seats",
                "max_seats", "max_course"]].to_string(index=False))
    print(f"\nE3 — Head (top {HEAD_TOP_N}) vs tail:")
    print(headtail[["term_label", "head_courses", "head_share_seats",
                    "tail_n_courses", "tail_n_subjects", "tail_seats",
                    "tail_mean_seats_per_course",
                    "tail_median_seats_per_course"]].to_string(index=False))

def _hhi(values: np.ndarray) -> float:
    """Herfindahl-Hirschman Index normalized to [0, 1]."""
    values = np.asarray(values, dtype=float)
    total = values.sum()
    if total <= 0:
        return float("nan")
    shares = values / total
    return float((shares ** 2).sum())


if __name__ == "__main__":
    run()
