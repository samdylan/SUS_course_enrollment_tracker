"""
Analysis C — SIA (Scientific Inquiry & Analysis / CFSI) supply over time.

By term and campus: SIA courses, sections, seats (capacity), enrollment,
overall fill rate. Courses entering or leaving the category. SUS share of SIA
enrollment, and shares of close substitutes.

Note: SIA coverage begins Fall 2025 (202601). No pre-202600 baseline available.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_load import OUT_DIR, load_sia_census, term_label, term_sort_key

CLOSE_SUBSTITUTES = {
    "BI 101": "BI 101",
    "CLIM 201": "CLIM 201",
    "ENSC 210": "ENSC 210",
    "GEO 100": "GEO 100",
    "GEO 101": "GEO 101",
    "GEO 202": "GEO 202",
    "GEOG 102": "GEOG 102",
    "H 244": "H 244",
    "OC 101": "OC 101",
}


def run():
    sia = load_sia_census()
    if sia.empty:
        print("No SIA data available.")
        return

    # ---------- Table C1: SIA totals by term and campus ----------
    tot_by_term_campus = (
        sia.groupby(["term_srcdb", "campus_simple"], as_index=False)
        .agg(
            n_courses=("course_code", "nunique"),
            n_sections=("crn", "nunique"),
            seats=("capacity", "sum"),
            enrolled=("enrolled", "sum"),
        )
    )
    tot_by_term_campus["fill_pct"] = (
        100.0 * tot_by_term_campus["enrolled"] / tot_by_term_campus["seats"]
    ).replace([np.inf, -np.inf], np.nan)
    tot_by_term_campus["term_label"] = tot_by_term_campus["term_srcdb"].map(term_label)
    tot_by_term_campus = tot_by_term_campus.sort_values(["term_srcdb", "campus_simple"])
    tot_by_term_campus.to_csv(OUT_DIR / "C1_sia_totals_by_term_campus.csv", index=False)

    # ---------- Table C2: SIA totals by term (all campuses) ----------
    tot_by_term = (
        sia.groupby("term_srcdb", as_index=False)
        .agg(
            n_courses=("course_code", "nunique"),
            n_sections=("crn", "nunique"),
            seats=("capacity", "sum"),
            enrolled=("enrolled", "sum"),
        )
    )
    tot_by_term["fill_pct"] = 100.0 * tot_by_term["enrolled"] / tot_by_term["seats"]
    tot_by_term["term_label"] = tot_by_term["term_srcdb"].map(term_label)
    tot_by_term = tot_by_term.sort_values("term_srcdb")
    tot_by_term.to_csv(OUT_DIR / "C2_sia_totals_by_term.csv", index=False)

    # ---------- Table C3: SIA by subject code ----------
    by_subj_term = (
        sia.groupby(["term_srcdb", "subject"], as_index=False)
        .agg(
            n_courses=("course_code", "nunique"),
            n_sections=("crn", "nunique"),
            seats=("capacity", "sum"),
            enrolled=("enrolled", "sum"),
        )
    )
    by_subj_term["term_label"] = by_subj_term["term_srcdb"].map(term_label)
    by_subj_term.to_csv(OUT_DIR / "C3_sia_by_subject_term.csv", index=False)

    # ---------- Table C4: course entry / exit across terms ----------
    courses_by_term = {
        t: set(sia.loc[sia["term_srcdb"] == t, "course_code"].unique())
        for t in sorted(sia["term_srcdb"].unique())
    }
    all_courses = sorted(set().union(*courses_by_term.values()))
    presence = pd.DataFrame(
        {t: [1 if c in courses_by_term[t] else 0 for c in all_courses]
         for t in courses_by_term},
        index=all_courses,
    )
    presence.index.name = "course_code"
    presence.to_csv(OUT_DIR / "C4_course_presence_by_term.csv")

    # Entries and exits (focusing on Fall-to-Fall and Winter-to-Winter)
    entries_exits = []
    terms_sorted = sorted(sia["term_srcdb"].unique())
    for i, t in enumerate(terms_sorted[1:], start=1):
        prev = terms_sorted[i - 1]
        new_in = courses_by_term[t] - courses_by_term[prev]
        gone = courses_by_term[prev] - courses_by_term[t]
        entries_exits.append({
            "term": t, "term_label": term_label(t),
            "prev_term": prev, "prev_label": term_label(prev),
            "n_new": len(new_in), "new_courses": "; ".join(sorted(new_in)),
            "n_removed": len(gone), "removed_courses": "; ".join(sorted(gone)),
        })
    pd.DataFrame(entries_exits).to_csv(
        OUT_DIR / "C4_course_entries_exits.csv", index=False
    )

    # ---------- Table C5: SUS share + close substitute shares ----------
    share_rows = []
    for t, group in sia.groupby("term_srcdb"):
        total_enr = group["enrolled"].sum(min_count=1)
        total_cap = group["capacity"].sum(min_count=1)
        sus_enr = group.loc[group["subject"] == "SUS", "enrolled"].sum()
        row = {
            "term_srcdb": t,
            "term_label": term_label(t),
            "sia_total_enrolled": total_enr,
            "sia_total_capacity": total_cap,
            "sus_enrolled": sus_enr,
            "sus_share_pct": 100.0 * sus_enr / total_enr if total_enr else np.nan,
        }
        for course_code in CLOSE_SUBSTITUTES:
            cenr = group.loc[group["course_code"] == course_code, "enrolled"].sum()
            row[f"{course_code}_enr"] = cenr
            row[f"{course_code}_share_pct"] = (
                100.0 * cenr / total_enr if total_enr else np.nan
            )
        share_rows.append(row)
    shares = pd.DataFrame(share_rows).sort_values("term_srcdb")
    shares.to_csv(OUT_DIR / "C5_sus_and_substitute_shares.csv", index=False)

    # ---------- Chart C1: SIA seats, enrolled, fill rate by term ----------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    terms_ordered = sorted(tot_by_term["term_srcdb"].unique(), key=term_sort_key)
    x = np.arange(len(terms_ordered))
    seats = [tot_by_term.loc[tot_by_term["term_srcdb"] == t, "seats"].iloc[0] for t in terms_ordered]
    enr = [tot_by_term.loc[tot_by_term["term_srcdb"] == t, "enrolled"].iloc[0] for t in terms_ordered]
    fill = [tot_by_term.loc[tot_by_term["term_srcdb"] == t, "fill_pct"].iloc[0] for t in terms_ordered]

    ax1.bar(x - 0.2, seats, width=0.4, label="Seats (capacity)", color="#b0c4de")
    ax1.bar(x + 0.2, enr, width=0.4, label="Enrolled", color="#4c78a8")
    ax1.set_ylabel("Headcount")
    ax1.set_title("SIA category: supply (capacity) vs demand (enrolled) by term")
    ax1.legend()
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.plot(x, fill, "o-", color="#e45756", linewidth=2)
    for xi, f in zip(x, fill):
        if pd.notna(f):
            ax2.text(xi, f + 1.5, f"{f:.0f}%", ha="center", fontsize=9)
    ax2.set_ylabel("Fill rate (%)")
    ax2.set_ylim(0, 110)
    ax2.axhline(100, color="grey", linestyle="--", alpha=0.5)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.set_xticks(x)
    ax2.set_xticklabels([term_label(t) for t in terms_ordered], rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "C1_chart_sia_supply_demand.png", dpi=150)
    plt.close(fig)

    # ---------- Chart C2: SUS share of SIA enrollment by term ----------
    fig, ax = plt.subplots(figsize=(10, 5))
    terms_ordered = sorted(shares["term_srcdb"].unique(), key=term_sort_key)
    x = np.arange(len(terms_ordered))
    sus_pct = [shares.loc[shares["term_srcdb"] == t, "sus_share_pct"].iloc[0] for t in terms_ordered]
    ax.plot(x, sus_pct, "D-", color="#4c78a8", linewidth=2.5, label="SUS")
    for substitute, _ in list(CLOSE_SUBSTITUTES.items())[:4]:
        pct = [shares.loc[shares["term_srcdb"] == t, f"{substitute}_share_pct"].iloc[0]
               for t in terms_ordered]
        ax.plot(x, pct, "o-", linewidth=1.5, alpha=0.7, label=substitute)
    ax.set_xticks(x)
    ax.set_xticklabels([term_label(t) for t in terms_ordered], rotation=45, ha="right")
    ax.set_ylabel("Share of total SIA enrollment (%)")
    ax.set_title("SUS share of SIA enrollment vs top substitutes by term")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "C2_chart_sus_vs_substitutes_share.png", dpi=150)
    plt.close(fig)

    # ---------- Chart C3: SIA seats by subject code (grouped bars by term) ----------
    # Top 10 subjects by overall enrollment
    top_subj = (
        by_subj_term.groupby("subject")["enrolled"].sum()
        .sort_values(ascending=False).head(10).index.tolist()
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    terms_ordered = sorted(by_subj_term["term_srcdb"].unique(), key=term_sort_key)
    x = np.arange(len(top_subj))
    width = 0.8 / len(terms_ordered)
    cmap = plt.cm.tab10
    for i, t in enumerate(terms_ordered):
        vals = []
        for subj in top_subj:
            v = by_subj_term.loc[
                (by_subj_term["term_srcdb"] == t) & (by_subj_term["subject"] == subj),
                "seats",
            ].sum()
            vals.append(v)
        ax.bar(x + i * width - 0.4 + width / 2, vals, width=width,
               label=term_label(t), color=cmap(i / max(1, len(terms_ordered) - 1)))
    ax.set_xticks(x)
    ax.set_xticklabels(top_subj)
    ax.set_ylabel("Seats (capacity)")
    ax.set_title("SIA seats by subject code and term (top 10 subjects)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "C3_chart_seats_by_subject.png", dpi=150)
    plt.close(fig)

    print("Analysis C complete.")
    print("\nSIA totals by term:")
    print(tot_by_term[["term_label", "n_courses", "n_sections", "seats", "enrolled", "fill_pct"]].to_string(index=False))
    print("\nShares:")
    print(shares[["term_label", "sia_total_enrolled", "sus_enrolled", "sus_share_pct"]].to_string(index=False))


if __name__ == "__main__":
    run()
