"""
Analysis D — Same-designator comparison (descriptive, not causal).

Classify SIA subject codes as single-course or multi-course within SIA.
For multi-course codes (where Core Ed's same-subject-only rule bites),
compare pre/post 202600 enrollment change, and within multi-course codes,
identify which course loses.

Caveats included in output and summary:
- Short post period (~1 academic year of SIA data).
- Bacc Core students still eligible and distort "post" numbers.
- Seat changes are endogenous (programs can expand sections in response).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_load import OUT_DIR, load_sia_census, season_of, term_label, term_sort_key


def run():
    sia = load_sia_census()
    if sia.empty:
        print("No SIA data available.")
        return

    # ---------- Classify subjects as single-course or multi-course within SIA ----------
    # Based on the latest Fall term available (represents offerings most comprehensively)
    latest_fall = max([t for t in sia["term_srcdb"].unique() if season_of(t) == "Fall"],
                      key=lambda t: int(t))
    fall_courses = sia[sia["term_srcdb"] == latest_fall]
    subj_course_counts = (
        fall_courses.groupby("subject")["course_code"].nunique().rename("n_courses_in_sia")
    )
    # Use union across all terms as supporting evidence
    all_subj_course_counts = (
        sia.groupby("subject")["course_code"].nunique().rename("n_courses_all_terms")
    )
    classification = pd.concat([subj_course_counts, all_subj_course_counts], axis=1).fillna(0)
    classification["is_multi_course"] = (classification["n_courses_all_terms"] >= 2).astype(int)
    classification = classification.sort_values("n_courses_all_terms", ascending=False)
    classification.to_csv(OUT_DIR / "D1_subject_classification.csv")

    print("Subject classification (SIA):")
    print(classification.to_string())

    multi_subj = classification[classification["is_multi_course"] == 1].index.tolist()
    single_subj = classification[classification["is_multi_course"] == 0].index.tolist()

    # ---------- Enrollment change by subject (Fall-to-Fall) ----------
    # Compare Fall 2025 (202601) vs Fall 2026 (202701) — only direct Fall-Fall comparison in SIA data.
    fall25 = sia[sia["term_srcdb"] == "202601"]
    fall26 = sia[sia["term_srcdb"] == "202701"]
    if fall25.empty or fall26.empty:
        print("Need both Fall 2025 and Fall 2026 for direct comparison.")
        return

    agg_subj_fall = []
    for subj in classification.index:
        f25_enr = fall25.loc[fall25["subject"] == subj, "enrolled"].sum()
        f26_enr = fall26.loc[fall26["subject"] == subj, "enrolled"].sum()
        f25_cap = fall25.loc[fall25["subject"] == subj, "capacity"].sum()
        f26_cap = fall26.loc[fall26["subject"] == subj, "capacity"].sum()
        f25_sec = fall25.loc[fall25["subject"] == subj, "crn"].nunique()
        f26_sec = fall26.loc[fall26["subject"] == subj, "crn"].nunique()
        agg_subj_fall.append({
            "subject": subj,
            "n_courses_in_sia": int(classification.loc[subj, "n_courses_all_terms"]),
            "is_multi_course": int(classification.loc[subj, "is_multi_course"]),
            "FA25_sections": f25_sec,
            "FA26_sections": f26_sec,
            "FA25_seats": f25_cap,
            "FA26_seats": f26_cap,
            "FA25_enrolled": f25_enr,
            "FA26_enrolled": f26_enr,
            "pct_change_enrolled": (
                100.0 * (f26_enr - f25_enr) / f25_enr if f25_enr else np.nan
            ),
            "pct_change_seats": (
                100.0 * (f26_cap - f25_cap) / f25_cap if f25_cap else np.nan
            ),
        })
    subj_change = pd.DataFrame(agg_subj_fall).sort_values("FA25_enrolled", ascending=False)
    subj_change.to_csv(OUT_DIR / "D2_fall25_vs_fall26_by_subject.csv", index=False)

    # ---------- Multi vs single course average change ----------
    def _mean_pct(df, col):
        vals = df[col].dropna()
        return vals.mean() if len(vals) else np.nan

    multi = subj_change[subj_change["is_multi_course"] == 1]
    single = subj_change[subj_change["is_multi_course"] == 0]
    summary_rows = [
        {"category": "multi-course codes",
         "n_subjects": len(multi),
         "FA25_enrolled_sum": multi["FA25_enrolled"].sum(),
         "FA26_enrolled_sum": multi["FA26_enrolled"].sum(),
         "pct_change_pooled": 100.0 * (multi["FA26_enrolled"].sum() - multi["FA25_enrolled"].sum())
                              / multi["FA25_enrolled"].sum() if multi["FA25_enrolled"].sum() else np.nan,
         "pct_change_mean_of_subj": _mean_pct(multi, "pct_change_enrolled"),
         "FA25_seats_sum": multi["FA25_seats"].sum(),
         "FA26_seats_sum": multi["FA26_seats"].sum(),
         "seats_pct_change_pooled": 100.0 * (multi["FA26_seats"].sum() - multi["FA25_seats"].sum())
                                     / multi["FA25_seats"].sum() if multi["FA25_seats"].sum() else np.nan,
         },
        {"category": "single-course codes",
         "n_subjects": len(single),
         "FA25_enrolled_sum": single["FA25_enrolled"].sum(),
         "FA26_enrolled_sum": single["FA26_enrolled"].sum(),
         "pct_change_pooled": 100.0 * (single["FA26_enrolled"].sum() - single["FA25_enrolled"].sum())
                              / single["FA25_enrolled"].sum() if single["FA25_enrolled"].sum() else np.nan,
         "pct_change_mean_of_subj": _mean_pct(single, "pct_change_enrolled"),
         "FA25_seats_sum": single["FA25_seats"].sum(),
         "FA26_seats_sum": single["FA26_seats"].sum(),
         "seats_pct_change_pooled": 100.0 * (single["FA26_seats"].sum() - single["FA25_seats"].sum())
                                     / single["FA25_seats"].sum() if single["FA25_seats"].sum() else np.nan,
         },
    ]
    pd.DataFrame(summary_rows).to_csv(
        OUT_DIR / "D3_multi_vs_single_course_summary.csv", index=False
    )

    # ---------- Within multi-course codes, which course loses? ----------
    within_rows = []
    for subj in multi_subj:
        subj_f25 = fall25[fall25["subject"] == subj]
        subj_f26 = fall26[fall26["subject"] == subj]
        courses_in_both = sorted(set(subj_f25["course_code"].unique()) &
                                 set(subj_f26["course_code"].unique()))
        for course in courses_in_both:
            e25 = subj_f25.loc[subj_f25["course_code"] == course, "enrolled"].sum()
            e26 = subj_f26.loc[subj_f26["course_code"] == course, "enrolled"].sum()
            within_rows.append({
                "subject": subj,
                "course_code": course,
                "FA25_enrolled": e25,
                "FA26_enrolled": e26,
                "change": e26 - e25,
                "pct_change": 100.0 * (e26 - e25) / e25 if e25 else np.nan,
            })
    within = pd.DataFrame(within_rows).sort_values(["subject", "FA25_enrolled"],
                                                   ascending=[True, False])
    within.to_csv(OUT_DIR / "D4_within_multi_course_detail.csv", index=False)

    # ---------- Chart D1: pct change by subject, multi vs single ----------
    fig, ax = plt.subplots(figsize=(12, 6))
    sc = subj_change[subj_change["FA25_enrolled"] >= 30].copy()  # drop tiny subjects
    sc = sc.sort_values("pct_change_enrolled")
    colors = ["#4c78a8" if m else "#b0b0b0" for m in sc["is_multi_course"]]
    bars = ax.barh(np.arange(len(sc)), sc["pct_change_enrolled"].values,
                   color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(np.arange(len(sc)))
    ax.set_yticklabels(sc["subject"].values)
    ax.set_xlabel("% change in enrollment, Fall 2025 → Fall 2026")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("SIA enrollment change by subject (FA25 → FA26, day-20 vs day-7)\n"
                 "Blue = multi-course within SIA (affected by same-designator rule)")
    # Annotate SUS
    if "SUS" in sc["subject"].values:
        sus_idx = sc["subject"].tolist().index("SUS")
        ax.annotate("SUS", xy=(sc["pct_change_enrolled"].iloc[sus_idx], sus_idx),
                    xytext=(10, 0), textcoords="offset points",
                    fontweight="bold", color="red", fontsize=11)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "D1_chart_subject_change.png", dpi=150)
    plt.close(fig)

    # ---------- Chart D2: within-SUS detail — 102 vs 103 ----------
    sus_within = within[within["subject"] == "SUS"]
    if not sus_within.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(sus_within))
        width = 0.35
        ax.bar(x - width/2, sus_within["FA25_enrolled"].values, width,
               label="FA25", color="#4c78a8")
        ax.bar(x + width/2, sus_within["FA26_enrolled"].values, width,
               label="FA26 (early)", color="#e45756")
        for i, (e25, e26, pct) in enumerate(zip(
            sus_within["FA25_enrolled"], sus_within["FA26_enrolled"],
            sus_within["pct_change"]
        )):
            ax.text(i, max(e25, e26) * 1.05, f"{pct:+.0f}%", ha="center", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(sus_within["course_code"].values, rotation=0)
        ax.set_ylabel("Enrolled")
        ax.set_title("SUS courses in SIA: FA25 (day-20) vs FA26 (day-7 early)")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "D2_chart_sus_within_detail.png", dpi=150)
        plt.close(fig)

    print("\nAnalysis D complete.")
    print("Subject-level FA25 vs FA26 change (top 15 by FA25 enrolment):")
    print(subj_change[["subject", "n_courses_in_sia", "is_multi_course",
                       "FA25_enrolled", "FA26_enrolled", "pct_change_enrolled",
                       "pct_change_seats"]].head(15).to_string(index=False))
    print("\nMulti vs single course summary:")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    run()
