"""
In-term rebalancing: projected final enrolment by course and modality.

Purpose
-------
SUS staffing is set for the year, with limited in-term flexibility
(overloads, swapping instructors between courses). This panel answers the
question that flexibility needs: a few weeks before term, which courses are
on track to run short of seats and which will have slack?

Method
------
For each course x modality in the target term, final (day-0) enrolment is
projected from today's count using prior terms of the same season:

    projected = today * (reference_final / reference_at_same_point)

averaged across reference terms; the min..max across references is the range.

"Same point" means days since registration opened, where that is known.
Opening dates move by several days year to year, and early in the ramp a
few days' misalignment swamps everything. A reference term whose logging
began after registration had already opened has no known opening date, so
it is aligned on days before term start instead, and labelled as such.

Data handling
-------------
- Lecture/primary sections only. schedule_type D/C1/C2/R are labs and
  recitations; where schedule_type is missing (legacy table), sections
  010-399 are treated as labs.
- The legacy table (sus_daily_registrations_2025) is an event log with one
  row per registration change, so each section is forward-filled onto a
  daily grid before summing. Summing only the sections that happened to
  transact on a given day produces nonsense.
- A reference term that sold out (final >= 95% of capacity) shows how many
  students its seats held, not how many wanted in, so a projection leaning
  on it understates demand. Such rows are flagged.
- Capacity for terms the tracker did not record can be supplied in
  data/manual_capacity.csv (term_srcdb, code, section, capacity).
- Cascades and other non-SUS-managed sections are excluded.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

LAB_TYPES = {"D", "C1", "C2", "R"}
OPEN_THRESHOLD = 20          # registration treated as open once the term's SUS lecture total exceeds this
SEASON = {"00": "Summer", "01": "Fall", "02": "Winter", "03": "Spring"}
MODES = ("On-campus", "Ecampus")
SLACK_FILL = 0.75            # projected fill at or below this is flagged as slack
FULL_FILL = 0.95             # at or above this counts as full / sold out
SMALL_COURSE_CAP = 30        # rows below this capacity are hidden by default


def term_short(t: str) -> str:
    """202702 -> 'WI27'."""
    t = str(t)
    code = t[-2:]
    yr = int(t[:4]) - 1 if code in ("00", "01") else int(t[:4])
    return f"{SEASON.get(code, code)[:2].upper()}{str(yr)[-2:]}"


def read_strength(season: str, mode: str, d: int, k: Optional[int]) -> str:
    """Rule of thumb, from 2024-26 data, for when a read can be trusted."""
    if season == "Fall":
        return "firm" if d >= (-60 if mode == "On-campus" else -45) else "early"
    if season == "Winter":
        if mode == "On-campus":
            return "firm" if (k is not None and k >= 7) else "early"
        return "firm" if d >= -30 else "early"
    if season == "Spring":
        return "firm" if d >= -15 else "early"
    return "unreliable"


def _mode(section) -> str:
    try:
        n = int(str(section).strip())
    except (TypeError, ValueError):
        return "On-campus"
    if 400 <= n <= 499:
        return "Ecampus"
    if 500 <= n <= 699:
        return "Cascades"
    if n >= 700:
        return "Other"
    return "On-campus"


def _lab_by_number(section) -> bool:
    try:
        n = int(str(section).strip())
    except (TypeError, ValueError):
        return False
    return 10 <= n < 400


def load_events(db_path: Path) -> pd.DataFrame:
    with sqlite3.connect(db_path) as c:
        L = pd.read_sql_query(
            "SELECT snapshot_date, term_srcdb, subject, course_number, section, enrolled "
            "FROM sus_daily_registrations_2025", c)
        V = pd.read_sql_query(
            "SELECT snapshot_date, term_srcdb, code, section, schedule_type, enrolled, capacity "
            "FROM enrollment", c)
    L["code"] = L.subject.astype(str).str.strip() + " " + L.course_number.astype(str).str.strip()
    L = L[~L.section.apply(_lab_by_number)].copy()
    L["capacity"] = np.nan
    L["src"] = "legacy"
    st_ = V.schedule_type.fillna("")
    n = pd.to_numeric(V.section, errors="coerce")
    V = V[~(st_.isin(LAB_TYPES) | (st_.eq("") & n.between(10, 399)))].copy()
    V["src"] = "live"
    cols = ["snapshot_date", "term_srcdb", "code", "section", "enrolled", "capacity", "src"]
    E = pd.concat([L[cols], V[cols]], ignore_index=True)
    E["term_srcdb"] = E.term_srcdb.astype(str)
    E["section"] = E.section.astype(str).str.strip()
    E["code"] = E.code.astype(str).str.strip()
    for col in ("enrolled", "capacity"):
        E[col] = pd.to_numeric(E[col], errors="coerce")
    E["ts"] = pd.to_datetime(E.snapshot_date, format="mixed", errors="coerce")
    E = E.dropna(subset=["ts"])
    E["day"] = E.ts.dt.normalize()
    # Prefer the live tracker for any term where it has a real trajectory.
    live_days = E[E.src == "live"].groupby("term_srcdb").day.nunique()
    live_terms = set(live_days[live_days > 5].index)
    E = E[~((E.src == "legacy") & E.term_srcdb.isin(live_terms))]
    E["mode"] = E.section.apply(_mode)
    E["key"] = E.code + "|" + E.section
    return E


def load_manual_capacity(db_path: Path) -> dict:
    p = Path(db_path).parent / "manual_capacity.csv"
    if not p.exists():
        return {}
    m = pd.read_csv(p, dtype=str)
    return {(r.term_srcdb.strip(), f"{r.code.strip()}|{r.section.strip()}"): float(r.capacity)
            for r in m.itertuples()}


def _grid(E: pd.DataFrame, term: str, start: pd.Timestamp):
    g = E[E.term_srcdb == term].sort_values("ts")
    if g.empty:
        return None, None
    enr = g.pivot_table(index="day", columns="key", values="enrolled", aggfunc="last")
    cap = g.pivot_table(index="day", columns="key", values="capacity", aggfunc="last")
    full = pd.date_range(enr.index.min(), enr.index.max(), freq="D")
    enr = enr.reindex(full).ffill()
    cap = cap.reindex(full).ffill().reindex(columns=enr.columns)
    off = (full - start).days
    enr.index = off
    cap.index = off
    return enr, cap


def _opening(enr: pd.DataFrame) -> Optional[int]:
    """First day the total exceeds the threshold; None if logging began after opening."""
    tot = enr.sum(axis=1, min_count=1).fillna(0)
    if tot.empty or tot.iloc[0] > OPEN_THRESHOLD:
        return None
    above = tot[tot > OPEN_THRESHOLD]
    return int(above.index.min()) if len(above) else None


def _cap_sum(cap, day, cols, term, manual):
    vals = []
    for c in cols:
        v = cap.loc[day, c] if (cap is not None and day in cap.index and c in cap.columns) else np.nan
        if pd.isna(v):
            v = manual.get((term, c), np.nan)
        vals.append(v)
    vals = [v for v in vals if not pd.isna(v)]
    return float(sum(vals)) if len(vals) == len(cols) and vals else np.nan


def build_projection(E, term, classes_begin: Callable[[str], Optional[pd.Timestamp]], manual=None):
    manual = manual or {}
    term = str(term)
    start = classes_begin(term)
    enr, cap = _grid(E, term, start)
    if enr is None:
        return pd.DataFrame(), {}
    season = SEASON.get(term[-2:], "?")
    d_now = int(min(enr.index.max(), 0))
    open_t = _opening(enr)
    k_now = (d_now - open_t) if open_t is not None else None

    # reference terms: same season, earlier, with a day-0 observation
    refs = []
    for r in sorted(E.term_srcdb.unique(), key=int, reverse=True):
        if r[-2:] != term[-2:] or int(r) >= int(term):
            continue
        rs = classes_begin(r)
        if rs is None:
            continue
        re_, rc_ = _grid(E, r, rs)
        if re_ is None or 0 not in re_.index:
            continue
        refs.append((r, re_, rc_, _opening(re_)))

    keymode = E.drop_duplicates("key").set_index("key")["mode"]
    keycode = E.drop_duplicates("key").set_index("key")["code"]
    rows = []
    for (code, mode), _ in pd.Series(1, index=pd.MultiIndex.from_arrays(
            [keycode.reindex(enr.columns).values, keymode.reindex(enr.columns).values])).groupby(level=[0, 1]):
        if mode not in MODES:
            continue
        cols = [c for c in enr.columns if keycode.get(c) == code and keymode.get(c) == mode]
        now = float(enr.loc[d_now, cols].sum())
        cap_now = _cap_sum(cap, d_now, cols, term, manual)
        projs, basis, sold_out, last_final = [], [], False, None
        for r, re_, rc_, open_r in refs:
            rcols = [c for c in re_.columns if keycode.get(c) == code and keymode.get(c) == mode]
            if not rcols:
                continue
            fin = float(re_.loc[0, rcols].sum())
            rcap = _cap_sum(rc_, 0, rcols, r, manual)
            if last_final is None:
                last_final = (r, fin, rcap)
            if open_r is not None and k_now is not None:
                d_ref, clock = open_r + k_now, f"open+{k_now}"
            else:
                d_ref, clock = d_now, f"d{d_now}"
            if d_ref not in re_.index:
                continue
            at = float(re_.loc[d_ref, rcols].sum())
            if at <= 0:
                continue
            ratio = fin / at
            capped = (not np.isnan(rcap)) and fin >= FULL_FILL * rcap
            sold_out |= capped
            projs.append(now * ratio)
            basis.append(f"{term_short(r)} {clock} x{ratio:.2f}" + (" (sold out)" if capped else ""))
        if d_now >= 0:
            proj, lo, hi = now, now, now
        elif projs and now > 0:
            proj, lo, hi = float(np.mean(projs)), float(min(projs)), float(max(projs))
        else:
            proj = lo = hi = np.nan
        fill_now = now / cap_now if cap_now and not np.isnan(cap_now) else np.nan
        if not np.isnan(fill_now) and fill_now >= FULL_FILL:
            status = "Full now"
        elif np.isnan(proj):
            status = "No basis" if now > 0 else "No registrations yet"
        elif not np.isnan(cap_now) and proj >= cap_now:
            status = "Likely to fill"
        elif not np.isnan(cap_now) and proj <= SLACK_FILL * cap_now:
            status = "Likely slack"
        else:
            status = "On track"
        rows.append(dict(
            Course=code, Mode=mode,
            Now=f"{now:.0f}/{cap_now:.0f}" if not np.isnan(cap_now) else f"{now:.0f}",
            FillNow=fill_now,
            Projected=proj, Low=lo, High=hi,
            Capacity=cap_now,
            Gap=(proj - cap_now) if not (np.isnan(proj) or np.isnan(cap_now)) else np.nan,
            Status=status,
            Read="final" if d_now >= 0 else read_strength(season, mode, d_now, k_now),
            LastFinal=(f"{last_final[1]:.0f}/{last_final[2]:.0f} ({term_short(last_final[0])})"
                       if last_final and not np.isnan(last_final[2])
                       else (f"{last_final[1]:.0f} ({term_short(last_final[0])})" if last_final else "")),
            Floor="yes" if sold_out else "",
            Basis="; ".join(basis),
        ))
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("Gap", ascending=False, na_position="last").reset_index(drop=True)
    meta = dict(term=term, season=season, d_now=d_now, open_day=open_t, k_now=k_now,
                as_of=(start + pd.Timedelta(days=d_now)).date() if start is not None else None,
                refs=[term_short(r) for r, *_ in refs])
    return out, meta


def render_rebalancing_panel(db_path, classes_begin):
    import streamlit as st
    st.header("Projected final enrolment")
    st.caption("For in-term rebalancing: which courses are heading short of seats, and which will have slack.")
    E = load_events(Path(db_path))
    manual = load_manual_capacity(Path(db_path))
    terms = sorted(E[(E.src == "live") & (E.enrolled > 0)].term_srcdb.unique(), key=int, reverse=True)
    if not terms:
        st.info("No live registration data yet.")
        return
    c1, c2 = st.columns([2, 3])
    with c1:
        term = st.selectbox("Term", terms, index=0, key="rebal_term",
                            format_func=lambda t: f"{term_short(t)}  ({t})")
    with c2:
        show_small = st.checkbox(f"Include courses under {SMALL_COURSE_CAP} seats", value=False,
                                 key="rebal_small")
    tbl, meta = build_projection(E, term, classes_begin, manual)
    if tbl.empty:
        st.info("No sections to project for this term.")
        return
    if not show_small:
        tbl = tbl[tbl.Capacity.isna() | (tbl.Capacity >= SMALL_COURSE_CAP)]
    opened = (f"registration opened day {meta['open_day']} ({meta['k_now']} days ago)"
              if meta["open_day"] is not None else "registration opening date not observed")
    st.caption(f"As of {meta['as_of']} · day {meta['d_now']} · {opened} · "
               f"reference terms: {', '.join(meta['refs']) or 'none'}. "
               "Rows marked early are directional: they show which way a course is heading, "
               "not by how much.")
    show = tbl.copy()
    show["Fill now"] = show.FillNow.map(lambda x: "" if pd.isna(x) else f"{x:.0%}")
    show["Projected"] = show.Projected.map(lambda x: "" if pd.isna(x) else f"{x:.0f}")
    show["Range"] = [("" if pd.isna(lo) else f"{lo:.0f}-{hi:.0f}") for lo, hi in zip(show.Low, show.High)]
    show["Capacity"] = show.Capacity.map(lambda x: "" if pd.isna(x) else f"{x:.0f}")
    show["Gap"] = show.Gap.map(lambda x: "" if pd.isna(x) else f"{x:+.0f}")
    show = show.rename(columns={"LastFinal": "Last same-season final", "Floor": "Ref sold out"})
    show = show[["Course", "Mode", "Now", "Fill now", "Projected", "Range", "Capacity", "Gap",
                 "Status", "Read", "Last same-season final", "Ref sold out", "Basis"]]
    colors = {"Likely to fill": "#f8d7da", "Full now": "#f8d7da",
              "Likely slack": "#d1ecf1", "On track": "", "No basis": "#eeeeee",
              "No registrations yet": "#eeeeee"}
    sty = show.style
    sty = (sty.map if hasattr(sty, "map") else sty.applymap)(
        lambda v: f"background-color: {colors.get(v, '')}" if v in colors and colors.get(v) else "",
        subset=["Status"])
    st.dataframe(sty, width="stretch", hide_index=True)
    with st.expander("How the projection works"):
        st.markdown(
            "- **Projected** = today's count x (reference term's final / its count at the same point), "
            "averaged over prior terms of the same season. **Range** is the spread across those terms.\n"
            "- **Same point** is days since registration opened where known; otherwise days before term "
            "start. The **Basis** column shows which, for each reference.\n"
            "- **Gap** = projected minus capacity; rows run from tightest to slackest. A positive gap means "
            "the course will likely fill. It does not measure unmet demand: early in the ramp the multiplier "
            "overstates it, and when the reference term sold out it understates it. **Last same-season final** "
            "is the better guide to how many students a course actually served.\n"
            f"- **Likely slack** means projected fill at or below {SLACK_FILL:.0%}.\n"
            "- **Ref sold out**: a reference term filled its seats, so it shows what the seats held, not "
            "what students wanted. The projection is then a floor.\n"
            "- **Read**: whether this point in the cycle has been reliable historically. Winter on-campus "
            "settles about a week after registration opens; Winter Ecampus around day -30; Fall from "
            "about day -60 (on-campus) and -45 (Ecampus); Spring only from about day -15.\n"
            "- Lecture and primary sections only. Cascades excluded.\n"
            "- Missing historical capacities can be added to `data/manual_capacity.csv`."
        )
