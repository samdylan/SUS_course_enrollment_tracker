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

"Same point" is defined by OSU's registration waves where they are on file
(data/registration_calendar.csv): seniors, juniors, sophomores and freshmen
open on different days, and Corvallis students cannot register for Ecampus
sections until Monday of week 10. Reference terms are compared at the same
time after the most recent wave that both terms have on record. Without a
calendar, alignment falls back to days since registration opened, or days
before term start.

Who has been able to register
-----------------------------
Until a wave opens, its students are missing from the counts, so projections
made before the sophomore and freshman waves (or, for Ecampus sections, before
week 10) are extrapolating from the wrong students. Each row therefore shows
which waves it is still waiting on, and such rows are marked provisional.

Data handling
-------------
- Lecture/primary sections only. schedule_type D/C1/C2/R are labs and
  recitations; where schedule_type is missing (legacy table), sections
  010-399 are treated as labs.
- The legacy table (sus_daily_registrations_2025) is an event log with one
  row per registration change, so each section is forward-filled onto a
  daily grid before summing.
- A reference term that sold out (final >= 95% of capacity) shows how many
  students its seats held, not how many wanted in. Such rows are flagged.
- Capacity for terms the tracker did not record can be supplied in
  data/manual_capacity.csv (term_srcdb, code, section, capacity).
- Snapshots are taken by the nightly GitHub Action at 17:00 UTC, i.e. about
  09:00 Pacific; a snapshot dated D reflects registrations up to that time.
- Cascades and other non-SUS-managed sections are excluded.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

LAB_TYPES = {"D", "C1", "C2", "R"}
OPEN_THRESHOLD = 20
SEASON = {"00": "Summer", "01": "Fall", "02": "Winter", "03": "Spring"}
MODES = ("On-campus", "Ecampus")
SLACK_FILL = 0.75
FULL_FILL = 0.95
SMALL_COURSE_CAP = 30
SNAPSHOT_HOUR_PT = 9          # nightly Action runs 17:00 UTC
SETTLE_DAYS = 3               # days after the last relevant wave before a read is "firm"
PT = "America/Los_Angeles"
METHOD_VERSION = "2026-11-23 waves-v1"
STATUS_ORDER = {"Full now": 0, "Likely to fill": 1, "On track": 2, "Likely slack": 3,
                "No basis": 4, "No registrations yet": 5}
STATUS_FILL = {"Full now": "#f1b0b7", "Likely to fill": "#f8d7da", "On track": "#d4edda",
               "Likely slack": "#d1ecf1", "No basis": "#eeeeee", "No registrations yet": "#eeeeee"}


def term_short(t: str) -> str:
    t = str(t)
    code = t[-2:]
    yr = int(t[:4]) - 1 if code in ("00", "01") else int(t[:4])
    return f"{SEASON.get(code, code)[:2].upper()}{str(yr)[-2:]}"


def now_pacific() -> pd.Timestamp:
    try:
        from zoneinfo import ZoneInfo
        return pd.Timestamp(datetime.now(ZoneInfo(PT)).replace(tzinfo=None))
    except Exception:
        return pd.Timestamp(datetime.utcnow()) - pd.Timedelta(hours=8)


def read_strength_fallback(season: str, mode: str, d: int, k: Optional[int]) -> str:
    """Rule of thumb for terms with no registration calendar on file."""
    if season == "Fall":
        return "firm" if d >= (-60 if mode == "On-campus" else -45) else "early"
    if season == "Winter":
        if mode == "On-campus":
            return "firm" if (k is not None and k >= 7) else "early"
        return "firm" if d >= -30 else "early"
    if season == "Spring":
        return "firm" if d >= -15 else "early"
    return "unreliable"


# ---------------------------------------------------------------- data

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


def load_calendar(db_path: Path) -> pd.DataFrame:
    p = Path(db_path).parent / "registration_calendar.csv"
    cols = ["term_srcdb", "order", "wave", "opens", "scope", "key", "source"]
    if not p.exists():
        return pd.DataFrame(columns=cols)
    c = pd.read_csv(p, dtype=str)
    c["term_srcdb"] = c.term_srcdb.str.strip()
    c["opens"] = pd.to_datetime(c.opens.str.strip())
    c["order"] = c.order.astype(int)
    c["scope"] = c.scope.str.strip().str.lower()
    c["key"] = c.key.str.strip().str.lower().eq("yes")
    return c.sort_values(["term_srcdb", "order"]).reset_index(drop=True)


# ---------------------------------------------------------------- waves

def wave_states(cal_t: pd.DataFrame, as_of_ts: pd.Timestamp, now_pt: pd.Timestamp) -> pd.DataFrame:
    """Each wave's state relative to the data snapshot and to today."""
    out = cal_t.copy()
    states = []
    for r in out.itertuples():
        covered = as_of_ts - r.opens
        if covered >= pd.Timedelta(hours=24):
            states.append("in data")
        elif r.opens <= as_of_ts:
            states.append("partly in data")
        elif r.opens <= now_pt:
            states.append("not in your data yet")
        else:
            states.append("upcoming")
    out["state"] = states
    return out


def _relevant(ws: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Key waves that govern a row: standing waves for everyone, plus the Ecampus release for Ecampus rows."""
    w = ws[ws.key]
    if mode != "Ecampus":
        w = w[w.scope != "ecampus"]
    return w


def _align_on_waves(cal_t, cal_r, mode, as_of_ts, target_start, ref_start):
    """Day in the reference term equivalent to as_of_ts in the target term.

    1. Anchor on the most recent relevant wave both terms have on file and take the
       same elapsed time after it; with no shared wave, use the same days before term.
    2. Never let the reference point fall after a wave that has not yet opened in the
       target: compare like with like, the same groups of students registered.

    Returns (d_ref, label), or None when either term has no calendar."""
    if cal_t is None or cal_r is None or cal_t.empty or cal_r.empty:
        return None
    rel_t = _relevant(cal_t.assign(state=""), mode)
    opened = rel_t[rel_t.opens <= as_of_ts]
    shared = opened[opened.wave.isin(set(cal_r.wave))]
    short = lambda w: {"Ecampus open to Corvallis students": "Ecampus release"}.get(w, w)
    if not shared.empty:
        w = shared.sort_values("opens").iloc[-1]
        dt = as_of_ts - w.opens
        ref_ts = cal_r[cal_r.wave == w.wave].opens.iloc[0] + dt
        label = f"{short(w.wave)}+{dt.days}d"
    else:
        ref_ts = ref_start + (as_of_ts - target_start)
        label = f"d{(as_of_ts.normalize() - target_start).days}"
    pending = rel_t[rel_t.opens > as_of_ts]
    caps = cal_r[cal_r.wave.isin(set(pending.wave))]
    if not caps.empty:
        c = caps.sort_values("opens").iloc[0]
        cap_ts = c.opens - pd.Timedelta(minutes=1)
        if ref_ts > cap_ts:
            ref_ts = cap_ts
            label += f", held before {short(c.wave)}"
    d_ref = int(np.floor((ref_ts - (ref_start + pd.Timedelta(hours=SNAPSHOT_HOUR_PT))) / pd.Timedelta(days=1)))
    return d_ref, label


# ---------------------------------------------------------------- grids

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


# ---------------------------------------------------------------- projection

def build_projection(E, term, classes_begin: Callable[[str], Optional[pd.Timestamp]],
                     manual=None, cal=None, now_pt=None):
    manual = manual or {}
    cal = cal if cal is not None else pd.DataFrame(columns=["term_srcdb"])
    now_pt = now_pt or now_pacific()
    term = str(term)
    start = classes_begin(term)
    enr, cap = _grid(E, term, start)
    if enr is None:
        return pd.DataFrame(), {}
    season = SEASON.get(term[-2:], "?")
    d_now = int(min(enr.index.max(), 0))
    as_of_ts = start + pd.Timedelta(days=d_now, hours=SNAPSHOT_HOUR_PT)
    open_t = _opening(enr)
    k_now = (d_now - open_t) if open_t is not None else None
    cal_t = cal[cal.term_srcdb == term] if not cal.empty else None
    ws = wave_states(cal_t, as_of_ts, now_pt) if cal_t is not None and not cal_t.empty else None

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
        cal_r = cal[cal.term_srcdb == r] if not cal.empty else None
        refs.append((r, rs, re_, rc_, _opening(re_), cal_r))

    keymode = E.drop_duplicates("key").set_index("key")["mode"]
    keycode = E.drop_duplicates("key").set_index("key")["code"]
    pairs = sorted({(keycode.get(c), keymode.get(c)) for c in enr.columns})
    rows = []
    for code, mode in pairs:
        if mode not in MODES:
            continue
        cols = [c for c in enr.columns if keycode.get(c) == code and keymode.get(c) == mode]
        now = float(enr.loc[d_now, cols].sum())
        cap_now = _cap_sum(cap, d_now, cols, term, manual)

        # who has been able to register
        if ws is not None:
            rel = _relevant(ws, mode)
            pending = rel[rel.state != "in data"]
            waiting = ", ".join({"Ecampus open to Corvallis students": "Ecampus release"}.get(w, w)
                                for w in pending.wave)
            last_rel = rel[rel.state == "in data"].opens.max() if (rel.state == "in data").any() else None
        else:
            pending, waiting, last_rel = None, None, None

        projs, basis, sold_out, last_final = [], [], False, None
        for r, rs, re_, rc_, open_r, cal_r in refs:
            rcols = [c for c in re_.columns if keycode.get(c) == code and keymode.get(c) == mode]
            if not rcols:
                continue
            fin = float(re_.loc[0, rcols].sum())
            rcap = _cap_sum(rc_, 0, rcols, r, manual)
            if last_final is None:
                last_final = (r, fin, rcap)
            aligned = _align_on_waves(cal_t, cal_r, mode, as_of_ts, start, rs)
            if aligned is not None:
                d_ref, clock = aligned
            elif open_r is not None and k_now is not None:
                d_ref, clock = open_r + k_now, f"open+{k_now}"
            else:
                d_ref, clock = d_now, f"d{d_now}"
            d_ref = min(d_ref, 0)
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

        if d_now >= 0:
            read = "final"
        elif ws is not None:
            if waiting:
                read = "waiting"
            elif last_rel is not None and (as_of_ts - last_rel) >= pd.Timedelta(days=SETTLE_DAYS):
                read = "firm"
            else:
                read = "settling"
        else:
            read = read_strength_fallback(season, mode, d_now, k_now)

        rows.append(dict(
            Course=code, Mode=mode,
            Now=f"{now:.0f}/{cap_now:.0f}" if not np.isnan(cap_now) else f"{now:.0f}",
            FillNow=fill_now, Projected=proj, Low=lo, High=hi, Capacity=cap_now,
            Gap=(proj - cap_now) if not (np.isnan(proj) or np.isnan(cap_now)) else np.nan,
            Status=status, Read=read,
            WaitingOn=("" if waiting is None else (waiting or "all in")),
            LastFinal=(f"{last_final[1]:.0f}/{last_final[2]:.0f} ({term_short(last_final[0])})"
                       if last_final and not np.isnan(last_final[2])
                       else (f"{last_final[1]:.0f} ({term_short(last_final[0])})" if last_final else "")),
            Floor="yes" if sold_out else "",
            Basis="; ".join(basis),
        ))
    out = pd.DataFrame(rows)
    if not out.empty:
        out["_o"] = out.Status.map(STATUS_ORDER).fillna(9)
        out = (out.sort_values(["_o", "Gap"], ascending=[True, False], na_position="last")
                  .drop(columns="_o").reset_index(drop=True))
    meta = dict(term=term, season=season, d_now=d_now, open_day=open_t, k_now=k_now,
                as_of=as_of_ts.date(), as_of_ts=as_of_ts, now_pt=now_pt, waves=ws,
                refs=[term_short(r) for r, *_ in refs])
    return out, meta


def forecast_records(tbl: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """The projection table plus the context needed to read it later: which data it was
    built from, which waves were in, which reference terms and method version."""
    out = tbl.copy()
    out.insert(0, "term_srcdb", meta["term"])
    out.insert(1, "data_as_of", str(meta["as_of"]))
    out.insert(2, "day", meta["d_now"])
    ws = meta.get("waves")
    out["waves_in_data"] = ("" if ws is None else
                            ", ".join(ws[ws.state == "in data"].wave))
    out["reference_terms"] = ", ".join(meta["refs"])
    out["method_version"] = METHOD_VERSION
    return out


# ---------------------------------------------------------------- rendering

_CHIP = {
    "in data": ("#d4edda", "#155724", "&#10003;"),
    "partly in data": ("#fff3cd", "#856404", "&#9680;"),
    "not in your data yet": ("#fff3cd", "#856404", "&#9680;"),
    "upcoming": ("#f8d7da", "#721c24", "&#9675;"),
}


def _wave_strip_html(ws: pd.DataFrame, now_pt: pd.Timestamp) -> str:
    chips = []
    for r in ws.itertuples():
        bg, fg, sym = _CHIP[r.state]
        name = {"Ecampus open to Corvallis students": "Ecampus to Corvallis students"}.get(r.wave, r.wave)
        when = r.opens.strftime("%a %d %b")
        if r.state == "upcoming":
            days = (r.opens.normalize() - now_pt.normalize()).days
            note = f"opens {when}" + (f", in {days} day{'s' if days != 1 else ''}" if days > 0 else ", today")
        elif r.state == "in data":
            note = f"opened {when}"
        elif r.state == "partly in data":
            note = f"opened {when}, under a day in your data"
        else:
            note = f"opened {when}, not in your data: git pull"
        weight = "600" if r.key else "400"
        chips.append(
            f'<span style="display:inline-block;padding:4px 11px;margin:3px 6px 3px 0;border-radius:14px;'
            f'background:{bg};color:{fg};font-size:0.85em;font-weight:{weight}">{sym} {name} '
            f'<span style="font-weight:400;opacity:0.85">· {note}</span></span>')
    return "".join(chips)


def render_rebalancing_panel(db_path, classes_begin):
    import streamlit as st
    st.header("Projected final enrolment")
    st.caption("For in-term rebalancing: which courses are heading short of seats, and which will have slack.")
    db_path = Path(db_path)
    E = load_events(db_path)
    manual = load_manual_capacity(db_path)
    cal = load_calendar(db_path)
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
    tbl, meta = build_projection(E, term, classes_begin, manual, cal)
    if tbl.empty:
        st.info("No sections to project for this term.")
        return

    st.markdown("**Who has been able to register**")
    if meta["waves"] is not None:
        st.markdown(_wave_strip_html(meta["waves"], meta["now_pt"]), unsafe_allow_html=True)
        st.caption("Green: in your data. Amber: opened, but not yet (fully) in the data you have; "
                   "run `git pull`. Red: not open yet. Bold waves drive the forecast.")
    else:
        st.caption("No registration calendar on file for this term "
                   "(add it to data/registration_calendar.csv).")

    if not show_small:
        tbl = tbl[tbl.Capacity.isna() | (tbl.Capacity >= SMALL_COURSE_CAP)]
    st.caption(f"Data as of {meta['as_of']} (snapshot ~09:00 Pacific) · day {meta['d_now']} · "
               f"reference terms: {', '.join(meta['refs']) or 'none'}. "
               "Grey numbers are provisional: some of the students who take that course "
               "have not been able to register yet.")

    show = tbl.copy()
    show["Fill now"] = show.FillNow.map(lambda x: "" if pd.isna(x) else f"{x:.0%}")
    show["Projected"] = show.Projected.map(lambda x: "" if pd.isna(x) else f"{x:.0f}")
    show["Range"] = [("" if pd.isna(lo) else f"{lo:.0f}-{hi:.0f}") for lo, hi in zip(show.Low, show.High)]
    show["Capacity"] = show.Capacity.map(lambda x: "" if pd.isna(x) else f"{x:.0f}")
    show["Gap"] = show.Gap.map(lambda x: "" if pd.isna(x) else f"{x:+.0f}")
    show = show.rename(columns={"WaitingOn": "Waiting on", "LastFinal": "Last same-season final",
                                "Floor": "Ref sold out"})
    cols = ["Course", "Mode", "Now", "Fill now", "Projected", "Range", "Capacity", "Gap",
            "Status", "Read", "Waiting on", "Last same-season final", "Ref sold out", "Basis"]
    show = show[cols]

    def _row_style(row):
        styles = [""] * len(row)
        provisional = row["Read"] == "waiting"
        for i, c in enumerate(row.index):
            if c == "Status" and row["Status"] in STATUS_FILL:
                styles[i] = f"background-color: {STATUS_FILL[row['Status']]}"
            elif provisional and c in ("Projected", "Range", "Gap", "Read"):
                styles[i] = "color: #9a9a9a"
            elif c == "Waiting on":
                if row["Waiting on"] == "all in":
                    styles[i] = "color: #155724"
                elif row["Waiting on"]:
                    styles[i] = "color: #a71d2a"
            elif c == "Read" and row["Read"] == "firm":
                styles[i] = "color: #155724; font-weight: 600"
        return styles

    st.dataframe(show.style.apply(_row_style, axis=1), width="stretch", hide_index=True)

    with st.expander("How the projection works"):
        st.markdown(
            "- **Projected** = today's count x (reference term's final / its count at the same point), "
            "averaged over prior terms of the same season. **Range** is the spread across those terms.\n"
            "- **Same point** is measured from the most recent registration wave both terms have on file "
            "(see **Basis**); otherwise days since registration opened, or days before term start.\n"
            "- **Waiting on** lists the registration waves not yet in your data for that row's students: "
            "the class-standing waves for every course, plus the week-10 release for Ecampus sections, "
            "before which Corvallis students cannot register for them. **Read** is *waiting* until those "
            f"are in, *settling* for {SETTLE_DAYS} days after the last one, then *firm*.\n"
            "- Rows are ordered by status (full now, likely to fill, on track, likely slack), then by gap.\n"
            "- **Gap** = projected minus capacity. A positive gap means "
            "the course will likely fill. It does not measure unmet demand: early in the ramp the multiplier "
            "overstates it, and when the reference term sold out it understates it. **Last same-season final** "
            "is the better guide to how many students a course actually served.\n"
            f"- **Likely slack** means projected fill at or below {SLACK_FILL:.0%}.\n"
            "- **Ref sold out**: a reference term filled its seats, so it shows what the seats held, not "
            "what students wanted.\n"
            "- Lecture and primary sections only. Cascades excluded.\n"
            "- Registration waves: `data/registration_calendar.csv` (from the registrar's priority "
            "registration page each term). Missing historical capacities: `data/manual_capacity.csv`."
        )
