"""
Seed data/forecast_errors.csv by replaying the rebalancing panel over past terms.

For each completed term, the panel is rebuilt as it would have looked each morning
before classes began, using only data logged up to that day, and every course x
modality forecast is scored against the enrolment actually reached on day 0.
The error used for confidence is |expected - final| / final, where expected is the
projection capped at the seats available that day (what the panel calls the
expected final).

Usage:
  python scripts/backtest_forecasts.py              # all completed terms
  python scripts/backtest_forecasts.py 202602 ...   # specific terms

Back-test rows for a term are replaced on each run; rows scored from the archived
daily forecasts (source = archive) are never touched. Re-run this after changing the
forecasting method, so confidence reflects the method the panel actually uses.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dashboards"))

import rebalancing as rb                      # noqa: E402
import osu_enrollment_dashboard_v5 as dash    # noqa: E402

OUT = ROOT / "data" / "forecast_errors.csv"
MIN_FINAL = 15          # tiny sections make percentage errors meaningless
COLS = ["term_srcdb", "season", "course", "mode", "day", "stage", "read", "projected",
        "capacity_at_forecast", "expected", "final", "abs_pct_error", "source",
        "method_version", "data_as_of"]


def finals(E, term, start):
    enr, _ = rb._grid(E, term, start)
    if enr is None or 0 not in enr.index:
        return {}
    keymode = E.drop_duplicates("key").set_index("key")["mode"]
    keycode = E.drop_duplicates("key").set_index("key")["code"]
    out = {}
    for c in enr.columns:
        k = (keycode.get(c), keymode.get(c))
        out[k] = out.get(k, 0.0) + float(np.nan_to_num(enr.loc[0, c]))
    return out


def has_reference(E, term):
    return any(r[-2:] == term[-2:] and int(r) < int(term) for r in E.term_srcdb.unique())


def backtest_term(E, term, manual, cal):
    cb = dash.estimate_classes_begin_from_srcdb
    start = cb(term)
    fin = finals(E, term, start)
    if not fin:
        return pd.DataFrame(columns=COLS)
    g = E[E.term_srcdb == term]
    first = int((g.day.min() - start).days)
    rows = []
    for d in range(max(first, -120), 0):
        cut = E[(E.term_srcdb != term) | (E.day <= start + pd.Timedelta(days=d))]
        now_pt = start + pd.Timedelta(days=d, hours=rb.SNAPSHOT_HOUR_PT)
        tbl, meta = rb.build_projection(cut, term, cb, manual, cal, now_pt=now_pt)
        if tbl.empty or meta["d_now"] != d:
            continue
        for r in tbl.itertuples():
            f = fin.get((r.Course, r.Mode))
            if f is None or f < MIN_FINAL or pd.isna(r.Expected):
                continue
            rows.append(dict(
                term_srcdb=term, season=meta["season"], course=r.Course, mode=r.Mode, day=d,
                stage=rb.stage_of(r.Read), read=r.Read, projected=round(r.Projected, 2),
                capacity_at_forecast=r.Capacity, expected=round(r.Expected, 2), final=f,
                abs_pct_error=round(abs(r.Expected - f) / f, 4), source="backtest",
                method_version=rb.METHOD_VERSION, data_as_of=str(meta["as_of"])))
    return pd.DataFrame(rows, columns=COLS)


def main(argv) -> int:
    db = Path(dash.DB_PATH)
    E = rb.load_events(db)
    manual = rb.load_manual_capacity(db)
    cal = rb.load_calendar(db)
    today = rb.now_pacific().normalize().tz_localize(None)
    if argv:
        terms = argv
    else:
        terms = []
        for t in sorted(E.term_srcdb.unique(), key=int):
            s = dash.estimate_classes_begin_from_srcdb(t)
            if s is not None and s <= today and has_reference(E, t):
                terms.append(t)
    old = pd.read_csv(OUT, dtype={"term_srcdb": str}) if OUT.exists() else pd.DataFrame(columns=COLS)
    for t in terms:
        new = backtest_term(E, t, manual, cal)
        old = old[~((old.term_srcdb == t) & (old.source == "backtest"))]
        if len(new):
            old = new if old.empty else pd.concat([old, new], ignore_index=True)
        print(f"{t}: {len(new)} back-test forecasts scored")
    old = old.sort_values(["term_srcdb", "source", "day", "course", "mode"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    old[COLS].to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(old)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
