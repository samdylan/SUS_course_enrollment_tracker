"""
Score archived daily forecasts once their term has begun.

Run nightly by the GitHub Action. For every term in forecasts/ that has reached
day 0 in the enrolment log, each saved forecast (forecasts/<term>/<date>.csv) is
compared with the enrolment actually reached on day 0, and the result is added to
data/forecast_errors.csv with source = archive. These rows take precedence over
the back-test replay for the same day, so the panel's Confidence column
recalibrates itself from real forecasts as terms go by.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dashboards"))
sys.path.insert(0, str(ROOT / "scripts"))

import rebalancing as rb                      # noqa: E402
import osu_enrollment_dashboard_v5 as dash    # noqa: E402
from backtest_forecasts import COLS, MIN_FINAL, OUT, finals   # noqa: E402


def main() -> int:
    fdir = ROOT / "forecasts"
    if not fdir.exists():
        print("no archived forecasts yet")
        return 0
    E = rb.load_events(Path(dash.DB_PATH))
    old = pd.read_csv(OUT, dtype={"term_srcdb": str}) if OUT.exists() else pd.DataFrame(columns=COLS)
    added = []
    for tdir in sorted(p for p in fdir.iterdir() if p.is_dir()):
        term = tdir.name
        start = dash.estimate_classes_begin_from_srcdb(term)
        if start is None:
            continue
        fin = finals(E, term, start)
        if not fin:
            continue                       # day 0 not logged yet
        done = set(old[(old.term_srcdb == term) & (old.source == "archive")].data_as_of.astype(str))
        for f in sorted(tdir.glob("*.csv")):
            rec = pd.read_csv(f, dtype={"term_srcdb": str})
            if rec.empty or str(rec.data_as_of.iloc[0]) in done:
                continue
            for r in rec.itertuples():
                final = fin.get((r.Course, r.Mode))
                proj = getattr(r, "Projected", np.nan)
                if final is None or final < MIN_FINAL or pd.isna(proj):
                    continue
                cap = getattr(r, "Capacity", np.nan)
                exp = getattr(r, "Expected", np.nan)
                if pd.isna(exp):
                    exp = min(proj, cap) if not pd.isna(cap) else proj
                read = str(getattr(r, "Read", ""))
                sc = rb.score_forecast(exp, final, cap)
                added.append(dict(
                    term_srcdb=term, season=rb.SEASON.get(term[-2:], "?"), course=r.Course,
                    mode=r.Mode, day=int(r.day), stage=rb.stage_of(read), read=read,
                    projected=round(proj, 2), capacity_at_forecast=cap, expected=round(exp, 2),
                    final=final, final_at_seats=sc[0], beyond_seats=sc[1],
                    abs_pct_error=round(sc[2], 4), source="archive",
                    method_version=getattr(r, "method_version", ""), data_as_of=str(r.data_as_of)))
    if not added:
        print("no newly scorable forecasts")
        return 0
    new = pd.DataFrame(added, columns=COLS)
    out = new if old.empty else pd.concat([old, new], ignore_index=True)
    out = out.sort_values(["term_srcdb", "source", "day", "course", "mode"])
    out[COLS].to_csv(OUT, index=False)
    print(f"scored {len(new)} archived forecasts; {OUT.relative_to(ROOT)} now has {len(out)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
