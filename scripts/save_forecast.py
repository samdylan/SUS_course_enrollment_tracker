"""
Save today's projected-final-enrolment table for each upcoming term.

Run nightly by the GitHub Action after the snapshot. Writes
forecasts/<term_srcdb>/<data date>.csv once per data date. Existing files are
never overwritten, so the archive is a faithful record of what the dashboard
panel showed on each day, including which registration waves were in and
which reference terms and method version were used.

The projection logic and the term start dates both come from the dashboard
code, so there is one source of truth for each.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dashboards"))

import rebalancing as rb                      # noqa: E402
import osu_enrollment_dashboard_v5 as dash    # noqa: E402  (term start dates)


def main() -> int:
    db = Path(dash.DB_PATH)
    E = rb.load_events(db)
    manual = rb.load_manual_capacity(db)
    cal = rb.load_calendar(db)
    terms = sorted(E[(E.src == "live") & (E.enrolled > 0)].term_srcdb.unique(), key=int)
    wrote = 0
    for t in terms:
        if dash.estimate_classes_begin_from_srcdb(t) is None:
            continue
        tbl, meta = rb.build_projection(E, t, dash.estimate_classes_begin_from_srcdb, manual, cal)
        if tbl.empty or meta["d_now"] >= 0:      # only terms that have not started
            continue
        out = ROOT / "forecasts" / t / f"{meta['as_of']}.csv"
        if out.exists():
            print(f"exists, left unchanged: {out.relative_to(ROOT)}")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        rb.forecast_records(tbl, meta).to_csv(out, index=False)
        wrote += 1
        print(f"saved {out.relative_to(ROOT)} ({len(tbl)} rows, data as of {meta['as_of']})")
    if not wrote:
        print("no new forecasts to save")
    return 0


if __name__ == "__main__":
    sys.exit(main())
