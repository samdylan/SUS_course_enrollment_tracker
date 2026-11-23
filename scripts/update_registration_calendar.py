"""
Add the registrar's published priority-registration schedule to
data/registration_calendar.csv.

The registrar's page (https://registrar.oregonstate.edu/priority-registration)
shows the schedule for the next term only, so running this nightly captures
each term's schedule as soon as it is posted.

Deliberately cautious. It changes nothing unless it finds:
  - the term, from the "Week 10 ... for <season> <year>" sentence;
  - the Seniors, Juniors, Sophomores and Freshman rows, each with a date
    whose weekday matches the calendar (e.g. "Monday, November 16");
  - those four in chronological order.
Otherwise it prints why and exits without writing. It only ever replaces
rows for that one term whose source starts with "registrar"; rows dated
from the data ("inferred") are kept unless the registrar supplies that wave.
"""
import re
import sys
from datetime import datetime
from html import unescape
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
CAL = ROOT / "data" / "registration_calendar.csv"
URL = "https://registrar.oregonstate.edu/priority-registration"
SOURCE = "registrar priority-registration page (auto)"

MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], start=1)}
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
GROUPS = {  # exact cell text -> (wave, order, key)
    "seniors": ("Seniors", 1, "yes"),
    "juniors": ("Juniors", 2, "yes"),
    "sophomores": ("Sophomores", 3, "yes"),
    "freshman": ("Freshmen", 4, "yes"),
    "freshmen": ("Freshmen", 4, "yes"),
    "nondegree students": ("Nondegree", 6, "no"),
    "nondegree": ("Nondegree", 6, "no"),
}
SEASON_CODE = {"winter": "02", "spring": "03", "summer": "00", "fall": "01"}
DATE_RE = re.compile(
    r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+"
    r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+"
    r"(\d{1,2}),?\s+(\d{1,2}):(\d{2})\s*([AaPp])\.?\s*[Mm]\.?")
WEEK10_RE = re.compile(
    r"Week\s*10\s+of\s+the\s+term\s*\(\s*"
    r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+"
    r"(\d{1,2})\s+at\s+(\d{1,2}):(\d{2})\s*([ap])\.?\s*m\.?\s*\(PT\)\s+for\s+"
    r"(winter|spring|summer|fall)\s+(20\d\d)\s*\)", re.I)


def _text(html: str) -> str:
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", html))).strip()


def term_srcdb(season: str, year: int) -> str:
    season = season.lower()
    label_year = year + 1 if season in ("summer", "fall") else year
    return f"{label_year}{SEASON_CODE[season]}"


def _dt(year, month, day, hh, mm, ap):
    h = int(hh) % 12 + (12 if ap.lower() == "p" else 0)
    return datetime(year, MONTHS[month], int(day), h, int(mm))


APPROX_TERM_START = {"winter": (1, 5), "spring": (3, 30), "summer": (6, 22), "fall": (9, 23)}


def _resolve_year(weekday, month, day, hh, mm, ap, season, term_year):
    """Registration happens in the months before the term. Accept a date only if it falls
    within 200 days before the term starts AND its weekday matches the published one."""
    m_, d_ = APPROX_TERM_START[season.lower()]
    start = datetime(term_year, m_, d_)
    for y in (term_year - 1, term_year):
        try:
            d = _dt(y, month, day, hh, mm, ap)
        except ValueError:
            continue
        if 0 <= (start - d).days <= 200:
            return d if DAYS[d.weekday()] == weekday else None
    return None


def parse(html: str):
    text = _text(html)
    m = WEEK10_RE.search(text)
    if not m:
        return None, "could not find the 'Week 10 ... for <season> <year>' sentence"
    e_month, e_day, e_hh, e_mm, e_ap, season, year = m.groups()
    year = int(year)
    term = term_srcdb(season, year)

    rows = {}
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.S | re.I):
        cells = [_text(c) for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, flags=re.S | re.I)]
        if len(cells) < 2:
            continue
        g = cells[0].strip().lower()
        if g not in GROUPS:
            continue
        dm = DATE_RE.search(" ".join(cells[1:]))
        if not dm:
            continue
        d = _resolve_year(*dm.groups(), season, year)
        if d is None:
            return None, f"weekday does not match calendar for '{cells[0]}: {dm.group(0)}'"
        wave, order, key = GROUPS[g]
        rows[wave] = (order, d, key)

    needed = ["Seniors", "Juniors", "Sophomores", "Freshmen"]
    missing = [w for w in needed if w not in rows]
    if missing:
        return None, f"table rows not found for: {', '.join(missing)}"
    seq = [rows[w][1] for w in needed]
    if seq != sorted(seq):
        return None, "class-standing waves are not in chronological order"

    for y in (year - 1, year):
        try:
            e = _dt(y, e_month, e_day, e_hh, e_mm, e_ap)
        except ValueError:
            continue
        if e.weekday() == 0 and seq[0] < e:          # week 10 is a Monday, after seniors open
            rows["Ecampus open to Corvallis students"] = (5, e, "yes")
            break
    out = pd.DataFrame([
        dict(term_srcdb=term, order=o, wave=w, opens=d.strftime("%Y-%m-%d %H:%M"),
             scope="ecampus" if w.startswith("Ecampus") else "all", key=k, source=SOURCE)
        for w, (o, d, k) in rows.items()]).sort_values("order")
    return out, f"parsed {len(out)} waves for {season.title()} {year} ({term})"


def merge(new: pd.DataFrame) -> bool:
    cur = pd.read_csv(CAL, dtype=str) if CAL.exists() else pd.DataFrame(columns=new.columns)
    term = new.term_srcdb.iloc[0]
    this = cur[cur.term_srcdb == term]
    keep_other_terms = cur[cur.term_srcdb != term]
    keep_inferred = this[~this.source.str.startswith("registrar") & ~this.wave.isin(set(new.wave))]
    merged_term = pd.concat([keep_inferred, new.astype(str)], ignore_index=True)
    old_cmp = this.sort_values(["wave"])[["wave", "opens"]].reset_index(drop=True)
    new_cmp = merged_term.sort_values(["wave"])[["wave", "opens"]].reset_index(drop=True)
    if old_cmp.equals(new_cmp):
        return False
    out = pd.concat([keep_other_terms, merged_term], ignore_index=True)
    out["order"] = out.order.astype(int)
    out = out.sort_values(["term_srcdb", "order"], ascending=[False, True])
    out.to_csv(CAL, index=False)
    return True


def main() -> int:
    try:
        r = requests.get(URL, timeout=45, headers={"User-Agent": "Mozilla/5.0 SUS-enrollment-tracker"})
        r.raise_for_status()
    except Exception as e:  # network trouble must never break the nightly job
        print(f"could not fetch {URL}: {e}; calendar unchanged")
        return 0
    new, msg = parse(r.text)
    print(msg)
    if new is None:
        print("calendar unchanged")
        return 0
    print("calendar updated" if merge(new) else "calendar already up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
