"""
osu_enrollment_snapshot_classes_api.py

Version that:
- Uses the classes.oregonstate.edu search endpoint to find sections
- Uses the details endpoint with the CORRECT payload:
    {
      "group":   "code:{CODE}",
      "key":     "crn:{CRN}",
      "srcdb":   "{TERM_SRCDB}",
      "matched": "crn:CRN1,CRN2,..."
    }
- Extracts true enrollment ("enrollment") and capacity ("max_enroll")
- Logs snapshots into a local SQLite DB for the Streamlit dashboard.

Run with:
    python osu_enrollment_snapshot_classes_api.py
"""

import datetime as dt
import json
import pathlib
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Iterable

from zoneinfo import ZoneInfo

# OSU local timezone (Corvallis, OR)
OSU_TZ = ZoneInfo("America/Los_Angeles")

def osu_now() -> dt.datetime:
    """Current datetime in OSU local time (PST/PDT)."""
    return dt.datetime.now(dt.timezone.utc).astimezone(OSU_TZ)

def osu_today() -> dt.date:
    """Today's date in OSU local time (PST/PDT)."""
    return osu_now().date()
import pandas as pd
import requests


# ---------- CONFIG ----------

CLASSES_API_URL = "https://classes.oregonstate.edu/api/"

SEARCH_QUERY = {
    "page": "fose",
    "route": "search",
}

DETAILS_QUERY = {
    "page": "fose",
    "route": "details",
}

TRACK_SUBJECTS = {"SUS"}          # all SUS courses
TRACK_EXACT_CODES = {"AEC 230X"}  # explicit extra codes

# Campus filter – None means "all campuses"
CAMP_FILTER = None

DB_PATH = pathlib.Path("osu_enrollment_log_classes.db")
print(f"[api] DB_PATH = {DB_PATH.resolve()}")

# ---------- TERM / DATE LOGIC ----------

TERM_CODE_MAP = {
    "Fall": "01",
    "Winter": "02",
    "Spring": "03",
}


def estimate_classes_begin(year: int, term: str) -> dt.date:
    """
    Estimate the 'classes begin' date from OSU's stable quarter pattern.

    Based on OSU multi-year calendars:
      - Fall:   last Wednesday in September
      - Winter: first Monday on or after Jan 3
      - Spring: first Monday on or after Mar 29
    """
    term = term.lower()
    if term == "fall":
        # Last Wednesday in September
        next_month = dt.date(year, 10, 1)
        last_day = next_month - dt.timedelta(days=1)
        d = last_day
        while d.weekday() != 2:  # 2 = Wednesday
            d -= dt.timedelta(days=1)
        return d

    elif term == "winter":
        # First Monday on or after Jan 3
        d = dt.date(year, 1, 3)
        offset = (0 - d.weekday()) % 7  # 0 = Monday
        return d + dt.timedelta(days=offset)

    elif term == "spring":
        # First Monday on or after Mar 29
        d = dt.date(year, 3, 29)
        offset = (0 - d.weekday()) % 7  # 0 = Monday
        return d + dt.timedelta(days=offset)

    else:
        raise ValueError(f"Unsupported term for classes_begin: {term}")


def determine_term_for_today(today: Optional[dt.date] = None) -> Dict[str, Any]:
    """
    Determine which term we should track today, based on a sliding window
    around the estimated first day of classes.

    For each candidate term (Fall, Winter, Spring) in years [today.year-1,
    today.year, today.year+1], we compute:

        classes_begin = estimate_classes_begin(year, term_name)
        window_start  = classes_begin - 50 days
        window_end    = classes_begin + 6 days

    If today's date falls inside any window, we pick that term (earliest
    classes_begin if more than one). Otherwise, we pick the future term
    whose classes_begin is nearest in time, and mark within_window=False.
    """
    if today is None:
        today = osu_today()

    candidates: List[Dict[str, Any]] = []

    for year in range(today.year - 1, today.year + 2):
        for term_name, code in TERM_CODE_MAP.items():
            classes_begin = estimate_classes_begin(year, term_name)
            window_start = classes_begin - dt.timedelta(days=50)
            window_end = classes_begin + dt.timedelta(days=6)
            srcdb = f"{classes_begin.year}{code}"
            candidates.append(
                {
                    "srcdb": srcdb,
                    "term_name": term_name,
                    "year": classes_begin.year,
                    "classes_begin": classes_begin,
                    "window_start": window_start,
                    "window_end": window_end,
                    "term_label": f"{term_name} {classes_begin.year}",
                }
            )

    # Check if today is inside any tracking window
    in_window = [
        c for c in candidates if c["window_start"] <= today <= c["window_end"]
    ]
    if in_window:
        chosen = sorted(in_window, key=lambda c: c["classes_begin"])[0]
        chosen["within_window"] = True
        return chosen

    # Otherwise pick nearest future term
    future_terms = [c for c in candidates if c["classes_begin"] >= today]
    if future_terms:
        chosen = sorted(future_terms, key=lambda c: c["classes_begin"])[0]
    else:
        # Fallback: nearest by absolute distance
        chosen = sorted(
            candidates,
            key=lambda c: abs((c["classes_begin"] - today).days),
        )[0]

    chosen["within_window"] = False
    return chosen


# ---------- API CALLS ----------

def build_search_payload(srcdb: str, keyword: str, camp: str = CAMP_FILTER) -> Dict[str, Any]:
    """
    Build the JSON payload for the search POST request, matching DevTools:

        {
          "other": {"srcdb": "202602"},
          "criteria": [
            {"field": "keyword", "value": "SUS"},
            {"field": "camp",    "value": "DB,DI,DR"}
          ]
        }
    """
    criteria = [
        {
            "field": "keyword",
            "value": keyword,
        }
    ]

    # Add campus filter if provided
    if camp:
        criteria.append(
            {
                "field": "camp",
                "value": camp,
            }
        )

    return {
        "other": {
            "srcdb": srcdb,
        },
        "criteria": criteria,
    }

import textwrap  # keep this near the top or here; either is fine

def fetch_class_search(srcdb: str, keyword: str) -> dict:
    """
    Call the OSU class search API and return parsed JSON.

    Mirrors the browser request seen in DevTools:

      curl 'https://classes.oregonstate.edu/api/?page=fose&route=search&keyword=SUS&camp=DB%2CDI%2CDR' \
        -X POST \
        --data-raw '{"other":{"srcdb":"202602"},"criteria":[{"field":"keyword","value":"SUS"},{"field":"camp","value":"DB,DI,DR"}]}'

    We:
      - send `keyword` and `camp` as query params, AND
      - include them in the JSON body via build_search_payload.
    """
    # JSON payload (body)
    payload = build_search_payload(srcdb, keyword, camp=CAMP_FILTER)

    # Query string params: page, route, keyword, camp
    params = {
    **SEARCH_QUERY,  # {"page": "fose", "route": "search"}
    }
    
    headers = {
        # Match the browser UA from your cURL
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                      "Version/26.1 Safari/605.1.15",
        "Content-Type": "application/json",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "X-Requested-With": "XMLHttpRequest",
        "Origin": "https://classes.oregonstate.edu",
        "Referer": f"https://classes.oregonstate.edu/?keyword={keyword}&srcdb={srcdb}&camp={CAMP_FILTER}",

        # *** IMPORTANT: paste the cookie string from your cURL here ***
        # Everything after "Cookie: " in your curl, *as one long string*.
        "Cookie": (
            "_ga_RHQKNVWJCF=GS2.1.s1765463753$o103$g0$t1765463754$j59$l0$h0; "
            "_fbp=fb.1.1755495424633.3934555457627417; "
            "ttcsid=1765427835915::Vd8KuwFkJuyK5cbUQ_HB.92.1765427859929.0; "
            "ttcsid_BSLGAR0QH7P7CAP6MM20=1765427835914::T3Ly4ErQbauqZv1dibS1.2.1765427859929.1; "
            "_ga=GA1.1.1946022616.1755495424; "
            "_scid_r=kHNJm9hlVc1MQDXELS-1jnAIBRuQQQrT6BfCtg; "
            "_tt_enable_cookie=1; "
            "_ttp=01K2XTWRBYVGS58AHCQ40Y9D06_.tt.1; "
            "_ga_FKJ26XQJCS=GS2.1.s1765427820$o21$g1$t1765427821$j59$l0$h0; "
            "_screload=; "
            "_gid=GA1.2.167845864.1765425133; "
            "_ga_24GLZC9ZR6=GS2.1.s1765272398$o23$g0$t1765272406$j52$l0$h0; "
            "_sctr=1%7C1765123200000; "
            "_ga_QYPPJQ8GWS=GS2.1.s1764916612$o10$g0$t1764916612$j60$l0$h0; "
            "_ga_LDRC9SJDTS=GS2.1.s1763531309$o1$g1$t1763532110$j59$l0$h0; "
            "_gcl_au=1.1.1004346326.1763342953; "
            "_ga_P4TKPDKRPV=GS2.1.s1763158649$o2$g1$t1763158675$j34$l0$h0; "
            "_ga_D7QRR38Y63=GS2.1.s1763110259$o1$g0$t1763110259$j60$l0$h0; "
            "_ga_9RY9MJX4BH=GS2.1.s1760653131$o2$g1$t1760654477$j59$l0$h0; "
            "_ga_S1XHD0R88P=GS2.2.s1760413574$o1$g0$t1760413574$j60$l0$h0; "
            "_ga_7F0QY2CKEC=GS2.1.s1759499782$o3$g0$t1759499782$j60$l0$h0; "
            "_ga_C3GW4PLYQ1=GS2.1.s1758812785$o1$g1$t1758812928$j60$l0$h0; "
            "_ga_K3VPG0MT96=GS2.1.s1758808084$o2$g0$t1758808084$j60$l0$h0; "
            "_ga_9EQSEP9D7N=GS2.1.s1758678518$o3$g0$t1758678518$j60$l0$h0; "
            "_ga_03KTKDRW2Q=GS2.1.s1758678484$o2$g0$t1758678484$j60$l0$h0; "
            "apt.uid=AP-JB0A21DE9MYX-2-1755504221702-91619822.0.2.3635bc02-2ddf-4575-a526-097013e05729; "
            "_ga_CMEYDDKYP2=GS2.1.s1758101842$o2$g0$t1758101842$j60$l0$h0; "
            "_ga_8WH466WTJE=GS2.1.s1757472066$o1$g1$t1757472136$j60$l0$h0; "
            "__zlcmid=1TZnu7wXEQ62l2c; "
            "_ga_QC3MB7K855=GS2.1.s1757032019$o1$g0$t1757032019$j60$l0$h0; "
            "_ga_20HQG504MJ=GS2.1.s1756301931$o1$g0$t1756301931$j60$l0$h0; "
            "_ga_5VX9TV05LH=GS2.1.s1756181308$o1$g0$t1756181310$j58$l0$h0; "
            "ttcsid_CCUU293C77U2F908M0U0=1756181309353::tLVRgUeeyFUikSDg-h8P.1.1756181310436; "
            "__gsas=ID=387d2843ddc1d981:T=1755660004:RT=1755660004:S=ALNI_MYl6wWwRPaFCO4jqkOf5n1NHFBdeg; "
            "_scid=f_NJm9hlVc1MQDXELS-1jnAIBRuQQQrT"
        ),
    }

    print(f"Requesting search data for term {srcdb}, keyword '{keyword}'")
    resp = requests.post(
        CLASSES_API_URL,
        params=params,
        json=payload,
        headers=headers,
        timeout=30,
    )

    # HTTP error handling
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        # Show a raw slice so we can see HTML / error text
        preview = resp.text[:800]
        raise RuntimeError(
            f"Class search HTTP error for srcdb={srcdb}, keyword='{keyword}'. "
            f"Status={resp.status_code}, URL={resp.url}, "
            f"Content-Type={resp.headers.get('Content-Type')!r}. "
            f"Body preview: {preview}"
        ) from e

    # JSON parsing
    try:
        return resp.json()
    except json.JSONDecodeError as e:
        preview = textwrap.shorten(resp.text.replace("\n", " "), width=500)
        raise RuntimeError(
            f"Class search did not return valid JSON for srcdb={srcdb}, keyword='{keyword}'. "
            f"Status={resp.status_code}, URL={resp.url}, "
            f"Content-Type={resp.headers.get('Content-Type')!r}. "
            f"Body preview: {preview}"
        ) from e

def build_details_payload(srcdb: str, code: str, crn: str, matched_crns: List[str]) -> Dict[str, Any]:
    """
    Build the JSON payload for the details POST request, matching what the
    browser sends. From DevTools, this looks like (URL-decoded):

        {
          "group":   "code:SUS 102",
          "key":     "crn:37717",
          "srcdb":   "202602",
          "matched": "crn:37716,36727,37717,..."
        }
    """
    # ensure all CRNs are strings and non-empty
    crn_list = [str(c).strip() for c in matched_crns if str(c).strip()]
    if not crn_list:
        crn_list = [crn]

    matched_str = "crn:" + ",".join(crn_list)

    return {
        "group": f"code:{code}",
        "key": f"crn:{crn}",
        "srcdb": srcdb,
        "matched": matched_str,
    }


def fetch_section_details(srcdb: str, code: str, crn: str, matched_crns: List[str]) -> Dict[str, Any]:
    """
    Call the classes.oregonstate.edu details API for a single section,
    using the observed payload shape and the same browser-style headers/cookies
    as the search request.
    """
    payload = build_details_payload(srcdb, code, crn, matched_crns)

    # Reuse the same headers we used for search so the server treats us
    # like the browser (cookies, UA, X-Requested-With, etc.).
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                      "Version/26.1 Safari/605.1.15",
        "Content-Type": "application/json",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "X-Requested-With": "XMLHttpRequest",
        "Origin": "https://classes.oregonstate.edu",
        "Referer": (
            f"https://classes.oregonstate.edu/?keyword=SUS&srcdb={srcdb}"
            f"&camp={CAMP_FILTER}"
        ),
        # ⚠️ Same cookie string you already pasted into fetch_class_search
        "Cookie": (
            "_ga_RHQKNVWJCF=GS2.1.s1765463753$o103$g0$t1765463754$j59$l0$h0; "
            "_fbp=fb.1.1755495424633.3934555457627417; "
            "ttcsid=1765427835915::Vd8KuwFkJuyK5cbUQ_HB.92.1765427859929.0; "
            "ttcsid_BSLGAR0QH7P7CAP6MM20=1765427835914::T3Ly4ErQbauqZv1dibS1.2.1765427859929.1; "
            "_ga=GA1.1.1946022616.1755495424; "
            "_scid_r=kHNJm9hlVc1MQDXELS-1jnAIBRuQQQrT6BfCtg; "
            "_tt_enable_cookie=1; "
            "_ttp=01K2XTWRBYVGS58AHCQ40Y9D06_.tt.1; "
            "_ga_FKJ26XQJCS=GS2.1.s1765427820$o21$g1$t1765427821$j59$l0$h0; "
            "_screload=; "
            "_gid=GA1.2.167845864.1765425133; "
            "_ga_24GLZC9ZR6=GS2.1.s1765272398$o23$g0$t1765272406$j52$l0$h0; "
            "_sctr=1%7C1765123200000; "
            "_ga_QYPPJQ8GWS=GS2.1.s1764916612$o10$g0$t1764916612$j60$l0$h0; "
            "_ga_LDRC9SJDTS=GS2.1.s1763531309$o1$g1$t1763532110$j59$l0$h0; "
            "_gcl_au=1.1.1004346326.1763342953; "
            "_ga_P4TKPDKRPV=GS2.1.s1763158649$o2$g1$t1763158675$j34$l0$h0; "
            "_ga_D7QRR38Y63=GS2.1.s1763110259$o1$g0$t1763110259$j60$l0$h0; "
            "_ga_9RY9MJX4BH=GS2.1.s1760653131$o2$g1$t1760654477$j59$l0$h0; "
            "_ga_S1XHD0R88P=GS2.2.s1760413574$o1$g0$t1760413574$j60$l0$h0; "
            "_ga_7F0QY2CKEC=GS2.1.s1759499782$o3$g0$t1759499782$j60$l0$h0; "
            "_ga_C3GW4PLYQ1=GS2.1.s1758812785$o1$g1$t1758812928$j60$l0$h0; "
            "_ga_K3VPG0MT96=GS2.1.s1758808084$o2$g0$t1758808084$j60$l0$h0; "
            "_ga_9EQSEP9D7N=GS2.1.s1758678518$o3$g0$t1758678518$j60$l0$h0; "
            "_ga_03KTKDRW2Q=GS2.1.s1758678484$o2$g0$t1758678484$j60$l0$h0; "
            "apt.uid=AP-JB0A21DE9MYX-2-1755504221702-91619822.0.2.3635bc02-2ddf-4575-a526-097013e05729; "
            "_ga_CMEYDDKYP2=GS2.1.s1758101842$o2$g0$t1758101842$j60$l0$h0; "
            "_ga_8WH466WTJE=GS2.1.s1757472066$o1$g1$t1757472136$j60$l0$h0; "
            "__zlcmid=1TZnu7wXEQ62l2c; "
            "_ga_QC3MB7K855=GS2.1.s1757032019$o1$g0$t1757032019$j60$l0$h0; "
            "_ga_20HQG504MJ=GS2.1.s1756301931$o1$g0$t1756301931$j60$l0$h0; "
            "_ga_5VX9TV05LH=GS2.1.s1756181308$o1$g0$t1756181310$j58$l0$h0; "
            "ttcsid_CCUU293C77U2F908M0U0=1756181309353::tLVRgUeeyFUikSDg-h8P.1.1756181310436; "
            "__gsas=ID=387d2843ddc1d981:T=1755660004:RT=1755660004:S=ALNI_MYl6wWwRPaFCO4jqkOf5n1NHFBdeg; "
            "_scid=f_NJm9hlVc1MQDXELS-1jnAIBRuQQQrT"
        ),
    }

    resp = requests.post(
        CLASSES_API_URL,
        params=DETAILS_QUERY,
        json=payload,
        headers=headers,
        timeout=30,
    )

    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        # We let enrich_with_details() handle the exception, but give a readable message.
        raise RuntimeError(
            f"Details HTTP error for code={code}, crn={crn}, srcdb={srcdb}. "
            f"Status={resp.status_code}, URL={resp.url}"
        ) from e

    try:
        return resp.json()
    except json.JSONDecodeError as e:
        # Again, let the caller catch this, but include a useful preview.
        preview = resp.text[:800]
        raise RuntimeError(
            f"Details did not return valid JSON for code={code}, crn={crn}, srcdb={srcdb}. "
            f"Status={resp.status_code}, Content-Type={resp.headers.get('Content-Type')!r}. "
            f"Body preview: {preview}"
        ) from e


# ---------- PARSING ----------

def safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        s = str(value).strip()
        if not s:
            return None
        return int(s)
    except Exception:
        return None


def split_code(code: str) -> Tuple[Optional[str], Optional[str]]:
    """Split 'SUS 304' -> ('SUS', '304')."""
    if not code:
        return None, None
    parts = code.split()
    if len(parts) >= 2:
        return parts[0], " ".join(parts[1:])
    return None, code


def filter_section(rec: Dict[str, Any]) -> bool:
    """
    Decide whether to keep a section in our log:
      - Subject in TRACK_SUBJECTS or full code in TRACK_EXACT_CODES
      - Not cancelled (isCancelled != '1')
    """
    code = rec.get("code", "")
    subject, _ = split_code(code)

    if TRACK_SUBJECTS and subject not in TRACK_SUBJECTS and code not in TRACK_EXACT_CODES:
        return False

    if str(rec.get("isCancelled", "")).strip() == "1":
        return False

    return True


def normalize_results(raw_json: Dict[str, Any], srcdb: str) -> pd.DataFrame:
    """
    Convert the raw search JSON ('results') into a DataFrame of basic
    section rows (no enrollment yet).
    """
    results = raw_json.get("results", [])
    records: List[Dict[str, Any]] = []

    for rec in results:
        if not filter_section(rec):
            continue

        code = rec.get("code", "")
        subject, course_number = split_code(code)

        record: Dict[str, Any] = {
            "term_srcdb": srcdb or rec.get("srcdb"),
            "key": rec.get("key"),
            "subject": subject,
            "course_number": course_number,
            "code": code,
            "crn": rec.get("crn"),
            "section": rec.get("no"),
            "title": rec.get("title"),
            "campus_code": rec.get("camp"),
            "schedule_type": rec.get("schd"),
            "status_code": rec.get("stat"),
            "is_cancelled": str(rec.get("isCancelled", "")).strip() == "1",
            "meets": rec.get("meets"),
            "instructor": rec.get("instr"),
            "start_date": rec.get("start_date"),
            "end_date": rec.get("end_date"),
        }

        records.append(record)

    if not records:
        return pd.DataFrame()

    return pd.DataFrame.from_records(records)


def find_value_by_keys(obj: Any, keys: Iterable[str]) -> Optional[Any]:
    """
    Recursively search a nested dict/list structure for the first occurrence
    of any key in `keys`, and return its value.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in keys:
                return v
            found = find_value_by_keys(v, keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = find_value_by_keys(item, keys)
            if found is not None:
                return found
    return None


def enrich_with_details(df: pd.DataFrame, srcdb: str) -> pd.DataFrame:
    """
    For each row in df (which has 'code' and 'crn'), call the details
    endpoint to fetch true enrollment and capacity fields, and merge them.
    """
    if df.empty:
        return df

    df = df.copy()

    # For the "matched" field we need the list of CRNs for each course code
    code_to_crns: Dict[str, List[str]] = (
        df.groupby("code")["crn"]
        .apply(lambda s: [str(x).strip() for x in s if str(x).strip()])
        .to_dict()
    )

    enrolled_list: List[Optional[int]] = []
    capacity_list: List[Optional[int]] = []

    for idx, (_, row) in enumerate(df.iterrows()):
        code = row.get("code")
        crn = str(row.get("crn") or "").strip()

        if not code or not crn:
            enrolled_list.append(None)
            capacity_list.append(None)
            continue

        matched_crns = code_to_crns.get(code, [crn])

        try:
            details = fetch_section_details(srcdb, code, crn, matched_crns)
        except Exception as e:
            print(f"Warning: details fetch failed for code={code!r}, crn={crn!r}: {e}")
            enrolled_list.append(None)
            capacity_list.append(None)
            continue

        # Debug print for first few rows
        if idx < 3:
            try:
                snippet = json.dumps(details, indent=2)[:1000]
            except TypeError:
                snippet = str(details)[:1000]
            print(f"\nDEBUG details for code={code}, crn={crn}")
            print(snippet)

        # Extract enrollment and capacity
        enrolled_val = find_value_by_keys(
            details, {"enrollment", "enrolled", "enrl", "enroll"}
        )
        capacity_val = find_value_by_keys(
            details, {"max_enroll", "capacity", "cap", "max_enrollment"}
        )

        if idx < 3:
            print(f"  -> extracted enrolled={enrolled_val}, capacity={capacity_val}\n")

        enrolled_list.append(safe_int(enrolled_val))
        capacity_list.append(safe_int(capacity_val))

    df["enrolled"] = enrolled_list
    df["capacity"] = capacity_list
    return df


# ---------- DB APPEND ----------

def append_snapshot_to_db(df: pd.DataFrame, db_path: pathlib.Path) -> None:
    """
    Append a snapshot (DataFrame) to the SQLite DB.

    Adds:
      - snapshot_date (YYYY-MM-DD)
      - timestamp (ISO datetime)
    """
    if df.empty:
        print("No rows after filtering; skipping DB append.")
        return

    df = df.copy()
    now = osu_now()
    df["snapshot_date"] = now.date().isoformat()
    df["timestamp"] = now.isoformat(timespec="seconds")

    # Consistent column order
    col_order = [
        "snapshot_date",
        "timestamp",
        "term_srcdb",
        "subject",
        "course_number",
        "code",
        "crn",
        "section",
        "title",
        "enrolled",
        "capacity",
        "campus_code",
        "schedule_type",
        "status_code",
        "is_cancelled",
        "meets",
        "instructor",
        "start_date",
        "end_date",
        "key",
    ]

    for col in col_order:
        if col not in df.columns:
            df[col] = None

    df = df[col_order]

    with sqlite3.connect(db_path) as conn:
        df.to_sql("enrollment", conn, if_exists="append", index=False)
        # Ensure a friendly view name for SUS enrollment
        conn.execute(
            "CREATE VIEW IF NOT EXISTS sus_enrollment AS SELECT * FROM enrollment"
        )

    print(f"Appended {len(df)} rows to {db_path} (table 'enrollment').")


# ---------- MAIN ----------

def main() -> None:
    today = osu_today()
    term_info = determine_term_for_today(today)

    srcdb = term_info["srcdb"]
    term_label = term_info["term_label"]
    classes_begin = term_info["classes_begin"]
    window_start = term_info["window_start"]
    window_end = term_info["window_end"]
    within_window = term_info["within_window"]

    print(
        f"[{today.isoformat()}] Tracking term: {term_label} "
        f"(srcdb={srcdb}); window {window_start} → {window_end} "
        f"(classes begin {classes_begin})"
    )

    if not within_window:
        print(
            "Today is outside the snapshot window for this term. "
            "Skipping fetch and DB write."
        )
        return

    # ----- NEW: do the search call safely -----
    keyword = "SUS"

    print(f"Requesting search data for term {srcdb}, keyword '{keyword}'")

    try:
        raw_json = fetch_class_search(srcdb=srcdb, keyword=keyword)
    except Exception as e:
        print("\nERROR: Search API request failed.")
        print("Details:", e)
        print("Aborting snapshot without writing to DB.\n")
        return

    # Convert the raw JSON to a DataFrame of sections
    df = normalize_results(raw_json, srcdb=srcdb)

    print(f"Parsed {len(df)} sections after filtering (before details).")
    if df.empty:
        print("No sections to enrich; exiting.")
        return

    # ----- Fetch enrollment / capacity details -----
    print("Fetching details (enrollment / max_enroll) per section...")
    df = enrich_with_details(df, srcdb=srcdb)

    print("Sample rows after enrichment:")
    print(df[["code", "section", "crn", "enrolled", "capacity"]].head())

    # ----- Append snapshot to DB (append-only) -----
    append_snapshot_to_db(df, DB_PATH)

if __name__ == "__main__":
    main()
