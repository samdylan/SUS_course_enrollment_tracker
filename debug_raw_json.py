import json
from osu_enrollment_snapshot_classes_api import fetch_class_search, TERM_SRCDB, DEFAULT_KEYWORD

if __name__ == "__main__":
    raw = fetch_class_search(srcdb=TERM_SRCDB, keyword=DEFAULT_KEYWORD)

    # Just print the keys we care about for the first few records
    for rec in raw.get("results", [])[:10]:
        print(
            rec.get("code"),
            rec.get("no"),
            {
                k: rec.get(k)
                for k in rec.keys()
                if k.lower() in {"total", "cap", "enrl", "enroll", "enrolled", "avail", "seats"}
            }
        )