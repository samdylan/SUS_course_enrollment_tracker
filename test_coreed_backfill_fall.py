import requests
import json
from datetime import datetime
from zoneinfo import ZoneInfo

API_URL = "https://classes.oregonstate.edu/api/"

def search_coreed(term_srcdb: str, coreed_attr: str):
    payload = {
        "page": "fose",
        "route": "search",
        "other": {"srcdb": term_srcdb},
        "criteria": [
            {"field": f"attributes_{coreed_attr}", "value": "Y"}
        ]
    }

    print(f"\n⏳ Searching term={term_srcdb}, attribute={coreed_attr} ...")
    print("Payload:", json.dumps(payload))

    r = requests.post(API_URL, json=payload)
    print("Status:", r.status_code)
    print("Content-Type:", r.headers.get("Content-Type"))

    text_preview = r.text[:600]
    print("\n=== Raw response preview (first 600 chars) ===")
    print(text_preview)
    print("=== END preview ===\n")

    try:
        data = r.json()
    except Exception as e:
        print("❌ Still cannot parse JSON:", repr(e))
        return None

    if "fatal" in data:
        print("❌ API fatal message:", data["fatal"])

    rows = data.get("data") or data.get("results") or []
    print(f"✓ Parsed JSON; found {len(rows)} sections.")
    return rows


def main():
    term = "202601"   # Fall 2025
    attr = "CFSI"

    timestamp = datetime.now(ZoneInfo("America/Los_Angeles")).isoformat()
    print(f"=== Test run at {timestamp} ===")

    rows = search_coreed(term, attr)

    if rows:
        print("\n🔍 First 5 parsed rows:")
        print(json.dumps(rows[:5], indent=2))
    else:
        print("\nNo rows parsed.\n")


if __name__ == "__main__":
    main()