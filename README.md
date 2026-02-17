# SUS & CoreEd Enrollment Snapshot System

Automated enrollment and capacity snapshots for OSU Sustainability (SUS) courses and Core Education (CoreEd) categories, with two Streamlit dashboards for visualization.

**Version:** Clean Build 1.0
**Last Updated:** February 17, 2026

---

## Quick Start

```bash
source .venv/bin/activate
pip install -r requirements.txt

# SUS + CoreEd Daily Dashboard
streamlit run dashboards/osu_enrollment_dashboard_v5.py

# CoreEd Capacity Dashboard
streamlit run dashboards/osu_coreed_capacity_dashboard_v5.py
```

Both dashboards open at `http://localhost:8501` by default.

### Daily snapshot (manual)

```bash
source .venv/bin/activate
python scripts/osu_enrollment_snapshot_classes_api_coreed.py
```

Automated daily snapshots run via GitHub Actions (see `.github/workflows/`).

---

## Project Structure

```
SUS_course_enrollment_tracker/
├── data/
│   └── osu_enrollment_log_classes.db   # Source-of-truth SQLite database
│
├── scripts/
│   └── osu_enrollment_snapshot_classes_api_coreed.py   # Daily snapshot script
│
├── dashboards/
│   ├── osu_enrollment_dashboard_v5.py          # SUS + CoreEd Daily Dashboard
│   └── osu_coreed_capacity_dashboard_v5.py     # CoreEd Capacity Dashboard
│
├── .github/workflows/
│   └── daily-sus-enrollment.yml        # GitHub Actions: daily at 17:00 UTC
│
├── docs/
│   └── reference/                      # User manuals, specs, screengrabs
│
├── dev/                                # DEVELOPMENT AREA
│   ├── scripts/                        # Diagnostic and utility scripts
│   └── section_fill_forecasting/       # Experimental forecasting project
│
├── archive/                            # Superseded versions and old data
│
├── requirements.txt                    # pandas, requests, streamlit, altair
├── .gitignore
└── README.md                           # This file
```

---

## Database Tables

### `enrollment`
- **Purpose**: Daily SUS course enrollment snapshots
- **Source**: Keyword search `"SUS"` via OSU Classes API
- **Used by**: SUS Daily Dashboard

### `coreed_daily_sections`
- **Purpose**: Daily CoreEd snapshots for terms with active registration
- **Source**: Attribute searches CFSI / CSSS / CSDP / CFSS via OSU Classes API
- **Authoritative for**: Current-term CoreEd enrollment and capacity
- **Used by**: Both dashboards

### `coreed_capacity`
- **Purpose**: Point-in-time CoreEd capacity snapshots (historical + lookahead)
- **Refresh**: Automatically on the 1st and 15th of each month; manually via `--coreed-capacity-refresh`
- **Used by**: CoreEd Capacity Dashboard (backbone for historical/future terms)

### `coreed_capacity_backfill`
- **Purpose**: QA/repair table for historical CoreEd data
- **Refresh**: Manual only (`--coreed-backfill`)
- **Dashboard**: Configured via `BACKFILL_TERMS` in the Capacity Dashboard

### `sus_daily_registrations_2025`
- **Purpose**: Legacy SUS daily registration data (imported from external spreadsheet)
- **Used by**: SUS Daily Dashboard (prior-term comparison lines)

---

## Term Logic

OSU term codes follow the pattern `YYYYTT`:
- `YYYY` = academic label year
- `TT` = `00` Summer, `01` Fall, `02` Winter, `03` Spring

Examples: `202602` = Winter 2026, `202603` = Spring 2026, `202700` = Summer 2026.

### Snapshot script
- **Current term**: determined dynamically based on a 56-day enrollment window (50 days before through 6 days after classes begin)
- **Lookahead term**: current + 1 quarter, refreshed on 1st/15th
- **Prior-term catch**: first day of a new window triggers a final refresh for the previous term

### Dashboard defaults
- **SUS Daily Dashboard**: defaults to the most recent term with non-zero enrollment
- **CoreEd Capacity Dashboard**: defaults to the 5 most recent terms (reverse chronological); terms with zero capacity are excluded automatically

---

## Running the System

### Prerequisites

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Snapshot commands

```bash
# Standard daily snapshot
python scripts/osu_enrollment_snapshot_classes_api_coreed.py

# Force capacity refresh (lookahead term)
python scripts/osu_enrollment_snapshot_classes_api_coreed.py --coreed-capacity-refresh

# Backfill a historical term (QA only)
python scripts/osu_enrollment_snapshot_classes_api_coreed.py 202601 --coreed-backfill
```

### Environment variable overrides

Both dashboards and the snapshot script resolve the database path relative to the repo structure by default. Override with:

```bash
export ENROLLMENT_DB_PATH="/custom/path/to/osu_enrollment_log_classes.db"
export COREED_DB_PATH="/custom/path/to/osu_enrollment_log_classes.db"
```

---

## GitHub Actions

The workflow `.github/workflows/daily-sus-enrollment.yml` runs daily at 17:00 UTC (9:00 AM PST):

1. Checks out the repo
2. Installs Python 3.11 + dependencies
3. Runs `scripts/osu_enrollment_snapshot_classes_api_coreed.py`
4. If the DB changed, commits and pushes `data/osu_enrollment_log_classes.db`

Pull the latest data locally:

```bash
git pull
```

---

## Dashboard Features

### SUS + CoreEd Daily Dashboard (`dashboards/osu_enrollment_dashboard_v5.py`)

- **SUS section**: enrollment timeseries by section/course/campus, aligned by "days from start of term"
- **CoreEd section**: capacity + enrollment timeseries by CoreEd category and campus
- **Term alignment**: current and prior terms overlaid using days-from-start x-axis
- **Smart defaults**: auto-selects the most recent term with enrollment data
- **Filters**: terms, campus, courses, labs toggle, aggregation level

### CoreEd Capacity Dashboard (`dashboards/osu_coreed_capacity_dashboard_v5.py`)

- **Clustered bar charts**: capacity by term and campus, with enrollment overlay (black tick marks)
- **Course-level charts**: per-course capacity breakdown by campus
- **Active term indicator**: terms with daily enrollment updates (within last 7 days) marked with `*`
- **Smart defaults**: 5 most recent terms; reverse-chronological filter; zero-capacity terms excluded
- **Data sources**: `coreed_capacity` (historical/future) → `coreed_daily_sections` (current term) → `coreed_capacity_backfill` (repairs)
- **Filters**: terms (multiselect), campus, courses, CAS-only toggle, labs toggle
- **Diagnostics**: sidebar section below filters with DB info, snapshot dates, NULL counts

---

## Data Governance

- The committed SQLite database (`data/osu_enrollment_log_classes.db`) is the source of truth
- GitHub Actions handles daily snapshots; manual runs supplement as needed
- Dashboard reads are read-only; no dashboard writes to the DB

---

## Dev and Archive

- `dev/scripts/` contains diagnostic utilities (DB peek, duplicate checks, backfill tools)
- `dev/section_fill_forecasting/` is an experimental section-fill prediction project
- `archive/` contains all superseded dashboard versions, old snapshot scripts, and historical DB backups
- Neither `dev/` nor `archive/` is tracked in git
