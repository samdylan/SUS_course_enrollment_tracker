# OSU SUS & CoreEd Enrollment Tracker

This project contains a small Streamlit dashboard plus supporting scripts to monitor
enrollment and capacity for:

- The **Sustainability Double Degree (SUS)**, and
- Selected **CoreEd (Core Education)** categories (CFSI, CSSS, CSDP, CFSS).

The dashboard reads from a local SQLite database that is populated by separate
snapshot scripts.

---

## Project structure

From the root folder `SUS_course_enrollment_tracker/`:

- `osu_enrollment_log_classes.db`  
  SQLite database containing snapshot tables, including:

  - `coreed_daily_sections`: per-section CoreEd daily snapshots
    (term, subject, course_number, section, campus, coreed_cat4, enrolled, capacity, snapshot_date, …).

- `CoreEd Daily Tracker/osu_enrollment_dashboard_v5.py`  
  Streamlit app for SUS + CoreEd enrollment trends and summaries.

- `inspect_coreed_other_campus.py`  
  Diagnostic helper to verify that the campus classification logic does not
  produce any “Other” rows for CoreEd data.

Older app versions (e.g. `osu_enrollment_dashboard_v3.py`, `osu_enrollment_dashboard_v4.py`)
can be moved to an `archive/` folder for reference.

---

## Data flow

1. **Snapshot scripts (outside this README)** query OSU systems and write into
   `osu_enrollment_log_classes.db`, at minimum populating `coreed_daily_sections`.

2. The dashboard:

   - Reads `osu_enrollment_log_classes.db` via
     ```python
     DB_PATH = Path(__file__).resolve().parents[1] / "osu_enrollment_log_classes.db"
     ```
   - Derives `campus_simple` (Corvallis, Ecampus, Cascades) from raw fields.
   - Computes `days_from_start` for each snapshot as:
     ```python
     days_from_start = (snapshot_date - classes_begin).days
     ```
   - Aggregates to build SUS and CoreEd charts.

3. The dashboard **does not** write to the DB; it is read-only. Updates happen
   only when snapshot scripts are run.

---

## Running the dashboard

From the project root:

```bash
cd /path/to/SUS_course_enrollment_tracker
source .venv/bin/activate      # if using a virtualenv
# or however you activate your environment

# Install dependencies once (example)
pip install streamlit pandas altair

# (Optional) Clear Streamlit's cached data if switching DBs
streamlit cache-data clear     # or: streamlit cache clear

# Run the app
streamlit run "CoreEd Daily Tracker/osu_enrollment_dashboard_v5.py"