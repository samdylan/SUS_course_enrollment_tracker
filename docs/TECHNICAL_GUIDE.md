# Technical Guide — SUS & CoreEd Enrollment Tracker

Step-by-step instructions for setting up, running, and maintaining the enrollment tracking system.

---

## 1. First-Time Setup

All commands assume you are in the project root directory:

```bash
cd "/Users/bellsa/Dropbox/OSU/SUS program Admin (internal)/Coding Projects/SUS_course_enrollment_tracker"
```

### Create the virtual environment (one time only)

```bash
python3 -m venv .venv
```

### Activate the virtual environment

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

This installs: `pandas`, `requests`, `streamlit`, `altair`.

---

## 2. Running the Dashboards

Always activate the virtual environment first:

```bash
source .venv/bin/activate
```

### SUS + CoreEd Daily Dashboard

```bash
streamlit run dashboards/osu_enrollment_dashboard_v5.py
```

Opens at **http://localhost:8501**. Shows:
- SUS enrollment timeseries by section/course/campus
- CoreEd capacity + enrollment timeseries by category and campus
- Term alignment using days-from-start x-axis

### CoreEd Capacity Dashboard

```bash
streamlit run dashboards/osu_coreed_capacity_dashboard_v5.py
```

Opens at **http://localhost:8501** (or next available port if the first dashboard is running). Shows:
- Clustered bar charts: capacity by term and campus for each CoreEd category
- Course-level capacity breakdown
- Enrollment overlay from daily snapshots
- Active registration terms marked with `*`

### Running both dashboards simultaneously

Run the first dashboard, then open a new terminal tab and run the second on a different port:

```bash
# Terminal 1
source .venv/bin/activate
streamlit run dashboards/osu_enrollment_dashboard_v5.py

# Terminal 2
source .venv/bin/activate
streamlit run dashboards/osu_coreed_capacity_dashboard_v5.py --server.port 8502
```

---

## 3. Running the Snapshot Script (Manual)

The daily snapshot is automated via GitHub Actions, but you can run it manually:

```bash
source .venv/bin/activate
python scripts/osu_enrollment_snapshot_classes_api_coreed.py
```

This will:
- Query the OSU Classes API for SUS courses and CoreEd sections
- Append results to `data/osu_enrollment_log_classes.db`
- On the 1st and 15th of the month, also refresh lookahead-term capacity

### Optional flags

```bash
# Force a capacity refresh for the lookahead term (runs even outside normal schedule)
python scripts/osu_enrollment_snapshot_classes_api_coreed.py --coreed-capacity-refresh

# Force a specific term code
python scripts/osu_enrollment_snapshot_classes_api_coreed.py 202603

# Backfill a historical term into the QA/repair table
python scripts/osu_enrollment_snapshot_classes_api_coreed.py 202601 --coreed-backfill
```

---

## 4. GitHub Actions (Automated Daily Snapshots)

The workflow at `.github/workflows/daily-sus-enrollment.yml` runs automatically:

- **Schedule**: Daily at 17:00 UTC (9:00 AM PST)
- **What it does**: Runs the snapshot script, commits and pushes the updated DB if data changed
- **Manual trigger**: Go to the repo's Actions tab on GitHub and click "Run workflow"

### Pulling the latest data locally

After GitHub Actions runs (or after any remote change):

```bash
git pull
```

Then re-run the dashboards to see updated data.

---

## 5. Updating Data After a New Term Begins

When a new term's registration window opens, the system handles it automatically:

1. **Snapshot script** detects the new term based on the 56-day enrollment window
2. **Daily dashboard** defaults to the most recent term with non-zero enrollment
3. **Capacity dashboard** shows the 5 most recent terms; the new term appears when it has capacity data

No manual intervention is needed. If you want to verify:

```bash
# Check which term the script thinks is current
source .venv/bin/activate
python scripts/osu_enrollment_snapshot_classes_api_coreed.py
# Look for the "[api] Current term: ..." line in the output
```

---

## 6. Database Location

The production database is at:

```
data/osu_enrollment_log_classes.db
```

All three production files (two dashboards + snapshot script) resolve this path automatically relative to the project structure.

### Environment variable overrides

For non-standard deployments (e.g., Streamlit Cloud, a different machine):

```bash
# For the daily dashboard
export ENROLLMENT_DB_PATH="/absolute/path/to/osu_enrollment_log_classes.db"

# For the capacity dashboard
export COREED_DB_PATH="/absolute/path/to/osu_enrollment_log_classes.db"
```

---

## 7. Troubleshooting

### Dashboard shows no charts / blank page
- Check that the virtual environment is activated (`source .venv/bin/activate`)
- Check that dependencies are installed (`pip install -r requirements.txt`)
- Look at terminal output for Python errors
- Confirm the DB exists: `ls data/osu_enrollment_log_classes.db`

### Dashboard defaults to a term with no data
- This means enrollment hasn't started for the newest term yet
- The smart defaults should skip it automatically; if not, select a different term from the sidebar

### GitHub Actions fails
- Check the Actions tab on GitHub for error logs
- Most common issue: API temporarily unavailable (the script has retry logic)
- Verify the workflow file paths match the current directory structure

### "Database file not found" error
- The DB path resolves relative to the script's location
- Make sure you're running from the project root: `cd` to `SUS_course_enrollment_tracker/`
- Or set the environment variable override (see Section 6)

### Stale data / dashboard not showing recent snapshots
- Run `git pull` to get the latest DB from GitHub
- Or run the snapshot script manually (see Section 3)

---

## 8. File Reference

| File | Purpose | Run command |
|------|---------|-------------|
| `dashboards/osu_enrollment_dashboard_v5.py` | SUS + CoreEd Daily Dashboard | `streamlit run dashboards/osu_enrollment_dashboard_v5.py` |
| `dashboards/osu_coreed_capacity_dashboard_v5.py` | CoreEd Capacity Dashboard | `streamlit run dashboards/osu_coreed_capacity_dashboard_v5.py` |
| `scripts/osu_enrollment_snapshot_classes_api_coreed.py` | Daily snapshot script | `python scripts/osu_enrollment_snapshot_classes_api_coreed.py` |
| `data/osu_enrollment_log_classes.db` | SQLite database (source of truth) | — |
| `.github/workflows/daily-sus-enrollment.yml` | GitHub Actions workflow | Runs automatically |
| `requirements.txt` | Python dependencies | `pip install -r requirements.txt` |
