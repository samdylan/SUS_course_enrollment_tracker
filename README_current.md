# SUS & CoreEd Enrollment Snapshot System

This repository maintains automated enrollment and capacity snapshots for:

- **Sustainability (SUS)** courses
- **Core Education (CoreEd)** courses:
  - CFSI (Foundations – Science & Innovation)
  - CSSS (Foundations – Social Sciences)
  - CSDP (Foundations – Design & Professional Skills)
  - CFSS (Foundations – Social & Cultural Perspectives)

The system is designed to support **daily operational monitoring**, **near-term planning**, and **historical reference** for College and University leadership.  
Data are stored in a local SQLite database and visualized via Streamlit dashboards.

---

## 1. High-level architecture

### Data source
All data are pulled from:

https://classes.oregonstate.edu/api/

The script uses browser-mimicking POST requests to match observed site behavior and avoid intermittent HTML / 202 responses.

### Core components
- **Snapshot script**: `osu_enrollment_snapshot_classes_api_coreed.py`
- **Database**: `osu_enrollment_log_classes.db` (SQLite)
- **Dashboards**:
  - SUS Daily Enrollment Dashboard
  - CoreEd Daily / Capacity Dashboard

---

## 2. Database tables and their roles

### 2.1 `enrollment`
- **Purpose**: Daily SUS course enrollment snapshots
- **Frequency**: Daily (during enrollment window)
- **Source logic**:
  - Keyword search: `"SUS"`
  - Details API for enrolled / capacity
- **Used by**: SUS dashboard
- **Authoritative for**: SUS enrollment and capacity

---

### 2.2 `coreed_daily_sections`
- **Purpose**: Daily CoreEd snapshots for the *current term*
- **Frequency**: Daily during the enrollment window
- **Source logic**:
  - Attribute searches: CFSI / CSSS / CSDP / CFSS
  - Details API for enrolled / capacity
- **Authoritative for**:
  - Current-term CoreEd enrollment
  - Current-term CoreEd capacity
- **Used by**: CoreEd dashboard (terms marked “(interim)”)

---

### 2.3 `coreed_capacity`
- **Purpose**: Point-in-time CoreEd capacity snapshots
- **Frequency**:
  - Automatically on the **1st and 15th** of each month (OSU local time)
  - Manually via CLI when needed
- **Typical use cases**:
  - Lookahead term (current term + 1)
  - Historical reference for completed terms
- **Used by**: CoreEd dashboard (historical and future terms)

---

### 2.4 `coreed_capacity_backfill`
- **Purpose**: QA and repair table for historical CoreEd data
- **Frequency**: Manual only
- **Used for**:
  - Repairing broken or incomplete historical capacity snapshots
  - Comparing legacy vs corrected data
- **Not used directly by dashboards**

---

## 3. Term logic

- **Current term**: dynamic, daily snapshots during enrollment window
- **Lookahead term**: current + 1, refreshed twice monthly
- **Final catch**: prior term refreshed on first day of next enrollment window

---

## 4. Why row counts may appear inflated

Raw table row counts reflect:
- Multiple snapshots per term
- Historical debug runs
- Intentional audit retention

Dashboards always select the **latest snapshot per term**.

---

## 5. Running the system locally

Activate environment:

```bash
source .venv/bin/activate
```

Run daily snapshot:

```bash
python osu_enrollment_snapshot_classes_api_coreed.py
```

Force lookahead refresh:

```bash
python osu_enrollment_snapshot_classes_api_coreed.py --coreed-capacity-refresh
```

Backfill a historical term (QA only):

```bash
python osu_enrollment_snapshot_classes_api_coreed.py 202601 --coreed-backfill
```

---

## 6. Data governance

- The committed SQLite database is treated as the **current source of truth**
- Manual repairs should be committed
- GitHub Actions continues daily snapshots thereafter

---

## 7. Design philosophy

The system prioritizes:
- Accuracy for decision-making
- Transparency over aggressive deduplication
- Reproducibility of *process*, not necessarily of historical states

