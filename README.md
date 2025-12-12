# OSU CoreEd Capacity Dashboards

This repo includes Streamlit dashboards for CoreEd capacity monitoring. The current version in use is `osu_coreed_capacity_dashboard_v4_b.py`.

## Quickstart (local)
1) Create/activate a virtualenv (optional but recommended):
```bash
python3 -m venv .venv
source .venv/bin/activate
```
2) Install dependencies:
```bash
pip install -r requirements.txt
```
3) Ensure the SQLite data file is present in the repo root: `osu_enrollment_log_classes.db`.
4) Run the dashboard locally:
```bash
streamlit run osu_coreed_capacity_dashboard_v4_b.py
```
Then open the URL shown in the terminal (default http://localhost:8501).

## Share on your network
Run Streamlit bound to all interfaces and a fixed port, then share your machine’s IP:
```bash
streamlit run osu_coreed_capacity_dashboard_v4_b.py --server.address 0.0.0.0 --server.port 8501
```
Share `http://<your-ip>:8501`. Your machine must stay on and reachable; firewall/VPN rules may apply.

## Deploy options
- **Streamlit Cloud:** push to GitHub (include `requirements.txt` and the SQLite file or provide it via secrets/storage), then deploy selecting `osu_coreed_capacity_dashboard_v4_b.py` as the entry point.
- **Self-host:** run on a VM/container with the repo and DB mounted, and reverse-proxy port 8501 behind HTTPS.
