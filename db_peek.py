import sqlite3, pandas as pd

conn = sqlite3.connect("osu_enrollment_log_classes.db")
df = pd.read_sql_query(
    "SELECT snapshot_date, term_srcdb, code, section, enrolled, capacity "
    "FROM enrollment "
    "WHERE term_srcdb = '202602' "
    "ORDER BY snapshot_date DESC, code, section "
    "LIMIT 40",
    conn,
)
conn.close()

print(df)