"""Feedback store: remembers every recovery and corrects the framework's estimates.

Every recovery (service, zone, measured delay, total recovery time, cost, carbon) is
saved in a SQLite file. Before the next decision, the framework asks for a corrected
zone table: each zone's recovery delay is replaced by a moving average of what was
actually measured there.

    estimate = (PRIOR_WEIGHT * configured_delay + sum(last WINDOW measurements))
               / (PRIOR_WEIGHT + number_of_those_measurements)

With no history the estimate equals the configured value. Each new measurement
pulls it toward reality, and old measurements drop out after WINDOW recoveries,
so the estimate also follows a zone whose speed changes over time.
"""
import os
import sqlite3
import pandas as pd

WINDOW = 20        # moving average over the last 20 recoveries of each zone
PRIOR_WEIGHT = 3   # configured value counts like 3 observations


class FeedbackStore:
    def __init__(self, path="feedback/recovery_history.db", window=WINDOW, prior_weight=PRIOR_WEIGHT):
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        self.path = path
        self.window = window
        self.prior_weight = prior_weight
        with self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS recoveries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    logged_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    run_id INTEGER,
                    scenario TEXT,
                    strategy TEXT,
                    service TEXT,
                    priority TEXT,
                    load REAL,
                    zone TEXT,
                    status TEXT,
                    measured_delay REAL,
                    recovery_sec REAL,
                    cost REAL,
                    carbon_g REAL
                )""")

    def _conn(self):
        return sqlite3.connect(self.path)

    # ---- write ------------------------------------------------------------
    def log_run(self, run_id, scenario, strategy, result):
        """Save one recovery run (the DataFrame returned by recover())."""
        rows = []
        for _, r in result.iterrows():
            rows.append((int(run_id), scenario, strategy, r["service"], r["priority"],
                         float(r["load"]), r["to_zone"], r["status"],
                         None if pd.isna(r["measured_delay"]) else float(r["measured_delay"]),
                         None if pd.isna(r["recovery_sec"]) else float(r["recovery_sec"]),
                         float(r["cost"]), float(r["carbon_g"])))
        with self._conn() as c:
            c.executemany("""INSERT INTO recoveries (run_id, scenario, strategy, service, priority,
                             load, zone, status, measured_delay, recovery_sec, cost, carbon_g)
                             VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""", rows)

    # ---- read -------------------------------------------------------------
    def count(self):
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM recoveries WHERE status='RECOVERED'").fetchone()[0]

    def history(self):
        with self._conn() as c:
            return pd.read_sql_query("SELECT * FROM recoveries ORDER BY id", c)

    def recent_delays(self, zone):
        with self._conn() as c:
            rows = c.execute("""SELECT measured_delay FROM recoveries
                                WHERE zone=? AND status='RECOVERED' AND measured_delay IS NOT NULL
                                ORDER BY id DESC LIMIT ?""", (zone, self.window)).fetchall()
        return [r[0] for r in rows]

    def estimate_delay(self, zone, configured):
        obs = self.recent_delays(zone)
        return (self.prior_weight * configured + sum(obs)) / (self.prior_weight + len(obs))

    def corrected_zones(self, zones):
        """Copy of the zone table with recovery_delay replaced by the learned estimate.
        Pass in the ORIGINAL configured table (configs/zones.csv), not an already-corrected one."""
        out = zones.copy()
        out["recovery_delay"] = [self.estimate_delay(r["zone"], r["recovery_delay"])
                                for _, r in zones.iterrows()]
        return out

    def reset(self):
        with self._conn() as c:
            c.execute("DELETE FROM recoveries")