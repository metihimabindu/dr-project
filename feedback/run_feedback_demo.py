"""Shows the feedback loop working.

The same 80 random failures are replayed twice:
  OFF  the framework trusts the configured recovery delays forever
  ON   after every recovery it saves what was measured and corrects its estimates

SIMULATION ASSUMPTION: the real recovery delay of a zone can differ from the value in
configs/zones.csv (a congested region, a slow network, ...). We invent such a difference
in TRUTH below so there is something to learn. Say this clearly in the report.

Run from the dr-project folder:   python feedback/run_feedback_demo.py
Sanity check (nothing to learn):  python feedback/run_feedback_demo.py --no-drift
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator.predictor import Predictor                       # noqa: E402
from simulator.recovery import Cloud, recover, PRIORITY_WEIGHT  # noqa: E402
from feedback.store import FeedbackStore                        # noqa: E402

SEED = 42
N_RUNS = 80
NOISE_SIGMA = 0.15            # each measurement varies about +-15% around the true delay
ROLL = 15                     # rolling-average window for the chart
# Real mean recovery delay per zone (sec). Configured values are A20 B30 C40 D50.
TRUTH = {"A": 55, "B": 30, "C": 45, "D": 30}
DB_PATH = "results/feedback_demo.db"

NO_DRIFT = "--no-drift" in sys.argv
os.makedirs("results", exist_ok=True)

zones = pd.read_csv("configs/zones.csv")
services = pd.read_csv("configs/services.csv")
zone_ids = list(zones["zone"])
if NO_DRIFT:
    TRUTH = {r["zone"]: float(r["recovery_delay"]) for _, r in zones.iterrows()}
    print("NO-DRIFT sanity check: real delays equal the configured ones.")
tag = "_nodrift" if NO_DRIFT else ""


def inject(cloud, kind, target):
    if kind == "zone_down":
        return cloud.zone_down(target)
    if kind == "cpu_overload":
        return cloud.cpu_overload(target)
    return cloud.network_loss(target)


# ---- build the shared list of failures -------------------------------------
rng = np.random.default_rng(SEED)
predictor = Predictor()
if predictor.load_note:
    print("NOTE:", predictor.load_note)
print(f"Preparing {N_RUNS} random failures (LSTM predictions take a little while)...")
scenarios = []
while len(scenarios) < N_RUNS:
    kind = str(rng.choice(["zone_down", "cpu_overload", "network_loss"], p=[0.5, 0.25, 0.25]))
    target = str(rng.choice(list(services["service"]) if kind == "network_loss" else zone_ids))
    t = predictor.random_time(rng)
    util = {z: max(predictor.predict(z, t)[:2]) / 100 for z in zone_ids}
    noise = {z: float(rng.lognormal(0, NOISE_SIGMA)) for z in zone_ids}
    if not inject(Cloud(zones, services, util), kind, target):
        continue                                        # nothing failed; draw another
    scenarios.append({"kind": kind, "target": target, "util": util, "noise": noise})

# ---- replay: feedback OFF vs ON -----------------------------------------------
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)
store = FeedbackStore(DB_PATH)
true_vec = np.array([TRUTH[z] for z in zone_ids], dtype=float)
conf_vec = zones.set_index("zone").loc[zone_ids, "recovery_delay"].values.astype(float)


def weighted_time(res):
    ok = res["status"] == "RECOVERED"
    if not ok.any():
        return np.nan
    w = res.loc[ok, "priority"].map(PRIORITY_WEIGHT)
    return float((w * res.loc[ok, "recovery_sec"]).sum() / w.sum())


def critical_time(res):
    ok = (res["status"] == "RECOVERED") & (res["priority"] == "Critical")
    return float(res.loc[ok, "recovery_sec"].mean()) if ok.any() else np.nan


rows = []
for i, sc in enumerate(scenarios, start=1):
    name = f"{sc['kind']}:{sc['target']}"
    out = {}
    for arm in ["off", "on"]:
        table = zones if arm == "off" else store.corrected_zones(zones)
        cloud = Cloud(table, services, sc["util"], truth=TRUTH, noise=sc["noise"])
        failed = inject(cloud, sc["kind"], sc["target"])
        res = recover(cloud, failed, strategy="topsis", rng=np.random.default_rng(SEED + i))
        out[arm] = res
        if arm == "on":
            store.log_run(i, name, "topsis", res)
    est = store.corrected_zones(zones).set_index("zone").loc[zone_ids, "recovery_delay"].values
    rows.append({
        "run": i, "scenario": name,
        "wtime_off": weighted_time(out["off"]), "wtime_on": weighted_time(out["on"]),
        "crit_off": critical_time(out["off"]), "crit_on": critical_time(out["on"]),
        "est_err_off": float(np.abs(conf_vec - true_vec).mean()),
        "est_err_on": float(np.abs(est - true_vec).mean()),
        **{f"est_{z}": float(e) for z, e in zip(zone_ids, est)},
    })
runs = pd.DataFrame(rows)
runs.to_csv(f"results/feedback_runs{tag}.csv", index=False)

# ---- summary table -------------------------------------------------------------
block = 20
runs["block"] = ((runs["run"] - 1) // block)
summary = runs.groupby("block").agg(
    runs=("run", "count"),
    weighted_time_off=("wtime_off", "mean"), weighted_time_on=("wtime_on", "mean"),
    critical_time_off=("crit_off", "mean"), critical_time_on=("crit_on", "mean"),
    estimate_error_on=("est_err_on", "last"),
).round(1)
summary.index = [f"runs {b * block + 1}-{b * block + block}" for b in summary.index]
summary["estimate_error_off"] = round(runs["est_err_off"].iloc[0], 1)
summary.to_csv(f"results/feedback_summary{tag}.csv")
print("\nAverages per block of 20 failures (seconds, lower is better):")
print(summary.to_string())
final = runs.iloc[-1]
print("\nLearned recovery delay per zone after all runs vs the real value:")
print(pd.DataFrame({"configured": conf_vec, "learned": [round(final[f"est_{z}"], 1) for z in zone_ids],
                    "real": true_vec}, index=zone_ids).to_string())
print(f"\n{store.count()} recoveries saved in {DB_PATH}")

# ---- chart -----------------------------------------------------------------------
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"      # validated pair; ON is always blue, OFF always orange
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), facecolor=SURFACE)
for ax in axes:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Failure number", color=INK2, fontsize=10)


def end_label(ax, x, y, text, color_dot):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center",
                fontsize=9, color=INK)
    ax.plot([x], [y], "o", color=color_dot, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)


a = axes[0]
off = runs["wtime_off"].rolling(ROLL, min_periods=5).mean()
on = runs["wtime_on"].rolling(ROLL, min_periods=5).mean()
a.plot(runs["run"], off, color=ORANGE, linewidth=2, label="Feedback off")
a.plot(runs["run"], on, color=BLUE, linewidth=2, label="Feedback on")
end_label(a, runs["run"].iloc[-1], off.iloc[-1], "off", ORANGE)
end_label(a, runs["run"].iloc[-1], on.iloc[-1], "on", BLUE)
a.set_title("Recovery time (priority-weighted, seconds; lower is better)", loc="left",
            color=INK, fontsize=11, fontweight="bold")
a.set_ylabel(f"Rolling average of {ROLL} failures", color=INK2, fontsize=10)

b = axes[1]
b.plot(runs["run"], runs["est_err_off"], color=ORANGE, linewidth=2, label="Feedback off")
b.plot(runs["run"], runs["est_err_on"], color=BLUE, linewidth=2, label="Feedback on")
end_label(b, runs["run"].iloc[-1], runs["est_err_off"].iloc[-1], "off", ORANGE)
end_label(b, runs["run"].iloc[-1], runs["est_err_on"].iloc[-1], "on", BLUE)
b.set_title("Error in the framework's delay estimates (seconds)", loc="left",
            color=INK, fontsize=11, fontweight="bold")
b.set_ylabel("Average gap to the real delay, 4 zones", color=INK2, fontsize=10)
b.set_ylim(bottom=0)

for ax in axes:
    ax.set_xlim(1, N_RUNS + 6)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False, fontsize=10,
           labelcolor=INK, bbox_to_anchor=(0.99, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(f"results/feedback_learning{tag}.png", dpi=160, facecolor=SURFACE)
print(f"Saved: results/feedback_runs{tag}.csv, results/feedback_summary{tag}.csv, "
      f"results/feedback_learning{tag}.png")