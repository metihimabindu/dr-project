"""Step 11: the main experiment.

The SAME random failures are given to every recovery method. For each method we record
availability, recovery time, cost, carbon and zone utilization, then report the mean with
a 95% confidence interval.

Three conditions are run on the SAME failures:
  no_drift  the configured recovery delays (configs/zones.csv) are correct
  drift     some zones really recover slower/faster than configured (assumption, see TRUTH_DRIFT)
  stress    drift, plus extra load from other tenants in every zone, so free capacity runs short
            and priority-aware placement matters (assumption, see STRESS_EXTRA)
Feedback can only help in "drift"/"stress"; "no_drift" shows it does no harm when there is nothing to learn.
For every condition a per-failure-type breakdown is printed and saved.

Run from the dr-project folder:   python simulator/run_experiment.py          (200 failures)
                                  python simulator/run_experiment.py 30       (quick test)
                                  python simulator/run_experiment.py 200 stress=0.08   (change the stress level)
"""
import math
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
from decision.paper_baseline import (WEIGHT_SETS, BASE_LATENCY_MS,  # noqa: E402
                                     CUTOFF_MS)

N = int(next((a for a in sys.argv[1:] if a.isdigit()), 200))
SEED = 2026
NOISE_SIGMA = 0.15                            # measured delay varies about +-15%
TRUTH_DRIFT = {"A": 55, "B": 30, "C": 45, "D": 30}   # real mean delays (configured: 20/30/40/50)
KINDS = ["zone_down", "two_zones_down", "cpu_overload", "network_loss"]
KIND_P = [0.4, 0.2, 0.2, 0.2]
# stress condition: extra share of every zone's capacity used by other tenants (0.05 = 5%)
STRESS_EXTRA = float(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("stress=")), 0.05))

# key, label for tables, short label for charts, recover() strategy, learns from feedback?, in chart?
METHODS = [
    ("random", "Random zone", "Random", "random", False, True),
    ("cheapest", "Cheapest zone", "Cheapest", "cheapest", False, True),
    ("paper", "Base paper Eq.1 (equal weights)", "Base paper Eq.1", "paper", False, True),
    ("topsis_basic", "TOPSIS", "TOPSIS", "topsis_basic", False, True),
    ("topsis_la", "TOPSIS + look-ahead", "TOPSIS + look-ahead", "topsis", False, True),
    ("topsis_fb", "TOPSIS + look-ahead + feedback", "TOPSIS + look-ahead\n+ feedback", "topsis", True, True),
    ("paper_cpu", "Base paper Eq.1 (CPU-heavy weights)", "", "paper:cpu_heavy", False, False),
    ("paper_lat", "Base paper Eq.1 (latency-heavy weights)", "", "paper:latency_heavy", False, False),
    ("paper_cut", "Base paper Eq.1 (with latency cutoff)", "", "paper:cutoff", False, False),
]
LABEL = {m[0]: m[1] for m in METHODS}
REFERENCE = "topsis_fb"

os.makedirs("results", exist_ok=True)
zones = pd.read_csv("configs/zones.csv")
services = pd.read_csv("configs/services.csv")
zone_ids = list(zones["zone"])
TRUTH_NONE = {r["zone"]: float(r["recovery_delay"]) for _, r in zones.iterrows()}


def inject(cloud, kind, target):
    if kind == "zone_down":
        return cloud.zone_down(target[0])
    if kind == "two_zones_down":
        return cloud.zone_down(target[0]) + cloud.zone_down(target[1])
    if kind == "cpu_overload":
        return cloud.cpu_overload(target[0])
    return cloud.network_loss(target[0])


def stressed(sc, extra):
    """Same failure, but every zone is busier by `extra` (the LSTM's own prediction is shifted)."""
    cap = lambda d: {z: min(v + extra, 0.85) for z, v in d.items()}
    return {**sc, "util": cap(sc["util"]),
            "pred": {"cpu": cap(sc["pred"]["cpu"]), "mem": cap(sc["pred"]["mem"])}}


# ---- 1. the shared list of failures ------------------------------------------------
rng = np.random.default_rng(SEED)
predictor = Predictor()
if predictor.load_note:
    print("NOTE:", predictor.load_note)
print(f"Preparing {N} random failures (the LSTM predictions take a few minutes)...")
scenarios = []
while len(scenarios) < N:
    kind = str(rng.choice(KINDS, p=KIND_P))
    if kind == "two_zones_down":
        target = [str(x) for x in rng.choice(zone_ids, 2, replace=False)]
    elif kind == "network_loss":
        target = [str(rng.choice(list(services["service"])))]
    else:
        target = [str(rng.choice(zone_ids))]
    t = predictor.random_time(rng)
    p = {z: predictor.predict(z, t)[:2] for z in zone_ids}
    pred = {"cpu": {z: p[z][0] / 100 for z in zone_ids}, "mem": {z: p[z][1] / 100 for z in zone_ids}}
    util = {z: max(pred["cpu"][z], pred["mem"][z]) for z in zone_ids}
    noise = {z: float(rng.lognormal(0, NOISE_SIGMA)) for z in zone_ids}
    if not inject(Cloud(zones, services, util), kind, target):
        continue                                           # nothing failed; draw another
    scenarios.append({"kind": kind, "target": target, "util": util, "pred": pred, "noise": noise})
print("Failure mix:", pd.Series([s["kind"] for s in scenarios]).value_counts().to_dict())
stress_scenarios = [stressed(s, STRESS_EXTRA) for s in scenarios]


# ---- 2. metrics of one recovery run -------------------------------------------------
def metrics(cloud, res):
    ok = res["status"] == "RECOVERED"
    crit = res["priority"] == "Critical"
    w = res["priority"].map(PRIORITY_WEIGHT)
    recovered_load = res.loc[ok, "load"].sum()
    util = [1 - cloud.free(z) / cloud.zones.loc[z, "capacity"]
            for z in cloud.zones.index if z not in cloud.down]
    return {
        "w_avail": 100 * w[ok].sum() / w.sum(),
        "crit_total": int(crit.sum()),
        "crit_ok": int((ok & crit).sum()),
        "crit_time": res.loc[ok & crit, "recovery_sec"].mean() if (ok & crit).any() else np.nan,
        "w_time": (w[ok] * res.loc[ok, "recovery_sec"]).sum() / w[ok].sum() if ok.any() else np.nan,
        "cost_per_unit": res["cost"].sum() / recovered_load if recovered_load > 0 else np.nan,
        "carbon_per_unit": res["carbon_g"].sum() / recovered_load if recovered_load > 0 else np.nan,
        "peak_util": 100 * max(util),
    }


def run_condition(cond, truth, scens):
    db = f"results/experiment_feedback_{cond}.db"
    if os.path.exists(db):
        os.remove(db)
    store = FeedbackStore(db)
    rows = []
    for i, sc in enumerate(scens, start=1):
        for key, _, _, strategy, learns, _ in METHODS:
            table = store.corrected_zones(zones) if learns else zones
            cloud = Cloud(table, services, sc["util"], truth=truth, noise=sc["noise"], pred=sc["pred"])
            failed = inject(cloud, sc["kind"], sc["target"])
            if not failed:                      # nothing failed in this condition: skip for every method
                continue
            res = recover(cloud, failed, strategy=strategy, rng=np.random.default_rng(SEED + i))
            if learns:
                store.log_run(i, sc["kind"] + ":" + "+".join(sc["target"]), strategy, res)
            rows.append({"scenario": i, "kind": sc["kind"], "method": key, **metrics(cloud, res)})
    return pd.DataFrame(rows)


# ---- 3. statistics ---------------------------------------------------------------------
def ci95(x):
    x = x.dropna()
    return 1.96 * x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else float("nan")


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


MEAN_COLS = ["w_avail", "crit_time", "w_time", "cost_per_unit", "carbon_per_unit", "peak_util"]


def summarize(raw):
    out = []
    for key, g in raw.groupby("method", sort=False):
        row = {"method": key}
        for c in MEAN_COLS:
            row[c] = g[c].mean()
            row[c + "_ci"] = ci95(g[c])
        k, n = g["crit_ok"].sum(), g["crit_total"].sum()
        lo, hi = wilson(k, n)
        row["crit_avail"] = 100 * k / n if n else np.nan
        row["crit_avail_ci"] = 50 * (hi - lo) if n else np.nan
        out.append(row)
    return pd.DataFrame(out)


def paired(raw):
    ref = raw[raw["method"] == REFERENCE].set_index("scenario")
    rows = []
    for key in raw["method"].unique():
        if key == REFERENCE:
            continue
        other = raw[raw["method"] == key].set_index("scenario")
        row = {"vs": key}
        for c in ["w_avail", "crit_time", "w_time", "cost_per_unit", "carbon_per_unit"]:
            d = (ref[c] - other[c])
            row[c] = d.mean()
            row[c + "_ci"] = ci95(d)
        rows.append(row)
    return pd.DataFrame(rows)


def fmt(m, c, d=1):
    return "-" if pd.isna(m) else (f"{m:.{d}f} ± {c:.{d}f}" if not pd.isna(c) else f"{m:.{d}f}")


def pretty(summary):
    t = pd.DataFrame({
        "method": [LABEL[k] for k in summary["method"]],
        "weighted availability %": [fmt(m, c) for m, c in zip(summary.w_avail, summary.w_avail_ci)],
        "critical availability %": [fmt(m, c) for m, c in zip(summary.crit_avail, summary.crit_avail_ci)],
        "critical recovery s": [fmt(m, c) for m, c in zip(summary.crit_time, summary.crit_time_ci)],
        "weighted recovery s": [fmt(m, c) for m, c in zip(summary.w_time, summary.w_time_ci)],
        "cost / load unit": [fmt(m, c, 4) for m, c in zip(summary.cost_per_unit, summary.cost_per_unit_ci)],
        "carbon g / load unit": [fmt(m, c, 2) for m, c in zip(summary.carbon_per_unit, summary.carbon_per_unit_ci)],
        "peak zone util %": [fmt(m, c) for m, c in zip(summary.peak_util, summary.peak_util_ci)],
    })
    return t


MAIN_KEYS = [m[0] for m in METHODS if m[5]]


def by_kind(raw):
    """Mean of three metrics per failure type, for the main methods (one table per metric)."""
    main = raw[raw["method"].isin(MAIN_KEYS)]
    present = [k for k in KINDS if k in set(main["kind"])]
    tables = []
    for col, name, d in [("w_avail", "weighted availability %", 1),
                         ("crit_time", "critical recovery s", 1),
                         ("carbon_per_unit", "carbon g / load unit", 2)]:
        p = main.pivot_table(index="method", columns="kind", values=col, aggfunc="mean")
        p = p.reindex(index=MAIN_KEYS, columns=present)   # a type with no value stays as an empty column
        p.index = [LABEL[k] for k in p.index]
        tables.append((name, p.round(d)))
    counts = main[main["method"] == REFERENCE]["kind"].value_counts().reindex(present).fillna(0).astype(int)
    return tables, counts


# ---- 4. chart ------------------------------------------------------------------------------
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
COLOR = {"random": "#b4b3ad", "cheapest": "#b4b3ad", "paper": "#eb6834",
         "topsis_basic": "#a9c9f2", "topsis_la": "#5f9ce3", "topsis_fb": "#2a78d6"}
PANELS = [("w_avail", "Weighted availability, % (higher is better)", 1),
          ("crit_time", "Critical-service recovery time, s (lower is better)", 1),
          ("cost_per_unit", "Cost per load unit recovered, USD (lower is better)", 4),
          ("carbon_per_unit", "Carbon per load unit recovered, g (lower is better)", 2)]


def chart(summary, path, title):
    main = [m for m in METHODS if m[5]]
    s = summary.set_index("method")
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.2), facecolor=SURFACE)
    for ax, (col, ptitle, dec) in zip(axes.ravel(), PANELS):
        ax.set_facecolor(SURFACE)
        ys = np.arange(len(main))
        vals = [s.loc[m[0], col] for m in main]
        errs = [s.loc[m[0], col + "_ci"] for m in main]
        ax.barh(ys, vals, xerr=errs, height=0.62, color=[COLOR[m[0]] for m in main],
                error_kw={"ecolor": INK2, "elinewidth": 1.2, "capsize": 3})
        top = max(v + (0 if pd.isna(e) else e) for v, e in zip(vals, errs))
        for y, v, e in zip(ys, vals, errs):
            ax.text(v + (0 if pd.isna(e) else e) + top * 0.015, y, f"{v:.{dec}f}", va="center",
                    fontsize=9, color=INK)
        ax.set_yticks(ys)
        ax.set_yticklabels([m[2] for m in main], fontsize=9, color=INK)
        ax.invert_yaxis()
        ax.set_xlim(0, top * 1.13)
        ax.set_title(ptitle, loc="left", fontsize=11, fontweight="bold", color=INK)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="x", colors=INK2, labelsize=9)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.005, f"{N} random failures, mean with 95% confidence interval. Gray = simple baselines, "
             "orange = base paper, blue = this project.", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.025, 1, 0.96))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


# ---- 5. run both conditions -------------------------------------------------------------------
print("\nASSUMPTIONS (state them in the report):")
print(f"  * weights of the base-paper score are not given in the paper: tested {list(WEIGHT_SETS)}")
print(f"  * latency is simulated (base ms {BASE_LATENCY_MS}); the main base-paper baseline has NO latency cutoff,")
print(f"    a version with a {CUTOFF_MS} ms cutoff is shown as a sensitivity row")
print(f"  * 'drift' condition real delays {TRUTH_DRIFT}, measurement noise +-{int(NOISE_SIGMA * 100)}%")
print(f"  * 'stress' condition: drift plus {int(STRESS_EXTRA * 100)}% of every zone's capacity used by other tenants")
TITLES = {"no_drift": "configured delays correct", "drift": "zone delays drift from configured",
          "stress": f"delays drift and every zone is {int(STRESS_EXTRA * 100)}% busier"}
for cond, truth, name, scens in [
        ("no_drift", TRUTH_NONE, "configured delays are correct", scenarios),
        ("drift", TRUTH_DRIFT, "some zones really recover slower or faster", scenarios),
        ("stress", TRUTH_DRIFT, f"drift plus {int(STRESS_EXTRA * 100)}% extra tenant load in every zone",
         stress_scenarios)]:
    print("\n" + "=" * 100)
    print(f"CONDITION {cond}: {name}  ({N} failures)")
    raw = run_condition(cond, truth, scens)
    summary = summarize(raw)
    table = pretty(summary)
    pair = paired(raw)
    raw.to_csv(f"results/experiment_raw_{cond}.csv", index=False)
    summary.to_csv(f"results/experiment_summary_{cond}.csv", index=False)
    table.to_csv(f"results/experiment_table_{cond}.csv", index=False)
    pair.to_csv(f"results/experiment_paired_{cond}.csv", index=False)
    chart(summary, f"results/experiment_{cond}.png", f"Recovery methods compared ({TITLES[cond]})")
    pd.set_option("display.width", 250)
    print(table.to_string(index=False))
    print(f"\nPAIRED DIFFERENCE: '{LABEL[REFERENCE]}' minus each other method, same failures")
    print("(availability: positive = better; time, cost, carbon: negative = better; ± is the 95% CI)")
    pp = pd.DataFrame({
        "vs": [LABEL[k] for k in pair["vs"]],
        "weighted availability %": [fmt(m, c) for m, c in zip(pair.w_avail, pair.w_avail_ci)],
        "critical recovery s": [fmt(m, c) for m, c in zip(pair.crit_time, pair.crit_time_ci)],
        "weighted recovery s": [fmt(m, c) for m, c in zip(pair.w_time, pair.w_time_ci)],
        "cost / unit": [fmt(m, c, 4) for m, c in zip(pair.cost_per_unit, pair.cost_per_unit_ci)],
        "carbon g / unit": [fmt(m, c, 2) for m, c in zip(pair.carbon_per_unit, pair.carbon_per_unit_ci)],
    })
    print(pp.to_string(index=False))
    tables, counts = by_kind(raw)
    print(f"\nBY FAILURE TYPE (mean per failure; failures per type: {counts.to_dict()})")
    long = []
    for tname, tab in tables:
        print(f"\n  {tname}")
        print(tab.to_string(na_rep="-"))
        long.append(tab.assign(metric=tname).reset_index().rename(columns={"index": "method"}))
    pd.concat(long).to_csv(f"results/experiment_by_kind_{cond}.csv", index=False)
print("\nSaved in results/: experiment_table_*.csv, experiment_paired_*.csv, experiment_summary_*.csv,"
      " experiment_raw_*.csv, experiment_by_kind_*.csv, experiment_no_drift.png, experiment_drift.png,"
      " experiment_stress.png")