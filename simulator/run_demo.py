"""Run a few failure scenarios and show how the framework recovers.

Run from the dr-project folder:   python simulator/run_demo.py
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator.predictor import Predictor          # noqa: E402
from simulator.recovery import Cloud, recover, summarize  # noqa: E402

SEED = 42
rng = np.random.default_rng(SEED)
os.makedirs("results", exist_ok=True)

zones = pd.read_csv("configs/zones.csv")
services = pd.read_csv("configs/services.csv")

# 1) Predict each zone's busy level at a random moment from the test period
predictor = Predictor()
if predictor.load_note:
    print("NOTE:", predictor.load_note)
t = predictor.random_time(rng)
predicted_util, sources = {}, set()
for z in zones["zone"]:
    cpu, mem, src = predictor.predict(z, t)
    predicted_util[z] = max(cpu, mem) / 100
    sources.add(src)
print(f"\nPredictions made at data row {t} using: {', '.join(sorted(sources))}")


def fresh_cloud():
    return Cloud(zones, services, predicted_util)


c0 = fresh_cloud()
print("\nZone state before any failure (units):")
state = pd.DataFrame({
    "zone": c0.zones.index,
    "predicted_busy_%": [round(100 * c0.busy[z], 1) for z in c0.zones.index],
    "free_units": [round(c0.free(z), 1) for z in c0.zones.index],
})
print(state.to_string(index=False))
if (state["free_units"] < 0).any():
    print("WARNING: a zone is already over capacity before any failure. Lower BASE_BUSY in simulator/recovery.py")

# 2) Failure scenarios
scenarios = [
    ("Zone A goes down", lambda c: c.zone_down("A")),
    ("Zone B goes down", lambda c: c.zone_down("B")),
    ("CPU overload in Zone D", lambda c: c.cpu_overload("D")),
    ("Network loss on Payment", lambda c: c.network_loss("Payment")),
    ("Zones B and D go down together", lambda c: c.zone_down("B") + c.zone_down("D")),
]

all_rows, compare = [], []
for name, inject in scenarios:
    print("\n" + "=" * 68)
    print("SCENARIO:", name)
    for strategy in ["topsis", "topsis_basic", "cheapest", "random"]:
        cloud = fresh_cloud()
        failed = inject(cloud)
        result = recover(cloud, failed, strategy=strategy, rng=np.random.default_rng(SEED))
        if strategy == "topsis":
            print("Failed services:", ", ".join(failed) if failed else "none")
            if len(result):
                print(result.round(1).to_string(index=False))
        res = result.copy()
        res.insert(0, "strategy", strategy)
        res.insert(0, "scenario", name)
        all_rows.append(res)
        compare.append({"scenario": name, "strategy": strategy, **summarize(result)})

print("\n" + "=" * 68)
print("COMPARISON (same failures, four ways of choosing the recovery zone)")
print("topsis = TOPSIS + look-ahead | topsis_basic = TOPSIS only | cheapest / random = baselines")
comp = pd.DataFrame(compare)
print(comp.to_string(index=False))

pd.concat(all_rows).to_csv("results/demo_scenarios.csv", index=False)
comp.to_csv("results/demo_comparison.csv", index=False)
print("\nSaved: results/demo_scenarios.csv, results/demo_comparison.csv")