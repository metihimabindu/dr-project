import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

KEYS = ["trial", "scenario", "kind", "service", "priority"]

for condition in ["no_drift", "drift", "stress"]:
    path = RESULTS / f"decision_trace_{condition}.csv"
    df = pd.read_csv(path)

    df["candidates"] = df["candidates"].apply(
        lambda x: json.loads(x) if isinstance(x, str) and x.strip() else {}
    )

    full = df[df.method == "topsis_la"].set_index(KEYS)
    no_cost = df[df.method == "topsis_no_cost"].set_index(KEYS)

    rows = []

    for key in full.index.intersection(no_cost.index):
        a = full.loc[key]
        b = no_cost.loc[key]

        candidates = a["candidates"]

        if set(candidates) != set(b["candidates"]):
            continue

        values = list(candidates.values())

        if len(values) < 2:
            continue

        costs = [x["cost"] for x in values]
        carbons = [x["carbon"] for x in values]
        delays = [x["recovery_delay"] for x in values]
        capacities = [x["capacity"] for x in values]

        rows.append({
            "priority": key[-1],
            "candidate_count": len(values),
            "cost_range": max(costs) - min(costs),
            "carbon_range": max(carbons) - min(carbons),
            "delay_range": max(delays) - min(delays),
            "capacity_range": max(capacities) - min(capacities),
            "cost_carbon_corr": (
                pd.Series(costs).corr(pd.Series(carbons))
                if len(set(costs)) > 1 and len(set(carbons)) > 1
                else None
            ),
            "full_selected": a["selected_zone"],
            "no_cost_selected": b["selected_zone"],
        })

    result = pd.DataFrame(rows)

    print("\n" + "=" * 65)
    print("CONDITION:", condition)

    if result.empty:
        print("No comparable multi-candidate decisions.")
        continue

    print("\nBY PRIORITY:")
    print(
        result.groupby("priority").agg(
            decisions=("priority", "size"),
            mean_cost_range=("cost_range", "mean"),
            mean_carbon_range=("carbon_range", "mean"),
            mean_delay_range=("delay_range", "mean"),
            mean_capacity_range=("capacity_range", "mean"),
            mean_cost_carbon_corr=("cost_carbon_corr", "mean"),
        ).round(4).to_string()
    )

    print("\nEXAMPLE CANDIDATE SETS:")
    print(result.head(8).round(4).to_string(index=False))

    print(
        "\nDecisions with different selections:",
        (result.full_selected != result.no_cost_selected).sum()
    )