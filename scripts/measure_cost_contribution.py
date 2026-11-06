import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

WEIGHTS = {
    "Critical":  [0.35, 0.40, 0.15, 0.10],
    "Important": [0.30, 0.25, 0.25, 0.20],
    "Normal":    [0.15, 0.10, 0.35, 0.40],
}

# availability, recovery_time, cost, carbon
COLUMNS = ["capacity", "recovery_delay", "cost", "carbon"]
BENEFIT = np.array([True, False, False, False])


def calculate_scores(candidates, priority, include_cost=True):
    zones = list(candidates)

    matrix = np.array([
        [candidates[z][column] for column in COLUMNS]
        for z in zones
    ], dtype=float)

    weights = np.array(WEIGHTS[priority], dtype=float)

    if not include_cost:
        keep = [0, 1, 3]
        matrix = matrix[:, keep]
        weights = weights[keep]
        benefit = BENEFIT[keep]
    else:
        benefit = BENEFIT

    weights = weights / weights.sum()

    denominator = np.sqrt((matrix ** 2).sum(axis=0))
    denominator[denominator == 0] = 1

    normalized = (matrix / denominator) * weights

    best = np.where(
        benefit,
        normalized.max(axis=0),
        normalized.min(axis=0),
    )

    worst = np.where(
        benefit,
        normalized.min(axis=0),
        normalized.max(axis=0),
    )

    d_best = np.sqrt(((normalized - best) ** 2).sum(axis=1))
    d_worst = np.sqrt(((normalized - worst) ** 2).sum(axis=1))

    total = d_best + d_worst
    total[total == 0] = 1

    scores = d_worst / total

    return dict(zip(zones, scores))


def main():
    for condition in ["no_drift", "drift", "stress"]:
        path = RESULTS / f"decision_trace_{condition}.csv"
        df = pd.read_csv(path)

        df = df[df["method"] == "topsis_la"]

        records = []

        for _, row in df.iterrows():
            candidates = json.loads(row["candidates"])

            if len(candidates) < 2:
                continue

            priority = row["priority"]

            full = calculate_scores(
                candidates, priority, include_cost=True
            )

            no_cost = calculate_scores(
                candidates, priority, include_cost=False
            )

            full_order = sorted(
                full, key=full.get, reverse=True
            )

            no_cost_order = sorted(
                no_cost, key=no_cost.get, reverse=True
            )

            top = full_order[0]
            second = full_order[1]

            records.append({
                "priority": priority,
                "candidate_count": len(candidates),
                "mean_abs_score_change": np.mean([
                    abs(full[z] - no_cost[z])
                    for z in candidates
                ]),
                "top_score_change": full[top] - no_cost[top],
                "top_margin_full": full[top] - full[second],
                "top_margin_no_cost": (
                    no_cost[top] - no_cost[second]
                ),
                "margin_change": (
                    (full[top] - full[second])
                    - (no_cost[top] - no_cost[second])
                ),
                "top_choice_changed": (
                    full_order[0] != no_cost_order[0]
                ),
            })

        result = pd.DataFrame(records)

        print("\n" + "=" * 70)
        print("CONDITION:", condition)

        if result.empty:
            print("No multi-candidate decisions.")
            continue

        print("\nBY PRIORITY:")
        print(
            result.groupby("priority").agg(
                decisions=("priority", "size"),
                mean_abs_score_change=(
                    "mean_abs_score_change", "mean"
                ),
                mean_top_score_change=(
                    "top_score_change", "mean"
                ),
                mean_full_top_margin=(
                    "top_margin_full", "mean"
                ),
                mean_no_cost_top_margin=(
                    "top_margin_no_cost", "mean"
                ),
                mean_margin_change=("margin_change", "mean"),
                top_choice_changes=("top_choice_changed", "sum"),
            ).round(6).to_string()
        )

        print("\nTOTAL TOP-CHOICE CHANGES:",
              int(result["top_choice_changed"].sum()),
              "out of", len(result))


if __name__ == "__main__":
    main()