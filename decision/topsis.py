
import numpy as np
import pandas as pd

WEIGHTS = {
    "Critical":  [0.35, 0.40, 0.15, 0.10],
    "Important": [0.30, 0.25, 0.25, 0.20],
    "Normal":    [0.15, 0.10, 0.35, 0.40],
}

# Internal criterion names and their corresponding zone-table columns.
CRITERIA = {
    "availability": ("capacity", True),
    "recovery_time": ("recovery_delay", False),
    "cost": ("cost", False),
    "carbon": ("carbon", False),
}

ALL_CRITERIA = tuple(CRITERIA.keys())


def topsis(matrix, weights, benefit):
    m = np.asarray(matrix, dtype=float)
    w = np.asarray(weights, dtype=float)
    benefit = np.asarray(benefit, dtype=bool)

    if m.ndim != 2 or m.shape[1] != len(w):
        raise ValueError("Matrix columns must match the number of weights.")
    if len(benefit) != len(w):
        raise ValueError("Benefit flags must match the number of weights.")
    if np.any(w < 0) or w.sum() <= 0:
        raise ValueError("Weights must be non-negative and have a positive sum.")

    denom = np.sqrt((m ** 2).sum(axis=0))
    denom[denom == 0] = 1

    v = (m / denom) * w
    best = np.where(benefit, v.max(axis=0), v.min(axis=0))
    worst = np.where(benefit, v.min(axis=0), v.max(axis=0))

    d_best = np.sqrt(((v - best) ** 2).sum(axis=1))
    d_worst = np.sqrt(((v - worst) ** 2).sum(axis=1))

    total = d_best + d_worst
    total[total == 0] = 1

    return d_worst / total


def rank_zones(zones, priority, criteria=None):
    """Rank zones using all criteria or a selected subset.

    Available criteria:
      availability, recovery_time, cost, carbon

    Removed criteria are excluded and remaining priority weights are
    renormalized to sum to one.
    """
    if priority not in WEIGHTS:
        raise ValueError(f"Unknown service priority: {priority}")

    selected = tuple(criteria or ALL_CRITERIA)

    if not selected:
        raise ValueError("At least one TOPSIS criterion is required.")

    unknown = set(selected) - set(ALL_CRITERIA)
    if unknown:
        raise ValueError(f"Unknown TOPSIS criteria: {sorted(unknown)}")

    if len(set(selected)) != len(selected):
        raise ValueError("Duplicate TOPSIS criteria are not allowed.")

    original_weights = dict(zip(ALL_CRITERIA, WEIGHTS[priority]))
    weights = np.array(
        [original_weights[name] for name in selected],
        dtype=float
    )
    weights /= weights.sum()

    columns = [CRITERIA[name][0] for name in selected]
    benefit = [CRITERIA[name][1] for name in selected]

    matrix = zones[columns].to_numpy(dtype=float)

    out = zones[["zone", "region"]].copy()
    out["score"] = topsis(matrix, weights, benefit).round(6)

    return out.sort_values(
        "score", ascending=False
    ).reset_index(drop=True)


if __name__ == "__main__":
    zones = pd.read_csv("configs/zones.csv")

    for priority in ["Critical", "Important", "Normal"]:
        print(f"\n--- Ranking for a {priority} service ---")
        print(rank_zones(zones, priority).to_string(index=False))
