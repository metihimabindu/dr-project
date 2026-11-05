import numpy as np
import pandas as pd

# Weights from design.md: availability, recovery time, cost, carbon
WEIGHTS = {
    "Critical":  [0.35, 0.40, 0.15, 0.10],
    "Important": [0.30, 0.25, 0.25, 0.20],
    "Normal":    [0.15, 0.10, 0.35, 0.40],
}
BENEFIT = np.array([True, False, False, False])  # only availability: higher is better

def topsis(matrix, weights, benefit=BENEFIT):
    m = np.asarray(matrix, dtype=float)
    denom = np.sqrt((m ** 2).sum(axis=0))
    denom[denom == 0] = 1
    v = (m / denom) * np.asarray(weights)
    best = np.where(benefit, v.max(axis=0), v.min(axis=0))
    worst = np.where(benefit, v.min(axis=0), v.max(axis=0))
    d_best = np.sqrt(((v - best) ** 2).sum(axis=1))
    d_worst = np.sqrt(((v - worst) ** 2).sum(axis=1))
    total = d_best + d_worst
    total[total == 0] = 1
    return d_worst / total          # 0 to 1, higher is better

def rank_zones(zones, priority):
    matrix = zones[["capacity", "recovery_delay", "cost", "carbon"]].values
    out = zones[["zone", "region"]].copy()
    out["score"] = topsis(matrix, WEIGHTS[priority]).round(3)
    return out.sort_values("score", ascending=False).reset_index(drop=True)

if __name__ == "__main__":
    zones = pd.read_csv("configs/zones.csv")
    for p in ["Critical", "Important", "Normal"]:
        print(f"\n--- Ranking for a {p} service ---")
        print(rank_zones(zones, p).to_string(index=False))