"""Cloud simulator + priority-based recovery.

Failure types (from design.md): zone_down, cpu_overload, network_loss.
Recovery: Critical services first; each service gets its zone from TOPSIS.

Strategies:
  "topsis"        TOPSIS ranking + LOOK-AHEAD (never picks a zone that blocks later services)
  "topsis_basic"  TOPSIS ranking only, one service at a time (kept for comparison)
  "cheapest"      baseline: cheapest zone that has room
  "random"        baseline: random zone that has room
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decision.topsis import rank_zones  # noqa: E402
from decision.paper_baseline import (zone_score, WEIGHT_SETS,  # noqa: E402
                                     BASE_LATENCY_MS, CUTOFF_MS)

# ---- Simulation assumptions (state these in the report) -------------------
BASE_BUSY = 0.10            # share of each zone's capacity used by other tenants
OVERLOAD_EXTRA = 0.35       # extra busy share when a zone has a CPU overload
ENERGY_KWH_PER_UNIT = 0.05  # energy used by 1 load unit in 1 hour
HOURS = 1                   # recovery cost/carbon counted for 1 hour
STEP_GAP_SEC = 2            # orchestration time between consecutive recoveries
LOOKAHEAD_MAX = 8           # how many later services the look-ahead considers
PRIORITY_WEIGHT = {"Critical": 3, "Important": 2, "Normal": 1}


class Cloud:
    def __init__(self, zones, services, predicted_util, truth=None, noise=None, pred=None):
        """predicted_util: {zone: fraction 0..1} from the LSTM (higher = busier).

        zones["recovery_delay"] is what the framework BELIEVES (its estimate).
        truth: optional {zone: real mean recovery delay in sec}. When given, the delay
               actually measured after a recovery comes from truth (x noise[zone]),
               not from the belief. Without it, measured delay = belief (old behaviour).
        """
        self.truth = truth
        self.noise = noise or {}
        self.pred = pred            # {"cpu": {zone: frac}, "mem": {zone: frac}}, needed by "paper"
        self.zones = zones.set_index("zone")
        self.services = services.set_index("service")
        self.busy = {z: min(BASE_BUSY + predicted_util.get(z, 0.0), 0.9) for z in self.zones.index}
        self.placement = {s: r["home_zone"] for s, r in self.services.iterrows()}
        self.down = set()
        self.excluded = {}          # service -> zones it may not go to

    def placed_load(self, zone):
        return sum(self.services.loc[s, "load"] for s, z in self.placement.items() if z == zone)

    def free(self, zone):
        return self.zones.loc[zone, "capacity"] * (1 - self.busy[zone]) - self.placed_load(zone)

    # ---- failure injection -------------------------------------------------
    def zone_down(self, zone):
        self.down.add(zone)
        failed = [s for s, z in self.placement.items() if z == zone]
        for s in failed:
            del self.placement[s]
            self.excluded.setdefault(s, set()).add(zone)
        return failed

    def cpu_overload(self, zone):
        self.busy[zone] = min(self.busy[zone] + OVERLOAD_EXTRA, 0.95)
        failed = []
        while self.free(zone) < 0:
            here = [s for s, z in self.placement.items() if z == zone]
            if not here:
                break
            # evict the least important service first (lowest weight, then biggest load)
            victim = sorted(here, key=lambda s: (PRIORITY_WEIGHT[self.services.loc[s, "priority"]],
                                                 -self.services.loc[s, "load"]))[0]
            del self.placement[victim]
            self.excluded.setdefault(victim, set()).add(zone)
            failed.append(victim)
        return failed

    def network_loss(self, service):
        zone = self.placement.pop(service)
        self.excluded.setdefault(service, set()).add(zone)
        return [service]


# ---- zone choice ----------------------------------------------------------
def _topsis_order(cloud, candidates, priority):
    """Candidate zones, best first, ranked by TOPSIS on free capacity, delay, cost, carbon."""
    table = cloud.zones.loc[candidates].reset_index()
    table["capacity"] = [cloud.free(z) for z in candidates]   # "availability" = free capacity now
    return list(rank_zones(table, priority)["zone"])


def _future_value(cloud, free, remaining):
    """Best total priority-weight the `remaining` services can still recover from the
    free capacities in `free`. Services are placed in order and must be placed when a
    zone has room, exactly like recover() does, so the answer is reachable in practice."""
    zone_ids = list(free)
    memo = {}

    def go(i, free_t):
        if i == len(remaining):
            return 0
        key = (i, free_t)
        if key in memo:
            return memo[key]
        s = remaining[i]
        w = PRIORITY_WEIGHT[cloud.services.loc[s, "priority"]]
        load = cloud.services.loc[s, "load"]
        banned = cloud.excluded.get(s, set())
        options = [k for k, z in enumerate(zone_ids) if z not in banned and free_t[k] >= load]
        if not options:
            best = go(i + 1, free_t)                      # nothing fits: service waits
        else:
            best = 0
            for k in options:
                nxt = list(free_t)
                nxt[k] = round(nxt[k] - load, 6)
                best = max(best, w + go(i + 1, tuple(nxt)))
        memo[key] = best
        return best

    return go(0, tuple(round(free[z], 6) for z in zone_ids))


def _choose(cloud, candidates, service, strategy, rng, later):
    priority = cloud.services.loc[service, "priority"]
    load = cloud.services.loc[service, "load"]
    if strategy == "cheapest":
        return min(candidates, key=lambda z: cloud.zones.loc[z, "cost"])
    if strategy == "random":
        return candidates[int(rng.integers(len(candidates)))]
    if strategy.startswith("paper"):
        # base paper, Eq. 1: "paper" (equal weights, no latency cutoff), "paper:cpu_heavy",
        # "paper:latency_heavy", "paper:cutoff" (equal weights + latency cutoff)
        parts = strategy.split(":")
        weights = WEIGHT_SETS[parts[1]] if len(parts) > 1 and parts[1] in WEIGHT_SETS else WEIGHT_SETS["equal"]
        cutoff = CUTOFF_MS if "cutoff" in parts else None
        best, best_score = None, 0.0
        for z in candidates:
            extra = (cloud.placed_load(z) + load) / cloud.zones.loc[z, "capacity"]
            c = BASE_BUSY + cloud.pred["cpu"][z] + extra     # utilization after the move
            m = BASE_BUSY + cloud.pred["mem"][z] + extra
            s = zone_score(c, m, BASE_LATENCY_MS.get(z, 50), weights, cutoff)
            if s > best_score:
                best, best_score = z, s
        return best                                          # None = every score was 0
    ranked = _topsis_order(cloud, candidates, priority)
    if strategy == "topsis_basic" or len(ranked) == 1 or not later:
        return ranked[0]
    # look-ahead: among the TOPSIS-ranked zones, keep the best one that does not cost
    # us any later service (ties go to the higher TOPSIS rank)
    base_free = {z: cloud.free(z) for z in cloud.zones.index if z not in cloud.down}
    best_zone, best_val = ranked[0], -1
    for z in ranked:
        free = dict(base_free)
        free[z] = free[z] - load
        val = _future_value(cloud, free, later[:LOOKAHEAD_MAX])
        if val > best_val:
            best_zone, best_val = z, val
    return best_zone


def recover(cloud, failed, strategy="topsis", rng=None):
    """Recover failed services in priority order. Returns one result row per service."""
    rng = rng if rng is not None else np.random.default_rng(0)
    order = sorted(failed, key=lambda s: (-PRIORITY_WEIGHT[cloud.services.loc[s, "priority"]],
                                          -cloud.services.loc[s, "load"]))
    rows, done = [], 0
    for idx, s in enumerate(order):
        priority = cloud.services.loc[s, "priority"]
        load = cloud.services.loc[s, "load"]
        candidates = [z for z in cloud.zones.index
                      if z not in cloud.down and z not in cloud.excluded.get(s, set())
                      and cloud.free(z) >= load]
        row = {"service": s, "priority": priority, "load": load}
        zone = _choose(cloud, candidates, s, strategy, rng, order[idx + 1:]) if candidates else None
        if zone is None:
            why = "no capacity" if not candidates else "no suitable zone"
            row.update(to_zone="-", status=f"WAITING ({why})", measured_delay=np.nan,
                       recovery_sec=np.nan, cost=0.0, carbon_g=0.0)
        else:
            cloud.placement[s] = zone
            z = cloud.zones.loc[zone]
            if cloud.truth is None:
                measured = float(z["recovery_delay"])
            else:
                measured = float(cloud.truth[zone] * cloud.noise.get(zone, 1.0))
            row.update(to_zone=zone, status="RECOVERED", measured_delay=measured,
                       recovery_sec=float(measured + STEP_GAP_SEC * done),
                       cost=float(z["cost"] * load * HOURS),
                       carbon_g=float(z["carbon"] * load * ENERGY_KWH_PER_UNIT * HOURS))
            done += 1
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(result):
    """One line of metrics for a recovery run."""
    w = result["priority"].map(PRIORITY_WEIGHT)
    ok = result["status"] == "RECOVERED"
    crit = result["priority"] == "Critical"
    return {
        "failed": len(result),
        "recovered": int(ok.sum()),
        "critical_recovered": f"{int((ok & crit).sum())}/{int(crit.sum())}",
        "weighted_availability_%": round(100 * w[ok].sum() / w.sum(), 1) if w.sum() else 100.0,
        "avg_recovery_sec": round(result.loc[ok, "recovery_sec"].mean(), 1) if ok.any() else np.nan,
        "cost": round(result["cost"].sum(), 1),
        "carbon_g": round(result["carbon_g"].sum(), 1),
    }