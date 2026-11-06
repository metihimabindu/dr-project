
"""Cloud simulator + priority-based recovery with RTO-feasible placement.

Failure types (from design.md): zone_down, cpu_overload, network_loss.
Recovery: Critical services first; each service gets its zone from TOPSIS.

Strategies:
  "topsis"        TOPSIS ranking + LOOK-AHEAD
  "topsis_basic"  TOPSIS ranking only
  "cheapest"      cheapest zone that has room
  "random"        random zone that has room

RTO is enforced as a candidate-feasibility constraint using the framework's
estimated zone recovery delay plus orchestration time already consumed by
earlier successful recoveries in the same recovery run.
"""

import os
import sys
from functools import lru_cache

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from decision.topsis import rank_zones  # noqa: E402
from decision.paper_baseline import (  # noqa: E402
    zone_score,
    WEIGHT_SETS,
    BASE_LATENCY_MS,
    CUTOFF_MS,
)

# ---- Simulation assumptions (state these in the report) -------------------

BASE_BUSY = 0.10
OVERLOAD_EXTRA = 0.35
ENERGY_KWH_PER_UNIT = 0.05
HOURS = 1
STEP_GAP_SEC = 2
LOOKAHEAD_MAX = 8

PRIORITY_WEIGHT = {
    "Critical": 3,
    "Important": 2,
    "Normal": 1,
}


@lru_cache(maxsize=1)
def _configured_rto_seconds():
    """Read per-service RTO targets from the repository's YAML declarations."""
    from ci.validate_services import load_declarations

    declarations = load_declarations(ROOT)

    rtos = {
        declaration["name"]: float(
            declaration["recovery"]["rto_seconds"]
        )
        for declaration in declarations.values()
    }

    if not rtos:
        raise RuntimeError(
            "No service RTO declarations were loaded from "
            "configs/services/*.yaml"
        )

    return rtos


class Cloud:
    def __init__(
        self,
        zones,
        services,
        predicted_util,
        truth=None,
        noise=None,
        pred=None,
        rto_seconds=None,
    ):
        """Create the simulated cloud.

        zones["recovery_delay"] is the delay the framework BELIEVES.
        truth is optional simulated actual delay and is used only after placement.
        rto_seconds optionally overrides/adds per-service RTO values.
        """
        self.truth = truth
        self.noise = noise or {}
        self.pred = pred

        self.zones = zones.set_index("zone")
        self.services = services.set_index("service")

        self.busy = {
            z: min(BASE_BUSY + predicted_util.get(z, 0.0), 0.9)
            for z in self.zones.index
        }

        self.placement = {
            s: r["home_zone"]
            for s, r in self.services.iterrows()
        }

        self.down = set()
        self.excluded = {}

        self.rto_seconds = dict(_configured_rto_seconds())

        if rto_seconds:
            self.rto_seconds.update(
                {
                    name: float(value)
                    for name, value in rto_seconds.items()
                }
            )

    def placed_load(self, zone):
        return sum(
            self.services.loc[s, "load"]
            for s, z in self.placement.items()
            if z == zone
        )

    def free(self, zone):
        return (
            self.zones.loc[zone, "capacity"] * (1 - self.busy[zone])
            - self.placed_load(zone)
        )

    # ---- Failure injection -----------------------------------------------

    def zone_down(self, zone):
        self.down.add(zone)

        failed = [
            s for s, z in self.placement.items()
            if z == zone
        ]

        for service in failed:
            del self.placement[service]
            self.excluded.setdefault(service, set()).add(zone)

        return failed

    def cpu_overload(self, zone):
        self.busy[zone] = min(
            self.busy[zone] + OVERLOAD_EXTRA,
            0.95,
        )

        failed = []

        while self.free(zone) < 0:
            here = [
                s for s, z in self.placement.items()
                if z == zone
            ]

            if not here:
                break

            victim = sorted(
                here,
                key=lambda s: (
                    PRIORITY_WEIGHT[
                        self.services.loc[s, "priority"]
                    ],
                    -self.services.loc[s, "load"],
                ),
            )[0]

            del self.placement[victim]
            self.excluded.setdefault(victim, set()).add(zone)
            failed.append(victim)

        return failed

    def network_loss(self, service):
        zone = self.placement.pop(service)
        self.excluded.setdefault(service, set()).add(zone)
        return [service]


# ---- Zone choice ----------------------------------------------------------


def _topsis_order(cloud, candidates, priority, criteria=None):
    """Candidate zones, best first, using the selected TOPSIS criteria."""

    table = cloud.zones.loc[candidates].reset_index()
    table["capacity"] = [
        cloud.free(zone)
        for zone in candidates
    ]

    ranked_table = rank_zones(
        table,
        priority,
        criteria=criteria,
    )

    return list(ranked_table["zone"])


def _future_value(cloud, free, remaining, already_recovered):
    """Best future priority weight under both capacity and RTO constraints."""

    zone_ids = list(free)
    memo = {}

    def go(i, free_t, recovered_count):
        if i == len(remaining):
            return 0

        key = (i, free_t, recovered_count)

        if key in memo:
            return memo[key]

        service = remaining[i]

        weight = PRIORITY_WEIGHT[
            cloud.services.loc[service, "priority"]
        ]

        load = cloud.services.loc[service, "load"]
        rto = cloud.rto_seconds.get(service, float("inf"))
        banned = cloud.excluded.get(service, set())

        options = [
            k
            for k, zone in enumerate(zone_ids)
            if (
                zone not in banned
                and free_t[k] >= load
                and float(
                    cloud.zones.loc[zone, "recovery_delay"]
                )
                + STEP_GAP_SEC
                * (already_recovered + recovered_count)
                <= rto
            )
        ]

        if not options:
            best = go(
                i + 1,
                free_t,
                recovered_count,
            )
        else:
            best = 0

            for k in options:
                nxt = list(free_t)
                nxt[k] = round(nxt[k] - load, 6)

                best = max(
                    best,
                    weight + go(
                        i + 1,
                        tuple(nxt),
                        recovered_count + 1,
                    ),
                )

        memo[key] = best
        return best

    return go(
        0,
        tuple(round(free[z], 6) for z in zone_ids),
        0,
    )


def _choose(
    cloud,
    candidates,
    service,
    strategy,
    rng,
    later,
    done,
    decision_trace=None,
):
    """Choose a recovery zone using the requested strategy."""

    priority = cloud.services.loc[service, "priority"]
    load = cloud.services.loc[service, "load"]

    # Single-objective baselines.
    if strategy in ("cheapest", "cost_only"):
        return min(
            candidates,
            key=lambda z: float(
                cloud.zones.loc[z, "cost"]
            ),
        )

    if strategy == "carbon_only":
        return min(
            candidates,
            key=lambda z: float(
                cloud.zones.loc[z, "carbon"]
            ),
        )

    if strategy == "random":
        return candidates[
            int(rng.integers(len(candidates)))
        ]

    # Base-paper baseline.
    if strategy.startswith("paper"):
        parts = strategy.split(":")

        weights = (
            WEIGHT_SETS[parts[1]]
            if len(parts) > 1 and parts[1] in WEIGHT_SETS
            else WEIGHT_SETS["equal"]
        )

        cutoff = CUTOFF_MS if "cutoff" in parts else None

        best, best_score = None, 0.0

        for zone in candidates:
            extra = (
                cloud.placed_load(zone) + load
            ) / cloud.zones.loc[zone, "capacity"]

            cpu = (
                BASE_BUSY
                + cloud.pred["cpu"][zone]
                + extra
            )

            mem = (
                BASE_BUSY
                + cloud.pred["mem"][zone]
                + extra
            )

            score = zone_score(
                cpu,
                mem,
                BASE_LATENCY_MS.get(zone, 50),
                weights,
                cutoff,
            )

            if score > best_score:
                best, best_score = zone, score

        return best

    # Multi-objective TOPSIS ablations.
    criteria_by_strategy = {
        "topsis_no_cost": (
            "availability",
            "recovery_time",
            "carbon",
        ),
        "topsis_no_carbon": (
            "availability",
            "recovery_time",
            "cost",
        ),
        "topsis_no_cost_carbon": (
            "availability",
            "recovery_time",
        ),
    }

    criteria = criteria_by_strategy.get(strategy)

    # IMPORTANT: Calculate the TOPSIS ranking before using it.
    ranked = _topsis_order(
        cloud,
        candidates,
        priority,
        criteria=criteria,
    )

    # Collect candidate information for optional decision tracing.
    candidate_details = {
        zone: {
            "rank": rank + 1,
            "cost": float(
                cloud.zones.loc[zone, "cost"]
            ),
            "carbon": float(
                cloud.zones.loc[zone, "carbon"]
            ),
            "capacity": float(
                cloud.free(zone)
            ),
            "recovery_delay": float(
                cloud.zones.loc[zone, "recovery_delay"]
            ),
            "future_value": None,
        }
        for rank, zone in enumerate(ranked)
    }

    # Basic TOPSIS does not use look-ahead.
    if (
        strategy == "topsis_basic"
        or len(ranked) == 1
        or not later
    ):
        chosen = ranked[0]

        if decision_trace is not None:
            decision_trace.append({
                "service": service,
                "priority": priority,
                "strategy": strategy,
                "candidates": candidate_details,
                "selected_zone": chosen,
                "lookahead_used": False,
                "lookahead_overrode_topsis": False,
            })

        return chosen

    # Full or ablated TOPSIS with the same look-ahead constraint.
    base_free = {
        zone: cloud.free(zone)
        for zone in cloud.zones.index
        if zone not in cloud.down
    }

    best_zone = ranked[0]
    best_val = -1

    for zone in ranked:
        free = dict(base_free)
        free[zone] -= load

        value = _future_value(
            cloud,
            free,
            later[:LOOKAHEAD_MAX],
            done + 1,
        )

        candidate_details[zone]["future_value"] = value

        # Strict comparison preserves TOPSIS order when values tie.
        if value > best_val:
            best_zone = zone
            best_val = value

    # Optional tracing: records the decision without changing it.
    if decision_trace is not None:
        decision_trace.append({
            "service": service,
            "priority": priority,
            "strategy": strategy,
            "candidates": candidate_details,
            "selected_zone": best_zone,
            "lookahead_used": True,
            "lookahead_overrode_topsis": (
                best_zone != ranked[0]
            ),
        })

    return best_zone


# ---- Recovery orchestration ----------------------------------------------


def recover(
    cloud,
    failed,
    strategy="topsis",
    rng=None,
    decision_trace=None,
):
    """Recover services in priority order, rejecting candidates that miss RTO."""

    rng = (
        rng
        if rng is not None
        else np.random.default_rng(0)
    )

    order = sorted(
        failed,
        key=lambda s: (
            -PRIORITY_WEIGHT[
                cloud.services.loc[s, "priority"]
            ],
            -cloud.services.loc[s, "load"],
        ),
    )

    rows = []
    done = 0

    for idx, service in enumerate(order):
        priority = cloud.services.loc[service, "priority"]
        load = cloud.services.loc[service, "load"]
        rto = cloud.rto_seconds.get(
            service,
            float("inf"),
        )

        # First apply physical/placement constraints.
        capacity_candidates = [
            zone
            for zone in cloud.zones.index
            if (
                zone not in cloud.down
                and zone not in cloud.excluded.get(service, set())
                and cloud.free(zone) >= load
            )
        ]

        # Then enforce RTO, including orchestration time already consumed.
        candidates = [
            zone
            for zone in capacity_candidates
            if (
                float(
                    cloud.zones.loc[zone, "recovery_delay"]
                )
                + STEP_GAP_SEC * done
            ) <= rto
        ]

        row = {
            "service": service,
            "priority": priority,
            "load": load,
            "rto_sec": (
                rto if np.isfinite(rto) else np.nan
            ),
        }

        zone = (
            _choose(
                cloud,
                candidates,
                service,
                strategy,
                rng,
                order[idx + 1:],
                done,
                decision_trace,
            )
            if candidates
            else None
        )

        if zone is None:
            if capacity_candidates and not candidates:
                why = "RTO constraint"
            elif not capacity_candidates:
                why = "no capacity"
            else:
                why = "no suitable zone"

            row.update(
                to_zone="-",
                status=f"WAITING ({why})",
                estimated_recovery_sec=np.nan,
                measured_delay=np.nan,
                recovery_sec=np.nan,
                cost=0.0,
                carbon_g=0.0,
            )

        else:
            cloud.placement[service] = zone
            zone_row = cloud.zones.loc[zone]

            estimated = (
                float(zone_row["recovery_delay"])
                + STEP_GAP_SEC * done
            )

            if cloud.truth is None:
                measured = float(
                    zone_row["recovery_delay"]
                )
            else:
                measured = float(
                    cloud.truth[zone]
                    * cloud.noise.get(zone, 1.0)
                )

            row.update(
                to_zone=zone,
                status="RECOVERED",
                estimated_recovery_sec=estimated,
                measured_delay=measured,
                recovery_sec=float(
                    measured + STEP_GAP_SEC * done
                ),
                cost=float(
                    zone_row["cost"] * load * HOURS
                ),
                carbon_g=float(
                    zone_row["carbon"]
                    * load
                    * ENERGY_KWH_PER_UNIT
                    * HOURS
                ),
            )

            done += 1

        rows.append(row)

    return pd.DataFrame(rows)


def summarize(result):
    """One line of metrics for a recovery run."""

    weights = result["priority"].map(PRIORITY_WEIGHT)
    recovered = result["status"] == "RECOVERED"
    critical = result["priority"] == "Critical"

    return {
        "failed": len(result),
        "recovered": int(recovered.sum()),
        "critical_recovered": (
            f"{int((recovered & critical).sum())}/"
            f"{int(critical.sum())}"
        ),
        "weighted_availability_%": (
            round(
                100 * weights[recovered].sum() / weights.sum(),
                1,
            )
            if weights.sum()
            else 100.0
        ),
        "avg_recovery_sec": (
            round(
                result.loc[
                    recovered,
                    "recovery_sec",
                ].mean(),
                1,
            )
            if recovered.any()
            else np.nan
        ),
        "cost": round(result["cost"].sum(), 1),
        "carbon_g": round(result["carbon_g"].sum(), 1),
    }
