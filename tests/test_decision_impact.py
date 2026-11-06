
"""Decision-impact tests for cost and carbon criteria.

Run from the dr-project root:
    python tests/test_decision_impact.py

These tests isolate TOPSIS selection by recovering one service at a time.
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from simulator.recovery import Cloud, recover


STRATEGIES = [
    "topsis",
    "topsis_no_cost",
    "topsis_no_carbon",
    "topsis_no_cost_carbon",
]


def run_recovery(zones, strategy):
    services = [
        {
            "service": "Payment",
            "priority": "Normal",
            "load": 10,
            "home_zone": "HOME",
        }
    ]

    cloud = Cloud(
        pd.DataFrame(zones),
        pd.DataFrame(services),
        {z["zone"]: 0.0 for z in zones},
        rto_seconds={"Payment": 100},
    )

    failed = cloud.zone_down("HOME")

    result = recover(
        cloud,
        failed,
        strategy=strategy,
        rng=np.random.default_rng(0),
    )

    return result.loc[0, "to_zone"]


def make_zones(costs, carbons):
    zones = []

    for name, cost, carbon in zip(
        ["A", "B", "C", "D"], costs, carbons
    ):
        zones.append({
            "zone": name,
            "region": name,
            "cost": cost,
            "carbon": carbon,
            "capacity": 100,
            "recovery_delay": 10,
        })

    zones.append({
        "zone": "HOME",
        "region": "Home",
        "cost": 1,
        "carbon": 1,
        "capacity": 100,
        "recovery_delay": 10,
    })

    return zones


def test_cost_changes_actual_recovery_zone():
    # Carbon, capacity and recovery time are identical.
    zones = make_zones(
        costs=[10, 8, 1, 6],
        carbons=[500, 500, 500, 500],
    )

    selected = {
        strategy: run_recovery(zones, strategy)
        for strategy in STRATEGIES
    }

    print("\nCost-impact selections:", selected)

    assert selected["topsis"] == "C"
    assert selected["topsis_no_cost"] == "A"
    assert selected["topsis"] != selected["topsis_no_cost"]


def test_carbon_changes_actual_recovery_zone():
    # Cost, capacity and recovery time are identical.
    zones = make_zones(
        costs=[5, 5, 5, 5],
        carbons=[700, 500, 100, 400],
    )

    selected = {
        strategy: run_recovery(zones, strategy)
        for strategy in STRATEGIES
    }

    print("\nCarbon-impact selections:", selected)

    assert selected["topsis"] == "C"
    assert selected["topsis_no_carbon"] == "A"
    assert selected["topsis"] != selected["topsis_no_carbon"]


def test_lookahead_overrides_cost_carbon_preferences():
    zones = [
        {
            "zone": "A", "region": "x", "cost": 1, "carbon": 700,
            "capacity": 100, "recovery_delay": 10,
        },
        {
            "zone": "B", "region": "y", "cost": 1, "carbon": 100,
            "capacity": 16, "recovery_delay": 17,
        },
        {
            "zone": "C", "region": "z", "cost": 10, "carbon": 700,
            "capacity": 16, "recovery_delay": 19,
        },
    ]

    services = [
        {
            "service": "Payment", "priority": "Critical",
            "load": 10, "home_zone": "A",
        },
        {
            "service": "Login", "priority": "Critical",
            "load": 5, "home_zone": "A",
        },
    ]

    selected = {}

    for strategy in STRATEGIES:
        cloud = Cloud(
            pd.DataFrame(zones),
            pd.DataFrame(services),
            {z["zone"]: 0.0 for z in zones},
            rto_seconds={"Payment": 30, "Login": 19},
        )

        result = recover(
            cloud,
            cloud.zone_down("A"),
            strategy=strategy,
            rng=np.random.default_rng(0),
        ).set_index("service")

        selected[strategy] = (
            result.loc["Payment", "to_zone"],
            result.loc["Login", "to_zone"],
        )

        assert result.loc["Payment", "status"] == "RECOVERED"
        assert result.loc["Login", "status"] == "RECOVERED"

    print("\nLook-ahead interaction selections:", selected)

    # Future Login recovery requires preserving zone B.
    # Therefore Payment must use C, regardless of TOPSIS ablation.
    for strategy in STRATEGIES:
        assert selected[strategy] == ("C", "B")

if __name__ == "__main__":
    tests = [
        test_cost_changes_actual_recovery_zone,
        test_carbon_changes_actual_recovery_zone,
        test_lookahead_overrides_cost_carbon_preferences,
    ]

    failures = 0

    for test in tests:
        try:
            test()
            print("PASS", test.__name__)
        except Exception as exc:
            failures += 1
            print("FAIL", test.__name__, "-", repr(exc))

    print(f"\n{len(tests) - failures} of {len(tests)} tests passed")
    sys.exit(1 if failures else 0)