"""Targeted tests for RTO-aware recovery placement.

Run from the dr-project root:
    python tests/test_rto.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from simulator.recovery import Cloud, recover  # noqa: E402


def make_cloud(zones, services, rtos):
    return Cloud(
        pd.DataFrame(zones),
        pd.DataFrame(services),
        {z["zone"]: 0.0 for z in zones},
        rto_seconds=rtos,
    )


def test_candidate_exceeding_rto_is_rejected():
    zones = [
        {"zone": "A", "region": "x", "cost": 1, "carbon": 700,
         "capacity": 100, "recovery_delay": 10},
        {"zone": "B", "region": "y", "cost": 1, "carbon": 480,
         "capacity": 100, "recovery_delay": 50},
        {"zone": "C", "region": "z", "cost": 10, "carbon": 300,
         "capacity": 100, "recovery_delay": 20},
    ]
    services = [
        {"service": "Payment", "priority": "Critical",
         "load": 10, "home_zone": "A"},
    ]
    cloud = make_cloud(zones, services, {"Payment": 30})
    result = recover(
        cloud, cloud.network_loss("Payment"),
        strategy="cheapest", rng=np.random.default_rng(0),
    )
    # B is cheaper but its estimated 50s delay exceeds the 30s RTO.
    assert result.loc[0, "status"] == "RECOVERED"
    assert result.loc[0, "to_zone"] == "C"
    assert result.loc[0, "estimated_recovery_sec"] == 20


def test_waits_when_every_capacity_candidate_violates_rto():
    zones = [
        {"zone": "A", "region": "x", "cost": 1, "carbon": 700,
         "capacity": 100, "recovery_delay": 10},
        {"zone": "B", "region": "y", "cost": 1, "carbon": 480,
         "capacity": 100, "recovery_delay": 40},
    ]
    services = [
        {"service": "Payment", "priority": "Critical",
         "load": 10, "home_zone": "A"},
    ]
    cloud = make_cloud(zones, services, {"Payment": 30})
    result = recover(
        cloud, cloud.network_loss("Payment"),
        strategy="topsis", rng=np.random.default_rng(0),
    )
    assert result.loc[0, "status"] == "WAITING (RTO constraint)"
    assert result.loc[0, "to_zone"] == "-"
    assert "Payment" not in cloud.placement


def test_orchestration_gap_counts_toward_later_service_rto():
    zones = [
        {"zone": "A", "region": "x", "cost": 1, "carbon": 700,
         "capacity": 100, "recovery_delay": 10},
        {"zone": "B", "region": "y", "cost": 1, "carbon": 480,
         "capacity": 100, "recovery_delay": 19},
    ]
    services = [
        {"service": "Payment", "priority": "Critical",
         "load": 10, "home_zone": "A"},
        {"service": "Login", "priority": "Critical",
         "load": 5, "home_zone": "A"},
    ]
    cloud = make_cloud(
        zones, services, {"Payment": 20, "Login": 20}
    )
    failed = cloud.zone_down("A")
    result = recover(
        cloud, failed, strategy="topsis_basic",
        rng=np.random.default_rng(0),
    ).set_index("service")

    # Payment is processed first (larger load): 19s is feasible.
    assert result.loc["Payment", "status"] == "RECOVERED"
    assert result.loc["Payment", "estimated_recovery_sec"] == 19

    # Login follows after a 2s orchestration gap: 19 + 2 = 21s > 20s.
    assert result.loc["Login", "status"] == "WAITING (RTO constraint)"


def test_lookahead_preserves_zone_needed_for_later_service_rto():
    zones = [
        {"zone": "A", "region": "x", "cost": 1, "carbon": 700,
         "capacity": 100, "recovery_delay": 10},
        # B is the best TOPSIS choice for Payment, but is the only RTO-feasible
        # option for Login after Payment's recovery and orchestration gap.
        {"zone": "B", "region": "y", "cost": 1, "carbon": 100,
         "capacity": 16, "recovery_delay": 17},
        {"zone": "C", "region": "z", "cost": 10, "carbon": 700,
         "capacity": 16, "recovery_delay": 19},
    ]
    services = [
        {"service": "Payment", "priority": "Critical",
         "load": 10, "home_zone": "A"},
        {"service": "Login", "priority": "Critical",
         "load": 5, "home_zone": "A"},
    ]
    cloud = make_cloud(
        zones, services, {"Payment": 30, "Login": 19}
    )
    failed = cloud.zone_down("A")
    result = recover(
        cloud, failed, strategy="topsis",
        rng=np.random.default_rng(0),
    ).set_index("service")

    assert result.loc["Payment", "status"] == "RECOVERED"
    assert result.loc["Payment", "to_zone"] == "C"
    assert result.loc["Login", "status"] == "RECOVERED"
    assert result.loc["Login", "to_zone"] == "B"
    assert result.loc["Login", "estimated_recovery_sec"] == 19


if __name__ == "__main__":
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    failures = 0
    for test in tests:
        try:
            test()
            print("PASS", test.__name__)
        except Exception as exc:
            failures += 1
            print("FAIL", test.__name__, "-", repr(exc))
    print(f"\n{len(tests) - failures} of {len(tests)} RTO tests passed")
    sys.exit(1 if failures else 0)
