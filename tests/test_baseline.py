"""Tests for the base-paper baseline (Equation 1). Run from the dr-project folder:

    python tests/test_baseline.py        (or:  python -m pytest tests)
"""
import os
import sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from decision.paper_baseline import zone_score, WEIGHT_SETS   # noqa: E402
from simulator.recovery import Cloud, recover                   # noqa: E402

ZONES = pd.DataFrame({
    "zone": ["A", "B"], "region": ["x", "y"], "cost": [10, 12], "carbon": [700, 480],
    "capacity": [100, 100], "recovery_delay": [20, 30],
})
SERVICES = pd.DataFrame({
    "service": ["Payment", "Search"], "priority": ["Critical", "Important"],
    "load": [10, 10], "home_zone": ["A", "B"],
})


def test_score_matches_equation_by_hand():
    # utilization 0.5, base latency 40 ms -> latency 80 ms, scaled 80/200 = 0.4
    # (1/3)/0.5 + (1/3)/0.5 + (1/3)/0.4 = 2.1667
    s = zone_score(0.5, 0.5, 40, WEIGHT_SETS["equal"])
    assert abs(s - 2.16667) < 1e-4, s


def test_less_loaded_zone_scores_higher():
    assert zone_score(0.2, 0.2, 40, WEIGHT_SETS["equal"]) > zone_score(0.6, 0.6, 40, WEIGHT_SETS["equal"])


def test_latency_cutoff_gives_zero():
    assert zone_score(0.5, 0.5, 40, WEIGHT_SETS["equal"], cutoff_ms=50) == 0.0
    assert zone_score(0.5, 0.5, 40, WEIGHT_SETS["equal"], cutoff_ms=500) > 0.0


def test_weights_sum_to_one():
    for w in WEIGHT_SETS.values():
        assert abs(sum(w) - 1.0) < 1e-9


def test_paper_strategy_picks_the_less_loaded_zone():
    pred = {"cpu": {"A": 0.05, "B": 0.05}, "mem": {"A": 0.60, "B": 0.10}}
    cloud = Cloud(ZONES, SERVICES, {"A": 0.6, "B": 0.1}, pred=pred)
    res = recover(cloud, cloud.network_loss("Payment"), strategy="paper")
    assert res.loc[0, "to_zone"] == "B" and res.loc[0, "status"] == "RECOVERED"


def test_all_zero_scores_means_service_waits():
    pred = {"cpu": {"A": 0.9, "B": 0.9}, "mem": {"A": 0.9, "B": 0.9}}
    cloud = Cloud(ZONES, SERVICES, {"A": 0.0, "B": 0.0}, pred=pred)
    res = recover(cloud, cloud.zone_down("A"), strategy="paper:cutoff")
    assert res.loc[0, "status"] == "WAITING (no suitable zone)"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"\n{len(tests)} tests passed")