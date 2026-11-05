"""Tests for the feedback store. Run from the dr-project folder:

    python tests/test_feedback.py        (or:  python -m pytest tests)
"""
import os
import sys
import tempfile
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from feedback.store import FeedbackStore            # noqa: E402
from simulator.recovery import Cloud, recover        # noqa: E402

ZONES = pd.DataFrame({
    "zone": ["A", "B"], "region": ["x", "y"], "cost": [10, 12], "carbon": [700, 480],
    "capacity": [100, 100], "recovery_delay": [20, 30],
})
SERVICES = pd.DataFrame({
    "service": ["Payment", "Search"], "priority": ["Critical", "Important"],
    "load": [10, 10], "home_zone": ["A", "B"],
})


def _store():
    return FeedbackStore(os.path.join(tempfile.mkdtemp(), "t.db"))


def _result(zone, delay, status="RECOVERED"):
    ok = status == "RECOVERED"
    return pd.DataFrame([{
        "service": "Payment", "priority": "Critical", "load": 10.0,
        "to_zone": zone if ok else "-", "status": status,
        "measured_delay": delay if ok else np.nan,
        "recovery_sec": delay if ok else np.nan, "cost": 100.0 if ok else 0.0,
        "carbon_g": 50.0 if ok else 0.0}])


def test_empty_history_keeps_configured_values():
    corrected = _store().corrected_zones(ZONES)
    assert list(corrected["recovery_delay"]) == [20, 30]


def test_estimate_formula():
    s = _store()
    s.log_run(1, "t", "topsis", _result("A", 60))
    s.log_run(2, "t", "topsis", _result("A", 60))
    # (3*20 + 60 + 60) / (3 + 2) = 36
    assert abs(s.estimate_delay("A", 20) - 36.0) < 1e-9
    assert abs(s.estimate_delay("B", 30) - 30.0) < 1e-9      # untouched zone


def test_only_last_window_counts():
    s = FeedbackStore(os.path.join(tempfile.mkdtemp(), "t.db"), window=5, prior_weight=0.0001)
    for i in range(10):
        s.log_run(i, "t", "topsis", _result("A", 100 if i < 5 else 40))
    assert abs(s.estimate_delay("A", 20) - 40.0) < 0.01      # the old 100s dropped out


def test_waiting_rows_are_saved_but_not_learned_from():
    s = _store()
    s.log_run(1, "t", "topsis", _result("A", 0, status="WAITING (no capacity)"))
    assert len(s.history()) == 1 and s.count() == 0
    assert s.estimate_delay("A", 20) == 20.0


def test_cloud_without_truth_measures_the_belief():
    cloud = Cloud(ZONES, SERVICES, {"A": 0.0, "B": 0.0})
    failed = cloud.zone_down("A")
    res = recover(cloud, failed)
    assert res.loc[0, "measured_delay"] == 30.0              # only zone B is left, belief = 30


def test_cloud_with_truth_measures_reality():
    cloud = Cloud(ZONES, SERVICES, {"A": 0.0, "B": 0.0}, truth={"A": 20, "B": 90}, noise={"B": 1.1})
    res = recover(cloud, cloud.zone_down("A"))
    assert abs(res.loc[0, "measured_delay"] - 99.0) < 1e-9


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"\n{len(tests)} tests passed")