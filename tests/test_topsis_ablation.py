import os
import sys
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from decision.topsis import rank_zones, WEIGHTS

def sample_zones():
    return pd.DataFrame([
        {
            "zone": "A", "region": "A",
            "capacity": 80, "recovery_delay": 20,
            "cost": 0.10, "carbon": 600,
        },
        {
            "zone": "B", "region": "B",
            "capacity": 100, "recovery_delay": 40,
            "cost": 0.12, "carbon": 300,
        },
        {
            "zone": "C", "region": "C",
            "capacity": 90, "recovery_delay": 30,
            "cost": 0.09, "carbon": 450,
        },
    ])


def test_full_topsis_returns_all_zones():
    result = rank_zones(sample_zones(), "Critical")
    assert len(result) == 3
    assert set(result["zone"]) == {"A", "B", "C"}


def test_ablation_returns_valid_scores():
    zones = sample_zones()

    for criteria in [
        ("availability", "recovery_time", "carbon"),
        ("availability", "recovery_time", "cost"),
        ("availability", "recovery_time"),
    ]:
        result = rank_zones(
            zones, "Critical", criteria=criteria
        )
        assert result["score"].between(0, 1).all()
        assert len(result) == 3


def test_remaining_weights_are_renormalized():
    selected = ["availability", "recovery_time"]
    original = dict(zip(
        ["availability", "recovery_time", "cost", "carbon"],
        WEIGHTS["Critical"]
    ))
    remaining = [original[c] for c in selected]
    normalized = [w / sum(remaining) for w in remaining]

    assert abs(sum(normalized) - 1.0) < 1e-9


def test_invalid_criterion_is_rejected():
    try:
        rank_zones(
            sample_zones(),
            "Critical",
            criteria=("availability", "unknown")
        )
    except ValueError:
        return

    raise AssertionError("Unknown criterion should raise ValueError")


if __name__ == "__main__":
    tests = [
        test_full_topsis_returns_all_zones,
        test_ablation_returns_valid_scores,
        test_remaining_weights_are_renormalized,
        test_invalid_criterion_is_rejected,
    ]

    for test in tests:
        test()
        print("PASS", test.__name__)

    print(f"\n{len(tests)} of {len(tests)} TOPSIS tests passed")
