"""CI checks: the service declarations and a simulated failure for every failure type.

Run from the dr-project folder:   python tests/test_ci.py
It needs only pandas, numpy and PyYAML (no TensorFlow), so GitHub Actions can run it on every push.
"""
import os
import shutil
import sys
import tempfile
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ci.validate_services import validate, load_declarations      # noqa: E402
from simulator.recovery import Cloud, recover, PRIORITY_WEIGHT   # noqa: E402

ZONES = pd.read_csv(os.path.join(ROOT, "configs", "zones.csv"))
SERVICES = pd.read_csv(os.path.join(ROOT, "configs", "services.csv"))
ZONE_IDS = list(ZONES["zone"])


def cloud_with_load(level):
    """A cloud where every zone is `level` busy from other tenants' predicted load (fixed, no LSTM needed)."""
    util = {z: level for z in ZONE_IDS}
    pred = {"cpu": {z: level for z in ZONE_IDS}, "mem": {z: level for z in ZONE_IDS}}
    return Cloud(ZONES, SERVICES, util, pred=pred)


def weighted_availability(res):
    w = res["priority"].map(PRIORITY_WEIGHT)
    return w[res["status"] == "RECOVERED"].sum() / w.sum()


def copy_configs(dest):
    shutil.copytree(os.path.join(ROOT, "configs"), os.path.join(dest, "configs"))


# ---- the service declarations --------------------------------------------------------------
def test_every_service_has_a_yaml_declaration():
    names = {d["name"].lower() for d in load_declarations(ROOT).values()}
    assert names == {s.lower() for s in SERVICES["service"]}, names


def test_declarations_are_valid():
    problems = validate(ROOT)
    assert problems == [], problems


def test_validator_rejects_a_bad_tier():
    tmp = tempfile.mkdtemp()
    try:
        copy_configs(tmp)
        f = os.path.join(tmp, "configs", "services", "payment.yaml")
        text = open(f, encoding="utf-8").read().replace("tier: Critical", "tier: Urgent")
        open(f, "w", encoding="utf-8").write(text)
        problems = validate(tmp)
        assert any("tier" in p for p in problems), problems
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_validator_rejects_a_mismatch_with_services_csv():
    tmp = tempfile.mkdtemp()
    try:
        copy_configs(tmp)
        f = os.path.join(tmp, "configs", "services", "login.yaml")
        text = open(f, encoding="utf-8").read().replace("home_zone: A", "home_zone: B")
        open(f, "w", encoding="utf-8").write(text)
        problems = validate(tmp)
        assert any("home_zone" in p and "services.csv" in p for p in problems), problems
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_validator_rejects_a_missing_declaration():
    tmp = tempfile.mkdtemp()
    try:
        copy_configs(tmp)
        os.remove(os.path.join(tmp, "configs", "services", "reports.yaml"))
        problems = validate(tmp)
        assert any("Reports" in p and "no YAML" in p for p in problems), problems
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---- simulated failures ---------------------------------------------------------------------
def test_zone_down_recovers_critical_services_first_and_within_their_rto():
    rto = {d["name"]: d["recovery"]["rto_seconds"] for d in load_declarations(ROOT).values()}
    for zone in ZONE_IDS:
        cloud = cloud_with_load(0.2)
        failed = cloud.zone_down(zone)
        res = recover(cloud, failed, strategy="topsis", rng=np.random.default_rng(0))
        weights = list(res["priority"].map(PRIORITY_WEIGHT))
        assert weights == sorted(weights, reverse=True), f"zone {zone}: not in priority order"
        for _, r in res[res["priority"] == "Critical"].iterrows():
            assert r["status"] == "RECOVERED", f"zone {zone}: {r['service']} was not recovered"
            assert r["recovery_sec"] <= rto[r["service"]], \
                f"zone {zone}: {r['service']} took {r['recovery_sec']} s, target {rto[r['service']]} s"


def test_every_failure_type_gives_a_consistent_result():
    cases = [("zone_down", ["A"]), ("two_zones_down", ["A", "B"]), ("cpu_overload", ["D"]),
             ("network_loss", ["Payment"])]
    for kind, target in cases:
        cloud = cloud_with_load(0.2)
        if kind == "zone_down":
            failed = cloud.zone_down(target[0])
        elif kind == "two_zones_down":
            failed = cloud.zone_down(target[0]) + cloud.zone_down(target[1])
        elif kind == "cpu_overload":
            failed = cloud.cpu_overload(target[0])
        else:
            failed = cloud.network_loss(target[0])
        assert failed, f"{kind}: nothing failed"
        res = recover(cloud, failed, strategy="topsis", rng=np.random.default_rng(0))
        assert len(res) == len(failed), kind
        assert set(res["status"].str.split(" ").str[0]) <= {"RECOVERED", "WAITING"}, kind
        for _, r in res[res["status"] == "RECOVERED"].iterrows():
            assert r["to_zone"] not in cloud.down, f"{kind}: {r['service']} sent to a failed zone"
            assert r["to_zone"] not in cloud.excluded.get(r["service"], set()), kind
        for z in ZONE_IDS:
            if z not in cloud.down:
                assert cloud.free(z) >= -1e-9, f"{kind}: zone {z} is over capacity"


def test_look_ahead_never_recovers_less_than_plain_topsis():
    pairs = [[a, b] for i, a in enumerate(ZONE_IDS) for b in ZONE_IDS[i + 1:]]
    for level in (0.1, 0.3, 0.5):
        for targets in [[z] for z in ZONE_IDS] + pairs:
            wa = {}
            for strategy in ("topsis", "topsis_basic"):
                cloud = cloud_with_load(level)
                failed = []
                for z in targets:
                    failed += cloud.zone_down(z)
                res = recover(cloud, failed, strategy=strategy, rng=np.random.default_rng(0))
                wa[strategy] = weighted_availability(res)
            assert wa["topsis"] >= wa["topsis_basic"] - 1e-9, (level, targets, wa)


if __name__ == "__main__":
    tests = sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f))
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("PASS", name)
        except Exception as exc:
            failed += 1
            print("FAIL", name, "-", repr(exc))
    print(f"\n{len(tests) - failed} of {len(tests)} tests passed")
    sys.exit(1 if failed else 0)