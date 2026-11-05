"""One-time helper: create configs/services/<name>.yaml for every service in configs/services.csv.

Each YAML file is the service's disaster-recovery declaration. CI checks these files on every push.

Run from the dr-project folder:
    python ci/make_service_yaml.py            (does not overwrite existing files)
    python ci/make_service_yaml.py --force    (overwrites them)
"""
import os
import sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RTO_SECONDS = {"Critical": 60, "Important": 120, "Normal": 300}   # example targets per tier; edit in the files

TEMPLATE = """# Disaster-recovery declaration for the {name} service.
# CI (ci/validate_services.py) checks this file on every push.
name: {name}
tier: {tier:<11}# Critical | Important | Normal
load_units: {load:<6}# capacity the service needs, in the simulator's units
home_zone: {zone:<7}# zone where the service normally runs
recovery:
  rto_seconds: {rto:<6}# target: back in service within this time after a failure
"""


def main():
    force = "--force" in sys.argv
    services = pd.read_csv(os.path.join(ROOT, "configs", "services.csv"))
    out_dir = os.path.join(ROOT, "configs", "services")
    os.makedirs(out_dir, exist_ok=True)
    for _, r in services.iterrows():
        f = os.path.join(out_dir, r["service"].lower() + ".yaml")
        if os.path.exists(f) and not force:
            print("kept   ", os.path.relpath(f, ROOT))
            continue
        with open(f, "w", encoding="utf-8") as fh:
            fh.write(TEMPLATE.format(name=r["service"], tier=r["priority"], load=int(r["load"]),
                                     zone=r["home_zone"], rto=RTO_SECONDS[r["priority"]]))
        print("created", os.path.relpath(f, ROOT))


if __name__ == "__main__":
    main()