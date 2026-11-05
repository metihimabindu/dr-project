"""Validate the disaster-recovery declarations in configs/services/*.yaml.

CI runs this on every push:   python ci/validate_services.py
It exits with an error if a declaration is missing, malformed, or disagrees with the files the
simulator uses (configs/services.csv and configs/zones.csv).
"""
import glob
import os
import sys
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIERS = ("Critical", "Important", "Normal")


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def load_declarations(root=ROOT):
    """Return {file name: parsed YAML} for every file in configs/services/."""
    out = {}
    for f in sorted(glob.glob(os.path.join(root, "configs", "services", "*.y*ml"))):
        with open(f, encoding="utf-8") as fh:
            out[os.path.basename(f)] = yaml.safe_load(fh)
    return out


def validate(root=ROOT):
    """Return a list of problems (an empty list means everything is fine)."""
    errors = []
    zones_f = os.path.join(root, "configs", "zones.csv")
    services_f = os.path.join(root, "configs", "services.csv")
    for f in (zones_f, services_f):
        if not os.path.exists(f):
            return [f"missing file: {os.path.relpath(f, root)}"]
    zones = pd.read_csv(zones_f)
    csv = pd.read_csv(services_f)
    zone_ids = set(zones["zone"])

    try:
        decl = load_declarations(root)
    except yaml.YAMLError as exc:
        return [f"a YAML file could not be read: {exc}"]
    if not decl:
        return ["no declarations found in configs/services/ (run: python ci/make_service_yaml.py)"]

    seen = {}
    for fname, d in decl.items():
        where = f"{fname}:"
        if not isinstance(d, dict):
            errors.append(f"{where} the file must contain key: value pairs")
            continue
        missing = [k for k in ("name", "tier", "load_units", "home_zone", "recovery") if k not in d]
        if missing:
            errors.append(f"{where} missing {', '.join(missing)}")
            continue
        if d["tier"] not in TIERS:
            errors.append(f"{where} tier '{d['tier']}' must be one of {', '.join(TIERS)}")
        if not _is_number(d["load_units"]) or d["load_units"] <= 0:
            errors.append(f"{where} load_units must be a number above 0")
        if d["home_zone"] not in zone_ids:
            errors.append(f"{where} home_zone '{d['home_zone']}' is not a zone in configs/zones.csv")
        rec = d["recovery"]
        if not isinstance(rec, dict) or not _is_number(rec.get("rto_seconds")) or rec["rto_seconds"] <= 0:
            errors.append(f"{where} recovery.rto_seconds must be a number above 0")
        key = str(d["name"]).lower()
        if key in seen:
            errors.append(f"{where} the name '{d['name']}' is already used in {seen[key]}")
        seen[key] = fname

    # the declarations must match what the simulator actually uses
    csv_rows = {r["service"].lower(): r for _, r in csv.iterrows()}
    for key in sorted(set(csv_rows) - set(seen)):
        errors.append(f"service '{csv_rows[key]['service']}' is in configs/services.csv but has no YAML file")
    for key in sorted(set(seen) - set(csv_rows)):
        errors.append(f"{seen[key]}: service '{key}' is not in configs/services.csv")
    for fname, d in decl.items():
        if not isinstance(d, dict) or str(d.get("name", "")).lower() not in csv_rows:
            continue
        r = csv_rows[str(d["name"]).lower()]
        for yaml_key, csv_key in (("tier", "priority"), ("load_units", "load"), ("home_zone", "home_zone")):
            if yaml_key in d and d[yaml_key] != r[csv_key]:
                errors.append(f"{fname}: {yaml_key} is {d[yaml_key]} but configs/services.csv says {r[csv_key]}")

    # each zone must be able to hold the services that live in it
    try:
        sys.path.insert(0, root)
        from simulator.recovery import BASE_BUSY
    except Exception:
        BASE_BUSY = 0.10
    load_by_zone = {}
    for d in decl.values():
        if isinstance(d, dict) and _is_number(d.get("load_units")) and d.get("home_zone") in zone_ids:
            load_by_zone[d["home_zone"]] = load_by_zone.get(d["home_zone"], 0) + d["load_units"]
    for _, z in zones.iterrows():
        room = z["capacity"] * (1 - BASE_BUSY)
        if load_by_zone.get(z["zone"], 0) > room:
            errors.append(f"zone {z['zone']} cannot hold its services: {load_by_zone[z['zone']]} units "
                          f"declared, {room:.0f} available")
    return errors


def main():
    errors = validate(ROOT)
    decl = load_declarations(ROOT) if not errors else {}
    if errors:
        print(f"{len(errors)} problem(s) found:")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    print(f"{len(decl)} service declarations are valid:")
    for fname, d in decl.items():
        print(f"  {d['name']:<16} {d['tier']:<10} {d['load_units']:>3} units  home zone {d['home_zone']}  "
              f"RTO {d['recovery']['rto_seconds']} s")


if __name__ == "__main__":
    main()