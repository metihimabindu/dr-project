import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

CONDITIONS = ["no_drift", "drift", "stress"]

PAIRS = {
    "cost_impact": ("topsis_la", "topsis_no_cost"),
    "carbon_impact": ("topsis_la", "topsis_no_carbon"),
    "combined_impact": ("topsis_la", "topsis_no_cost_carbon"),
    "feedback_impact": ("topsis_fb", "topsis_la"),
    "lookahead_impact": ("topsis_la", "topsis_basic"),
}

KEYS = [
    "trial",
    "scenario",
    "kind",
    "service",
    "priority",
]


def load_trace(condition):
    path = RESULTS / f"decision_trace_{condition}.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    df["candidates"] = df["candidates"].apply(
        lambda value: json.loads(value)
        if isinstance(value, str) and value.strip()
        else {}
    )

    return df


def candidate_order(candidates):
    return sorted(
        candidates,
        key=lambda zone: candidates[zone]["rank"]
    )


def compare_pair(df, pair_name, reference, alternative):
    ref = df[df["method"] == reference].copy()
    alt = df[df["method"] == alternative].copy()

    merged = ref.merge(
        alt,
        on=KEYS,
        suffixes=("_ref", "_alt"),
        how="inner",
    )

    records = []

    for _, row in merged.iterrows():
        ref_candidates = row["candidates_ref"]
        alt_candidates = row["candidates_alt"]

        ref_zones = set(ref_candidates)
        alt_zones = set(alt_candidates)

        same_candidates = ref_zones == alt_zones

        ref_first = (
            candidate_order(ref_candidates)[0]
            if ref_candidates else None
        )

        alt_first = (
            candidate_order(alt_candidates)[0]
            if alt_candidates else None
        )

        ref_selected = row["selected_zone_ref"]
        alt_selected = row["selected_zone_alt"]

        ref_selected_data = ref_candidates.get(ref_selected, {})
        alt_selected_data = alt_candidates.get(alt_selected, {})

        records.append({
            "condition": row["condition_ref"],
            "trial": row["trial"],
            "scenario": row["scenario"],
            "kind": row["kind"],
            "service": row["service"],
            "priority": row["priority"],
            "comparison": pair_name,
            "reference_method": reference,
            "alternative_method": alternative,
            "same_candidate_set": same_candidates,
            "reference_first_choice": ref_first,
            "alternative_first_choice": alt_first,
            "reference_selected": ref_selected,
            "alternative_selected": alt_selected,
            "ranking_changed": ref_first != alt_first,
            "selection_changed": ref_selected != alt_selected,
            "reference_lookahead": row["lookahead_used_ref"],
            "alternative_lookahead": row["lookahead_used_alt"],
            "reference_override": row["lookahead_overrode_topsis_ref"],
            "alternative_override": row["lookahead_overrode_topsis_alt"],
            "reference_selected_cost": ref_selected_data.get("cost"),
            "alternative_selected_cost": alt_selected_data.get("cost"),
            "reference_selected_carbon": ref_selected_data.get("carbon"),
            "alternative_selected_carbon": alt_selected_data.get("carbon"),
            "reference_selected_delay": ref_selected_data.get("recovery_delay"),
            "alternative_selected_delay": alt_selected_data.get("recovery_delay"),
        })

    return pd.DataFrame(records)


def summarize(results):
    if results.empty:
        return {
            "matched_decisions": 0,
            "same_candidate_sets": 0,
            "ranking_changes": 0,
            "selection_changes": 0,
        }

    same = results[results["same_candidate_set"]]

    return {
        "matched_decisions": len(results),
        "same_candidate_sets": int(
            results["same_candidate_set"].sum()
        ),
        "ranking_changes_same_candidates": int(
            same["ranking_changed"].sum()
        ),
        "selection_changes_same_candidates": int(
            same["selection_changed"].sum()
        ),
        "selection_change_pct_same_candidates": round(
            100 * same["selection_changed"].mean(), 2
        ) if len(same) else None,
    }


def main():
    all_details = []
    summary_rows = []

    for condition in CONDITIONS:
        print("\n" + "=" * 75)
        print("CONDITION:", condition)

        df = load_trace(condition)

        for pair_name, (reference, alternative) in PAIRS.items():
            result = compare_pair(
                df,
                pair_name,
                reference,
                alternative,
            )

            if result.empty:
                print(f"\n{pair_name}: no matched decisions")
                continue

            all_details.append(result)

            stats = summarize(result)
            stats["condition"] = condition
            stats["comparison"] = pair_name

            summary_rows.append(stats)

            print(f"\n{pair_name}: {reference} vs {alternative}")

            for key, value in stats.items():
                if key not in ("condition", "comparison"):
                    print(f"  {key}: {value}")

    if all_details:
        details = pd.concat(all_details, ignore_index=True)
        details.to_csv(
            RESULTS / "decision_trace_paired_details.csv",
            index=False,
        )

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(
            RESULTS / "decision_trace_paired_summary.csv",
            index=False,
        )

        print("\n" + "=" * 75)
        print("FINAL PAIRED DECISION SUMMARY")
        print(summary.to_string(index=False))

    print("\nSaved paired analysis in results/.")


if __name__ == "__main__":
    main()