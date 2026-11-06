
"""Recalculate experiment confidence intervals using Student's t.

Reads the existing raw CSV files and writes corrected summary and
paired-comparison CSV files. Original experiment outputs are preserved.
"""

from pathlib import Path
import math

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

CONDITIONS = [
    "no_drift",
    "drift",
    "stress",
]

REFERENCE = "topsis_fb"

MEAN_COLS = [
    "w_avail",
    "crit_time",
    "w_time",
    "cost_per_unit",
    "carbon_per_unit",
    "peak_util",
]

COMPARISON_COLS = [
    "w_avail",
    "crit_time",
    "w_time",
    "cost_per_unit",
    "carbon_per_unit",
]


def ci95(values):
    """Two-sided 95% Student's t confidence interval half-width."""

    values = pd.Series(values).dropna()
    n = len(values)

    if n <= 1:
        return float("nan")

    standard_error = (
        values.std(ddof=1) / math.sqrt(n)
    )

    critical_value = student_t.ppf(
        0.975,
        df=n - 1
    )

    return critical_value * standard_error


def summarize(raw):
    """Average within each trial, then calculate intervals across trials."""

    output = []

    for method, group in raw.groupby("method", sort=False):

        trial_rows = []

        for trial, trial_group in group.groupby("trial"):

            row = {"trial": trial}

            for col in MEAN_COLS:
                row[col] = trial_group[col].mean()

            critical_total = trial_group["crit_total"].sum()
            critical_ok = trial_group["crit_ok"].sum()

            row["crit_total"] = critical_total
            row["crit_ok"] = critical_ok

            row["crit_avail"] = (
                100 * critical_ok / critical_total
                if critical_total > 0
                else np.nan
            )

            trial_rows.append(row)

        trial_summary = pd.DataFrame(trial_rows)

        result = {
            "method": method,
            "trials": len(trial_summary),
        }

        for col in MEAN_COLS:
            result[col] = trial_summary[col].mean()
            result[col + "_ci"] = ci95(trial_summary[col])

        result["crit_avail"] = trial_summary["crit_avail"].mean()
        result["crit_avail_ci"] = ci95(
            trial_summary["crit_avail"]
        )

        output.append(result)

    return pd.DataFrame(output)


def paired(raw):
    """Reference-minus-other paired comparisons across independent trials."""

    reference = raw[raw["method"] == REFERENCE]

    if reference.empty:
        raise ValueError(
            f"Reference method '{REFERENCE}' was not found."
        )

    output = []

    for method in raw["method"].unique():

        if method == REFERENCE:
            continue

        other = raw[raw["method"] == method]

        merged = reference.merge(
            other,
            on=["trial", "scenario"],
            suffixes=("_ref", "_other"),
            how="inner",
            validate="one_to_one",
        )

        if merged.empty:
            raise ValueError(
                f"No paired scenarios found for method '{method}'."
            )

        row = {"vs": method}

        for col in COMPARISON_COLS:

            difference = (
                merged[col + "_ref"]
                - merged[col + "_other"]
            )

            trial_differences = (
                pd.DataFrame({
                    "trial": merged["trial"],
                    "difference": difference,
                })
                .groupby("trial")["difference"]
                .mean()
            )

            row[col] = trial_differences.mean()
            row[col + "_ci"] = ci95(trial_differences)

        output.append(row)

    return pd.DataFrame(output)


def main():

    print("Student's t confidence interval recalculation")
    print("Confidence level: 95%")
    print("Reference method:", REFERENCE)
    print()

    for condition in CONDITIONS:

        raw_path = RESULTS / f"experiment_raw_{condition}.csv"

        if not raw_path.exists():
            raise FileNotFoundError(
                f"Missing raw experiment file: {raw_path}"
            )

        raw = pd.read_csv(raw_path)

        required = {
            "trial",
            "scenario",
            "method",
            "crit_total",
            "crit_ok",
            *MEAN_COLS,
        }

        missing = required - set(raw.columns)

        if missing:
            raise ValueError(
                f"{condition}: missing columns: {sorted(missing)}"
            )

        trials = sorted(raw["trial"].dropna().unique())

        if len(trials) != 5:
            raise ValueError(
                f"{condition}: expected 5 independent trials, "
                f"found {len(trials)}."
            )

        print(f"Processing: {condition}")
        print("  Raw rows:", len(raw))
        print("  Trials:", trials)

        summary = summarize(raw)
        paired_results = paired(raw)

        summary_path = (
            RESULTS / f"experiment_summary_t95_{condition}.csv"
        )

        paired_path = (
            RESULTS / f"experiment_paired_t95_{condition}.csv"
        )

        summary.to_csv(summary_path, index=False)
        paired_results.to_csv(paired_path, index=False)

        print("  Summary saved:", summary_path.name)
        print("  Paired results saved:", paired_path.name)
        print("  Methods:", len(summary))
        print("  Paired comparisons:", len(paired_results))
        print()

    print("SUCCESS: All three conditions recalculated.")
    print("Original experiment CSV files were not modified.")


if __name__ == "__main__":
    main()
