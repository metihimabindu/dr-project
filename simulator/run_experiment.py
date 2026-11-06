
"""
Step 11: Main disaster-recovery experiment.

The same randomly generated failures are given to every recovery method
within each trial.

Each independent trial uses:
- A different random seed.
- Its own generated failure scenarios.
- Its own feedback database for each condition.

Three conditions:
  no_drift : configured recovery delays are correct
  drift    : actual recovery delays differ from configured delays
  stress   : drift plus extra tenant load in every zone

Results include:
- Availability, recovery time, cost, carbon and utilization.
- Mean and 95% confidence intervals across independent trials.
- Paired comparisons using the same trial and failure scenarios.
- Per-failure-type breakdowns.
- Detailed TOPSIS decision traces.
- Look-ahead usage and override statistics.

Run from the dr-project folder:

    python simulator/run_experiment.py
    python simulator/run_experiment.py 30
    python simulator/run_experiment.py 200 trials=5
    python simulator/run_experiment.py 200 trials=5 stress=0.08
"""

import math
import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy.stats import t as student_t


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from simulator.predictor import Predictor
from simulator.recovery import Cloud, recover, PRIORITY_WEIGHT
from feedback.store import FeedbackStore

from decision.paper_baseline import (
    WEIGHT_SETS,
    BASE_LATENCY_MS,
    CUTOFF_MS,
)


# ---------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------

def argument_value(prefix, default):
    for arg in sys.argv[1:]:
        if arg.startswith(prefix):
            return arg.split("=", 1)[1]
    return default


N = int(next(
    (arg for arg in sys.argv[1:] if arg.isdigit()),
    200
))

BASE_SEED = 2026

TRIALS = int(argument_value("trials=", "5"))

SEEDS = [
    BASE_SEED + i
    for i in range(TRIALS)
]

NOISE_SIGMA = 0.15

TRUTH_DRIFT = {
    "A": 55,
    "B": 30,
    "C": 45,
    "D": 30,
}

KINDS = [
    "zone_down",
    "two_zones_down",
    "cpu_overload",
    "network_loss",
]

KIND_P = [
    0.4,
    0.2,
    0.2,
    0.2,
]

STRESS_EXTRA = float(argument_value("stress=", "0.05"))

if N < 1:
    raise ValueError("Number of failures must be at least 1.")

if TRIALS < 1:
    raise ValueError("Number of trials must be at least 1.")

if not 0 <= STRESS_EXTRA < 1:
    raise ValueError("stress must be between 0 and 1.")

if not math.isclose(sum(KIND_P), 1.0):
    raise ValueError("Failure probabilities must sum to 1.")


# ---------------------------------------------------------------------
# Recovery methods
#
# key, display label, chart label, strategy, uses feedback, show in chart
# ---------------------------------------------------------------------

METHODS = [
    ("random", "Random zone", "Random",
     "random", False, True),

    ("cheapest", "Cheapest zone", "Cheapest",
     "cheapest", False, True),

    ("cost_only", "Cost-only recovery", "Cost-only",
     "cost_only", False, True),

    ("carbon_only", "Carbon-only recovery", "Carbon-only",
     "carbon_only", False, True),

    ("paper", "Base paper Eq.1 (equal weights)", "Base paper Eq.1",
     "paper", False, True),

    ("topsis_basic", "TOPSIS", "TOPSIS",
     "topsis_basic", False, True),

    ("topsis_la", "TOPSIS + look-ahead", "TOPSIS + look-ahead",
     "topsis", False, True),

    ("topsis_no_cost", "TOPSIS without cost", "No cost",
     "topsis_no_cost", False, True),

    ("topsis_no_carbon", "TOPSIS without carbon", "No carbon",
     "topsis_no_carbon", False, True),

    ("topsis_no_cost_carbon",
     "TOPSIS without cost and carbon",
     "No cost/carbon",
     "topsis_no_cost_carbon", False, True),

    ("topsis_fb",
     "TOPSIS + look-ahead + feedback",
     "TOPSIS + look-ahead\n+ feedback",
     "topsis", True, True),

    ("paper_cpu",
     "Base paper Eq.1 (CPU-heavy weights)",
     "",
     "paper:cpu_heavy", False, False),

    ("paper_lat",
     "Base paper Eq.1 (latency-heavy weights)",
     "",
     "paper:latency_heavy", False, False),

    ("paper_cut",
     "Base paper Eq.1 (with latency cutoff)",
     "",
     "paper:cutoff", False, False),
]

LABEL = {
    method[0]: method[1]
    for method in METHODS
}

REFERENCE = "topsis_fb"

MAIN_KEYS = [
    method[0]
    for method in METHODS
    if method[5]
]


# ---------------------------------------------------------------------
# Load configuration and predictor
# ---------------------------------------------------------------------

os.makedirs("results", exist_ok=True)

zones = pd.read_csv("configs/zones.csv")
services = pd.read_csv("configs/services.csv")

zone_ids = list(zones["zone"])

TRUTH_NONE = {
    row["zone"]: float(row["recovery_delay"])
    for _, row in zones.iterrows()
}

predictor = Predictor()

if predictor.load_note:
    raise RuntimeError(
        "LSTM models could not be loaded: "
        + predictor.load_note
    )

missing = set(zone_ids) - set(predictor.models.keys())

if missing:
    raise RuntimeError(
        f"Missing trained LSTM models for zones: {missing}"
    )

print("All four trained LSTM models loaded successfully.")


# ---------------------------------------------------------------------
# Failure injection
# ---------------------------------------------------------------------

def inject(cloud, kind, target):

    if kind == "zone_down":
        return cloud.zone_down(target[0])

    if kind == "two_zones_down":
        return (
            cloud.zone_down(target[0])
            + cloud.zone_down(target[1])
        )

    if kind == "cpu_overload":
        return cloud.cpu_overload(target[0])

    if kind == "network_loss":
        return cloud.network_loss(target[0])

    raise ValueError(f"Unknown failure type: {kind}")


# ---------------------------------------------------------------------
# Generate scenarios independently for each trial
# ---------------------------------------------------------------------

def build_scenarios(seed):

    rng = np.random.default_rng(seed)

    scenarios = []

    while len(scenarios) < N:

        kind = str(
            rng.choice(KINDS, p=KIND_P)
        )

        if kind == "two_zones_down":

            target = [
                str(x)
                for x in rng.choice(
                    zone_ids,
                    2,
                    replace=False
                )
            ]

        elif kind == "network_loss":

            target = [
                str(
                    rng.choice(
                        list(services["service"])
                    )
                )
            ]

        else:

            target = [
                str(rng.choice(zone_ids))
            ]

        t = predictor.random_time(rng)

        p = {
            z: predictor.predict(z, t)[:2]
            for z in zone_ids
        }

        pred = {
            "cpu": {
                z: p[z][0] / 100
                for z in zone_ids
            },

            "mem": {
                z: p[z][1] / 100
                for z in zone_ids
            },
        }

        util = {
            z: max(
                pred["cpu"][z],
                pred["mem"][z]
            )
            for z in zone_ids
        }

        # Mean-corrected lognormal noise.
        # This makes the expected multiplier approximately 1.0.
        noise = {
            z: float(
                rng.lognormal(
                    -0.5 * NOISE_SIGMA ** 2,
                    NOISE_SIGMA
                )
            )
            for z in zone_ids
        }

        # Confirm that this scenario actually produces a failure.
        test_cloud = Cloud(
            zones,
            services,
            util
        )

        if not inject(test_cloud, kind, target):
            continue

        scenarios.append({
            "kind": kind,
            "target": target,
            "util": util,
            "pred": pred,
            "noise": noise,
        })

    failure_mix = pd.Series(
        [scenario["kind"] for scenario in scenarios]
    ).value_counts().to_dict()

    print(
        f"Seed {seed} failure mix: {failure_mix}"
    )

    return scenarios


# ---------------------------------------------------------------------
# Stress scenarios
# ---------------------------------------------------------------------

def stressed(scenario, extra):

    def increase_utilization(values):
        return {
            z: min(value + extra, 0.85)
            for z, value in values.items()
        }

    return {
        **scenario,

        "util": increase_utilization(
            scenario["util"]
        ),

        "pred": {
            "cpu": increase_utilization(
                scenario["pred"]["cpu"]
            ),

            "mem": increase_utilization(
                scenario["pred"]["mem"]
            ),
        },
    }


# ---------------------------------------------------------------------
# Metrics for one recovery run
# ---------------------------------------------------------------------

def metrics(cloud, res):

    ok = res["status"] == "RECOVERED"

    crit = res["priority"] == "Critical"

    w = res["priority"].map(PRIORITY_WEIGHT)

    recovered_load = res.loc[
        ok, "load"
    ].sum()

    available_util = [
        1 - cloud.free(z) / cloud.zones.loc[z, "capacity"]
        for z in cloud.zones.index
        if z not in cloud.down
    ]

    peak_util = (
        100 * max(available_util)
        if available_util
        else np.nan
    )

    return {

        "w_avail":
            100 * w[ok].sum() / w.sum()
            if w.sum() > 0 else np.nan,

        "crit_total":
            int(crit.sum()),

        "crit_ok":
            int((ok & crit).sum()),

        "crit_time":
            res.loc[
                ok & crit,
                "recovery_sec"
            ].mean()
            if (ok & crit).any()
            else np.nan,

        "w_time":
            (
                w[ok] * res.loc[
                    ok, "recovery_sec"
                ]
            ).sum() / w[ok].sum()
            if ok.any() and w[ok].sum() > 0
            else np.nan,

        "cost_per_unit":
            res["cost"].sum() / recovered_load
            if recovered_load > 0
            else np.nan,

        "carbon_per_unit":
            res["carbon_g"].sum() / recovered_load
            if recovered_load > 0
            else np.nan,

        "peak_util":
            peak_util,
    }


# ---------------------------------------------------------------------
# Run one condition for one independent trial
# ---------------------------------------------------------------------

def run_condition(cond, truth, scens, trial, trial_seed):

    db = (
        f"results/experiment_feedback_"
        f"{cond}_trial_{trial}.db"
    )

    if os.path.exists(db):
        os.remove(db)

    store = FeedbackStore(db)

    rows = []
    trace_rows = []

    for i, scenario in enumerate(scens, start=1):

        for key, _, _, strategy, learns, _ in METHODS:

            table = (
                store.corrected_zones(zones)
                if learns
                else zones
            )

            cloud = Cloud(
                table,
                services,
                scenario["util"],
                truth=truth,
                noise=scenario["noise"],
                pred=scenario["pred"],
            )

            failed = inject(
                cloud,
                scenario["kind"],
                scenario["target"]
            )

            if not failed:
                continue

            # Same random seed for corresponding methods
            # on the same trial and scenario.
            scenario_seed = trial_seed + i

            # Collect optional TOPSIS decision traces.
            decision_trace = []

            res = recover(
                cloud,
                failed,
                strategy=strategy,
                rng=np.random.default_rng(scenario_seed),
                decision_trace=decision_trace,
            )

            # Save individual decision records.
            for decision in decision_trace:

                trace_rows.append({
                    "condition": cond,
                    "trial": trial,
                    "scenario": i,
                    "kind": scenario["kind"],
                    "method": key,
                    "service": decision["service"],
                    "priority": decision["priority"],
                    "strategy": decision["strategy"],
                    "selected_zone": decision["selected_zone"],
                    "lookahead_used": decision["lookahead_used"],
                    "lookahead_overrode_topsis": (
                        decision["lookahead_overrode_topsis"]
                    ),
                    "candidates": json.dumps(
                        decision["candidates"]
                    ),
                })

            if learns:

                store.log_run(
                    i,
                    scenario["kind"]
                    + ":"
                    + "+".join(scenario["target"]),
                    strategy,
                    res,
                )

            rows.append({
                "trial": trial,
                "scenario": i,
                "kind": scenario["kind"],
                "method": key,
                **metrics(cloud, res),
            })

    trace_columns = [
        "condition",
        "trial",
        "scenario",
        "kind",
        "method",
        "service",
        "priority",
        "strategy",
        "selected_zone",
        "lookahead_used",
        "lookahead_overrode_topsis",
        "candidates",
    ]

    return (
        pd.DataFrame(rows),
        pd.DataFrame(trace_rows, columns=trace_columns),
    )


# ---------------------------------------------------------------------
# Confidence intervals across independent trials
# ---------------------------------------------------------------------

def ci95(values):

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


def wilson(k, n, z=1.96):

    if n == 0:
        return float("nan"), float("nan")

    p = k / n

    centre = (
        p + z * z / (2 * n)
    ) / (
        1 + z * z / n
    )

    half = (
        z * math.sqrt(
            p * (1 - p) / n
            + z * z / (4 * n * n)
        )
    ) / (
        1 + z * z / n
    )

    return centre - half, centre + half


MEAN_COLS = [
    "w_avail",
    "crit_time",
    "w_time",
    "cost_per_unit",
    "carbon_per_unit",
    "peak_util",
]


# ---------------------------------------------------------------------
# Summary: average within each trial first, then across trials
# ---------------------------------------------------------------------

def summarize(raw):

    out = []

    for method, group in raw.groupby(
        "method",
        sort=False
    ):

        trial_rows = []

        for trial, trial_group in group.groupby("trial"):

            row = {
                "trial": trial
            }

            for col in MEAN_COLS:
                row[col] = trial_group[col].mean()

            critical_total = trial_group[
                "crit_total"
            ].sum()

            critical_ok = trial_group[
                "crit_ok"
            ].sum()

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

            result[col + "_ci"] = ci95(
                trial_summary[col]
            )

        result["crit_avail"] = (
            trial_summary["crit_avail"].mean()
        )

        result["crit_avail_ci"] = ci95(
            trial_summary["crit_avail"]
        )

        out.append(result)

    return pd.DataFrame(out)


# ---------------------------------------------------------------------
# Paired comparisons
#
# Reference minus other method.
# Match the same trial and scenario before calculating differences.
# Confidence intervals are calculated across trial-level differences.
# ---------------------------------------------------------------------

def paired(raw):

    reference = raw[
        raw["method"] == REFERENCE
    ]

    rows = []

    comparison_cols = [
        "w_avail",
        "crit_time",
        "w_time",
        "cost_per_unit",
        "carbon_per_unit",
    ]

    for key in raw["method"].unique():

        if key == REFERENCE:
            continue

        other = raw[
            raw["method"] == key
        ]

        merged = reference.merge(
            other,
            on=["trial", "scenario"],
            suffixes=("_ref", "_other"),
            how="inner",
        )

        row = {
            "vs": key
        }

        for col in comparison_cols:

            merged["difference"] = (
                merged[col + "_ref"]
                - merged[col + "_other"]
            )

            trial_differences = (
                merged.groupby("trial")["difference"]
                .mean()
            )

            row[col] = trial_differences.mean()

            row[col + "_ci"] = ci95(
                trial_differences
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------

def fmt(mean, ci, decimals=1):

    if pd.isna(mean):
        return "-"

    if pd.isna(ci):
        return f"{mean:.{decimals}f}"

    return (
        f"{mean:.{decimals}f} "
        f"± {ci:.{decimals}f}"
    )


def pretty(summary):

    return pd.DataFrame({

        "method": [
            LABEL[key]
            for key in summary["method"]
        ],

        "weighted availability %": [
            fmt(mean, ci)
            for mean, ci in zip(
                summary.w_avail,
                summary.w_avail_ci
            )
        ],

        "critical availability %": [
            fmt(mean, ci)
            for mean, ci in zip(
                summary.crit_avail,
                summary.crit_avail_ci
            )
        ],

        "critical recovery s": [
            fmt(mean, ci)
            for mean, ci in zip(
                summary.crit_time,
                summary.crit_time_ci
            )
        ],

        "weighted recovery s": [
            fmt(mean, ci)
            for mean, ci in zip(
                summary.w_time,
                summary.w_time_ci
            )
        ],

        "cost / load unit": [
            fmt(mean, ci, 4)
            for mean, ci in zip(
                summary.cost_per_unit,
                summary.cost_per_unit_ci
            )
        ],

        "carbon g / load unit": [
            fmt(mean, ci, 2)
            for mean, ci in zip(
                summary.carbon_per_unit,
                summary.carbon_per_unit_ci
            )
        ],

        "peak zone util %": [
            fmt(mean, ci)
            for mean, ci in zip(
                summary.peak_util,
                summary.peak_util_ci
            )
        ],
    })


# ---------------------------------------------------------------------
# Breakdown by failure type
# ---------------------------------------------------------------------

def by_kind(raw):

    main = raw[
        raw["method"].isin(MAIN_KEYS)
    ]

    present = [
        kind
        for kind in KINDS
        if kind in set(main["kind"])
    ]

    tables = []

    definitions = [
        ("w_avail", "weighted availability %", 1),
        ("crit_time", "critical recovery s", 1),
        ("carbon_per_unit", "carbon g / load unit", 2),
    ]

    for col, name, decimals in definitions:

        table = main.pivot_table(
            index="method",
            columns="kind",
            values=col,
            aggfunc="mean",
        )

        table = table.reindex(
            index=MAIN_KEYS,
            columns=present,
        )

        table.index = [
            LABEL[key]
            for key in table.index
        ]

        tables.append(
            (name, table.round(decimals))
        )

    counts = (
        main[
            main["method"] == REFERENCE
        ]["kind"]
        .value_counts()
        .reindex(present)
        .fillna(0)
        .astype(int)
    )

    return tables, counts


# ---------------------------------------------------------------------
# Chart configuration
# ---------------------------------------------------------------------

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"

COLOR = {
    "random": "#b4b3ad",
    "cheapest": "#b4b3ad",
    "cost_only": "#b4b3ad",
    "carbon_only": "#b4b3ad",
    "paper": "#eb6834",
    "topsis_basic": "#a9c9f2",
    "topsis_la": "#5f9ce3",
    "topsis_fb": "#2a78d6",
}

PANELS = [
    (
        "w_avail",
        "Weighted availability, % (higher is better)",
        1,
    ),

    (
        "crit_time",
        "Critical-service recovery time, s (lower is better)",
        1,
    ),

    (
        "cost_per_unit",
        "Cost per load unit recovered, USD (lower is better)",
        4,
    ),

    (
        "carbon_per_unit",
        "Carbon per load unit recovered, g (lower is better)",
        2,
    ),
]


def chart(summary, path, title):

    main = [
        method
        for method in METHODS
        if method[5]
    ]

    summary_indexed = summary.set_index("method")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 8.2),
        facecolor=SURFACE,
    )

    for ax, (col, panel_title, decimals) in zip(
        axes.ravel(),
        PANELS,
    ):

        ax.set_facecolor(SURFACE)

        ys = np.arange(len(main))

        vals = [
            summary_indexed.loc[method[0], col]
            for method in main
        ]

        errs = [
            summary_indexed.loc[
                method[0],
                col + "_ci"
            ]
            for method in main
        ]

        safe_errs = [
            0 if pd.isna(error) else error
            for error in errs
        ]

        ax.barh(
            ys,
            vals,
            xerr=safe_errs,
            height=0.62,
            color=[
                COLOR.get(method[0], "#b4b3ad")
                for method in main
            ],
            error_kw={
                "ecolor": INK2,
                "elinewidth": 1.2,
                "capsize": 3,
            },
        )

        top = max(
            (
                value + error
                for value, error in zip(
                    vals,
                    safe_errs,
                )
            ),
            default=1,
        )

        for y, value, error in zip(
            ys,
            vals,
            safe_errs,
        ):

            ax.text(
                value + error + top * 0.015,
                y,
                f"{value:.{decimals}f}",
                va="center",
                fontsize=9,
                color=INK,
            )

        ax.set_yticks(ys)

        ax.set_yticklabels(
            [
                method[2]
                for method in main
            ],
            fontsize=9,
            color=INK,
        )

        ax.invert_yaxis()

        ax.set_xlim(
            0,
            top * 1.13 if top > 0 else 1,
        )

        ax.set_title(
            panel_title,
            loc="left",
            fontsize=11,
            fontweight="bold",
            color=INK,
        )

        for spine in (
            "top",
            "right",
            "left",
        ):
            ax.spines[spine].set_visible(False)

        ax.spines["bottom"].set_color(GRID)

        ax.tick_params(
            axis="x",
            colors=INK2,
            labelsize=9,
        )

        ax.tick_params(
            axis="y",
            length=0,
        )

        ax.grid(
            axis="x",
            color=GRID,
            linewidth=0.8,
        )

        ax.set_axisbelow(True)

    fig.suptitle(
        title,
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
        color=INK,
    )

    fig.text(
        0.01,
        0.005,
        (
            f"{N} failures per trial, "
            f"{TRIALS} independent trials. "
            "Mean with 95% confidence interval. "
            "Gray = simple baselines, "
            "orange = base paper, "
            "blue = this project."
        ),
        fontsize=9,
        color=INK2,
    )

    fig.tight_layout(
        rect=(0, 0.025, 1, 0.96)
    )

    fig.savefig(
        path,
        dpi=160,
        facecolor=SURFACE,
    )

    plt.close(fig)


# ---------------------------------------------------------------------
# Decision-impact audit
# ---------------------------------------------------------------------

def summarize_decision_trace(decision_traces):

    if decision_traces.empty:

        return pd.DataFrame(
            columns=[
                "method",
                "priority",
                "total_decisions",
                "lookahead_decisions",
                "lookahead_overrides",
                "lookahead_rate_pct",
                "override_rate_pct",
                "override_when_lookahead_pct",
            ]
        )

    trace = decision_traces.copy()

    trace["lookahead_used"] = (
        trace["lookahead_used"].astype(bool)
    )

    trace["lookahead_overrode_topsis"] = (
        trace["lookahead_overrode_topsis"].astype(bool)
    )

    summary = (
        trace
        .groupby(
            ["method", "priority"],
            as_index=False
        )
        .agg(
            total_decisions=(
                "selected_zone",
                "count"
            ),
            lookahead_decisions=(
                "lookahead_used",
                "sum"
            ),
            lookahead_overrides=(
                "lookahead_overrode_topsis",
                "sum"
            ),
        )
    )

    summary["lookahead_rate_pct"] = (
        100
        * summary["lookahead_decisions"]
        / summary["total_decisions"]
    )

    summary["override_rate_pct"] = (
        100
        * summary["lookahead_overrides"]
        / summary["total_decisions"]
    )

    summary["override_when_lookahead_pct"] = (
        100
        * summary["lookahead_overrides"]
        / summary["lookahead_decisions"].replace(0, np.nan)
    )

    return summary


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

print("\nEXPERIMENT CONFIGURATION")

print(f"Failures per trial: {N}")
print(f"Independent trials: {TRIALS}")
print(f"Random seeds: {SEEDS}")

print("\nASSUMPTIONS (state these in the report):")

print(
    "  * Base-paper score weights are not specified; "
    f"tested: {list(WEIGHT_SETS)}"
)

print(
    f"  * Latency is simulated (base ms {BASE_LATENCY_MS}); "
    "the main baseline has NO latency cutoff."
)

print(
    f"  * A sensitivity version uses a {CUTOFF_MS} ms cutoff."
)

print(
    f"  * Drift condition actual delays: {TRUTH_DRIFT}"
)

print(
    f"  * Measurement noise: approximately "
    f"{int(NOISE_SIGMA * 100)}% variation"
)

print(
    f"  * Stress condition adds "
    f"{STRESS_EXTRA * 100:.1f}% tenant utilization."
)


# Generate a separate failure set for every independent trial.
SCENARIOS_BY_TRIAL = {}

STRESS_BY_TRIAL = {}

for trial, seed in enumerate(SEEDS, start=1):

    print("\n" + "-" * 80)

    print(
        f"GENERATING TRIAL {trial}/{TRIALS} "
        f"(seed={seed})"
    )

    scenarios = build_scenarios(seed)

    SCENARIOS_BY_TRIAL[trial] = scenarios

    STRESS_BY_TRIAL[trial] = [
        stressed(scenario, STRESS_EXTRA)
        for scenario in scenarios
    ]


TITLES = {
    "no_drift": "configured delays correct",

    "drift": "zone delays drift from configured",

    "stress": (
        f"delays drift and every zone is "
        f"{int(STRESS_EXTRA * 100)}% busier"
    ),
}


CONDITIONS = [
    (
        "no_drift",
        TRUTH_NONE,
        "configured delays are correct",
        SCENARIOS_BY_TRIAL,
    ),

    (
        "drift",
        TRUTH_DRIFT,
        "some zones really recover slower or faster",
        SCENARIOS_BY_TRIAL,
    ),

    (
        "stress",
        TRUTH_DRIFT,
        (
            "drift plus "
            f"{int(STRESS_EXTRA * 100)}% extra tenant load"
        ),
        STRESS_BY_TRIAL,
    ),
]


# ---------------------------------------------------------------------
# Run each condition across every independent trial
# ---------------------------------------------------------------------

for cond, truth, description, scenario_sets in CONDITIONS:

    print("\n" + "=" * 100)

    print(
        f"CONDITION {cond}: {description}"
    )

    trial_results = []
    trial_traces = []

    for trial, seed in enumerate(SEEDS, start=1):

        print(
            f"\nRunning {cond}: "
            f"trial {trial}/{TRIALS}, "
            f"seed={seed}"
        )

        raw_trial, trace_trial = run_condition(
            cond,
            truth,
            scenario_sets[trial],
            trial,
            seed,
        )

        trial_results.append(raw_trial)
        trial_traces.append(trace_trial)

    raw = pd.concat(
        trial_results,
        ignore_index=True,
    )

    decision_traces = pd.concat(
        trial_traces,
        ignore_index=True,
    )

    summary = summarize(raw)

    table = pretty(summary)

    pair = paired(raw)

    # -------------------------------------------------------------
    # Decision-impact audit
    # -------------------------------------------------------------

    decision_summary = summarize_decision_trace(
        decision_traces
    )

    # Save detailed candidate-level decisions.
    decision_traces.to_csv(
        f"results/decision_trace_{cond}.csv",
        index=False,
    )

    # Save aggregate decision-impact statistics.
    decision_summary.to_csv(
        f"results/decision_trace_summary_{cond}.csv",
        index=False,
    )

    print("\nDECISION-IMPACT AUDIT")

    if decision_summary.empty:
        print("No TOPSIS decision records were collected.")
    else:
        print(
            decision_summary.to_string(index=False)
        )

    # Save raw results and summaries.
    raw.to_csv(
        f"results/experiment_raw_{cond}.csv",
        index=False,
    )

    summary.to_csv(
        f"results/experiment_summary_{cond}.csv",
        index=False,
    )

    table.to_csv(
        f"results/experiment_table_{cond}.csv",
        index=False,
    )

    pair.to_csv(
        f"results/experiment_paired_{cond}.csv",
        index=False,
    )

    # Generate chart.
    chart(
        summary,
        f"results/experiment_{cond}.png",
        f"Recovery methods compared ({TITLES[cond]})",
    )

    pd.set_option(
        "display.width",
        250,
    )

    print("\nSUMMARY: MEAN ACROSS INDEPENDENT TRIALS")

    print(
        table.to_string(index=False)
    )

    print(
        "\nPAIRED DIFFERENCE: "
        f"'{LABEL[REFERENCE]}' minus each other method"
    )

    print(
        "Availability: positive = better. "
        "Time, cost and carbon: negative = better."
    )

    print(
        "Confidence intervals are calculated "
        "across independent trial-level differences."
    )

    paired_table = pd.DataFrame({

        "vs": [
            LABEL[key]
            for key in pair["vs"]
        ],

        "weighted availability %": [
            fmt(mean, ci)
            for mean, ci in zip(
                pair.w_avail,
                pair.w_avail_ci,
            )
        ],

        "critical recovery s": [
            fmt(mean, ci)
            for mean, ci in zip(
                pair.crit_time,
                pair.crit_time_ci,
            )
        ],

        "weighted recovery s": [
            fmt(mean, ci)
            for mean, ci in zip(
                pair.w_time,
                pair.w_time_ci,
            )
        ],

        "cost / unit": [
            fmt(mean, ci, 4)
            for mean, ci in zip(
                pair.cost_per_unit,
                pair.cost_per_unit_ci,
            )
        ],

        "carbon g / unit": [
            fmt(mean, ci, 2)
            for mean, ci in zip(
                pair.carbon_per_unit,
                pair.carbon_per_unit_ci,
            )
        ],
    })

    print(
        paired_table.to_string(index=False)
    )

    # Failure-type breakdown.
    tables, counts = by_kind(raw)

    print(
        "\nBY FAILURE TYPE "
        f"(reference-method failure counts: {counts.to_dict()})"
    )

    long_tables = []

    for table_name, failure_table in tables:

        print(f"\n{table_name}")

        print(
            failure_table.to_string(
                na_rep="-"
            )
        )

        long_tables.append(
            failure_table
            .assign(metric=table_name)
            .reset_index()
            .rename(
                columns={
                    "index": "method"
                }
            )
        )

    pd.concat(
        long_tables,
        ignore_index=True,
    ).to_csv(
        f"results/experiment_by_kind_{cond}.csv",
        index=False,
    )


# ---------------------------------------------------------------------
# Completion
# ---------------------------------------------------------------------

print("\n" + "=" * 100)

print("EXPERIMENT COMPLETED.")

print(
    f"Completed {TRIALS} independent trials "
    f"with {N} failures per trial."
)

print("\nSaved in results/:")

print(
    "  experiment_table_*.csv"
)

print(
    "  experiment_paired_*.csv"
)

print(
    "  experiment_summary_*.csv"
)

print(
    "  experiment_raw_*.csv"
)

print(
    "  experiment_by_kind_*.csv"
)

print(
    "  decision_trace_*.csv"
)

print(
    "  decision_trace_summary_*.csv"
)

print(
    "  experiment_no_drift.png"
)

print(
    "  experiment_drift.png"
)

print(
    "  experiment_stress.png"
)

print(
    "  experiment_feedback_*_trial_*.db"
)
