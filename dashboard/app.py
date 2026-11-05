"""Disaster-recovery dashboard (Streamlit).

Run from the dr-project folder:
    streamlit run dashboard/app.py

Tabs
  1. Failure simulation   pick a failure, see the zones, the ranked zones and the recovery plan
  2. Recovery history     every recovery run in this session, and what the feedback has learned
  3. Experiment results   the saved results of simulator/run_experiment.py
  4. LSTM and data        the saved LSTM results and charts
  5. Design and sources   design.md and sources.md
"""
import os
import sys
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from decision.topsis import rank_zones                                   # noqa: E402
from simulator.recovery import Cloud, recover, PRIORITY_WEIGHT           # noqa: E402

st.set_page_config(page_title="Disaster recovery dashboard", layout="wide")

TRUTH_DRIFT = {"A": 55, "B": 30, "C": 45, "D": 30}   # same assumption as simulator/run_experiment.py
FAILURES = {
    "Zone down": "zone_down",
    "Two zones down": "two_zones_down",
    "CPU overload in a zone": "cpu_overload",
    "Network loss on one service": "network_loss",
}
METHODS = {
    "TOPSIS + look-ahead (full method)": "topsis",
    "TOPSIS only": "topsis_basic",
    "Cheapest zone": "cheapest",
    "Random zone": "random",
    "Base paper Eq. 1": "paper",
}
METHOD_LABEL = {v: k for k, v in METHODS.items()}
RESULT_LABEL = {
    "random": "Random zone", "cheapest": "Cheapest zone", "paper": "Base paper Eq.1 (equal weights)",
    "topsis_basic": "TOPSIS", "topsis_la": "TOPSIS + look-ahead",
    "topsis_fb": "TOPSIS + look-ahead + feedback", "paper_cpu": "Base paper Eq.1 (CPU-heavy weights)",
    "paper_lat": "Base paper Eq.1 (latency-heavy weights)", "paper_cut": "Base paper Eq.1 (with latency cutoff)",
}
CONDITIONS = {
    "no_drift": "Configured delays are correct",
    "drift": "Delays differ from the configured ones",
    "stress": "Delays differ and every zone is busier",
}


def path(*parts):
    return os.path.join(ROOT, *parts)


def exists(*parts):
    return os.path.exists(path(*parts))


# ---- loading ----------------------------------------------------------------
@st.cache_data
def load_configs():
    return pd.read_csv(path("configs", "zones.csv")), pd.read_csv(path("configs", "services.csv"))


@st.cache_data
def load_trace():
    f = path("data", "zone_data.csv")
    if not os.path.exists(f):
        return None
    df = pd.read_csv(f, parse_dates=["timestamp"])
    return {z: g.sort_values("timestamp").reset_index(drop=True) for z, g in df.groupby("zone")}


@st.cache_resource(show_spinner="Loading the trained LSTM models (first time only)...")
def load_predictor():
    try:
        from simulator.predictor import Predictor     # imports TensorFlow, so only when needed
        p = Predictor()
        return p, (getattr(p, "load_note", "") or "")
    except Exception as exc:                          # TensorFlow missing or blocked
        return None, f"The LSTM could not be loaded ({type(exc).__name__}). The last reading is used instead."


def get_store():
    """A feedback store that lives for this browser session only."""
    if "store" not in st.session_state:
        try:
            from feedback.store import FeedbackStore
            db = os.path.join(tempfile.mkdtemp(prefix="dr_dashboard_"), "feedback.db")
            st.session_state["store"] = FeedbackStore(db)
        except Exception:
            st.session_state["store"] = None
    return st.session_state["store"]


# ---- simulation helpers ---------------------------------------------------------
def get_loads(zone_ids, source, t, trace, manual, predictor):
    """Return ({zone: cpu %}, {zone: memory %}, note)."""
    cpu, mem, note = {}, {}, ""
    for z in zone_ids:
        if source == "Manual":
            cpu[z] = mem[z] = float(manual[z])
            continue
        row = trace[z].loc[t]
        c, m = float(row["cpu"]), float(row["memory"])
        if source == "LSTM prediction" and predictor is not None:
            try:
                c, m = [float(v) for v in predictor.predict(z, t)[:2]]
            except Exception as exc:
                note = f"The LSTM prediction failed ({type(exc).__name__}); the last reading is used."
        cpu[z], mem[z] = c, m
    return cpu, mem, note


def make_cloud(zones, services, cpu, mem, truth):
    util = {z: max(cpu[z], mem[z]) / 100 for z in cpu}
    pred = {"cpu": {z: cpu[z] / 100 for z in cpu}, "mem": {z: mem[z] / 100 for z in mem}}
    return Cloud(zones, services, util, truth=truth, noise=None, pred=pred)


def inject(cloud, kind, target):
    if kind == "zone_down":
        return cloud.zone_down(target[0])
    if kind == "two_zones_down":
        return cloud.zone_down(target[0]) + cloud.zone_down(target[1])
    if kind == "cpu_overload":
        return cloud.cpu_overload(target[0])
    return cloud.network_loss(target[0])


def zone_table(cloud, cpu, mem):
    rows = []
    for z, r in cloud.zones.iterrows():
        here = [s for s, zz in cloud.placement.items() if zz == z]
        rows.append({
            "zone": f"{z} ({r['region']})",
            "state": "DOWN" if z in cloud.down else "up",
            "CPU %": round(cpu[z], 1),
            "memory %": round(mem[z], 1),
            "busy % (tenants + load)": round(100 * cloud.busy[z], 1),
            "free units": round(cloud.free(z), 1),
            "services running": ", ".join(here) if here else "-",
        })
    return pd.DataFrame(rows)


def recovery_order(cloud, failed):
    return sorted(failed, key=lambda s: (-PRIORITY_WEIGHT[cloud.services.loc[s, "priority"]],
                                         -cloud.services.loc[s, "load"]))


def ranking(cloud, service):
    """TOPSIS ranking of the zones that could take `service` right now."""
    load = cloud.services.loc[service, "load"]
    priority = cloud.services.loc[service, "priority"]
    cands = [z for z in cloud.zones.index
             if z not in cloud.down and z not in cloud.excluded.get(service, set()) and cloud.free(z) >= load]
    if not cands:
        return None, priority
    table = cloud.zones.loc[cands].reset_index()
    table["capacity"] = [cloud.free(z) for z in cands]     # "availability" = free capacity now
    r = rank_zones(table, priority)
    detail = table.set_index("zone").loc[r["zone"]]
    out = pd.DataFrame({
        "rank": range(1, len(r) + 1),
        "zone": [f"{z} ({reg})" for z, reg in zip(r["zone"], r["region"])],
        "TOPSIS score": r["score"].values,
        "free units": detail["capacity"].round(1).values,
        "believed delay s": detail["recovery_delay"].round(1).values,
        "cost USD/h": detail["cost"].values,
        "carbon g/kWh": detail["carbon"].values,
    })
    return out, priority


def run_metrics(res):
    w = res["priority"].map(PRIORITY_WEIGHT)
    ok = res["status"] == "RECOVERED"
    crit = res["priority"] == "Critical"
    return {
        "recovered": f"{int(ok.sum())}/{len(res)}",
        "critical recovered": f"{int((ok & crit).sum())}/{int(crit.sum())}",
        "weighted availability %": round(100 * w[ok].sum() / w.sum(), 1) if w.sum() else 100.0,
        "avg recovery s": round(float(res.loc[ok, "recovery_sec"].mean()), 1) if ok.any() else None,
        "est. cost USD (1 h)": round(float(res["cost"].sum()), 3),
        "est. carbon g (1 h)": round(float(res["carbon_g"].sum()), 1),
    }


def plan_table(res, zones):
    region = dict(zip(zones["zone"], zones["region"]))
    out = pd.DataFrame({
        "service": res["service"],
        "priority": res["priority"],
        "load": res["load"],
        "moved to": [f"{z} ({region.get(z, '')})" if z in region else "-" for z in res["to_zone"]],
        "status": ["✅ RECOVERED" if s == "RECOVERED" else "⏳ " + s for s in res["status"]],
        "recovery s": res["recovery_sec"].round(1),
        "cost USD (1 h)": res["cost"].round(3),
        "carbon g (1 h)": res["carbon_g"].round(1),
    })
    return out


def read_csv(*parts):
    return pd.read_csv(path(*parts)) if exists(*parts) else None


def show_image(*parts, caption=None):
    if exists(*parts):
        st.image(path(*parts), caption=caption)
    else:
        st.caption(f"(missing file: {'/'.join(parts)})")


# ---- page ------------------------------------------------------------------------
try:
    zones, services = load_configs()
except Exception as exc:
    st.error(f"Could not read configs/zones.csv and configs/services.csv ({exc}). "
             "Run this from the dr-project folder: streamlit run dashboard/app.py")
    st.stop()

zone_ids = list(zones["zone"])
service_names = list(services["service"])
trace = load_trace()
st.session_state.setdefault("history", [])
st.session_state.setdefault("run_id", 0)

st.title("Adaptive disaster recovery: business-aware, cost- and carbon-optimized")
st.caption("Everything here is simulated: failures, recovery delays and capacity. Zone load comes from the real "
           "Bitbrains trace, and cost and carbon come from published values. See the Design and sources tab.")

# ---- sidebar: what to simulate ----------------------------------------------------
with st.sidebar:
    st.header("1. Failure")
    kind_label = st.selectbox("Failure type", list(FAILURES))
    kind = FAILURES[kind_label]
    if kind == "two_zones_down":
        target = st.multiselect("Zones that fail (pick 2)", zone_ids, default=zone_ids[:2])
    elif kind == "network_loss":
        target = [st.selectbox("Service that loses its network", service_names)]
    else:
        target = [st.selectbox("Zone", zone_ids)]
    valid_target = len(target) == (2 if kind == "two_zones_down" else 1)
    if not valid_target:
        st.warning("Pick exactly 2 zones.")

    st.header("2. Zone load")
    sources = ["Manual"] if trace is None else ["LSTM prediction", "Last reading", "Manual"]
    source = st.radio("Zone load comes from", sources)
    t, manual, predictor, predictor_note = None, {}, None, ""
    if trace is None:
        st.caption("data/zone_data.csv not found, so the load is set by hand.")
    if source != "Manual":
        n_rows = len(trace[zone_ids[0]])
        split = int(n_rows * 0.7)
        t = st.slider("Moment in the trace (data row)", 12, n_rows - 1, min(max(6289, 12), n_rows - 1))
        st.caption(f"Time: {trace[zone_ids[0]].loc[t, 'timestamp']}. Rows from {split} on were never used "
                   "to train the LSTM.")
    if source == "LSTM prediction":
        predictor, predictor_note = load_predictor()
        if predictor_note:
            st.caption(predictor_note)
    if source == "Manual":
        for z in zone_ids:
            manual[z] = st.slider(f"Zone {z}: load from other tenants (%)", 0, 80, 20)

    st.header("3. Recovery")
    method_label = st.selectbox("Method", list(METHODS))
    strategy = METHODS[method_label]
    drift = st.checkbox("Real delays differ from the configured ones (drift)", value=False,
                        help=f"Uses real delays {TRUTH_DRIFT} instead of the values in configs/zones.csv.")
    use_fb = st.checkbox("Learn from past recoveries (feedback)", value=False,
                         help="The delay estimate of each zone becomes the average of its recent measured delays.")
    run = st.button("Run recovery", type="primary", disabled=not valid_target)

cpu, mem, load_note = get_loads(zone_ids, source, t, trace, manual, predictor)
truth = TRUTH_DRIFT if drift else None
store = get_store() if use_fb else None
if use_fb and store is None:
    st.sidebar.warning("The feedback store could not be loaded, so feedback is off.")
zones_belief = store.corrected_zones(zones) if store is not None else zones

tab_sim, tab_hist, tab_exp, tab_lstm, tab_docs = st.tabs(
    ["Failure simulation", "Recovery history", "Experiment results", "LSTM and data", "Design and sources"])

# ---- tab 1 --------------------------------------------------------------------------
with tab_sim:
    if load_note:
        st.warning(load_note)
    before = make_cloud(zones_belief, services, cpu, mem, truth)
    st.subheader("Zones before the failure")
    st.dataframe(zone_table(before, cpu, mem), hide_index=True)

    if run and valid_target:
        cloud = make_cloud(zones_belief, services, cpu, mem, truth)
        failed = inject(cloud, kind, target)
        if not failed:
            st.session_state["last"] = {"empty": True, "label": f"{kind_label}: {', '.join(target)}"}
        else:
            order = recovery_order(cloud, failed)
            rank_df, rank_priority = ranking(cloud, order[0])
            free_before = [round(before.free(z), 1) for z in zone_ids]
            res = recover(cloud, failed, strategy=strategy, rng=np.random.default_rng(0))
            state_after = zone_table(cloud, cpu, mem)
            compare = []
            for lab, strat in METHODS.items():
                c2 = make_cloud(zones_belief, services, cpu, mem, truth)
                f2 = inject(c2, kind, target)
                r2 = recover(c2, f2, strategy=strat, rng=np.random.default_rng(0))
                compare.append({"method": lab, **run_metrics(r2)})
            if store is not None:
                st.session_state["run_id"] += 1
                store.log_run(st.session_state["run_id"], f"{kind}:{'+'.join(target)}", strategy, res)
            m = run_metrics(res)
            st.session_state["history"].append({
                "time": datetime.now().strftime("%H:%M:%S"), "failure": f"{kind_label}: {', '.join(target)}",
                "method": method_label, "feedback": "on" if store is not None else "off", **m})
            st.session_state["last"] = {
                "empty": False, "label": f"{kind_label}: {', '.join(target)}", "method": method_label,
                "order": order, "rank_df": rank_df, "rank_service": order[0], "rank_priority": rank_priority,
                "free_before": free_before, "state_after": state_after, "res": res, "metrics": m,
                "compare": pd.DataFrame(compare)}

    last = st.session_state.get("last")
    if not last:
        st.info("Choose a failure on the left and press **Run recovery**.")
    elif last["empty"]:
        st.info(f"{last['label']}: no service went down at this moment. Try another moment, zone or failure.")
    else:
        st.divider()
        st.subheader(f"Result: {last['label']}  |  method: {last['method']}")
        mt = last["metrics"]
        c = st.columns(6)
        c[0].metric("Services recovered", mt["recovered"])
        c[1].metric("Critical recovered", mt["critical recovered"])
        c[2].metric("Weighted availability", f"{mt['weighted availability %']}%")
        c[3].metric("Avg recovery time", "-" if mt["avg recovery s"] is None else f"{mt['avg recovery s']} s")
        c[4].metric("Est. cost (1 h)", f"{mt['est. cost USD (1 h)']} USD")
        c[5].metric("Est. carbon (1 h)", f"{mt['est. carbon g (1 h)']} g")

        st.markdown("**Recovery plan** (Critical first, then Important, then Normal)")
        st.dataframe(plan_table(last["res"], zones), hide_index=True)

        st.markdown(f"**Why this zone?** TOPSIS ranking for the first service, "
                    f"**{last['rank_service']}** ({last['rank_priority']}), before any recovery started")
        if last["rank_df"] is None:
            st.caption("No zone had room for this service.")
        else:
            st.dataframe(last["rank_df"], hide_index=True)
            st.caption("Availability = free units in the zone right now. Critical services weigh recovery "
                       "time most; Normal services weigh cost and carbon most.")

        left, right = st.columns(2)
        with left:
            st.markdown("**Free capacity per zone**")
            chart = pd.DataFrame({"before failure": last["free_before"],
                                  "after recovery": last["state_after"]["free units"].values}, index=zone_ids)
            st.bar_chart(chart)
        with right:
            st.markdown("**Zones after recovery**")
            st.dataframe(last["state_after"], hide_index=True)

        st.markdown("**The same failure, every method**")
        st.dataframe(last["compare"], hide_index=True)
        st.caption("Costs and carbon are estimates for 1 hour. 'Random zone' uses a fixed seed here.")

# ---- tab 2 --------------------------------------------------------------------------
with tab_hist:
    st.subheader("Recovery runs in this session")
    if st.session_state["history"]:
        st.dataframe(pd.DataFrame(st.session_state["history"]), hide_index=True)
        if st.button("Clear history and feedback"):
            st.session_state["history"] = []
            st.session_state.pop("store", None)
            st.session_state.pop("last", None)
            st.rerun()
    else:
        st.info("No runs yet.")
    st.subheader("What the feedback has learned")
    store_now = st.session_state.get("store")
    if store_now is None:
        st.caption("Tick 'Learn from past recoveries' in the sidebar and run a few recoveries.")
    else:
        learned = store_now.corrected_zones(zones)[["zone", "region", "recovery_delay"]]
        out = zones[["zone", "region", "recovery_delay"]].rename(columns={"recovery_delay": "configured delay s"})
        out["believed delay s"] = learned["recovery_delay"].round(1).values
        if drift:
            out["real delay s (drift)"] = [TRUTH_DRIFT.get(z) for z in out["zone"]]
        st.dataframe(out, hide_index=True)
        st.caption("With drift switched on, the believed delay moves toward the real delay as recoveries are run. "
                   "Zones that are never chosen are never measured, so their estimate does not change.")

# ---- tab 3 --------------------------------------------------------------------------
with tab_exp:
    st.subheader("Saved experiment results")
    cond = st.selectbox("Condition", list(CONDITIONS), format_func=lambda k: f"{k}: {CONDITIONS[k]}")
    table = read_csv("results", f"experiment_table_{cond}.csv")
    if table is None:
        st.info("No results yet. Run: python simulator/run_experiment.py")
    else:
        st.caption("Mean with a 95% confidence interval over the same random failures for every method. "
                   "Gray = simple baselines, orange = base paper, blue = this project.")
        st.dataframe(table, hide_index=True)
        show_image("results", f"experiment_{cond}.png")
        paired = read_csv("results", f"experiment_paired_{cond}.csv")
        if paired is not None:
            with st.expander("Paired difference: full method minus each other method"):
                pp = pd.DataFrame({"vs": [RESULT_LABEL.get(k, k) for k in paired["vs"]]})
                for col, name, d in [("w_avail", "weighted availability % (+ is better)", 1),
                                     ("crit_time", "critical recovery s (- is better)", 1),
                                     ("w_time", "weighted recovery s (- is better)", 1),
                                     ("cost_per_unit", "cost per unit (- is better)", 4),
                                     ("carbon_per_unit", "carbon g per unit (- is better)", 2)]:
                    pp[name] = [f"{m:.{d}f} ± {ci:.{d}f}" for m, ci in zip(paired[col], paired[col + "_ci"])]
                st.dataframe(pp, hide_index=True)
                st.caption("A difference counts only if its interval does not include zero.")
        kinds = read_csv("results", f"experiment_by_kind_{cond}.csv")
        if kinds is not None:
            with st.expander("By failure type"):
                for metric, g in kinds.groupby("metric", sort=False):
                    st.markdown(f"**{metric}**")
                    g = g.drop(columns="metric").copy()
                    g["method"] = [RESULT_LABEL.get(k, k) for k in g["method"]]
                    st.dataframe(g, hide_index=True)
    fb = read_csv("results", "feedback_summary.csv")
    if exists("results", "feedback_learning.png"):
        with st.expander("Feedback demo: recovery gets better as outcomes are recorded"):
            show_image("results", "feedback_learning.png")
            if fb is not None:
                st.dataframe(fb, hide_index=True)

# ---- tab 4 --------------------------------------------------------------------------
with tab_lstm:
    st.subheader("LSTM prediction of CPU and memory")
    m3 = read_csv("results", "lstm_metrics_horizon3.csv")
    if m3 is None:
        m3 = read_csv("results", "lstm_metrics.csv")
    if m3 is None:
        st.info("No LSTM results yet. Run: python models/train_lstm.py")
    else:
        st.markdown("**Errors per zone** (lower is better; 'naive' repeats the last reading)")
        st.dataframe(m3, hide_index=True)
    m1 = read_csv("results", "lstm_metrics_horizon1.csv")
    if m1 is not None:
        with st.expander("Supporting result: 5 minutes ahead"):
            st.dataframe(m1, hide_index=True)
    cols = st.columns(2)
    for i, z in enumerate(zone_ids):
        with cols[i % 2]:
            show_image("results", f"lstm_zone_{z}.png")
    if exists("data", "zone_check.png"):
        with st.expander("The workload data, first 7 days per zone"):
            show_image("data", "zone_check.png")

# ---- tab 5 --------------------------------------------------------------------------
with tab_docs:
    for name in ["design.md", "sources.md"]:
        if exists(name):
            with st.expander(name, expanded=(name == "design.md")):
                st.markdown(open(path(name), encoding="utf-8").read())
        else:
            st.caption(f"({name} not found in the project folder)")