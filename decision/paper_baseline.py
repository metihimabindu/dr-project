"""Baseline taken from the base paper (Alomari & Wang, IEEE Trans. Cloud Computing, 2026, Eq. 1).

    S_i = w_c * (1 / C_i) + w_m * (1 / M_i) + w_l * (1 / L_i)     if L_i <= L_threshold
    S_i = 0                                                       otherwise

C_i, M_i = predicted CPU and memory utilization of zone i, L_i = predicted latency.
The scheduler picks the zone with the highest S_i; if every zone scores 0, no zone is
suitable and the service waits (as the paper says).

WHAT THE PAPER DOES NOT GIVE US (so these are OUR assumptions, state them in the report):
  * the weights w_c, w_m, w_l           -> three settings are tested (WEIGHT_SETS)
  * the latency threshold L_threshold   -> not given. Our latencies are SIMULATED, so a cutoff
                                           on them would be arbitrary: in our tests a 200 ms
                                           cutoff left 65% of services waiting, which says more
                                           about our made-up latencies than about the paper.
                                           The MAIN baseline therefore has NO cutoff (the most
                                           favourable reading for the paper). A version with a
                                           CUTOFF_MS cutoff is reported as a sensitivity row.
  * the latency of each zone            -> our data has no latency, so it is simulated:
                                           L_i = BASE_LATENCY_MS[zone] / (1 - utilization)
  * the units of L_i                    -> L_i is divided by L_REF_MS so that 1/C, 1/M and 1/L
                                           sit on a similar scale (scale only, not a cutoff)
"""

BASE_LATENCY_MS = {"A": 25, "B": 40, "C": 60, "D": 80}   # assumed, no public source
L_REF_MS = 200            # scale only
CUTOFF_MS = 500           # used only by the "cutoff" sensitivity variant
WEIGHT_SETS = {
    "equal": (1 / 3, 1 / 3, 1 / 3),
    "cpu_heavy": (0.6, 0.2, 0.2),
    "latency_heavy": (0.2, 0.2, 0.6),
}
_FLOOR, _CEIL = 0.02, 0.98       # keep 1/x finite


def predicted_latency_ms(base_ms, utilization):
    u = min(max(utilization, 0.0), 0.95)
    return base_ms / (1.0 - u)


def zone_score(cpu_util, mem_util, base_latency_ms, weights, cutoff_ms=None):
    """Equation 1 for one zone. cpu_util and mem_util are fractions 0..1.
    cutoff_ms=None means the latency threshold is switched off."""
    c = min(max(cpu_util, _FLOOR), _CEIL)
    m = min(max(mem_util, _FLOOR), _CEIL)
    latency = predicted_latency_ms(base_latency_ms, max(c, m))
    if cutoff_ms is not None and latency > cutoff_ms:
        return 0.0
    w_c, w_m, w_l = weights
    return w_c / c + w_m / m + w_l / (latency / L_REF_MS)