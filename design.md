# Design Decisions: Adaptive Business-Aware, Cost- and Carbon-Optimized Disaster Recovery

**Project:** An Adaptive Framework for Business-Aware, Cost and Carbon-Optimized Disaster Recovery in Cloud Environments
**Team (Batch A7-8):** Meti Hima Bindu, Kummari Krishna Sri, Harish Krishna Kundeti, Raja Saketh Ram
**Guide:** Mr. P. Venkateswarlu Reddy
**Version:** 2, updated 5 November 2026 (version 1, 4 November 2026, used placeholder cost and carbon values)

This file is the shared agreement for the whole team. Every program reads its numbers from `configs/zones.csv` and `configs/services.csv`, and this file explains them. If a value changes in the code, change it here too.

---

## 1. What changed since version 1

| Item | Version 1 | Version 2 |
|---|---|---|
| Zone cost | Made-up values in rupees | Real AWS On-Demand prices, USD per hour |
| Zone carbon | Made-up values | Real grid values (Ember 2025, EPA eGRID 2023) |
| TOPSIS "availability" | Not defined | Free capacity in the zone right now |
| Placement | One service at a time | Look-ahead, so one choice cannot block a later service |
| Feedback | Planned | Built: measured recovery delays correct the configured ones |
| Evaluation | Planned | Built: 200 random failures, 3 conditions, baselines, 95% confidence intervals |

All sources and read dates are in `sources.md`.

---

## 2. Cloud zones

| Zone | Region | AWS region | Cost (USD/hour) | Carbon (gCO2e/kWh) | Capacity (units) | Recovery delay (s) |
|---|---|---|---:|---:|---:|---:|
| A | Mumbai | ap-south-1 | 0.101 | 671 | 100 | 20 |
| B | Singapore | ap-southeast-1 | 0.120 | 497 | 100 | 30 |
| C | Frankfurt | eu-central-1 | 0.115 | 334 | 80 | 40 |
| D | Oregon | us-west-2 | 0.096 | 288.2 | 120 | 50 |

**Real values (cited in `sources.md`)**

- **Cost:** AWS Pricing Calculator, instance m5.large, Linux, On-Demand, read on 05-11-2026. The same instance is used in all four regions so the prices are comparable.
- **Carbon, Mumbai, Singapore, Frankfurt:** Ember Electricity Data Explorer, carbon emissions intensity for 2025 (India 671, Singapore 497, Germany 334).
- **Carbon, Oregon:** EPA eGRID2023 Summary Tables (revision 2), subregion NWPP (WECC Northwest), total output emission rate 635.267 lb/MWh CO2e, converted with 1 lb/MWh = 0.4536 g/kWh. This is 2023 data, while the other three are 2025 data. The Oregon state-only figure would be 165.5 g/kWh. The NWPP value was chosen because it describes the grid region Oregon belongs to.

**Simulation settings (not real measurements)**

- **Capacity** is in abstract load units.
- **Recovery delay** is the delay the framework *believes* at the start. It is not a provider guarantee. In the "drift" and "stress" conditions the real delays differ (section 7).

**Zone-to-region mapping.** The four zones are labelled with real regions for cost and carbon. The workload data comes from one data center (section 5), so the mapping of VMs to zones is a modelling choice and must be stated in the paper.

---

## 3. Services and business priorities

| Service | Priority | Weight | Load (units) | Home zone |
|---|---|---:|---:|---|
| Payment | Critical | 3 | 20 | A |
| Login | Critical | 3 | 10 | A |
| Orders | Critical | 3 | 25 | B |
| Search | Important | 2 | 30 | B |
| Notifications | Important | 2 | 15 | C |
| Reports | Normal | 1 | 20 | C |
| Analytics | Normal | 1 | 25 | D |
| Recommendations | Normal | 1 | 20 | D |
| **Total** | | | **165** | |

**Order of recovery (tie-break rule).** Failed services are recovered in this order: Critical, then Important, then Normal. Inside the same priority, the larger load goes first. A service that fits nowhere waits and is reported as `WAITING`. Nothing is ever silently assumed to be recovered.

---

## 4. Failure types

Failures are simulated, because no public dataset records "this zone failed and this recovery was chosen".

| Failure | What happens | Share of failures |
|---|---|---:|
| `zone_down` | Every service in one random zone fails | 40% |
| `two_zones_down` | Two random zones fail together | 20% |
| `cpu_overload` | One zone's load rises by 35% of its capacity; the least important services are evicted until it fits | 20% |
| `network_loss` | One random service becomes unreachable | 20% |

Each failure happens at a random moment of the real workload trace. The LSTM predicts each zone's CPU and memory at that moment, and the larger of the two sets how busy the zone is.

---

## 5. Workload data and the LSTM

**Data.** Bitbrains GWA-T-12 fastStorage trace (Grid Workload Archive), month folder 2013-8, 1,250 VM files. Each zone is the average of 10 consecutive VMs (the first 40 files in sorted order), resampled to 5 minutes. Each zone has 8,640 rows (30 days). Columns: `timestamp, zone, cpu, memory` (percent). Raw data is not stored in GitHub (`data/raw/` is in `.gitignore`).

**Model.** One LSTM per zone: input of the last 12 readings (1 hour), two LSTM layers of 50 units, one dense layer. The inputs are CPU, memory, and sine/cosine of hour and day of week. The scaler is fitted on the training part only. The split is 70% train and 30% test in time order, with no shuffling.

**Main setting.** Predict 3 readings ahead (15 minutes). The 5-minute result is kept as a supporting comparison. Metrics are RMSE and MAE, compared with a "repeat the last reading" baseline. At 15 minutes the LSTM beat the baseline in 7 of 8 comparisons (CPU and memory in four zones); CPU in Zone C was worse and must be reported as it is. Exact numbers: `results/lstm_metrics_horizon3.csv`.

**What is real and what is simulated**

| Part | Real or simulated |
|---|---|
| CPU and memory traces | Real (Bitbrains) |
| Cost and carbon per zone | Real published values |
| Failures, recovery delays, capacity, latency | Simulated |

---

## 6. How a recovery decision is made

1. Failed services are put in priority order (section 3).
2. For each service, only zones that are up, that the service did not just fail in, and that have enough free capacity are candidates. If there are none, the service waits.
3. **TOPSIS** ranks the candidates on four criteria:

| Criterion | Meaning | Direction |
|---|---|---|
| Availability | Free capacity in the zone right now (units), after other tenants and services already placed | Higher is better |
| Recovery time | Believed recovery delay (s) | Lower is better |
| Cost | USD per hour | Lower is better |
| Carbon | gCO2e/kWh | Lower is better |

   The weights depend on the priority of the service being recovered. Each row adds up to 1.

| Service priority | Availability | Recovery time | Cost | Carbon |
|---|---:|---:|---:|---:|
| Critical | 0.35 | 0.40 | 0.15 | 0.10 |
| Important | 0.30 | 0.25 | 0.25 | 0.20 |
| Normal | 0.15 | 0.10 | 0.35 | 0.40 |

4. **Look-ahead.** Among the TOPSIS-ranked candidates, the chosen zone is the best-ranked one that does not reduce what the next failed services (up to 8) can still recover. This stops a Critical service from taking the only space a bigger Important service needed.
5. **Feedback.** After every recovery, the measured delay is saved. The next decision uses the average of the last 20 measured delays for each zone in place of its configured delay. With no history, the configured value is used. Only the recovery delay is corrected. Cost and carbon are stored but do not change decisions.

---

## 7. Simulation assumptions (must be stated in the report)

| Setting | Value | Where |
|---|---|---|
| Share of each zone used by other tenants | 10% plus the LSTM-predicted load | `BASE_BUSY` |
| Extra load in a CPU overload | 35% of capacity | `OVERLOAD_EXTRA` |
| Energy of one load unit | 0.05 kWh per hour | `ENERGY_KWH_PER_UNIT` |
| Time counted for cost and carbon | 1 hour | `HOURS` |
| Orchestration gap between recoveries | 2 s | `STEP_GAP_SEC` |
| Look-ahead depth | 8 later services | `LOOKAHEAD_MAX` |
| Real delays in "drift" and "stress" | A 55 s, B 30 s, C 45 s, D 30 s | `TRUTH_DRIFT` |
| Measurement noise on delays | about ±15% | `NOISE_SIGMA` |
| Extra tenant load in "stress" | 5% of every zone (change with `stress=0.08`) | `STRESS_EXTRA` |
| Random seed and failures | 2026, 200 failures | `SEED`, `N` |

**Cost and carbon accounting.**

- Cost = zone price (USD/hour) × load units × 1 hour. One load unit is modelled as one m5.large instance-hour.
- Carbon = carbon intensity × load units × 0.05 kWh × 1 hour.
- Results are reported **per load unit recovered**, so a method that recovers fewer services does not look cheaper.
- Both are **estimates**, not billing or measured emissions, and the paper must say so.

---

## 8. Evaluation protocol

**Methods compared on the same failures**

| Method | Description |
|---|---|
| Random zone | Random zone with room |
| Cheapest zone | Cheapest zone with room |
| Base paper Eq. 1 | Score S = w_c/C + w_m/M + w_l/L, zero if latency L is above a threshold (Alomari and Wang, 2026) |
| TOPSIS | TOPSIS only, one service at a time |
| TOPSIS + look-ahead | The recovery logic of section 6, steps 1 to 4 |
| TOPSIS + look-ahead + feedback | The full method (the reference in all comparisons) |

The base paper gives **no weights and no latency threshold**, and this data has no latency. So the base-paper baseline uses equal weights with no cutoff (the most favourable reading for the paper). CPU-heavy weights, latency-heavy weights and a version with a latency cutoff are reported as sensitivity rows. Latency values are simulated. These are assumptions and must be stated.

**Conditions (same failures in each)**

| Condition | Meaning | Purpose |
|---|---|---|
| `no_drift` | Configured delays are correct | Shows feedback does no harm when there is nothing to learn |
| `drift` | Some zones really recover slower or faster than configured | Shows feedback correcting wrong settings |
| `stress` | Drift plus extra tenant load in every zone | Shows behaviour when free capacity runs short |

**Metrics.** Weighted availability (priority weights 3, 2, 1), Critical-service availability, Critical recovery time, weighted recovery time, cost per load unit, carbon per load unit, and peak zone utilization. Every number is a mean with a 95% confidence interval. A paired comparison (full method minus each other method, same failures) is also reported. A difference counts only if its interval does not include zero. A breakdown by failure type is saved for each condition.

---

## 9. Limitations to state honestly

- Failures, recovery delays, capacity and latency are simulated.
- The workload comes from one data center. Grouping VMs into zones is a modelling choice.
- Carbon years differ: three values are 2025, Oregon is 2023.
- The Oregon value uses the NWPP subregion. The state-only value is much lower (165.5 g/kWh), and a sensitivity run has not been done yet.
- Look-ahead raises availability but makes Critical recovery a few seconds slower than plain TOPSIS.
- Feedback can only learn about zones that are used. A zone wrongly believed to be slow may never be tried again.
- The full method is not better than every baseline on every metric. The result is a trade-off, not a win on all counts.

---

## 10. Still to do

1. Run and review the stress condition and the per-failure-type results.
2. Streamlit dashboard: zone health, ranked zones, recovery history.
3. CI/CD: one YAML file per service (tier and requirements) and a GitHub Actions workflow that validates it and runs a simulated failure.
4. Report, presentation and IEEE paper (with the guide as co-author).

---

## 11. Team approval

- [ ] Meti Hima Bindu
- [ ] Kummari Krishna Sri
- [ ] Harish Krishna Kundeti
- [ ] Raja Saketh Ram
- [ ] Guide: Mr. P. Venkateswarlu Reddy