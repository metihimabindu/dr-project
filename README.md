# Adaptive, business-aware, cost- and carbon-optimized disaster recovery

Capstone project, Mohan Babu University, Department of CSE (Batch A7-8).
A simulator that decides where to recover each failed service: an LSTM predicts zone load,
TOPSIS ranks the zones by business priority, cost and carbon, and every recovery outcome is fed back
to improve the next decision. Failures, delays and capacity are simulated; workload, cost and carbon
come from published data (see `sources.md`). The full design is in `design.md`.

## Folders

| Folder | What is in it |
|---|---|
| `configs/` | `zones.csv`, `services.csv`, and `services/*.yaml` (one recovery declaration per service) |
| `data/` | `build_zone_data.py`, `zone_data.csv` (raw Bitbrains files are not in Git) |
| `models/` | `train_lstm.py` and the trained LSTM models |
| `decision/` | `topsis.py`, `paper_baseline.py` |
| `simulator/` | failures and recovery (`recovery.py`), `predictor.py`, `run_experiment.py` |
| `feedback/` | the recovery history database and the learning demo |
| `dashboard/` | `app.py` (Streamlit) |
| `ci/`, `tests/`, `.github/workflows/` | checks that run on every push |
| `results/` | saved tables and charts |

## Setup

```
pip install -r requirements.txt
```

TensorFlow is only needed to train or use the LSTM. The checks and the dashboard (with "Last reading" or "Manual" load) run without it.

## Run

```
python data/build_zone_data.py          # needs the Bitbrains fastStorage download in data/raw
python models/train_lstm.py             # trains one LSTM per zone
python simulator/run_experiment.py      # 200 random failures, all methods, 3 conditions
streamlit run dashboard/app.py          # the dashboard
python ci/validate_services.py          # check the YAML declarations
python tests/test_ci.py                 # simulated failures
```

## Checks on every push

`.github/workflows/ci.yml` validates each `configs/services/*.yaml` against `configs/services.csv`,
runs a simulated failure of every type, and runs the unit tests.