"""Gives the recovery system a predicted CPU / memory level for each zone.

Uses the LSTM models saved by models/train_lstm.py. If TensorFlow or a model file
is not available, it falls back to "the last reading" so the rest of the project
still runs.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

LOOKBACK = 12   # must match models/train_lstm.py
HORIZON = 3     # must match models/train_lstm.py


def _time_features(ts):
    hour = ts.dt.hour + ts.dt.minute / 60
    dow = ts.dt.dayofweek
    return np.column_stack([
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow / 7),
        np.cos(2 * np.pi * dow / 7),
    ])


class Predictor:
    def __init__(self, data_path="data/zone_data.csv", model_dir="models"):
        df = pd.read_csv(data_path, parse_dates=["timestamp"])
        self.zones = {}
        for zone, g in df.groupby("zone"):
            g = g.sort_values("timestamp").reset_index(drop=True)
            values = g[["cpu", "memory"]].values
            split_row = int(len(g) * 0.7)
            scaler = MinMaxScaler().fit(values[:split_row])
            x_all = np.hstack([
                scaler.transform(values),
                _time_features(g["timestamp"])
            ])
            self.zones[zone] = {
                "values": values,
                "scaler": scaler,
                "x_all": x_all,
                "split_row": split_row,
                "n": len(g),
            }

        self.models = {}
        self.load_note = ""
        try:
            from tensorflow import keras
            for zone in self.zones:
                path = os.path.join(model_dir, f"lstm_zone_{zone}.keras")
                if os.path.exists(path):
                    self.models[zone] = keras.models.load_model(path)
            missing = [z for z in self.zones if z not in self.models]
            if missing:
                self.load_note = (
                    f"no model file for zones {missing}; using last reading there"
                )
        except Exception as e:
            self.load_note = (
                f"TensorFlow not usable ({type(e).__name__}); using last reading"
            )

    def random_time(self, rng):
        """Choose a forecast origin whose three target rows are all in the test period."""
        start = max(z["split_row"] for z in self.zones.values())
        n = min(z["n"] for z in self.zones.values())
        # rng.integers has an exclusive upper bound. t + HORIZON - 1 must be < n.
        stop = n - HORIZON + 1
        if start >= stop:
            raise ValueError("Not enough post-cutoff rows for a test forecast")
        return int(rng.integers(start, stop))

    def predict(self, zone, t):
        """Predict CPU/memory for rows t through t + HORIZON - 1."""
        z = self.zones[zone]
        if t < z["split_row"]:
            raise ValueError(
                f"Prediction origin {t} is before zone {zone}'s test cutoff "
                f"{z['split_row']}"
            )
        if t + HORIZON > z["n"]:
            raise ValueError(f"Prediction horizon exceeds available data for zone {zone}")

        model = self.models.get(zone)
        if model is None:
            last = z["values"][t - 1]
            return float(last[0]), float(last[1]), "last-reading"

        window = z["x_all"][t - LOOKBACK:t]
        pred = model.predict(window[None, :, :], verbose=0).reshape(HORIZON, 2)
        inv = np.clip(z["scaler"].inverse_transform(pred), 0, 100)
        return float(inv[:, 0].max()), float(inv[:, 1].max()), "lstm"
