import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

LOOKBACK = 12   # 12 readings x 5 min = last 1 hour as input
HORIZON = 3     # predict the next 3 reading (15 min ahead); try 3 later
EPOCHS = 60
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

df = pd.read_csv("data/zone_data.csv", parse_dates=["timestamp"])

def add_time_features(d):
    hour = d["timestamp"].dt.hour + d["timestamp"].dt.minute / 60
    dow = d["timestamp"].dt.dayofweek
    d["h_sin"] = np.sin(2 * np.pi * hour / 24)
    d["h_cos"] = np.cos(2 * np.pi * hour / 24)
    d["d_sin"] = np.sin(2 * np.pi * dow / 7)
    d["d_cos"] = np.cos(2 * np.pi * dow / 7)
    return d

def make_windows(X, lookback, horizon):
    xs, ys = [], []
    for i in range(len(X) - lookback - horizon + 1):
        xs.append(X[i:i + lookback])
        ys.append(X[i + lookback:i + lookback + horizon, :2].reshape(-1))
    return np.array(xs), np.array(ys)

def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

def mae(a, b):
    return float(np.mean(np.abs(a - b)))

results = []
for zone, g in df.groupby("zone"):
    g = add_time_features(g.sort_values("timestamp").reset_index(drop=True))
    values = g[["cpu", "memory"]].values
    split_row = int(len(g) * 0.7)                      # 70% train, 30% test
    scaler = MinMaxScaler().fit(values[:split_row])    # fit on training part only
    scaled = scaler.transform(values)
    X_all = np.hstack([scaled, g[["h_sin", "h_cos", "d_sin", "d_cos"]].values])

    X, y = make_windows(X_all, LOOKBACK, HORIZON)
    n_train = split_row - LOOKBACK - HORIZON + 1
    X_tr, y_tr, X_te, y_te = X[:n_train], y[:n_train], X[n_train:], y[n_train:]

    model = keras.Sequential([
        layers.Input(shape=(LOOKBACK, X.shape[2])),
        layers.LSTM(50, return_sequences=True),
        layers.LSTM(50),
        layers.Dense(HORIZON * 2),
    ])
    model.compile(optimizer="adam", loss="mse")
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                         restore_best_weights=True)
    print(f"\n=== Training Zone {zone}: {len(X_tr)} train / {len(X_te)} test windows ===")
    model.fit(X_tr, y_tr, epochs=EPOCHS, batch_size=64,
              validation_split=0.1, callbacks=[stop], verbose=2)
    model.save(f"models/lstm_zone_{zone}.keras")

    pred = model.predict(X_te, verbose=0)
    pred_inv = scaler.inverse_transform(pred.reshape(-1, 2)).reshape(-1, HORIZON, 2)
    true_inv = scaler.inverse_transform(y_te.reshape(-1, 2)).reshape(-1, HORIZON, 2)
    last_inv = scaler.inverse_transform(X_te[:, -1, :2])           # "same as last reading"
    naive_inv = np.repeat(last_inv[:, None, :], HORIZON, axis=1)

    row = {
        "zone": zone,
        "rmse_norm": rmse(pred, y_te),
        "cpu_rmse": rmse(pred_inv[:, :, 0], true_inv[:, :, 0]),
        "cpu_mae": mae(pred_inv[:, :, 0], true_inv[:, :, 0]),
        "mem_rmse": rmse(pred_inv[:, :, 1], true_inv[:, :, 1]),
        "mem_mae": mae(pred_inv[:, :, 1], true_inv[:, :, 1]),
        "naive_cpu_rmse": rmse(naive_inv[:, :, 0], true_inv[:, :, 0]),
        "naive_mem_rmse": rmse(naive_inv[:, :, 1], true_inv[:, :, 1]),
    }
    results.append(row)

    n = min(576, len(pred_inv))   # first 2 days of the test period
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax[0].plot(true_inv[:n, 0, 0], label="Actual"); ax[0].plot(pred_inv[:n, 0, 0], label="Predicted")
    ax[0].set_ylabel("CPU %"); ax[0].legend(); ax[0].set_title(f"Zone {zone}: LSTM prediction vs actual")
    ax[1].plot(true_inv[:n, 0, 1]); ax[1].plot(pred_inv[:n, 0, 1]); ax[1].set_ylabel("Memory %")
    plt.tight_layout(); plt.savefig(f"results/lstm_zone_{zone}.png", dpi=120); plt.close()

out = pd.DataFrame(results).round(4)
out.to_csv("results/lstm_metrics.csv", index=False)
print("\n", out.to_string(index=False))
print("\nSaved: models/lstm_zone_*.keras, results/lstm_metrics.csv, results/lstm_zone_*.png")