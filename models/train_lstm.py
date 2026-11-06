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
HORIZON = 3     # predict the next 3 readings (15 min forecast window)
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
    """Return input windows, flattened CPU/memory targets, and target start indices."""
    xs, ys, target_starts = [], [], []
    for i in range(len(X) - lookback - horizon + 1):
        target_start = i + lookback
        xs.append(X[i:i + lookback])
        ys.append(X[target_start:target_start + horizon, :2].reshape(-1))
        target_starts.append(target_start)
    return np.asarray(xs), np.asarray(ys), np.asarray(target_starts)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


results = []
for zone, g in df.groupby("zone"):
    g = add_time_features(g.sort_values("timestamp").reset_index(drop=True))
    values = g[["cpu", "memory"]].values

    split_row = int(len(g) * 0.7)  # first 70% rows are training period
    if split_row <= LOOKBACK + HORIZON:
        raise ValueError(f"Zone {zone}: not enough rows before the training cutoff")

    # Fit scaling parameters only on the training period.
    scaler = MinMaxScaler().fit(values[:split_row])
    scaled = scaler.transform(values)
    X_all = np.hstack([scaled, g[["h_sin", "h_cos", "d_sin", "d_cos"]].values])

    X, y, target_starts = make_windows(X_all, LOOKBACK, HORIZON)

    # Training targets must end before the cutoff.
    train_mask = target_starts + HORIZON <= split_row

    # Every test target must start at or after the cutoff. Test input windows
    # may use the preceding LOOKBACK observations as historical context.
    test_mask = target_starts >= split_row

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_te, y_te = X[test_mask], y[test_mask]

    if len(X_tr) == 0 or len(X_te) == 0:
        raise ValueError(f"Zone {zone}: empty train/test set after timestamp-safe split")

    # Explicitly verify that no test target timestamp is in the training period.
    first_test_target = int(target_starts[test_mask][0])
    last_test_target_end = int(target_starts[test_mask][-1] + HORIZON - 1)
    assert first_test_target >= split_row
    assert last_test_target_end < len(g)

    model = keras.Sequential([
        layers.Input(shape=(LOOKBACK, X.shape[2])),
        layers.LSTM(50, return_sequences=True),
        layers.LSTM(50),
        layers.Dense(HORIZON * 2),
    ])
    model.compile(optimizer="adam", loss="mse")
    stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    print(
        f"\n=== Training Zone {zone}: {len(X_tr)} train / {len(X_te)} test windows ==="
    )
    print(
        "Training target range:",
        g["timestamp"].iloc[LOOKBACK].isoformat(),
        "to",
        g["timestamp"].iloc[split_row - 1].isoformat(),
    )
    print(
        "Test target range:",
        g["timestamp"].iloc[first_test_target].isoformat(),
        "to",
        g["timestamp"].iloc[last_test_target_end].isoformat(),
    )

    model.fit(
        X_tr, y_tr, epochs=EPOCHS, batch_size=64,
        validation_split=0.1, callbacks=[stop], verbose=2
    )
    model.save(f"models/lstm_zone_{zone}.keras")

    pred = model.predict(X_te, verbose=0)
    pred_inv = scaler.inverse_transform(
        pred.reshape(-1, 2)
    ).reshape(-1, HORIZON, 2)
    true_inv = scaler.inverse_transform(
        y_te.reshape(-1, 2)
    ).reshape(-1, HORIZON, 2)

    # Last-value baseline: repeat the last observed CPU/memory value for all
    # three forecast steps.
    last_inv = scaler.inverse_transform(X_te[:, -1, :2])
    naive_inv = np.repeat(last_inv[:, None, :], HORIZON, axis=1)

    row = {
        "zone": zone,
        "train_windows": len(X_tr),
        "test_windows": len(X_te),
        "training_cutoff_timestamp": g["timestamp"].iloc[split_row].isoformat(),
        "first_test_target_timestamp": g["timestamp"].iloc[first_test_target].isoformat(),
        "last_test_target_timestamp": g["timestamp"].iloc[last_test_target_end].isoformat(),
        "cpu_rmse": rmse(pred_inv[:, :, 0], true_inv[:, :, 0]),
        "cpu_mae": mae(pred_inv[:, :, 0], true_inv[:, :, 0]),
        "mem_rmse": rmse(pred_inv[:, :, 1], true_inv[:, :, 1]),
        "mem_mae": mae(pred_inv[:, :, 1], true_inv[:, :, 1]),
        "naive_cpu_rmse": rmse(naive_inv[:, :, 0], true_inv[:, :, 0]),
        "naive_cpu_mae": mae(naive_inv[:, :, 0], true_inv[:, :, 0]),
        "naive_mem_rmse": rmse(naive_inv[:, :, 1], true_inv[:, :, 1]),
        "naive_mem_mae": mae(naive_inv[:, :, 1], true_inv[:, :, 1]),
    }

    # Also report errors separately for each forecast step, rather than hiding
    # horizon-specific behaviour inside one aggregate metric.
    for step in range(HORIZON):
        row[f"cpu_rmse_t+{(step + 1) * 5}min"] = rmse(
            pred_inv[:, step, 0], true_inv[:, step, 0]
        )
        row[f"mem_rmse_t+{(step + 1) * 5}min"] = rmse(
            pred_inv[:, step, 1], true_inv[:, step, 1]
        )
        row[f"naive_cpu_rmse_t+{(step + 1) * 5}min"] = rmse(
            naive_inv[:, step, 0], true_inv[:, step, 0]
        )
        row[f"naive_mem_rmse_t+{(step + 1) * 5}min"] = rmse(
            naive_inv[:, step, 1], true_inv[:, step, 1]
        )

    results.append(row)

    n = min(576, len(pred_inv))  # first 2 days of the test period
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax[0].plot(true_inv[:n, 0, 0], label="Actual")
    ax[0].plot(pred_inv[:n, 0, 0], label="LSTM")
    ax[0].plot(naive_inv[:n, 0, 0], label="Last-value baseline", alpha=0.7)
    ax[0].set_ylabel("CPU %")
    ax[0].legend()
    ax[0].set_title(f"Zone {zone}: 5-minute-ahead CPU prediction")
    ax[1].plot(true_inv[:n, 0, 1], label="Actual")
    ax[1].plot(pred_inv[:n, 0, 1], label="LSTM")
    ax[1].plot(naive_inv[:n, 0, 1], label="Last-value baseline", alpha=0.7)
    ax[1].set_ylabel("Memory %")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(f"results/lstm_zone_{zone}.png", dpi=120)
    plt.close()

out = pd.DataFrame(results).round(4)
out.to_csv("results/lstm_metrics.csv", index=False)
print("\n", out.to_string(index=False))
print(
    "\nSaved: models/lstm_zone_*.keras, results/lstm_metrics.csv, "
    "results/lstm_zone_*.png"
)
