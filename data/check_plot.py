import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("data/zone_data.csv", parse_dates=["timestamp"])
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
for z, g in df.groupby("zone"):
    g = g.iloc[:2016]  # first 7 days
    axes[0].plot(g["timestamp"], g["cpu"], label=f"Zone {z}")
    axes[1].plot(g["timestamp"], g["memory"], label=f"Zone {z}")
axes[0].set_ylabel("CPU %"); axes[1].set_ylabel("Memory %")
axes[0].legend(); axes[0].set_title("First 7 days per zone")
plt.tight_layout(); plt.savefig("data/zone_check.png", dpi=120)
print("saved data/zone_check.png")