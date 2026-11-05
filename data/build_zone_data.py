import glob, os
import pandas as pd

RAW = "data/raw/gwa_t_12_fastStorage/fastStorage/2013-8"   # folder with the VM csv files
N_PER_ZONE = 10                # VMs averaged into each zone
ZONES = ["A", "B", "C", "D"]

files = sorted(glob.glob(os.path.join(RAW, "**", "*.csv"), recursive=True))
print("VM files found:", len(files))
if len(files) < N_PER_ZONE * len(ZONES):
    raise SystemExit("Not enough files. Check the RAW path.")

def load_vm(path):
    df = pd.read_csv(path, sep=";", skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    ts = [c for c in df.columns if c.lower().startswith("timestamp")][0]
    cpu = [c for c in df.columns if "cpu usage [%]" in c.lower()][0]
    mem_used = [c for c in df.columns if "memory usage" in c.lower()][0]
    mem_cap = [c for c in df.columns if "memory capacity" in c.lower()][0]
    unit = "ms" if df[ts].iloc[0] > 1e11 else "s"
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df[ts], unit=unit),
        "cpu": df[cpu],
        "memory": 100 * df[mem_used] / df[mem_cap].replace(0, float("nan")),
    })
    return out.set_index("timestamp")

frames = []
for i, name in enumerate(ZONES):
    group = files[i * N_PER_ZONE:(i + 1) * N_PER_ZONE]
    vms = []
    for f in group:
        try:
            vms.append(load_vm(f).resample("5min").mean())
        except Exception as e:
            print("skipped", os.path.basename(f), e)
    zone_df = pd.concat(vms).groupby(level=0).mean()
    zone_df["zone"] = name
    frames.append(zone_df.reset_index())

result = pd.concat(frames)[["timestamp", "zone", "cpu", "memory"]].dropna()
result.to_csv("data/zone_data.csv", index=False)
print(result.groupby("zone").agg(rows=("cpu", "size"),
                                 cpu_mean=("cpu", "mean"),
                                 mem_mean=("memory", "mean")))