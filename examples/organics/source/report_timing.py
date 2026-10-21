"""Summarize warmed-up timing separately from JAX compilation and warmup."""

import argparse
import csv
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("run", type=Path)
a = p.parse_args()
cfg = json.loads((a.run / "config.json").read_text())
with (a.run / "history.csv").open() as f:
    rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
steady = rows[5:]
if len(steady) < 5:
    raise RuntimeError("Need at least 10 completed steps for a timing report")
result = dict(
    sites=cfg["sites"],
    t_prime=cfg["t_prime"],
    U=cfg["U"],
    channels=cfg["channels"],
    d=cfg["pairing_features"],
    C0=cfg["C0"],
    samples=cfg["samples"],
    complex_features=cfg["complex_features"],
    devices=cfg["device_kinds"],
    max_parallel=cfg["max_parallel"],
    parameters=cfg["parameters"],
    completed_steps=len(rows),
    excluded_initial_steps=5,
    first_step_seconds=rows[0]["seconds"],
    total_elapsed_seconds=rows[-1]["elapsed_seconds"],
)
for key in ["seconds", "sampling_seconds", "optimizer_seconds", "update_seconds"]:
    values = [r[key] for r in steady]
    result[key] = dict(
        median=float(np.median(values)),
        mean=float(np.mean(values)),
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
    )
result["projected_2000_steps_hours_excluding_startup"] = (
    result["seconds"]["median"] * 2000 / 3600
)
monitor = a.run.parents[1] / "gpu.csv"
if monitor.exists():
    with monitor.open() as f:
        gpu = list(csv.DictReader(f, skipinitialspace=True))
    memory = [float(r["memory.used [MiB]"]) for r in gpu]
    result["peak_observed_gpu_memory_MiB"] = max(memory) if memory else None
    util = [float(r["utilization.gpu [%]"]) for r in gpu]
    result["median_observed_gpu_utilization_percent"] = (
        float(np.median(util)) if util else None
    )
(a.run / "timing.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
