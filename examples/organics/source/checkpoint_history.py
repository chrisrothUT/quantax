"""Align durable training history with a committed model/optimizer/RNG boundary."""

import csv
from pathlib import Path
import time


def reconcile_history(out, iteration):
    path = Path(out) / "history.csv"
    if not path.exists() and iteration == 0:
        return []
    with path.open() as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = [{k: float(v) for k, v in r.items()} for r in reader]
    if len(rows) < iteration or any(
        int(r["iteration"]) != i + 1 for i, r in enumerate(rows[:iteration])
    ):
        raise ValueError("History does not contain the committed checkpoint prefix")
    if len(rows) > iteration:
        archive = path.with_name(f"history_uncommitted_{time.time_ns()}.csv")
        archive.write_bytes(path.read_bytes())
        temp = path.with_suffix(".csv.tmp")
        with temp.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows[:iteration])
        temp.replace(path)
    return rows[:iteration]
