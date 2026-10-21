"""Execute the combined notebook and each coupling notebook."""

import argparse
import json
from pathlib import Path
from make_notebook import make

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--root", type=Path, required=True)
a = p.parse_args()
experiment = json.loads((a.root / "experiment.json").read_text())
for U in [None, *sorted(experiment["U_order"])]:
    print(make(a.root, U, execute=True), flush=True)
