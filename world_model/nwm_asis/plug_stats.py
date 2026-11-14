"""metric_waypoint_spacing and per-step action min / max of our commanded hand displacements (training split), as NWM's
config/data_config.yaml holds them for its datasets -> plug_data_config.yaml next to this file."""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from plug_dataset import STATS, command_positions

roots = sys.argv[1:]
rng = np.random.default_rng(0)
steps, far = [], []
for root in roots:
    entries = [e for e in json.loads((Path(root) / "index.json").read_text()) if e["split"] == "train" and e["n_images"] > 8]
    for e in [entries[i] for i in rng.permutation(len(entries))[:600]]:
        try:
            _, P, R = command_positions(e["path"], root)
        except FileNotFoundError:
            continue
        steps.append(np.einsum("ti,tij->tj", P[1:] - P[:-1], R[:-1]))
        far.append(np.abs(P - P[0]).max())
d = np.concatenate(steps)
norm = np.linalg.norm(d, axis=1)
spacing = float(norm[norm > 2e-5].mean())
u = d / spacing
out = {"metric_waypoint_spacing": spacing,
       "action_stats": {"min": [float(v) for v in np.percentile(u, 0.5, axis=0)], "max": [float(v) for v in np.percentile(u, 99.5, axis=0)]}}
STATS.write_text(yaml.safe_dump(out))
print(out)
print("frames", len(d), "| moving share", round(float((norm > 2e-5).mean()), 3), "| step mm p50 / p99", [round(float(v) * 1000, 3) for v in np.percentile(norm, [50, 99])],
      "| largest excursion from the start, mm, median / max", round(float(np.median(far)) * 1000, 1), round(float(np.max(far)) * 1000, 1))
