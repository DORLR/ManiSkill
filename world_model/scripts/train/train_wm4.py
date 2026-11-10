"""Offline stage of the history world model (v4, wm/training/wm4.py), from scratch.

    python world_model/scripts/train/train_wm4.py --config world_model/configs/v4/wm.json --run $Q/runs/wm_v4
"""

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from wm.data.history import HistoryWindowDataset
from wm.training import common, wm4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int, help="smoke tests only")
    args = ap.parse_args()
    cfg = common.load_config(args.config)
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(cfg, indent=1))
    scale = json.loads((Path(cfg["latents"]) / "stats.json").read_text())["scale"]
    ds = HistoryWindowDataset(wm4.offline_files(cfg, "train"), scale, cfg["mask"], p_step=cfg["p_step"])
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True, num_workers=cfg["workers"],
                        persistent_workers=True, pin_memory=True)
    resume = str(run / "last.pt") if (run / "last.pt").exists() else None
    wm4.train(cfg, run, common.cycle(loader), max_steps=args.steps, resume=resume)


if __name__ == "__main__":
    main()
