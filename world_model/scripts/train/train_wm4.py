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
    scale = wm4.latent_scale(cfg)
    mk = lambda files: DataLoader(HistoryWindowDataset(files, scale, cfg["mask"], p_step=cfg["p_step"]),
                                  batch_size=cfg["batch_size"], shuffle=True, drop_last=True, num_workers=cfg["workers"],
                                  persistent_workers=True, pin_memory=True)
    batches = wm4.train_batches(cfg, mk)
    resume = str(run / "last.pt") if (run / "last.pt").exists() else None
    wm4.train(cfg, run, batches, max_steps=args.steps, resume=resume)


if __name__ == "__main__":
    main()
