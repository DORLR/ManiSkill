"""Our scripted trajectories in the form NWM's train.py expects from its TrainingDataset (see README.md)."""
import io
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset

STATS = Path(__file__).resolve().parent / "plug_data_config.yaml"


def command_positions(path: str, root: str):
    """-> control step of every image, commanded hand position (n_images, 3) and rotation (n_images, 3, 3) at those steps."""
    with np.load(path) as f:
        steps = f["img_step"].astype(int)
    with np.load(path.replace(root, root + "_ee", 1)) as g:
        cmd = g["hand_cmd"]
    s = np.clip(steps, 0, len(cmd) - 1)
    return steps, cmd[s][:, :3, 3].astype(np.float64), cmd[s][:, :3, :3].astype(np.float64)


class PlugTrainingDataset(Dataset):
    """Same constructor and outputs as nwm/datasets.py TrainingDataset. `data_split_folder` is the name of the split in
    our index (train / ood_val); `traj_names` and `predefined_index` are not used."""

    def __init__(self, data_folder, data_split_folder, dataset_name, image_size, min_dist_cat, max_dist_cat, len_traj_pred, traj_stride,
                 context_size, transform, traj_names="traj_names.txt", normalize=True, predefined_index=None, goals_per_obs=1):
        self.root, self.split, self.dataset_name = data_folder.rstrip("/"), str(data_split_folder).strip("/"), dataset_name
        self.context_size, self.goals_per_obs, self.normalize, self.transform = context_size, goals_per_obs, normalize, transform
        entries = [e for e in json.loads((Path(self.root) / "index.json").read_text())
                   if e["split"] == self.split and e["n_images"] > context_size and os.path.exists(e["path"].replace(self.root, self.root + "_ee", 1))]
        self.paths = [e["path"] for e in entries]
        self.index_to_data = []
        for i, e in enumerate(entries):                       # as their _build_index
            n = e["n_images"]
            for curr in range(context_size - 1, n - len_traj_pred, traj_stride):
                self.index_to_data.append((i, curr, max(min_dist_cat, -curr), min(max_dist_cat, n - curr - 1)))
        st = yaml.safe_load(STATS.read_text())
        self.spacing = float(st["metric_waypoint_spacing"])
        self.amin, self.amax = np.asarray(st["action_stats"]["min"], np.float64), np.asarray(st["action_stats"]["max"], np.float64)

    def __len__(self):
        return len(self.index_to_data)

    def action(self, P, R, curr, goal_time):
        d = (P[goal_time] - P[curr]) @ R[curr]                # the goal's commanded position in the commanded hand's frame now
        if self.normalize:
            d = d / self.spacing
        return (d - self.amin) / (self.amax - self.amin) * 2 - 1

    def images(self, path, times):
        with np.load(path) as f:
            off, jpeg = f["rgb_offsets"], f["rgb_jpeg"]
        return torch.stack([self.transform(Image.open(io.BytesIO(jpeg[off[t]:off[t + 1]].tobytes())).convert("RGB")) for t in times])

    def __getitem__(self, k):
        i, curr, lo, hi = self.index_to_data[k]
        path = self.paths[i]
        goal_offset = np.random.randint(lo, hi + 1, size=(self.goals_per_obs))
        goal_time = (curr + goal_offset).astype("int")
        rel_time = goal_offset.astype("float") / 128.0
        times = list(range(curr - self.context_size + 1, curr + 1)) + list(goal_time)
        _, P, R = command_positions(path, self.root)
        return (torch.as_tensor(self.images(path, times), dtype=torch.float32),
                torch.as_tensor(self.action(P, R, curr, goal_time), dtype=torch.float32),
                torch.as_tensor(rel_time, dtype=torch.float32))
