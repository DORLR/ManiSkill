"""NWM's train.py, unchanged, on our data: only its dataset class is replaced (see README.md).

    python world_model/nwm_asis/run_plug.py --config world_model/nwm_asis/plug_cdit_b.yaml --ckpt-every 5000 \
        --eval-every 100000000 --bfloat16 1 --epochs 300 --torch-compile 0
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NWM = os.environ.get("NWM_DIR", "/mnt/storage2/qingxin/code/nwm_orig")
sys.argv = [sys.argv[0]] + [os.path.abspath(a) if a.endswith(".yaml") else a for a in sys.argv[1:]]
sys.path.insert(0, HERE)
sys.path.insert(0, NWM)
os.chdir(NWM)                                   # train.py and misc.py open config/*.yaml relative to the repository

import torch  # noqa: E402

frac = os.environ.get("WM_GPU_MEM_FRACTION")    # shared GPU: fail on our side instead of crowding others
if frac:
    torch.cuda.set_per_process_memory_fraction(float(frac))

import datasets  # noqa: E402  (NWM's datasets.py)
from plug_dataset import PlugTrainingDataset  # noqa: E402

datasets.TrainingDataset = PlugTrainingDataset
import train  # noqa: E402  (NWM's train.py; it does `from datasets import TrainingDataset`)

if __name__ == "__main__":
    train.main(train.get_args_parser().parse_args())
