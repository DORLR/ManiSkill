"""The world model's labels, windows, training script, validation and resume on a CPU with the sample fast copy
(tests/test_demo320.py builds it). Tiny sizes; checks shapes, masks, label arithmetic and the run's bookkeeping.

    python world_model/tests/test_wm_cpu.py
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "train"))
import train_wm320 as T  # noqa: E402
from wm.tdmpc import demo320 as D  # noqa: E402
from wm.tdmpc import wmdata320 as WD  # noqa: E402
from wm.tdmpc.wm320 import load  # noqa: E402

FAST = Path(os.environ["WM_FIX_ROOT"]) / "data" / "wm_v1_fast_sample"
_split = D.split320
D.split320 = lambda port, pitch, pos, grasp: "select" if grasp == "G00_g02" else _split(port, pitch, pos, grasp)


def _phi(e, k):
    with np.load(FAST / e["path"]) as z:
        meta = json.loads(str(z["meta"]))
        return float(WD.decision_labels(meta, z["hand"], np.array([WD.EXECUTE * k]))["phi"][0])


def test_labels_and_windows():
    n_ok, kinds = 0, set()
    for e in D.load_entries(FAST):
        if e["steps"] < WD.EXECUTE:                         # e.g. an expert that failed before its first step: no decision
            continue
        with np.load(FAST / e["path"]) as z:
            meta = json.loads(str(z["meta"]))
            tr0 = WD.trajectory(z, meta, 0.0, 0.967)
            tr1 = WD.trajectory(z, meta, 0.3, 0.967)
        n = tr0["n"]
        gain = (tr0["depth"][-1] - tr0["depth"][0]) * 1000
        assert abs(tr0["reward"].sum() - gain - (10 if tr0["end_seated"] else 0)) < 1e-3, "reward = depth gained (+10)"
        assert np.isfinite(tr1["reward"]).all() and tr1["reward"].shape == (n,)
        want = 0.0 if e["kind"] == "random" else float(tr0["end_seated"])
        assert tr0["term"][-1] == want and not tr0["term"][:-1].any(), "term from FK only"
        if tr1["term"][-1]:                                  # potential 0 at a terminal state
            assert abs(tr1["reward"][-1] - (tr0["reward"][-1] + 0.3 * (0.967 * 0 - _phi(e, n - 1)))) < 1e-3
        else:                                                # a cut end keeps its potential
            assert abs(tr1["reward"][-1] - (tr0["reward"][-1] + 0.3 * (0.967 * _phi(e, n) - _phi(e, n - 1)))) < 1e-3
        L_mm = 1000 * e["pin_length_m"]
        jam = e["kind"] == "contact" and not tr0["end_seated"] and tr0["probe"][n, 0] >= -L_mm - 1.0
        assert tr1["boot_ok"][-1] == float(jam) and not tr1["boot_ok"][:-1].any(), "boot only at a jam's last decision"
        if jam:
            assert abs(tr1["boot"][-1] + 0.3 * _phi(e, n)) < 1e-3 and tr0["boot"][-1] == 0.0
        if e["outcome"] == "inserted" and e["kind"] != "contact":
            assert tr0["end_seated"], "a successful expert trajectory ends seated by the FK rule"
        s0, top = WD.action_scale([tr0])
        for k in range(n):
            w = WD.window(tr0, k, s0, top)
            assert list(w["slots"]) == list(np.clip(2 * k - 6 + 2 * np.arange(8), 0, 2 * n))
            for j in range(WD.H + 1):                       # observation j's pictures are the policy's context of decision k+j
                if k + j <= n:
                    assert [int(w["slots"][j + c]) for c in range(4)] == D.context_indices(2 * (k + j))
            m = int(w["valid"].sum())
            assert m == min(WD.H, n - k) and w["obs_valid"].sum() == min(WD.H, n - k) + 1
            assert np.allclose(w["reward"][:m], tr0["reward"][k:k + m]) and (np.abs(w["action"]) <= 1).all()
        kinds.add(e["kind"]); n_ok += 1
    return f"{n_ok} trajectories ({sorted(kinds)}): reward sums, terminal flags, window slots and masks"


def run(out, *extra):
    argv = ["train_wm320.py", "--data", str(FAST), "--out", str(out), "--stem", "standard", "--shift-px", "4", "--shaping", "0.3",
            "--wm-batch", "2", "--bc-batch", "2", "--accum", "1", "--workers-wm", "0", "--workers-bc", "0", "--eval-every", "2",
            "--eval-windows", "8", "--keep-every", "2", "--save-every", "1", "--td-warmup", "2", "--device", "cpu", *extra]
    old, sys.argv = sys.argv, argv
    try:
        T.main()
    finally:
        sys.argv = old


def test_training(tmp):
    out = tmp / "wm"
    run(out, "--steps", "3")
    log = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    vals = [r for r in log if r.get("event") == "val"]
    assert json.loads((out / "DONE.json").read_text()) == {"event": "done", "step": 3} and len(vals) == 2
    v = vals[-1]["heldout"]
    need = {"consistency", "q_loss", "v_loss", "done_loss", "act_err_mrad", "auc_V_before_face_contact"}
    assert need <= set(v) or need - {"auc_V_before_face_contact"} <= set(v), sorted(v)
    m = load(str(out / "step3.pt"), "cpu")
    assert m.cfg.gamma == 1 - 1 / 30 and not m._Qt.training
    (out / "DONE.json").unlink()
    run(out, "--steps", "4", "--resume")
    assert json.loads((out / "DONE.json").read_text())["step"] == 4
    return "3 steps (2 in warm-up, 1 with the built-in head's next action), 2 validations, checkpoint loads, resume to step 4; " + \
        ", ".join(f"{k}={v[k]:.3g}" for k in ("consistency", "q_loss", "done_loss", "act_err_mrad") if isinstance(v.get(k), float))


if __name__ == "__main__":
    tmp = Path(tempfile.mkdtemp(prefix="wm320_"))
    try:
        print("test_labels_and_windows OK:", test_labels_and_windows(), flush=True)
        print("test_training OK:", test_training(tmp), flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
