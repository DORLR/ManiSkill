"""The policy's training script, resume, validation and agent on a CPU with the sample data (a few tiny steps).
Needs the fast copy of the sample (tests/test_demo320.py builds it). Environment as test_demo320.py.

    python world_model/tests/test_policy_cpu.py
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "train"))
import train_policy320 as T  # noqa: E402
from wm.tdmpc import demo320 as D  # noqa: E402
from wm.tdmpc.policy320 import Policy320Agent, ResnetFMPolicy, evaluate_policy, policy_loader  # noqa: E402

FIX = Path(os.environ["WM_FIX_ROOT"])
FAST, SRC = FIX / "data" / "wm_v1_fast_sample", Path(os.environ["WM_SAMPLE_ROOT"])
_split = D.split320
D.split320 = lambda port, pitch, pos, grasp: "select" if grasp == "G00_g02" else _split(port, pitch, pos, grasp)   # the sample has no selection grasp


def run(out, *extra):
    argv = ["train_policy320.py", "--data", str(FAST), "--out", str(out), "--stem", "standard", "--shift-px", "4", "--batch", "2", "--accum", "2",
            "--effective-batch", "4", "--eval-every", "1", "--eval-samples", "4", "--keep-every", "1", "--workers", "0", "--device", "cpu", *extra]
    old, sys.argv = sys.argv, argv
    try:
        T.main()
    finally:
        sys.argv = old


def refused(out, *extra):
    try:
        run(out, *extra)
    except SystemExit as e:
        return bool(e.code)
    return False


def test_train_resume(tmp):
    out = tmp / "run"
    run(out, "--steps", "1")
    assert {"last.pt", "best.pt", "step1.pt", "DONE.json", "stats.json", "log.jsonl"} <= {p.name for p in out.iterdir()}
    assert json.loads((out / "DONE.json").read_text()) == {"event": "done", "step": 1}
    assert refused(out, "--steps", "3") and refused(out, "--steps", "3", "--resume"), "a finished run is not restarted"
    (out / "DONE.json").unlink()
    assert refused(out, "--steps", "3"), "an existing run needs --resume"
    a = torch.load(out / "last.pt", weights_only=False)
    run(out, "--steps", "3", "--resume")
    b = torch.load(out / "last.pt", weights_only=False)
    log = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    resume = [r for r in log if r.get("event") == "resume"]
    assert len(resume) == 1 and resume[0]["step"] == 1 and b["step"] == 3 and b["ema_n"] == 3 and a["ema_n"] == 1
    assert np.array_equal(a["mean"], b["mean"]) and np.array_equal(a["std"], b["std"]), "the action scales are kept"
    assert all(k in b for k in ("raw", "opt", "best", "bad", "skipped", "args")) and (out / "step3.pt").exists()
    vals = [r for r in log if r.get("event") == "val"]
    assert len(vals) == 3 and all(set(r) >= {"select", "heldout", "train"} for r in vals)
    assert refused(tmp / "bad", "--steps", "1", "--accum", "3"), "batch x accum must be the effective batch"
    (out / "DONE.json").unlink(); (out / "step3.pt").unlink(); (out / "best.pt").unlink()     # as after a kill between the files
    c = torch.load(out / "last.pt", weights_only=False)
    run(out, "--steps", "4", "--resume")
    assert (out / "step3.pt").exists() and (out / "best.pt").exists() or c["bad"] > 0, "files the last validation owed are rewritten"
    assert (out / "step3.pt").exists() and torch.load(out / "step3.pt", weights_only=False)["step"] == 3
    young = tmp / "young"                          # a run stopped before its first validation: a log, no last.pt
    young.mkdir(); (young / "log.jsonl").write_text("")
    assert refused(young, "--steps", "1")
    run(young, "--steps", "1", "--resume")
    assert (young / "DONE.json").exists(), "--resume starts such a run again"
    return f"1 step, refusals, resume to step 3; validation on {sorted(k for k in vals[0] if isinstance(vals[0][k], dict))}"


def test_validation_and_agent(tmp):
    ents = [e for e in D.load_entries(FAST) if e["bc"]]
    items = D.policy_items(ents)[:6]
    ck = torch.load(tmp / "run" / "step3.pt", weights_only=False)
    assert ck["format"] == "policy320-2" and set(ck) >= {"model", "ema", "mean", "std", "step", "info"} and ck["info"]["shift_px"] == 4
    model = ResnetFMPolicy(**ck["model"])
    model.load_state_dict(ck["ema"])
    mean, std = torch.tensor(ck["mean"]), torch.tensor(ck["std"])
    state = torch.random.get_rng_state()
    r1 = evaluate_policy(model, policy_loader(FAST, ents, 3, 0, shuffle=False, items=items), mean, std, "cpu")
    assert torch.equal(state, torch.random.get_rng_state()), "the validation leaves the random generator as it was"
    torch.rand(5)
    r2 = evaluate_policy(model, policy_loader(FAST, ents, 3, 0, shuffle=False, items=items), mean, std, "cpu")
    assert r1["samples"] == r2["samples"] == 6 and r1 == r2, (r1, r2)
    loader = policy_loader(FAST, ents, 4, 0, seed=3)
    first, second = list(loader.sampler), list(loader.sampler)
    assert sorted(first) == list(range(len(loader.dataset))) and first != second, "a pass draws every sample exactly once, the next pass in another order"
    assert list(policy_loader(FAST, ents, 4, 0, seed=3).sampler) == first, "the order depends on the seed only"
    with np.load(SRC / ents[0]["path"]) as z:
        frame = SimpleNamespace(rgb=D.decode_rgb(z["rgb_jpeg"], z["rgb_offsets"], 5), depth=z["depth"][5], qpos=z["qpos"][int(z["img_step"][5])])
    out = {}
    for jpeg in (True, False):
        agent = Policy320Agent(str(tmp / "run" / "step3.pt"), device="cpu", jpeg_roundtrip=jpeg)
        assert agent.reset() and agent.step == 3
        torch.manual_seed(0)
        out[jpeg] = agent.act(frame, 2)
        assert out[jpeg].shape == (2, 16, 7) and np.isfinite(out[jpeg]).all()
        assert np.abs(out[jpeg] - frame.qpos[:7]).max() < 1.0, "targets are joint angles near the measured ones"
    rt = D.jpeg_roundtrip(frame.rgb)
    assert rt.shape == frame.rgb.shape and 0 < np.abs(rt.astype(int) - frame.rgb.astype(int)).mean() < 3
    return f"validation repeatable ({r1['action_err_mrad']:.2f} mrad on 6 samples); agent output {out[True].shape}"


if __name__ == "__main__":
    tmp = Path(tempfile.mkdtemp(prefix="p320_"))
    try:
        for t in (test_train_resume, test_validation_and_agent):
            print(t.__name__, "OK:", t(tmp), flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
