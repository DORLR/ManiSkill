"""The training-only offset read-out (the user's decision 2026-11-18) on a CPU with the sample data: its labels are
wmdata320's FK "probe" of the right pictures; FM (head on the ResNet tokens) and WM (head on z, encoded and imagined)
train with it, store it, report its error at validation, refuse a resume with another weight; the agents load it
and never call it; checkpoints without it load as before. Environment of test_demo320.py.

    python world_model/tests/test_offset_cpu.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_policy_cpu as TP  # noqa: E402  (its sample split and run() helper)
import test_wm_cpu as TW  # noqa: E402
from wm.online.agent320 import WMAgent  # noqa: E402
from wm.tdmpc import demo320 as D  # noqa: E402
from wm.tdmpc import wmdata320 as WD  # noqa: E402
from wm.tdmpc.policy320 import PolicyData, ResnetFMPolicy, context_offsets  # noqa: E402
from wm.tdmpc.wm320 import WM320, load  # noqa: E402


def test_labels():
    ents = [e for e in D.load_entries(TP.FAST) if e["bc"]]
    e = ents[0]
    path = TP.FAST / e["path"]
    with np.load(path) as z:
        meta, img, hand = json.loads(str(z["meta"])), z["img_step"], z["hand"].astype(np.float64)
    for j in (0, 3, len(img) - 1):
        got = context_offsets(path, j)
        want = WD.decision_labels(meta, hand, img[D.context_indices(j)])["probe"]
        assert got.shape == (4, 4) and np.allclose(got, want)
    last = context_offsets(path, len(img) - 1)[-1]
    item = PolicyData(TP.FAST, [e], np.array([[0, len(img) - 1]]), offset=True)[0]
    assert torch.allclose(item["off"], torch.from_numpy(context_offsets(path, len(img) - 1))) and item["rgb"].shape == (4, 320, 320, 3)
    assert "off" not in PolicyData(TP.FAST, [e], np.array([[0, 0]]))[0], "without offset the items are as before"
    with np.load(path) as z:                          # WM windows and FM samples label the same pictures alike
        tr = WD.trajectory(z, meta, 0.3, 0.967)
    for k in range(min(tr["n"], 3)):
        w = WD.window(tr, k, np.ones(56, np.float32), 2 * np.ones(56, np.float32))
        for j in range(WD.H + 1):
            if k + j <= tr["n"]:
                assert np.allclose(w["probe"][j], context_offsets(path, 2 * (k + j))[-1]), (k, j)
    return f"labels = FK probe of the 4 context pictures, as the WM windows'; a demonstration's last picture is {np.round(last, 2).tolist()} from H*"


def test_policy(tmp):
    out = tmp / "fm"
    torch.manual_seed(0)
    init = ResnetFMPolicy("standard", offset=True).off.state_dict()      # train_policy320 seeds with --seed 0 likewise
    TP.run(out, "--steps", "2", "--offset-weight", "1.0")
    trained = torch.load(out / "last.pt", weights_only=False)["raw"]
    assert any(not torch.equal(init[k], trained[f"off.{k}"]) for k in init), "the offset head was trained"
    log = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    vals = [r for r in log if r.get("event") == "val"]
    assert vals and all(k in vals[-1]["select"] for k in ("offset_along_err_mm", "offset_lateral_err_mm", "offset_tilt_err_deg"))
    ck = torch.load(out / "step2.pt", weights_only=False)
    assert ck["model"] == {"stem": "standard", "offset": True} and ck["info"]["offset_weight"] == 1.0
    m = ResnetFMPolicy(**ck["model"])
    m.load_state_dict(ck["ema"])
    assert m.off is not None and any(k.startswith("off.") for k in ck["ema"])
    (out / "DONE.json").unlink()
    try:
        TP.run(out, "--steps", "3", "--resume")
        raise AssertionError("a resume without the offset weight was accepted")
    except SystemExit as e:
        assert "offset_weight" in str(e.code), e.code
    TP.run(out, "--steps", "3", "--resume", "--offset-weight", "1.0")
    calls = []                                         # the agent never reads the offset head
    agent = TP.Policy320Agent(str(out / "step3.pt"), "cpu")
    agent.model.off.register_forward_hook(lambda *a: calls.append(1))
    ents = [e for e in D.load_entries(TP.FAST) if e["bc"]]
    with np.load(TP.SRC / ents[0]["path"]) as z:
        frame = TP.SimpleNamespace(rgb=D.decode_rgb(z["rgb_jpeg"], z["rgb_offsets"], 5), depth=z["depth"][5], qpos=z["qpos"][int(z["img_step"][5])])
    agent.reset()
    a = agent.act(frame)
    assert a.shape == (1, D.CHUNK, 7) and not calls
    plain = tmp / "fm_plain"
    TP.run(plain, "--steps", "1")
    c = torch.load(plain / "step1.pt", weights_only=False)
    assert c["model"] == {"stem": "standard"} and ResnetFMPolicy(**c["model"]).off is None
    v = vals[-1]["select"]
    return (f"FM: trains, stores, refuses another weight, agent does not call it; read-out error after 2 steps "
            f"along {v['offset_along_err_mm']:.1f} mm, lateral {v['offset_lateral_err_mm']:.1f} mm, tilt {v['offset_tilt_err_deg']:.1f} deg")


def test_wm(tmp):
    out = tmp / "wm"
    TW.run(out, "--steps", "2", "--offset-weight", "1.0")
    log = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    vals = [r for r in log if r.get("event") == "val"]
    v = vals[-1]["heldout"]
    assert all(k in v for k in ("offset_loss", "offset_lat_err_mm_encoded", "offset_lat_err_mm_imagined_last")), sorted(v)
    ck = torch.load(out / "step2.pt", weights_only=False)
    assert ck["offset_weight"] == 1.0 and any(k.startswith("_offset.") for k in ck["state"])
    fresh = WM320("standard", ck["stats"], 0.3, offset_weight=1.0, **ck["cfg"])
    assert any(not torch.equal(v, ck["state"][f"_offset.{k}"]) for k, v in fresh._offset.state_dict().items() if torch.is_tensor(v)), \
        "the offset head was trained"
    m = load(str(out / "step2.pt"), "cpu")
    ag = WMAgent(str(out / "step2.pt"), "cpu")
    assert m._offset is not None and ag.m._offset is not None
    calls = []
    ag.m._offset.register_forward_hook(lambda *a: calls.append(1))
    ents = [x for x in D.load_entries(TP.FAST) if x["split"] == "train" and x["steps"] >= 3 * WD.EXECUTE]
    with np.load(TP.FAST / ents[0]["path"]) as z:
        rgb = np.stack([D.decode_rgb(z["rgb_jpeg"], z["rgb_offsets"], i) for i in range(5)])
        depth = np.stack([D.decode_depth(z["depth_png"], z["depth_offsets"], i) for i in range(5)])
        q = z["qpos"][16]
    idx = D.context_indices(4)
    zz = ag.encode(rgb[idx], depth[idx], q)
    p = ag.propose(zz)
    ag.score(zz, np.stack([p[:WD.EXECUTE], p[:WD.EXECUTE] + 1e-3]))
    ag.done(zz), ag.values(zz)
    assert not calls, "no decision of the agent reads the offset head"
    (out / "DONE.json").unlink()
    try:
        TW.run(out, "--steps", "3", "--resume")
        raise AssertionError("a resume without the offset weight was accepted")
    except SystemExit as err:
        assert "offset_weight" in str(err.code), err.code
    ck.pop("offset_weight")
    for k in [k for k in ck["state"] if k.startswith("_offset.")]:
        ck["state"].pop(k)
    p2 = tmp / "old.pt"
    torch.save(ck, p2)
    assert load(str(p2), "cpu")._offset is None, "an old checkpoint builds the model without the head"
    return (f"WM: trains, stores, agent never calls it, refuses another weight, old checkpoints plain; read-out error "
            f"encoded {v['offset_lat_err_mm_encoded']:.1f} mm, imagined {v['offset_lat_err_mm_imagined_last']:.1f} mm (2 steps)")


if __name__ == "__main__":
    tmp = Path(tempfile.mkdtemp(prefix="offset320_"))
    try:
        print("test_labels OK:", test_labels(), flush=True)
        print("test_policy OK:", test_policy(tmp), flush=True)
        print("test_wm OK:", test_wm(tmp), flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
