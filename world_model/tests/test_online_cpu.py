"""The online parts without a simulator, on a CPU with the sample fast copy (tests/test_demo320.py builds it):
an attempt written by run_wm320_attempts.write_fast reads back byte for byte and gives the labels of the same motion;
the agent's scores have the documented arithmetic; the joint offset moves the hand as asked; the learner mixes in
complete rounds of a pool only, and online attempts that ended seated join the imitation set.

    python world_model/tests/test_online_cpu.py
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "train"))
sys.path.insert(0, str(ROOT / "scripts" / "online"))
import run_wm320_attempts as RA  # noqa: E402
import train_wm320 as T  # noqa: E402
from wm.online.agent320 import WMAgent  # noqa: E402
from wm.online.common320 import jpeg, offset, unjpeg  # noqa: E402
from wm.tdmpc import demo320 as D  # noqa: E402
from wm.tdmpc import wmdata320 as WD  # noqa: E402

FAST = Path(os.environ["WM_FIX_ROOT"]) / "data" / "wm_v1_fast_sample"


def _train_entry(kind="contact"):
    for e in D.load_entries(FAST):
        if e["split"] == "train" and e["kind"] == kind and e["steps"] >= 3 * WD.EXECUTE:
            return e
    raise AssertionError(f"no training trajectory of kind {kind} in the sample")


def _as_attempt(e):
    """A sample trajectory dressed as an attempt's recording (pictures re-encoded with the attempt's codec)."""
    with np.load(FAST / e["path"]) as z:
        meta = json.loads(str(z["meta"]))
        rgb = np.stack([D.decode_rgb(z["rgb_jpeg"], z["rgb_offsets"], i) for i in range(len(z["img_step"]))])
        depth = np.stack([D.decode_depth(z["depth_png"], z["depth_offsets"], i) for i in range(len(z["img_step"]))])
        out = {k: z[k] for k in ("img_step", "qpos", "hand", "action")}
    out.update(rgb=rgb, depth=depth, rgb_jpeg_list=[jpeg(x) for x in rgb], dec_kind=np.zeros(len(out["action"]) // 8, np.int8))
    return out, meta


def test_write_fast(tmp):
    e = _train_entry()
    out, meta = _as_attempt(e)
    rel = "r1/traj/x/s0.npz"
    m = {**{k: meta[k] for k in ("port", "grasp", "condition", "pitch_deg", "grasp_pos_m", "H_star", "axis")},
         "pin_length_m": e["pin_length_m"], "kind": "online", "outcome": "seated_fk", "source": rel}
    ent = RA.write_fast(tmp / rel, out, m)
    assert ent["path"] == rel and ent["n_images"] == len(out["img_step"]) and ent["steps"] == len(out["action"]) and ent["kind"] == "online"
    with np.load(tmp / rel) as z:
        for i in range(len(out["img_step"])):
            assert np.array_equal(D.decode_rgb(z["rgb_jpeg"], z["rgb_offsets"], i), unjpeg(out["rgb_jpeg_list"][i]))
            assert np.array_equal(D.decode_depth(z["depth_png"], z["depth_offsets"], i), out["depth"][i])
        tr = WD.trajectory(z, json.loads(str(z["meta"])), 0.3, 0.967)
    with np.load(FAST / e["path"]) as z:
        tr0 = WD.trajectory(z, json.loads(str(z["meta"])), 0.3, 0.967)
    for k in ("delta", "q", "done", "depth", "probe"):
        assert np.allclose(tr[k], tr0[k]), k
    assert tr["term"][-1] == float(tr["end_seated"]), "an online attempt ends for good only when FK says seated"
    s = D.policy_sample(tmp / rel, len(out["img_step"]) - 1)
    assert s["rgb"].shape == (4, 320, 320, 3) and s["a"].shape == (D.CHUNK * D.JOINTS,)
    return f"{tr['n']} decisions, {ent['n_images']} pictures; labels equal to the source's"


def test_offset():
    rng = np.random.default_rng(0)
    for _ in range(20):
        J = rng.normal(size=(6, 7))
        d = rng.normal(size=3) * 1e-3
        dq = offset(J, d)
        assert np.allclose(J @ dq, np.r_[d, 0, 0, 0], atol=1e-8), "the hand moves by d, no rotation"
    return "20 random Jacobians: J dq = (d, 0)"


def test_agent(tmp):
    run = tmp / "wm"
    _run(run, "--steps", "2")
    ck = str(run / "step2.pt")
    ag = WMAgent(ck, "cpu", horizon=2)
    e = _train_entry()
    out, _ = _as_attempt(e)
    idx = D.context_indices(4)
    z = ag.encode(np.stack([unjpeg(out["rgb_jpeg_list"][i]) for i in idx]), out["depth"][idx], out["qpos"][16])   # picture 4 = step 16
    p = ag.propose(z)
    assert p.shape == (D.CHUNK, D.JOINTS) and 0.0 <= ag.done(z) <= 1.0 and ag.values(z).shape == (ag.m.cfg.num_q,)
    cands = np.stack([p[:WD.EXECUTE], p[:WD.EXECUTE] + 1e-3, p[:WD.EXECUTE] - 1e-3])
    sc = ag.score(z, cands)
    assert sc.shape == (3, ag.m.cfg.num_q) and np.isfinite(sc).all()
    m, g = ag.m, ag.gamma
    with torch.no_grad():                                  # the documented sum, written out for candidate 1
        a0 = m.scale(torch.from_numpy(cands[1:2].astype(np.float32).reshape(1, -1)))
        r0 = m.scalar(m.reward(z, a0))[0, 0]
        z1 = m.next(z, a0)
        c1 = 1 - torch.sigmoid(m._done(z1))[0, 0]
        a1 = m.policy_action(z1)
        r1 = m.scalar(m.reward(z1, a1))[0, 0]
        z2 = m.next(z1, a1)
        c2 = c1 * (1 - torch.sigmoid(m._done(z2))[0, 0])
        V = m.scalar(m.V(z2))[:, 0, 0]
        want = (r0 + c1 * g * r1 + c2 * g ** 2 * V).double().numpy()
    assert np.allclose(sc[1], want, atol=1e-5), (sc[1], want)
    return f"scores {sc.shape}, candidate 1 equals the written-out sum"


def test_online_learner(tmp):
    pool = tmp / "pool"
    e = _train_entry()
    out, meta = _as_attempt(e)
    ents = []
    for i, outcome in enumerate(("seated_fk", "not_seated")):
        rel = f"r1/traj/{e['port']}/{e['grasp']}/s{i}.npz"
        m = {**{k: meta[k] for k in ("port", "grasp", "condition", "pitch_deg", "grasp_pos_m", "H_star", "axis")},
             "pin_length_m": e["pin_length_m"], "kind": "online", "outcome": outcome, "source": rel}
        ents.append(RA.write_fast(pool / rel, out, m))
    (pool / "r1" / "manifest.json").write_text(json.dumps({"entries": ents}))
    (pool / "r2").mkdir()
    (pool / "r2" / "manifest.json").write_text(json.dumps({"entries": ents}))      # no DONE: not read
    got, names = T.online_entries(pool)
    assert names == [] and got == []
    (pool / "r1" / "DONE").write_text("{}")
    got, names = T.online_entries(pool)
    assert names == ["r1"] and len(got) == 2 and [g["bc"] for g in got] == [True, False] and all(Path(g["path"]).is_absolute() for g in got)
    run = tmp / "wm_online"
    _run(run, "--steps", "3", "--online", str(pool), "--online-frac", "0.5", "--wm-batch", "2", "--refresh-every", "2")
    log = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()]
    data = [r for r in log if r.get("event") == "data"]
    assert data and data[0]["online_trajectories"] == 2 and data[0]["online_seated"] == 1 and data[0]["online_windows_per_micro_batch"] == 1
    bad = dict(ents[0], pitch_deg=-40.0)                    # a held-out grasp (pitch strip) in the pool must stop the learner
    (pool / "r3").mkdir()
    (pool / "r3" / "manifest.json").write_text(json.dumps({"entries": [bad]}))
    (pool / "r3" / "DONE").write_text("{}")
    try:
        T.online_entries(pool)
        raise AssertionError("a non-training grasp was accepted")
    except SystemExit:
        pass
    return "only complete rounds are read; 1 of 2 micro-batch windows online; seated attempts join imitation; foreign grasps refused"


def _run(out, *extra):
    argv = ["train_wm320.py", "--data", str(FAST), "--out", str(out), "--stem", "standard", "--shift-px", "4", "--shaping", "0.3",
            "--wm-batch", "2", "--bc-batch", "2", "--accum", "1", "--workers-wm", "0", "--workers-bc", "0", "--eval-every", "2",
            "--eval-windows", "8", "--keep-every", "2", "--save-every", "1", "--td-warmup", "2", "--device", "cpu", *extra]
    old, sys.argv = sys.argv, argv
    try:
        T.main()
    finally:
        sys.argv = old


if __name__ == "__main__":
    tmp = Path(tempfile.mkdtemp(prefix="online320_"))
    try:
        print("test_write_fast OK:", test_write_fast(tmp), flush=True)
        print("test_offset OK:", test_offset(), flush=True)
        print("test_agent OK:", test_agent(tmp), flush=True)
        print("test_online_learner OK:", test_online_learner(tmp), flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
