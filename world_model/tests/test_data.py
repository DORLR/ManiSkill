"""Unit tests that need no simulator: store round trip, plan, sampling."""

import numpy as np

from wm.collect.plan import CollectConfig, TEST_PORTS, UNBUILDABLE, build, shapes
from wm.data import schema, store
from wm.env.sampling import CONDITIONS, sample_contact_offset, sample_start


def test_store_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    arrays = {"rgb": rng.integers(0, 255, (3, 32, 32, 3), dtype=np.uint8),
              "depth": rng.integers(0, 3000, (3, 32, 32), dtype=np.uint16),
              "action": rng.normal(size=(10, 7)).astype(np.float32)}
    store.save(tmp_path / "t.npz", arrays, {"kind": "insert"})
    out, meta = store.load(tmp_path / "t.npz")
    assert meta["kind"] == "insert" and meta["schema"] == schema.SCHEMA_VERSION
    assert np.array_equal(out["depth"], arrays["depth"])          # depth is lossless
    assert np.abs(out["rgb"].astype(int) - arrays["rgb"]).mean() < 25  # jpeg is lossy but close
    assert store.index(tmp_path)[0]["kind"] == "insert"


def test_plan_covers_all_shapes_and_conditions():
    cfg = CollectConfig()
    specs = build(cfg)
    ports = {s.port for s in specs}
    assert ports == set(shapes(cfg)) and not ports & UNBUILDABLE
    for p in TEST_PORTS:
        assert {s.condition for s in specs if s.port == p} == set(CONDITIONS)
    gen_conds = {s.condition for s in specs if s.port not in TEST_PORTS}
    assert gen_conds == set(CONDITIONS)
    assert len({s.seed for s in specs}) == len(specs)
    assert build(cfg)[5].trajectories[2].start == specs[5].trajectories[2].start  # deterministic


def test_sampling_ranges():
    rng = np.random.default_rng(1)
    for _ in range(200):
        s = sample_start(rng)
        assert 0.006 <= s.tip_gap_m <= 0.018 and max(map(abs, s.yz_m)) <= 0.010
        assert max(map(abs, s.rpy_deg)) <= 6.0
        o = sample_contact_offset(rng)
        assert np.hypot(*o.yz_m) <= 0.003 + 1e-9 and np.linalg.norm(o.rpy_deg) <= 3.0 + 1e-9


def test_phase_ids():
    assert schema.phase_id("random_motion_2") == schema.PHASES.index("random_motion")
    assert schema.phase_id("something_new") == 0
