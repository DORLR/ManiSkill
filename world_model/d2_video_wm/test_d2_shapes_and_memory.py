"""Fast (<30s on GPU) shape, zero-privilege gradient-isolation, compat-reload, VRAM & latency smoke test for D2.

Run modes:
  1. Static check (no PyTorch required; validates syntax & preset table):
       python test_d2_shapes_and_memory.py --static-only

  2. Fast structural GPU/CPU smoke test (no internet / no pretrained weights needed):
       python test_d2_shapes_and_memory.py --no-pretrained --wm-batch 4 --bc-batch 8

  3. Full pretrained `fir` `3g.40gb` / `h100:1` benchmark (after login-node pre-caching):
       python test_d2_shapes_and_memory.py --pretrained --local-files-only --wm-batch 16 --bc-batch 32
"""

from __future__ import annotations

import argparse
import json
import py_compile
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

DEFAULT_ARCHS = (
    "d2a_segformer",
    "d2a_resnet34",
    "d2b_segformer",
    "d2b_resnet18",
    "d2c_vjepa2_vitl",
)


def run_static_checks() -> None:
    files = [
        _THIS_DIR / "__init__.py",
        _THIS_DIR / "video_wm320.py",
        _THIS_DIR / "train_video_wm320.py",
        _THIS_DIR / "eval_video_wm320_compat.py",
        _THIS_DIR / "compat_hook" / "sitecustomize.py",
        _THIS_DIR / "test_d2_shapes_and_memory.py",
    ]
    for f in files:
        assert f.exists(), f"Missing deliverable file: {f}"
        py_compile.compile(str(f), doraise=True)
    print(f"[STATIC OK] Compiled {len(files)} Python files cleanly.")


def make_dummy_stats() -> dict:
    import numpy as np

    return {
        "s0": np.full(56, 0.02, dtype=np.float32),
        "top": np.full(56, 0.40, dtype=np.float32),
        "a_mean": np.zeros(112, dtype=np.float32),
        "a_std": np.full(112, 0.05, dtype=np.float32),
    }


def make_dummy_batches(wm_batch: int, bc_batch: int, dev, offset_mode: str = "gap"):
    import torch

    H, Ctx = 4, 4
    S = H + Ctx  # 8 frames per window
    L_mm = torch.full((wm_batch,), 18.0, dtype=torch.float32, device=dev)
    probe_along = torch.randn(wm_batch, H + 1, 4, dtype=torch.float32, device=dev) * 5.0
    probe_along[..., 0] = -L_mm[:, None] - torch.randn(wm_batch, H + 1, dtype=torch.float32, device=dev) * 4.0
    gap_mm = -probe_along[..., 0] - L_mm[:, None]
    probe_gap = torch.cat([gap_mm.unsqueeze(-1), probe_along[..., 1:]], dim=-1)
    wire_mode = "along" if offset_mode in ("along", "probe") else "gap"

    W = {
        "rgb": torch.randint(0, 256, (wm_batch, S, 320, 320, 3), dtype=torch.uint8, device=dev),
        "depth": torch.randint(700, 2500, (wm_batch, S, 320, 320), dtype=torch.int16, device=dev),
        "q": torch.randn(wm_batch, H + 1, 7, dtype=torch.float32, device=dev) * 0.5,
        "action": torch.randn(wm_batch, H, 56, dtype=torch.float32, device=dev).clamp(-1.0, 1.0),
        "reward": torch.randn(wm_batch, H, dtype=torch.float32, device=dev),
        "term": torch.zeros(wm_batch, H, dtype=torch.float32, device=dev),
        "valid": torch.ones(wm_batch, H, dtype=torch.float32, device=dev),
        "obs_valid": torch.ones(wm_batch, H + 1, dtype=torch.float32, device=dev),
        "v_valid": torch.ones(wm_batch, H + 1, dtype=torch.float32, device=dev),
        "done": torch.zeros(wm_batch, H + 1, dtype=torch.float32, device=dev),
        "fk_depth": torch.rand(wm_batch, H + 1, dtype=torch.float32, device=dev) * 0.015,
        "a_next": torch.randn(wm_batch, H + 1, 56, dtype=torch.float32, device=dev).clamp(-1.0, 1.0),
        "a_next_valid": torch.ones(wm_batch, H + 1, dtype=torch.float32, device=dev),
        "probe": probe_gap if wire_mode == "gap" else probe_along,
        "probe_gap": probe_gap,
        "probe_along": probe_along,
        "pin_length_mm": L_mm,
        "gap_mm": gap_mm,
    }
    W["done"][:, -1] = 1.0
    W["term"][:, -1] = 1.0

    Bc = {
        "rgb": torch.randint(0, 256, (bc_batch, Ctx, 320, 320, 3), dtype=torch.uint8, device=dev),
        "depth": torch.randint(700, 2500, (bc_batch, Ctx, 320, 320), dtype=torch.int16, device=dev),
        "q": torch.randn(bc_batch, 7, dtype=torch.float32, device=dev) * 0.5,
        "a": torch.randn(bc_batch, 112, dtype=torch.float32, device=dev) * 0.05,
    }
    return W, Bc


def test_window_probe_canonicalization() -> None:
    """Verify `canonicalize_window_probes` never double-negates across legacy vs D1-patched WindowData."""
    import torch

    from video_wm320 import canonicalize_window_probes

    L_mm = 18.0
    along = torch.tensor([-22.0, -20.0, -18.5, -17.0, -15.0], dtype=torch.float32)
    gap = -along - L_mm  # [4.0, 2.0, 0.5, -1.0, -3.0]
    rest = torch.tensor([[1.0, -2.0, 0.5]] * 5, dtype=torch.float32)

    # Case 1: Legacy unpatched base WindowData (base_offset_mode='along', no gap_mm)
    w_legacy = {"probe": torch.cat([along[:, None], rest], dim=-1)}
    canonicalize_window_probes(w_legacy, L_mm, base_offset_mode="along", target_offset_mode="gap")
    assert torch.allclose(w_legacy["gap_mm"], gap)
    assert torch.allclose(w_legacy["probe_gap"][:, 0], gap)
    assert torch.allclose(w_legacy["probe_along"][:, 0], along)
    assert torch.allclose(w_legacy["probe"][:, 0], gap)

    # Case 2: D1-patched base WindowData with base_offset_mode='gap' (probe[:,0] is already gap_mm!)
    w_d1_gap = {"probe": torch.cat([gap[:, None], rest], dim=-1), "gap_mm": gap.clone()}
    canonicalize_window_probes(w_d1_gap, L_mm, base_offset_mode="gap", target_offset_mode="gap")
    assert torch.allclose(w_d1_gap["gap_mm"], gap)
    assert torch.allclose(w_d1_gap["probe_gap"][:, 0], gap)
    assert torch.allclose(w_d1_gap["probe_along"][:, 0], along)
    assert torch.allclose(w_d1_gap["probe"][:, 0], gap)
    print("[PROBES OK] canonicalize_window_probes verified (zero double-negation on legacy & D1 WindowData).")


def test_single_arch(
    arch: str,
    dev,
    wm_batch: int,
    bc_batch: int,
    pretrained: bool,
    local_files_only: bool,
    offset_mode: str,
    bench_iters: int,
) -> Dict[str, object]:
    import numpy as np
    import torch

    from eval_video_wm320_compat import VideoWMAgent, load_any_wm320
    from video_wm320 import VideoWM320

    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)

    stats = make_dummy_stats()
    model = VideoWM320(
        arch=arch,
        stats=stats,
        shaping=0.3,
        offset_mode=offset_mode,
        offset_weight=1.0,
        sg_probe=True,
        pretrained=pretrained,
        local_files_only=local_files_only,
    ).to(dev)
    summary = model.architecture_summary()

    # 1. Shape & forward checks
    W, Bc = make_dummy_batches(wm_batch, bc_batch, dev, offset_mode=offset_mode)
    L_flat = model.packed_latent_dim
    model.train()

    z = model.observe(W["rgb"], W["depth"], W["q"])
    assert z.shape == (wm_batch, 5, L_flat), f"{arch}: bad observe shape {z.shape}"
    z1 = model.next(z[:, 0], W["action"][:, 0])
    assert z1.shape == (wm_batch, L_flat), f"{arch}: bad next shape {z1.shape}"
    # Anti-collapse step-0 sanity check for D2b/D2c: ST-DiT must be strictly non-identity & action-sensitive!
    if model.spatial_latent and model.dynamics_type == "st_dit":
        z1_alt = model.next(z[:, 0], -W["action"][:, 0])
        assert float((z1 - z1_alt).abs().max()) > 1e-4, f"{arch}: STDiT insensitive to action at step 0!"
        assert float((z1 - z[:, 0]).abs().max()) > 1e-3, f"{arch}: STDiT collapsed to identity at step 0!"
        # Verify q_t flows into z_0 directly from step 0 (not discarded!)
        z_q_alt = model.observe(W["rgb"][:1], W["depth"][:1], W["q"][:1] + 0.5)
        assert float((z_q_alt[:, 0] - z[:1, 0]).abs().max()) > 1e-4, f"{arch}: q_t not preserved in z_0!"
    r = model.reward(z[:, :-1].transpose(0, 1), W["action"].transpose(0, 1))
    assert r.shape == (4, wm_batch, model.cfg.num_bins), f"{arch}: bad reward shape {r.shape}"
    q_out = model.Q(z[:, :-1].transpose(0, 1), W["action"].transpose(0, 1))
    assert q_out.shape == (model.cfg.num_q, 4, wm_batch, model.cfg.num_bins), f"{arch}: bad Q shape {q_out.shape}"
    v_out = model.V(z.transpose(0, 1))
    assert v_out.shape == (model.cfg.num_q, 5, wm_batch, model.cfg.num_bins), f"{arch}: bad V shape {v_out.shape}"
    act_out = model.act_chunk(z[:, 0])
    assert act_out.shape == (wm_batch, 16, 7), f"{arch}: bad act_chunk shape {act_out.shape}"
    done_out = model._done(z)
    assert done_out.shape == (wm_batch, 5, 1), f"{arch}: bad _done shape {done_out.shape}"
    if model.offset_mode == "gap":
        gap_out = model.predict_gap_mm(z)
        assert gap_out is not None and gap_out.shape == (wm_batch, 5), f"{arch}: bad gap shape"
    else:
        assert model.predict_gap_mm(z) is None, f"{arch}: along mode must not expose zero-privilege predict_gap_mm"
    if model._offset is not None:
        off_out = model._offset(z)
        assert off_out.shape == (wm_batch, 5, 4), f"{arch}: bad _offset shape {off_out.shape}"

    # 2. Mode-specific gradient checks:
    #    - `gap` (D1b 4D parity): `_offset` backprops into `enc`/`dynamics` (no `_probe_sg`!)
    #    - `gap_1d` / `none`: `_probe_sg` MUST leak 0.0 gradient into `enc`/`dynamics`/`readout`/`_gap`!
    if model.video_offset_mode == "gap":
        assert model._probe_sg is None and model._gap is None and model._offset is not None
        model.zero_grad(set_to_none=True)
        z_iso = model.observe(W["rgb"][:2], W["depth"][:2], W["q"][:2])
        z_next = model.next(z_iso[:, 0], W["action"][:2, 0])
        model._offset(z_next).pow(2).mean().backward()
        dyn_has_grad = any(p.grad is not None and float(p.grad.abs().max()) > 0.0 for p in model._dynamics.parameters())
        assert dyn_has_grad, f"{arch}: D1b-parity gap mode 4D _offset must backprop into dynamics!"
    elif model._probe_sg is not None:
        model.zero_grad(set_to_none=True)
        z_iso = model.observe(W["rgb"][:2], W["depth"][:2], W["q"][:2])
        z_next = model.next(z_iso[:, 0], W["action"][:2, 0])
        sg_out = model._probe_sg(z_next)
        sg_out.pow(2).mean().backward()

        sg_ids = {id(p) for p in model._probe_sg.parameters()}
        leaked = [
            n
            for n, p in model.named_parameters()
            if id(p) not in sg_ids and p.grad is not None and float(p.grad.abs().max()) > 0.0
        ]
        assert not leaked, f"{arch}: PRIVILEGE LEAK! _probe_sg backpropped into: {leaked[:5]}"
        sg_has_grad = any(p.grad is not None and float(p.grad.abs().max()) > 0.0 for p in model._probe_sg.parameters())
        assert sg_has_grad, f"{arch}: _probe_sg received no gradients!"

    # 3. Full training step (loss + backward + grad check + optimizer step + target EMA)
    opt = torch.optim.Adam(model.param_groups(lr=3e-4, enc_lr_scale=0.3))
    opt.zero_grad(set_to_none=True)
    loss, info = model.loss(W, Bc, step=10, warmup=5, bc_weight=1.0, detail=True)
    assert torch.isfinite(loss), f"{arch}: non-finite loss {loss}"
    loss.backward()

    # Verify ZERO unused trainable parameters (critical for DDP without find_unused_parameters!)
    unused = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not unused, f"{arch}: unused trainable params (would break DDP!): {unused[:8]}"

    gn = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 20.0))
    assert np.isfinite(gn), f"{arch}: non-finite grad norm {gn}"
    opt.step()
    model.update_target()

    peak_train_gb = (torch.cuda.max_memory_allocated(dev) / 1e9) if dev.type == "cuda" else 0.0

    # 4. Checkpoint save/load + Drop-in `VideoWMAgent` zero-privilege D1 contract + Latency benchmark
    with tempfile.TemporaryDirectory() as td:
        ck_path = Path(td) / f"{arch}_{offset_mode}.pt"
        ck = {**model.checkpoint_meta(stats), "state": model.state_dict(), "step": 10}
        assert ck["offset_mode"] == model.offset_mode
        assert ck["video_offset_mode"] == model.video_offset_mode
        torch.save(ck, ck_path)

        m_reloaded = load_any_wm320(ck_path, device=dev, local_files_only=local_files_only)
        assert isinstance(m_reloaded, VideoWM320)
        assert m_reloaded.video_offset_mode == model.video_offset_mode
        assert m_reloaded.offset_mode == model.offset_mode

        agent = VideoWMAgent(str(ck_path), device=str(dev), horizon=3, local_files_only=local_files_only)
        assert agent.offset_mode == model.offset_mode
        rgb4 = np.random.randint(0, 256, (4, 320, 320, 3), dtype=np.uint8)
        dep4 = np.random.randint(700, 2500, (4, 320, 320), dtype=np.uint16)
        q7 = np.zeros(7, dtype=np.float32)
        deltas13 = np.random.randn(13, 8, 7).astype(np.float32) * 0.01

        # Verify D1 zero-privilege `VideoWMAgent.gap_mm` contract strictly!
        z_chk = agent.encode(rgb4, dep4, q7)
        if agent.offset_mode == "gap":
            g_ok = agent.gap_mm(z_chk)
            assert g_ok is not None and np.isfinite(g_ok)
            if model.video_offset_mode == "gap":
                o_chk = agent.offset_mm(z_chk)
                assert o_chk is not None and abs(g_ok - float(o_chk[0])) < 1e-5
            try:
                agent.gap_mm(z_chk, pin_length_mm=18.0)
                raise AssertionError(f"{arch}/{offset_mode}: gap mode MUST raise ValueError when pin_length_mm is passed!")
            except ValueError:
                pass
        else:
            try:
                agent.gap_mm(z_chk)
                raise AssertionError(f"{arch}/{offset_mode}: along mode MUST raise ValueError when pin_length_mm is omitted!")
            except ValueError:
                pass
            o_chk = agent.offset_mm(z_chk)
            g_ok = agent.gap_mm(z_chk, pin_length_mm=18.0)
            assert o_chk is not None and g_ok is not None and abs(g_ok - (-float(o_chk[0]) - 18.0)) < 1e-5

        def _decision():
            z_dec = agent.encode(rgb4, dep4, q7)
            _ = agent.propose(z_dec)
            _ = agent.done(z_dec)
            _ = agent.gap_mm(z_dec) if agent.offset_mode == "gap" else agent.gap_mm(z_dec, pin_length_mm=18.0)
            _ = agent.offset_mm(z_dec)
            _ = agent.values(z_dec)
            sc, G = agent.score(z_dec, deltas13, parts=True)
            return z_dec, sc, G

        # Warmup
        for _ in range(3):
            z_dec, sc, G = _decision()
        assert z_dec.shape == (1, L_flat)
        assert sc.shape == (13, model.cfg.num_q)
        assert G.shape == (13,)

        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        times_ms: List[float] = []
        for _ in range(bench_iters):
            t0 = time.perf_counter()
            _decision()
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            times_ms.append((time.perf_counter() - t0) * 1000.0)
        med_ms = float(np.median(times_ms))

    return {
        "arch": arch,
        "family": summary["family"],
        "latent_shape": summary["latent_shape"],
        "params_total_m": summary["params_total_m"],
        "params_trainable_m": summary["params_trainable_m"],
        "encoder_pretrained": summary["encoder_pretrained"],
        "transition_pretrained": summary["transition_pretrained"],
        "peak_train_vram_gb": round(peak_train_gb, 2),
        "decision_latency_ms": round(med_ms, 2),
        "loss_total": round(info["total"], 3),
        "grad_norm": round(gn, 3),
    }


def main():
    ap = argparse.ArgumentParser(description="Smoke test & VRAM/latency benchmark for VideoWM320 (D2a/D2b/D2c)")
    ap.add_argument("--static-only", action="store_true", help="Run syntax & file checks only (no PyTorch required)")
    ap.add_argument("--archs", nargs="+", default=list(DEFAULT_ARCHS), help="List of arch presets to test")
    ap.add_argument("--include-vitg", action="store_true", help="Also test d2c_vjepa2_ac_vitg")
    ap.add_argument("--device", default=None, help="cuda or cpu (auto-detects CUDA if available)")
    ap.add_argument("--wm-batch", type=int, default=4, help="WM micro-batch size for smoke test / VRAM check")
    ap.add_argument("--bc-batch", type=int, default=8, help="BC micro-batch size for smoke test / VRAM check")
    ap.add_argument("--pretrained", action="store_true", help="Load real pretrained HF/torchvision weights")
    ap.add_argument("--no-pretrained", action="store_true", help="Use structural random-init weights (fast offline test)")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--offset-mode", default="gap", choices=["gap", "gap_1d", "along", "probe", "none"])
    ap.add_argument("--bench-iters", type=int, default=8)
    args = ap.parse_args()

    run_static_checks()
    if args.static_only:
        return

    import torch

    test_window_probe_canonicalization()

    dev = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    pretrained = bool(args.pretrained) and not bool(args.no_pretrained)
    archs = list(args.archs)
    if args.include_vitg and "d2c_vjepa2_ac_vitg" not in archs:
        archs.append("d2c_vjepa2_ac_vitg")

    print(
        f"Running D2 smoke test on {dev} | pretrained={pretrained} | offset_mode={args.offset_mode} | "
        f"wm_batch={args.wm_batch}, bc_batch={args.bc_batch}"
    )
    rows = []
    for arch in archs:
        print(f"  -> testing {arch} (offset_mode={args.offset_mode}) ...", flush=True)
        res = test_single_arch(
            arch=arch,
            dev=dev,
            wm_batch=args.wm_batch,
            bc_batch=args.bc_batch,
            pretrained=pretrained,
            local_files_only=args.local_files_only,
            offset_mode=args.offset_mode,
            bench_iters=args.bench_iters,
        )
        rows.append(res)
        print(
            f"     OK | params={res['params_total_m']}M (train {res['params_trainable_m']}M) | "
            f"latent={res['latent_shape']} | VRAM={res['peak_train_vram_gb']} GB | "
            f"13x3 decision={res['decision_latency_ms']} ms",
            flush=True,
        )

    # Always cross-check `gap_1d` (0.0 grad isolation on _probe_sg) AND `along` (legacy contract) on a representative arch!
    rep_arch = "d2b_segformer" if "d2b_segformer" in archs else archs[0]
    for extra_mode in ("gap_1d", "along"):
        if extra_mode == args.offset_mode:
            continue
        print(f"  -> cross-checking {rep_arch} (offset_mode={extra_mode}) ...", flush=True)
        extra_res = test_single_arch(
            arch=rep_arch,
            dev=dev,
            wm_batch=min(args.wm_batch, 2),
            bc_batch=min(args.bc_batch, 4),
            pretrained=pretrained,
            local_files_only=args.local_files_only,
            offset_mode=extra_mode,
            bench_iters=2,
        )
        print(f"     OK ({extra_mode}) | loss={extra_res['loss_total']} | grad_norm={extra_res['grad_norm']}", flush=True)

    print("\n=== D2 Smoke Test Summary ===")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
