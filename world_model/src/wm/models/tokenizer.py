"""Frame -> 64 tokens: 63 relation tokens + 1 joint token.

1. Depth (0.1 mm) is back-projected to points in hand-centred port axes; points are
   split into plug (in front of the board) and port (board and holes) by the board
   depth, as in alignment_field's RelNet.
2. Each point carries xyz, RGB and frozen DINOv2 features; PointNet++ set
   abstraction gives plug tokens and port tokens.
3. Plug <-> port cross-attention blocks (RelNet's CrossBlock).
4. 63 learned relation queries read both token sets (self-attention among the
   queries, cross-attention to plug and port tokens): a fixed-size, fixed-order set,
   so the world model can treat it as an 8 x 8 grid together with the joint token.
5. The joint token embeds the 7 measured arm joint angles (the control signal).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .perception import (BOARD_BAND, BOX_YZ, DEPTH_UNIT_MM, PLUG_X, CrossBlock, DinoFeatures,
                         SetAbstraction, _sample, backproject_torch, mlp, to_port_frame)

N_TOKENS = 64


class RelationTokenizer(nn.Module):
    def __init__(self, d: int = 256, cross_layers: int = 4, query_layers: int = 2,
                 n_plug: int = 128, n_port: int = 256, n_cand: int = 2048, use_dino: bool = True):
        super().__init__()
        self.d, self.n_plug, self.n_port, self.n_cand = d, n_plug, n_port, n_cand
        self.dino = DinoFeatures() if use_dino else None
        fdim = (384 if use_dino else 0) + 6
        self.point_in = mlp((fdim, 128, 128), last_act=True)
        self.sa_plug, self.sa_port = SetAbstraction(128, d), SetAbstraction(128, d)
        self.type_emb = nn.Parameter(torch.zeros(2, d))
        self.cross = nn.ModuleList([CrossBlock(d) for _ in range(cross_layers)])
        self.queries = nn.Parameter(torch.randn(N_TOKENS - 1, d) * 0.02)
        self.query_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d, 8, 2 * d, batch_first=True, norm_first=True) for _ in range(query_layers)])
        self.joint_in = mlp((7, 128, d))

    def points(self, rgb, depth, K, cam_T_world, hand, R_P):
        """Hand-centred port-axis points and plug/port masks (geometry in float32)."""
        B = rgb.shape[0]
        with torch.autocast("cuda", enabled=False):
            pts_cam, valid = backproject_torch(depth.float() * DEPTH_UNIT_MM, K.float())
            p = to_port_frame(pts_cam, cam_T_world.float(), hand.float(), R_P.float()).reshape(B, -1, 3)
        valid = valid.reshape(B, -1)
        lateral = (p[..., 1].abs() < BOX_YZ) & (p[..., 2].abs() < BOX_YZ)
        far = valid & lateral & (p[..., 0] > 0.0)
        xs = torch.where(far, p[..., 0], torch.full_like(p[..., 0], float("nan")))
        x_b = torch.nanquantile(xs, 0.8, dim=1, keepdim=True).nan_to_num(0.05)   # board depth
        port = valid & lateral & ((p[..., 0] - x_b).abs() < BOARD_BAND)
        plug = valid & lateral & (p[..., 0] < x_b - BOARD_BAND) & (p[..., 0] > PLUG_X[0])
        return p, plug, port

    def _set(self, p, mask, rgbf, fmap, H, W, sa, n_centres):
        idx = _sample(mask, self.n_cand)
        xyz = torch.gather(p, 1, idx[..., None].expand(-1, -1, 3))
        m = torch.gather(mask, 1, idx)
        feats = [torch.gather(rgbf, 1, idx[..., None].expand(-1, -1, 3)), xyz / 0.05]
        if fmap is not None:
            vv, uu = (idx // W).float(), (idx % W).float()
            grid = torch.stack([(uu + 0.5) / W * 2 - 1, (vv + 0.5) / H * 2 - 1], -1)[:, None]
            feats.insert(0, F.grid_sample(fmap, grid, mode="bilinear", align_corners=False)[:, :, 0].transpose(1, 2))
        tok, _ = sa(xyz, self.point_in(torch.cat(feats, -1)), m, n_centres)
        return tok, xyz, m

    def forward(self, rgb, depth, K, cam_T_world, hand, R_P, qpos) -> dict:
        """rgb (B,H,W,3) uint8, depth (B,H,W) uint16 [0.1 mm], K (B,3,3), cam_T_world/hand (B,4,4),
        R_P (B,3,3), qpos (B,9). Returns tokens (B,64,d) and the sampled plug/port points."""
        B, H, W = depth.shape
        p, plug, port = self.points(rgb, depth, K, cam_T_world, hand, R_P)
        rgbf = rgb.reshape(B, -1, 3).float() / 255.0
        fmap = self.dino(rgb) if self.dino is not None else None
        t_plug, plug_xyz, plug_m = self._set(p, plug, rgbf, fmap, H, W, self.sa_plug, self.n_plug)
        t_port, port_xyz, port_m = self._set(p, port, rgbf, fmap, H, W, self.sa_port, self.n_port)
        a, b = t_plug + self.type_emb[0], t_port + self.type_emb[1]
        for blk in self.cross:
            a, b = blk(a, b)
        memory = torch.cat([a, b], 1)
        q = self.queries[None].expand(B, -1, -1)
        for layer in self.query_layers:
            q = layer(q, memory)
        joint = self.joint_in(qpos[:, :7].float())[:, None]
        return {"tokens": torch.cat([q, joint], 1), "plug_xyz": plug_xyz, "plug_mask": plug_m,
                "port_xyz": port_xyz, "port_mask": port_m}
