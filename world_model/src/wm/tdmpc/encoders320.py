"""The picture encoders added for the AutoMate experiment (2026-11-21, the user's decisions): ResNet-34 and SegFormer-B1.
Selected with policy320.PictureTokens(encoder=...); the legacy ResNet-18 ("resnet18", the default) is unchanged.

  resnet34      built exactly as the legacy ResNet-18 (policy320.PictureTokens) with torchvision's resnet34 trunk
                (blocks 3-4-6-3) and its ImageNet weights (IMAGENET1K_V1): group norm (8 groups) instead of batch
                norm, a 4-channel 7x7 stem with step 2, max-pool, layer1-4, a 1x1 convolution to token_dim; 10 x 10 =
                100 tokens per 320 x 320 picture; RGB normalised with ImageNet's mean and std. Two differences from
                the legacy encoder: the depth slice of the first convolution starts at zero (the RGB weights are
                copied), and the depth input is normalised by depth_z below.
  segformer_b1  the full pretrained SegFormer-B1 of Hugging Face (SEGFORMER_B1 at SEGFORMER_B1_REVISION: the MiT-B1
                encoder and the all-MLP decoder trained on ADE20K), built with transformers'
                SegformerForSemanticSegmentation from the checkpoint's own configuration (SEGFORMER_B1_CONFIG). The
                first overlapping patch embedding (7x7, step 4) takes 4 channels: RGB weights copied, depth slice
                zero. RGB normalised with the checkpoint's image-processor mean and std (= ImageNet's), depth by
                depth_z. Kept: the decoder's fused map, i.e. the output of linear_fuse -> batch_norm -> ReLU (256
                channels at stride 4: 80 x 80 x 256 for 320 x 320), before the decoder's dropout and the ADE20K
                classifier, which are both replaced by nn.Identity (the classifier would be an unused parameter, which
                DistributedDataParallel refuses). The decoder's batch norm keeps its structure and checkpoint keys,
                with its running statistics frozen (see BATCH NORM below).
                The fused map is reduced to 20 x 20 = 400 tokens by LocalRefine (TokenPacker-style).

BATCH NORM (segformer_b1 only; the user's decision 2026-11-21): the decoder's BatchNorm2d(256) is a
FrozenStatsBatchNorm2d (a subclass of nn.BatchNorm2d, the same parameters, buffers and checkpoint keys): it always
normalises with its running statistics (the ADE20K values copied by init_segformer_b1) and never updates them, in
train() as in eval(); its affine weight and bias are trained. So the training and evaluation outputs of the map
agree, the two forward calls of wm320.WM320.loss see the same normalisation, gradient accumulation gives the same
gradient as one large batch, and under DistributedDataParallel every process holds the same statistics. (The
encoder's stochastic depth, drop_path_rate 0.1 of the original configuration, still acts in train() only.)

Weights are read only from the local Hugging Face cache ($HF_HOME), at the pinned revision, from its
pytorch_model.bin (SEGFORMER_B1_LOAD: local_files_only, use_safetensors=False; the pinned revision has no
safetensors file, and the Hub's automatic conversion lives on another revision). They are downloaded once, by hand,
with `python -m wm.tdmpc.encoders320 --download` (download_segformer_b1); a training run, and each of its processes,
never contacts the Hub.

Transformers source cited below: transformers 4.57.6, models/segformer/modeling_segformer.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

SEGFORMER_B1 = "nvidia/segformer-b1-finetuned-ade-512-512"
SEGFORMER_B1_REVISION = "a775b4f5dcb2179e57006dca0e268281306412ef"      # the Hub commit the weights are read from
SEGFORMER_B1_FILES = ["config.json", "preprocessor_config.json", "pytorch_model.bin"]
SEGFORMER_B1_LOAD = dict(revision=SEGFORMER_B1_REVISION, local_files_only=True, use_safetensors=False)
# The checkpoint's config.json at that revision, without its label names (id2label / label2id: 150 ADE20K classes),
# "architectures", "torch_dtype" and "transformers_version". Held here so that a saved model is rebuilt without the
# Hub; tests/test_encoders_automate.py checks it against the Hub's file.
SEGFORMER_B1_CONFIG = dict(
    attention_probs_dropout_prob=0.0, classifier_dropout_prob=0.1, decoder_hidden_size=256, depths=[2, 2, 2, 2],
    downsampling_rates=[1, 4, 8, 16], drop_path_rate=0.1, hidden_act="gelu", hidden_dropout_prob=0.0,
    hidden_sizes=[64, 128, 320, 512], image_size=224, initializer_range=0.02, layer_norm_eps=1e-06,
    mlp_ratios=[4, 4, 4, 4], num_attention_heads=[1, 2, 5, 8], num_channels=3, num_encoder_blocks=4,
    patch_sizes=[7, 3, 3, 3], reshape_last_stage=True, sr_ratios=[8, 4, 2, 1], strides=[4, 2, 2, 2], num_labels=150,
)
# The checkpoint's preprocessor_config.json: image_mean / image_std (identical to ImageNet's); rescale 1/255.
SEGFORMER_B1_MEAN, SEGFORMER_B1_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
FUSED_CH, FUSED_STRIDE = 256, 4                       # decoder_hidden_size; the fused map is at the first stage's stride


def depth_z(depth: torch.Tensor, max_m: float, mean_m: float, std_m: float, invalid: float) -> torch.Tensor:
    """Depth input of the new encoders. depth (N, H, W) in 0.1 mm, 0 = invalid -> (N, H, W) float32:
    d = clip(depth * 1e-4 m, 0, max_m); z = (d - mean_m) / std_m with the fixed dataset mean and standard deviation
    (no per-picture normalisation); invalid pixels (depth == 0) set to the constant `invalid` after normalisation."""
    d = (depth.float() * 1e-4).clamp(0.0, max_m)
    return ((d - mean_m) / std_m).masked_fill(depth == 0, invalid)


class Float32Conv2d(nn.Conv2d):
    """nn.Conv2d that always runs in float32, also inside a bfloat16 autocast region: the first layer of the
    SegFormer, as the legacy encoder's stem runs outside autocast ("0.1 mm of depth must survive"). Same parameters
    and state_dict keys as the nn.Conv2d it replaces."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(x.device.type, enabled=False):
            return super().forward(x.float())


def _transformers():
    """transformers 4.x (written and tested with 4.57.6). Version 5 lays the SegFormer out differently (stages instead
    of segformer.encoder.patch_embeddings / block / layer_norm), so its modules and checkpoint keys would not match."""
    import transformers
    if int(transformers.__version__.split(".")[0]) != 4:
        raise ImportError(f"encoder segformer_b1 needs transformers 4.x (tested: 4.57.6), found {transformers.__version__}: "
                          "version 5 names the SegFormer's modules and checkpoint tensors differently")
    return transformers


class FrozenStatsBatchNorm2d(nn.BatchNorm2d):
    """nn.BatchNorm2d whose running statistics are fixed: the forward pass always uses running_mean / running_var
    (as nn.BatchNorm2d in eval()) whatever the module's train()/eval() mode, and never updates them or
    num_batches_tracked. weight and bias stay trainable. Same state_dict keys as nn.BatchNorm2d."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, False, 0.0, self.eps)


def segformer_b1() -> nn.Module:
    """The SegFormer-B1 architecture (SegformerForSemanticSegmentation of SEGFORMER_B1_CONFIG, weights not loaded:
    init_segformer_b1 copies them) with a 4-channel first patch embedding computed in float32, and the decoder's
    dropout and classifier replaced by nn.Identity, so that decode_head(...) returns the fused map
    (modeling_segformer.py lines 645-647: linear_fuse -> batch_norm -> activation; line 648 dropout and line 651
    classifier become identities), and the decoder's batch norm with frozen statistics (FrozenStatsBatchNorm2d)."""
    tf = _transformers()
    m = tf.SegformerForSemanticSegmentation(tf.SegformerConfig(**SEGFORMER_B1_CONFIG))
    pe = m.segformer.encoder.patch_embeddings[0]
    old = pe.proj                                             # Conv2d(3, 64, 7, stride 4, padding 3), with bias
    pe.proj = Float32Conv2d(4, old.out_channels, old.kernel_size, old.stride, old.padding, bias=old.bias is not None)
    m.decode_head.dropout = nn.Identity()
    m.decode_head.classifier = nn.Identity()
    bn = m.decode_head.batch_norm                             # BatchNorm2d(256), eps 1e-5, affine, running statistics
    m.decode_head.batch_norm = FrozenStatsBatchNorm2d(bn.num_features, eps=bn.eps, momentum=bn.momentum, affine=bn.affine,
                                                      track_running_stats=bn.track_running_stats)
    return m


@torch.no_grad()
def init_segformer_b1(m: nn.Module) -> int:
    """Copy the pretrained SEGFORMER_B1 (at SEGFORMER_B1_REVISION) into `m` (from segformer_b1()): every tensor
    except the ADE20K classifier's; the first patch embedding's RGB weights and bias copied, its depth slice zero.
    Batch-norm statistics included. Returns the number of tensors copied."""
    try:
        src = _transformers().SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_B1, **SEGFORMER_B1_LOAD).state_dict()
    except OSError as exc:
        raise OSError(f"{SEGFORMER_B1} at {SEGFORMER_B1_REVISION} is not in the local cache ($HF_HOME); download it once "
                      "with `python -m wm.tdmpc.encoders320 --download`") from exc
    sd = {k: v for k, v in src.items() if not k.startswith("decode_head.classifier.")}
    k0 = "segformer.encoder.patch_embeddings.0.proj.weight"
    w = sd[k0]
    sd[k0] = torch.cat([w, torch.zeros_like(w[:, :1])], 1)
    m.load_state_dict(sd, strict=True)                        # every tensor of m is copied, none left over
    return len(sd)


def segformer_fused(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """x (N, 4, H, W) normalised -> the decoder's fused map (N, 256, H/4, W/4), as
    SegformerForSemanticSegmentation.forward (lines 711-720: segformer(..., output_hidden_states=True), its
    hidden_states into decode_head) with the identities of segformer_b1()."""
    hs = m.segformer(x, output_hidden_states=True, return_dict=True).hidden_states
    return m.decode_head(hs)


class LocalRefine(nn.Module):
    """TokenPacker-style local refinement (Li et al. 2024, arXiv 2407.02392; code github.com/CircleRadon/TokenPacker
    at 771a16e, llava/model/multimodal_projector/builder.py, class TokenPacker), restricted to one feature map.

    F (N, C, H, W), cells of s x s (s = 4: the 80 x 80 fused map gives a 20 x 20 grid); cell (i, j) holds the fine
    features f_1..f_16 (row-major inside the cell):
      q_ij = AvgPool_s(F)_ij                                    coarse query (TokenPacker: bilinear down-sampling)
      t_ij = q_ij + MHA(LN_q(q_ij), LN_k(f_k + p_k), LN_v(f_k + p_k)),  k = 1..16 of cell (i, j) only
      token_ij = Linear(GELU(Linear(t_ij)))                     C -> out_dim -> out_dim
    p_k: a learned sub-position embedding (s*s, C). The attention is restricted to the cell by reshaping F into
    (N * cells, s*s, C) (TokenPacker's divide_feature does the same), not by masking a full attention.

    As TokenPacker: LayerNorms (eps 1e-6) on the query, the keys and the values before the attention (its ln_q_1,
    ln_k_1, ln_v_1); nn.MultiheadAttention (its own W_Q, W_K, W_V and output projection, no dropout) with 8 heads
    (TokenPacker: num_heads = 1024 // 128 = 8; here of width C / 8 = 32); after the attention its MLP
    Linear -> GELU -> Linear (its self.mlp, which maps to the output width; here out_dim = token_dim, so this MLP is
    also the final projection to token_dim); weights as its _init_weights (Linear weights truncated normal, std 0.02,
    biases 0; LayerNorm 1 and 0; the attention's input projection keeps PyTorch's default). On purpose not as
    TokenPacker (the user's design): the residual q + MHA (TokenPacker has none), the sub-position embedding (it has
    none), average pooling for the query, one feature map instead of its multi-level keys and values, and therefore
    no pre-projections (its q_proj_1 and the two-layer k_proj_1 / v_proj_1 map CLIP's 1024 / 4 x 1024 channels)."""

    HEADS = 8

    def __init__(self, dim: int = FUSED_CH, cell: int = 4, out_dim: int = 128):
        super().__init__()
        self.cell = cell
        self.sub_pos = nn.Parameter(torch.randn(cell * cell, dim) * 0.02)       # as wm320.Encoder.token_pos
        self.ln_q = nn.LayerNorm(dim, eps=1e-6)
        self.ln_k = nn.LayerNorm(dim, eps=1e-6)
        self.ln_v = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, self.HEADS, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):                                       # TokenPacker._init_weights, verbatim in effect
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """f (N, C, H, W), H and W multiples of the cell -> tokens (N, (H/s) * (W/s), out_dim), row-major."""
        N, C, H, W = f.shape
        s = self.cell
        h, w = H // s, W // s
        q = F.avg_pool2d(f, s).flatten(2).transpose(1, 2).reshape(N * h * w, 1, C)                 # (N*cells, 1, C)
        kv = f.reshape(N, C, h, s, w, s).permute(0, 2, 4, 3, 5, 1).reshape(N * h * w, s * s, C) + self.sub_pos
        t = q + self.attn(self.ln_q(q), self.ln_k(kv), self.ln_v(kv), need_weights=False)[0]
        return self.mlp(t).reshape(N, h * w, -1)


def download_segformer_b1() -> str:
    """Download the pinned revision's files into $HF_HOME once (the only function here that contacts the Hub).
    Refuses an unset $HF_HOME (the default cache is the account's home directory, a shared account)."""
    import os
    if not os.environ.get("HF_HOME"):
        raise SystemExit("set $HF_HOME to our own cache before downloading (the default lies in the account's home)")
    from huggingface_hub import snapshot_download
    return snapshot_download(SEGFORMER_B1, revision=SEGFORMER_B1_REVISION, allow_patterns=SEGFORMER_B1_FILES)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--download"]:
        print(download_segformer_b1())
    else:
        raise SystemExit("python -m wm.tdmpc.encoders320 --download")
