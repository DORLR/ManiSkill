"""Models.

    perception   point cloud + DINOv2 primitives (reused from alignment_field/paf) and port axes
    tokenizer    one frame -> 63 relation tokens + 1 joint token (plug/port set abstraction,
                 plug<->port cross-attention, learned relation queries)
    vae          token VAE: tokenizer -> per-token Gaussian latent (64 x C) -> decoders and
                 the plug-pose head (trained with the plug-point loss)
    cdit         NWM CDiT (vendored, unchanged) over the 8 x 8 latent grid, conditioned on the
                 joint-target chunk; Gaussian diffusion from NWM
"""
