"""Training (v3).

    common    seeding, GPU memory cap, logging, EMA, checkpoints
    losses    autoencoder reconstruction and pose losses
    vae       stage 1: autoencoder
    encode    frozen autoencoder -> latents (+ joints, hand poses, commanded hand poses)
    adapter   joint adapter
    wm        stage 2: world model (time shifts up to +-16 s, predicted-pose loss)
    policy    flow-matching policy (latent) and the latent-free baseline
"""
