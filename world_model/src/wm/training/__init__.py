"""Training.

    common    config loading, seeding, logging, checkpoints, EMA
    losses    Chamfer, plug-point loss, KL
    vae       stage 1: token VAE (tokenizer + latent + decoders + plug-pose head)
    encode    frozen VAE -> latents of every recorded image (input to stage 2)
    wm        stage 2: CDiT diffusion world model on latents
"""
