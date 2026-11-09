"""Policy improvement in the world model's imagination.

    imagine   batched imagined insertion environment (world model + joint adapter + FK + pose readout)
    residual  residual PPO on top of the frozen latent FM policy
"""
