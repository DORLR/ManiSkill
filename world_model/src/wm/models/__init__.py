"""Models (v3).

    encoder         wrist RGB-D frame -> 64 tokens (DINOv2 + raw RGB + raw depth + viewing rays,
                    self-attention, 64 learned queries); no segmentation, no point cloud
    vae             encoder -> latent (64 x C, deterministic) -> patch decoder (depth, validity,
                    RGB, DINO) + pose head (remaining motion to the inserted pose)
    cdit            world model: NWM's CDiT with time shift, robot-state / commanded-path
                    condition and context-noise level
    adapter         Ctrl-World-style joint adapter: joints + tracking history + targets -> joints
    policy          flow-matching policy on the latent + joints
    baseline_policy latent-free baseline: the same encoder trained end to end with the policy
    perception      port axes from one frame (board-plane fit + gravity)
"""
