"""Offline checks, run on held-out shapes (ood_val), no training of the checked models.

    probes     what the latent keeps: jam vs insert classifiers, offset regressions
    wm_checks  what the world model predicts: per-time-shift accuracy, jam vs insert separation,
               long imagined rollouts with the joint adapter
"""
