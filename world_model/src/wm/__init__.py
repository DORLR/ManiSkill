"""World model for plug insertion (see docs/README.md for the record of every run).

Subpackages
    wm.env       environment access, start/grasp sampling, the demonstration expert
    wm.data      on-disk schema, recorder, store, index/splits, forward kinematics, conditions, datasets
    wm.collect   dataset collection plan and multi-process runner
    wm.models    encoder, autoencoder, world model (CDiT), joint adapter, policies
    wm.training  training loops and latent encoding
    wm.eval      closed-loop evaluation in simulation
    wm.analysis  offline checks (what the latent keeps, what the world model predicts)
"""

__version__ = "0.3.0"
