"""Environment side: robot access, grasp/start sampling, the demonstration expert.

Only `wm.env.expert` and the placement helpers read simulator truth (the expert
stands in for a human teleoperator). Recorded model inputs never contain truth;
see wm.data.schema.
"""
