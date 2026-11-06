"""All experiment constants in one place (fixed before any field result)."""

import numpy as np

PORTS = ("A", "B", "C", "D")

# ---------------------------------------------------------------- splits
# Grasp pitch range of the environment is [-60, -20] deg, grasp position
# (housing X of the grip centre) is [-30, -10] mm.
PITCH_INTERP = (-42.0, -38.0)
PITCH_EXTRAP = ((-60.0, -55.0), (-25.0, -20.0))
POS_INTERP = (-0.021, -0.019)
POS_EXTRAP = ((-0.030, -0.028), (-0.012, -0.010))
PITCH_RANGE = (-60.0, -20.0)
POS_RANGE = (-0.030, -0.010)

# --------------------------------------------------------------- geometry
GAP0 = 0.003            # pre-insertion pose: pin tips 3 mm outside the board
SIGMA_T = 0.001         # 1 mm  } scale that makes translation and rotation
SIGMA_R = np.radians(1)  # 1 deg } comparable in the field metric
SIGMA = np.array([SIGMA_T] * 3 + [SIGMA_R] * 3)

# ------------------------------------------------------------- collection
N_SWEEP = 150           # labelled views per grasp
N_PUSH = 32             # straight-push trials per grasp (defines ID)
NEAR_T, NEAR_TX, NEAR_R = 0.0015, 0.0015, np.radians(1.5)
WIDE_T, WIDE_TX, WIDE_R = 0.010, 0.015, np.radians(8.0)
PUSH_T, PUSH_R = 0.0008, np.radians(1.0)
PUSH_STEP = 0.00025     # 5 mm/s at 20 Hz
PUSH_JAM_LAG = 0.0015   # stop a push when the hand lags the command by 1.5 mm
PUSH_SUCCESS_MARGIN = 0.001  # inserted if the hand advanced >= stroke - 1 mm

# ------------------------------------------------------------- perception
# Wrist camera: Intel RealSense D405 at the Trossen Stationary AI default stream
# (640x480, 87 deg horizontal FOV, square pixels), mounted where Chongyu's env
# mounts it. Depth closer than the D405 minimum range is invalid (0), as on the
# real sensor.
CAM_W, CAM_H, CAM_HFOV_DEG = 640, 480, 87.0
DEPTH_MIN_MM = 70
CROP = (200, 456, 256, 512)      # IL policy input: rows r0:r1, cols c0:c1 (256x256) around pins and holes
REL_CROP = (160, 480, 224, 544)  # relational net input (320x320); K is shifted accordingly
N_POINTS = 1024
PC_BOX = ((-0.02, 0.12), (-0.06, 0.06), (-0.06, 0.06))  # hand-centred port axes, m

# ---------------------------------------------------------------- closed loop
OOD_T = (0.003, 0.010)      # |t_y|, |t_z| of OOD starts
OOD_TX = (-0.015, -0.003)
OOD_R = np.radians(6.0)
LOOP_MAX_ITERS = 25
STEP_MAX = 3.0               # max field step per iteration, in sigma units


def in_band(x, band):
    return band[0] <= x <= band[1]


def pitch_split(p):
    if in_band(p, PITCH_INTERP):
        return "interp"
    if any(in_band(p, b) for b in PITCH_EXTRAP):
        return "extrap"
    return "train"


def pos_split(x):
    if in_band(x, POS_INTERP):
        return "interp"
    if any(in_band(x, b) for b in POS_EXTRAP):
        return "extrap"
    return "train"


def sample_train_grasp(rng):
    while True:
        p = rng.uniform(*PITCH_RANGE)
        x = rng.uniform(*POS_RANGE)
        if pitch_split(p) == "train" and pos_split(x) == "train":
            return float(p), float(x)


def sample_grasp(rng, pitch_kind, pos_kind):
    def draw(kind, full, interp, extrap, split):
        if kind == "train":
            while True:
                v = rng.uniform(*full)
                if split(v) == "train":
                    return v
        if kind == "interp":
            return rng.uniform(*interp)
        b = extrap[rng.integers(len(extrap))]
        return rng.uniform(*b)
    return (float(draw(pitch_kind, PITCH_RANGE, PITCH_INTERP, PITCH_EXTRAP, pitch_split)),
            float(draw(pos_kind, POS_RANGE, POS_INTERP, POS_EXTRAP, pos_split)))
