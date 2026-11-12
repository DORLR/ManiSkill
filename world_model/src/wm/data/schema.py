"""On-disk format of one recorded trajectory (one .npz file).

Model inputs (all obtainable on a real robot)
    rgb          (N, 320, 320, 3) uint8   wrist RGB crop (stored as JPEG, see store.py)
    depth        (N, 320, 320)    uint16  wrist depth crop, units of DEPTH_UNIT_M (0.1 mm), 0 = no return
    img_step     (N,)             int32   control step at which each image was taken
    cam_T_world  (N, 4, 4)        float32 world -> camera (OpenCV), from FK and the hand-eye mount
    K            (3, 3)           float32 intrinsics of the crop
    qpos         (T + 1, 9)       float32 measured joint angles at steps 0..T
    hand         (T + 1, 4, 4)    float32 hand (TCP) pose from FK at steps 0..T
    action       (T, 7)           float32 commanded arm joint targets (20 Hz)
    phase        (T,)             uint8   index into PHASES (expert phase label, bookkeeping only)

Labels (also real-robot obtainable)
    seg          (N, 320, 320)    uint8   per-pixel class (SEG_CLASSES), the training target of the
                                          segmentation network; on a real robot an annotation or an
                                          off-the-shelf segmenter (e.g. SAM) gives it
    in `meta`: H_star  inserted hand pose (4x4) of this grasp (FK of the one insertion)
               axis    insertion direction (unit, world) from the hand motion of that insertion

Evaluation only (simulator truth; never a model input, prefixed `eval_`)
    eval_rel     (T + 1, 7)       float32 charger pose in the socket-goal frame (x, y, z, qw, qx, qy, qz)

`meta` (JSON string) also holds: schema, grasp key, port, condition, pitch, grasp
position, seeds, trajectory kind and its parameters, outcome, steps.
"""

SCHEMA_VERSION = "wm-traj-2"   # 2: + seg
SEG_CLASSES = ("other", "plug", "robot", "board", "table")
DEPTH_UNIT_M = 1e-4
CONTROL_HZ = 20
IMAGE_EVERY = 4  # 20 Hz control -> 5 Hz images

TRAJECTORY_KINDS = (
    "reference",  # the one insertion per grasp, from the aligned safe standoff
    "insert",     # expert from a far start: align, insert
    "perturbed",  # expert from a far start with stochastic perturbation, recovery, insert
    "contact",    # align, offset by 0-3 mm / 0-3 deg, straight push until inserted or jammed
    "random",     # random motion in front of the board (no insertion)
    "explore",    # policy-independent contact exploration (collect/explore.py): random start near the
                  # board, truth-free random maneuver, straight push until inserted or blocked
)

PHASES = (
    "other",
    "continuous_approach",
    "recovery_align",
    "perturbation_stochastic",
    "axial_push",
    "final_verify",
    "contact_offset",
    "contact_push",
    "random_motion",
    "explore_maneuver",
    "explore_hold",
    "explore_push",
    "explore_contact",   # the contact phase of the contact set (collect/explore.py)
)
_PHASE_ID = {p: i for i, p in enumerate(PHASES)}

MODEL_INPUT_KEYS = ("rgb", "depth", "img_step", "cam_T_world", "K", "qpos", "hand", "action")
EVAL_PREFIX = "eval_"


def phase_id(phase: str) -> int:
    if phase.startswith("random_motion"):
        return _PHASE_ID["random_motion"]
    return _PHASE_ID.get(phase, 0)
