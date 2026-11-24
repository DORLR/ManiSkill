"""Grasp variables and their per-assembly ranges (no Isaac; numpy, scipy, trimesh, hpp-fcl).

Grasp of an episode: shape (assembly id) and three in-hand offsets of AutoMate's optimised grasp (plug_grasps.json):
    dtheta  clamp angle [deg]: rotation of the plug about the hand y axis (finger-closing axis) through the TCP
    dp      [m]: translation of the plug along its own axis; dp > 0 = the fingers sit closer to the plug tip
    dpsi    spin [deg]: rotation of the plug about a line parallel to plug-frame z through (x_c, y_c), chosen by the
            switch config.GRASP_SPIN_AXIS (default "plug_axis", approved by the user 2026-11-22):
              "plug_axis"  (default; approved by the user 2026-11-22) the plug's own axis, (x_c, y_c) = centroid of the tight-fit section
                           (assets.AssemblySpec.axis_xy_m, fit.py): what the spec text says ("spin about the plug's
                           own axis");
              "cad_origin" (option) (x_c, y_c) = (0, 0), the CAD origin: what the spec formula writes (T_c = I);
            the spec formula is to be written as T_hp below.
Plug pose in the TCP frame (TCP = panda_hand + 0.1034 m along hand z):
    T_hp = R_y(dtheta) * T_hp0 * Trans_z(dp) * T_c * R_z(dpsi) * T_c^-1,   T_c = Trans(x_c, y_c, 0)
    T_hp0 = T_tcp_fc * T_plug_fc0^-1,  T_tcp_fc = Trans_z(0.112071 - 0.1034)   (AutoMate's grasp of its fingertip body)
R_y acts in the TCP frame (left factor), Trans_z and the spin in the plug frame (right factors). The spec writes the
spin as R_z(dpsi) and calls it the spin about the plug's own axis; T_c * R_z * T_c^-1 is that rotation about the plug's
axis. For plugs whose CAD origin lies on that axis (x_c = y_c = 0) both switch values give R_z(dpsi). The assemblies
where they differ, with the in-hand shift 2 r sin(dpsi / 2) at the box's largest spin (r = |(x_c, y_c)|), are listed
by scripts/automate/diag/axis_offsets.py (m1b/v6/axis_offsets.json). Since the plug tip is at
the plug's lowest z (insertion direction -z_plug), Trans_z(+dp) moves the plug up relative to the fingers, i.e. the
grip point moves dp towards the tip. Seated TCP pose: H* = T_socket * T_hp^-1 (seated plug pose = socket pose).

Ranges. Base box: dtheta in [-20, 20] deg, dp in [-10, 10] mm, dpsi in [-s, s], s = min(P/4, 15 deg), P = 360/n the
rotational symmetry period of the plug's tight-fit cross-section (fit.py; round: s = 15 deg), n = assets.AssemblySpec.
symmetry_n: "round" for the 16 assemblies of assets.SPIN_ROUND_2026_11_23 (the user's decision 2026-11-23: their
plugs turn freely in the socket, physical check of fit.py), the plug rule's n for every other assembly. The box is shrunk to
the largest box (on a 7 x 7 x 5 grid of the base box, counting grid points) that contains (0, 0, 0) and in which
every grid point is collision-free at H*:
    free(v) = min distance between the socket mesh and {hand, left finger, right finger} collision shapes >= 0.5 mm
              (MARGIN_M; 1 mm until 2026-11-22, relaxed to 0.5 mm with the user's approval of 2026-11-22, to be
              confirmed in simulation: milestone-1b check (iii), see below), the shapes PhysX uses for
              franka_mimic.usd (exported from the USD: hand = convex hull of its
              mesh, each finger = the bounding box of its mesh, the "boundingCube" approximation of the USD; checked
              in simulation by batch_assemblies.py), hand at H*(v), fingers closed
              until they touch the plug mesh (q_f = largest finger opening at which a finger meets the plug,
              bisection to 0.01 mm; both fingers move together as in franka_mimic.usd)
then verified on a 9 x 9 x 9 grid spanning that box; while a verification point collides, all three ranges shrink by
10 % towards 0. Normalised variable u = (v - v_min) / (v_max - v_min). Held-out grasp rule (heldout(u)):
    held out  <=>  there is a variable i with 0.55 <= u_i <= 0.65      (ANY of the three, not all three)
ANY is the rule in use (decided with split "automate_v1", 2026-11-22: 12 train-range grasps with every u outside the
band plus 3 held-out grasps, one per variable in the band). A variable whose range has zero width (v_min = v_max, a
collapsed box face) has no u: normalise returns None for it (not 0.5, which would look like a mid-range draw), and the
held-out band cannot apply to it.
Simulation confirmation of the margin (milestone-1b check (iii), scripts/automate/batch_assemblies.py): at the box's
grid point with the smallest mesh margin, reset, move to H*, hold 5 control steps, and
    pass_iii = move_ok AND completed AND |depth_left| < 0.5 mm AND F_hand/finger-socket <= III_MAX_HAND_FINGER_N
with F_hand/finger-socket = max over the hold of both measures (contact sensor, largest body; contact report, sum over
robot-socket shape pairs). The 1 N threshold is an information-level threshold of this margin check only (it allows
numerical contact noise), not a task criterion (lead's resolution 2026-11-23).
RANGES_FILE (ranges_v5.json for "plug_axis", ranges_v5_cad_origin.json for "cad_origin": the box depends on the spin
axis) is written only by scripts/automate/analyse_assemblies.py (load_ranges(..., compute=True)); everywhere else,
collection workers included, load_ranges only reads it and raises if it is missing.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.spatial.transform import Rotation

from . import assets as A
from . import config as C

TCP_Z = 0.1034
FINGERTIP_Z = 0.112071
FINGER_JOINT_Z = 0.0584
FINGER_OPEN = 0.04
BASE = {"dtheta_deg": (-20.0, 20.0), "dp_m": (-0.010, 0.010)}
SPIN_MAX_DEG = 15.0
HELDOUT_U = (0.55, 0.65)
MARGIN_M = 0.0005         # required hand/finger-socket distance (approved by the user 2026-11-22; was 1 mm): the
                          # collision meshes are proxies of franka_mimic.usd
III_MAX_HAND_FINGER_N = 1.0   # check (iii): hand / finger - socket force allowed at the worst grasp (docstring); the
                              # inclusion gate (collect.py) must use this constant, not its own copy
# Assemblies excluded whatever the box (approved by the user 2026-11-22); ranges() gives them no box with this reason
EXCLUDED = {"00855": "plug wider than the socket collar in the CAD"}


def _T(R=None, p=None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = p
    return T


def Ry(deg):
    return _T(Rotation.from_euler("y", deg, degrees=True).as_matrix())


def Rz(deg):
    return _T(Rotation.from_euler("z", deg, degrees=True).as_matrix())


def Tz(m):
    return _T(p=[0.0, 0.0, m])


def T_hp0(assembly_id: str) -> np.ndarray:
    """AutoMate's grasp: plug pose in the TCP frame."""
    return Tz(FINGERTIP_Z - TCP_Z) @ np.linalg.inv(A.grasp_fingertip_in_plug(assembly_id))


def spin_axis_xy(spec, axis: str | None = None) -> tuple:
    """(x_c, y_c) of the spin axis in the plug frame for the switch config.GRASP_SPIN_AXIS (module docstring)."""
    axis = axis or C.GRASP_SPIN_AXIS
    assert axis in C.GRASP_SPIN_AXES, axis
    return tuple(float(v) for v in spec.axis_xy_m) if axis == "plug_axis" else (0.0, 0.0)


def T_hp(T0: np.ndarray, dtheta_deg: float, dp_m: float, dpsi_deg: float, axis_xy=(0.0, 0.0)) -> np.ndarray:
    """axis_xy: (x_c, y_c) of the spin axis in the plug frame (spin_axis_xy)."""
    Tc = _T(p=[float(axis_xy[0]), float(axis_xy[1]), 0.0])
    return Ry(dtheta_deg) @ T0 @ Tz(dp_m) @ Tc @ Rz(dpsi_deg) @ np.linalg.inv(Tc)


def spin_limit_deg(n) -> float:
    if n in (None, "round"):
        return SPIN_MAX_DEG
    return min(360.0 / n / 4.0, SPIN_MAX_DEG)


class FingerSocketCheck:
    """Collision of hand / fingers with the socket at H*, in the socket (= seated plug) frame."""

    def __init__(self, assembly_id: str):
        import hppfcl
        self.fcl = hppfcl
        load = lambda f: hppfcl.MeshLoader().load(str(f), np.ones(3))
        d = A.ASSET_DIR / "franka_mimic_collision"               # exported by scripts/automate/export_franka_collision.py
        self.hand, self.left, self.right = (load(d / "panda_hand.obj"), load(d / "panda_leftfinger.obj"),
                                            load(d / "panda_rightfinger.obj"))
        self.plug = load(A.fetch(f"{assembly_id}/plug.obj"))
        self.sock = load(A.fetch(f"{assembly_id}/socket.obj"))
        self.T0 = T_hp0(assembly_id)
        self.axis_xy = spin_axis_xy(A.load_spec(assembly_id))
        self.I = hppfcl.Transform3f(np.eye(3), np.zeros(3))

    def _obj(self, geom, T):
        return self.fcl.CollisionObject(geom, self.fcl.Transform3f(T[:3, :3].copy(), T[:3, 3].copy()))

    def _collide(self, geom, T, other) -> bool:
        r = self.fcl.CollisionResult()
        return bool(self.fcl.collide(self._obj(geom, T), self.fcl.CollisionObject(other, self.I),
                                     self.fcl.CollisionRequest(), r))

    def _dist(self, geom, T, other) -> float:
        r = self.fcl.DistanceResult()
        self.fcl.distance(self._obj(geom, T), self.fcl.CollisionObject(other, self.I), self.fcl.DistanceRequest(), r)
        return float(r.min_distance)

    def frames(self, v, q: float):
        Ts_tcp = np.linalg.inv(T_hp(self.T0, *v, axis_xy=self.axis_xy))      # seated: plug frame = socket frame
        Ts_hand = Ts_tcp @ np.linalg.inv(Tz(TCP_Z))
        left = Ts_hand @ _T(p=[0.0, q, FINGER_JOINT_Z])
        right = Ts_hand @ _T(p=[0.0, -q, FINGER_JOINT_Z])          # USD body frames (meshes exported in them)
        return Ts_hand, left, right

    def closing(self, v) -> tuple[float, str]:
        """Finger opening q_f [m] where the closing fingers first touch the plug."""
        qs = []
        for side in (1, 2):
            geom = self.left if side == 1 else self.right
            hit = lambda q: self._collide(geom, self.frames(v, q)[side], self.plug)
            if hit(FINGER_OPEN):
                return FINGER_OPEN, "plug touches the open finger"
            if not hit(0.0):
                qs.append(0.0)
                continue
            lo, hi = 0.0, FINGER_OPEN                              # hit(lo), not hit(hi)
            for _ in range(13):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if hit(mid) else (lo, mid)
            qs.append(hi)
        return max(qs), "ok" if max(qs) > 0 else "fingers close without touching the plug"

    def check(self, v) -> dict:
        q, note = self.closing(v)
        H, L, R = self.frames(v, q)
        parts = {"hand": (self.hand, H), "left": (self.left, L), "right": (self.right, R)}
        hits = {k: self._collide(g, T, self.sock) for k, (g, T) in parts.items()}
        margin = None if any(hits.values()) else min(self._dist(g, T, self.sock) for g, T in parts.values())
        ok = not any(hits.values()) and margin >= MARGIN_M and note != "plug touches the open finger"
        return {"free": ok, "q_f": q, "note": note,
                "hits": [k for k, h in hits.items() if h], "margin_m": margin}


def ranges(assembly_id: str, n_sym, grid=(7, 7, 5), verify=9) -> dict:
    if assembly_id in EXCLUDED:
        return {"n": n_sym, "box": None, "reason": "excluded: " + EXCLUDED[assembly_id], "margin_m": MARGIN_M}
    chk = FingerSocketCheck(assembly_id)
    s = spin_limit_deg(n_sym)
    axes = [np.linspace(*BASE["dtheta_deg"], grid[0]), np.linspace(*BASE["dp_m"], grid[1]), np.linspace(-s, s, grid[2])]
    nom = [len(a) // 2 for a in axes]
    free = np.zeros(grid, bool)
    for idx in itertools.product(*[range(g) for g in grid]):
        free[idx] = chk.check([axes[k][idx[k]] for k in range(3)])["free"]
    out = {"n": n_sym, "margin_m": MARGIN_M, "spin_axis": C.GRASP_SPIN_AXIS,
           "period_deg": None if n_sym in (None, "round") else 360.0 / n_sym, "spin_limit_deg": s,
           "base_free_fraction": float(free.mean()), "nominal_free": bool(free[tuple(nom)])}
    if not free[tuple(nom)]:
        out.update(box=None, reason="nominal grasp collides at H*: " + str(chk.check([0.0, 0.0, 0.0])))
        return out
    best, best_score = None, -1
    for lo in itertools.product(*[range(n + 1) for n in nom]):
        for hi in itertools.product(*[range(n, g) for n, g in zip(nom, grid)]):
            sub = free[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1]
            score = sub.size
            if score > best_score and sub.all():
                best, best_score = (lo, hi), score
    lo, hi = best
    box = [[float(axes[k][lo[k]]), float(axes[k][hi[k]])] for k in range(3)]
    shrinks, worst = 0, None
    while True:
        pts = [np.linspace(b[0], b[1], verify) if b[1] > b[0] else np.array([b[0]]) for b in box]
        bad, worst = None, (np.inf, None)
        for v in itertools.product(*pts):
            r = chk.check(list(v))
            if not r["free"]:
                bad = (list(v), r)
                break
            if r["margin_m"] is not None and r["margin_m"] < worst[0]:
                worst = (r["margin_m"], list(v))
        if bad is None or shrinks >= 10:
            break
        shrinks += 1
        box = [[0.9 * b[0], 0.9 * b[1]] for b in box]
    names = ("dtheta_deg", "dp_m", "dpsi_deg")
    rng = {n_: b for n_, b in zip(names, box)}
    held = {n_: [b[0] + HELDOUT_U[0] * (b[1] - b[0]), b[0] + HELDOUT_U[1] * (b[1] - b[0])] for n_, b in zip(names, box)}
    out.update(box=rng, heldout=held, verified=bad is None, shrinks=shrinks, failing=None if bad is None else bad,
               worst_margin_m=None if worst[1] is None else worst[0], worst_point=worst[1],
               nominal=chk.check([0.0, 0.0, 0.0]))
    return out


def heldout(u: dict) -> bool:
    """The held-out grasp rule of the module docstring: True if 0.55 <= u_i <= 0.65 for ANY grasp variable i; a
    zero-width variable (u_i None) never counts."""
    return any(x is not None and HELDOUT_U[0] <= x <= HELDOUT_U[1] for x in u.values())


def normalise(v: float, lo: float, hi: float) -> float | None:
    """u = (v - v_min) / (v_max - v_min); None for a zero-width range (v_min = v_max), where u is undefined."""
    return (v - lo) / (hi - lo) if hi > lo else None


RANGES_FILES = {"plug_axis": "ranges_v5.json", "cad_origin": "ranges_v5_cad_origin.json"}
RANGES_FILE = RANGES_FILES[C.GRASP_SPIN_AXIS]
# v5 (2026-11-23): n from assets.SPEC_FILE = geometry_v7; ranges_v4 byte for byte except the 16 assemblies of
# assets.SPIN_ROUND_2026_11_23, recomputed by ranges(id, "round") (scripts/automate/diag/spin_round_v7.py; no v5 file
# exists for "cad_origin", which is not an approved switch value); v4 (2026-11-22, round 2): margin 0.5 mm, n and axis
# from geometry_v6 (centroid with the interior rings filled); v3: spin about the geometry_v5 axis; v2: about the CAD
# origin, n from geometry_v4. Older files are never overwritten.


def ranges_file(assembly_id: str):
    return A.ASSET_DIR / assembly_id / RANGES_FILE


def load_ranges(assembly_id: str, compute: bool = False) -> dict:
    """ASSET_DIR/<id>/RANGES_FILE. Read-only unless compute=True (scripts/automate/analyse_assemblies.py only): then
    a missing file is computed with ranges() and written. Collection workers never compute or write it."""
    import json
    f = ranges_file(assembly_id)
    if f.exists():
        return json.loads(f.read_text())
    if not compute:
        raise FileNotFoundError(f"{f} is missing: {RANGES_FILE} is written by scripts/automate/analyse_assemblies.py, "
                                "never by a reader")
    out = ranges(assembly_id, A.load_spec(assembly_id).symmetry_n)
    f.write_text(json.dumps(out, default=float))
    return out
