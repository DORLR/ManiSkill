"""Tight-fit depth, rotational symmetry and shape category of an AutoMate assembly, from cross-sections of the meshes
(no Isaac; trimesh + shapely). Units: metres, degrees. Frame: the assembled frame of plug.obj / socket.obj (seated
plug = socket pose, insertion direction -z, see assets.py).

Sections. S(z) = socket material in the plane at height z, P(h) = plug material at plug-local height h (both shapely
(Multi)Polygons in x, y), on a grid of DZ = 0.2 mm, offset by DZ/2 from the grid origin so that no plane lies on a
mesh face.

Lift sweep (tight-fit depth). With the plug lifted by d along +z (out of the socket, axis kept),
    g(d) = min over z of dist(P(z - d), S(z))        (one-sided lateral gap; 0 if the sections overlap; +inf if no z
                                                      has both sections)
    L    = min { d >= 0 : g(d) > 2 * c_ref },   c_ref = max(c, 0.1 mm),  c = radial clearance per side (assets.py)
i.e. the tight fit ends where the lateral gap first exceeds twice the clearance. d is scanned every 1 mm, then the
first crossing is refined on the 0.2 mm grid. The entrance (start of the tight fit) is at height z_ent = min_z(plug) + L.

Plug axis. The line parallel to plug-frame z through (x_c, y_c), the area-weighted centroid of the tight-fit band
sections P(h) below WITH THEIR INTERIOR RINGS FILLED (union of the polygons' exteriors; axis_xy_m). Holes are not
counted, so that an eccentric bore inside a round sleeve does not move the axis away from the outer surface that the
socket holds (00077: the area centroid with the hole lay 0.30 mm from the outer circle's centre; geometry_v5 used it).
It is the axis of the spin variable (grasp.py, "plug_axis") and of the symmetry test. (Until 2026-11-22 both used
plug-frame z through the CAD origin, which for plugs whose origin lies off the plug labelled round pins n = 1 "keyed"
and made a spin also shift the plug sideways.)

Symmetry. For plug sections P(h) at h = min_z(plug) + L * {0.1, 0.3, 0.5, 0.7, 0.9}, rotation by phi about the plug
axis (through (x_c, y_c)):
    e(phi) = max_h area(P(h) xor Rot(phi) P(h)) / area(P(h))      (full sections, holes included)
    spread(ring, o) = (r_max - r_min) / r_max of the ring sampled every 0.1 mm, r = distance to the point o
    round      if every P(h) is one polygon, spread(exterior, (x_c, y_c)) < 0.025, AND every interior ring with >= 5 %
               of its polygon's filled area has its centroid within c_ref of (x_c, y_c) and spread(ring, its own
               centroid) < 0.025 (a round bore close to the axis; an eccentric or non-round bore keys the part)
               (a tessellated circle with >= 16 sides passes, a 12-gon does not)
    n          = the largest n in 2..12 with e(360/n) < TOL, else 1;    TOL = 0.03
    period P   = 360 / n deg (round: unlimited)
Known limit of the bore rule (audit 2026-11-23): it looks at the plug alone. A bore keys the part only if socket
material inside it blocks rotation; for 00320 (n = 12) and 00863 (n = 8) the bore is empty or fits loosely around a
round post (rotated plug sections do not overlap the socket sections at any angle), so physically they rotate freely.
00081 (n = 5) and 00175 (n = 1) also changed n with this rule, with unchanged boxes.
The user's decision of 2026-11-23 (r4-decisions-06, option 2 on a named list): analyse() keeps the plug rule above for
every assembly; assets.SPIN_ROUND_2026_11_23 (16 assemblies named by the user: 15 with a 12-gon plug and 00863, an
8-gon sleeve; for each the physical check below says "round" with zero overlap) sets symmetry_n = "round" for the
spin box and the spin information columns (assets.mesh_geometry, geometry_v7 / ranges_v5). The category below is not changed by it. The physical rule is a
check, not applied to the other assemblies. Physical symmetry (physical_symmetry below, run for every assembly by
scripts/automate/diag/bore_physical_n.py):
    o(phi) = max_z area(Rot(phi) P(z) & S(z)) / area(P(z))           (full sections, holes included)
    blocked(phi)  <=>  o(phi) > o(0) + TOL_OVERLAP,   TOL_OVERLAP = 1e-4   (o(0): overlap of the CAD seated pose;
                  0.5 % let 01102's square plug count as n = 5: its 72 deg rotation overlaps 0.32 %, rs2 2026-11-23)
    n_phys = "round"  if no phi of the 0.5 deg scan of (0, 360) is blocked,
             else the largest n in 2..12 with not blocked(360 / n), else 1
             None ("unreliable") when o(0) > O0_MAX = 0.01: the seated CAD sections already overlap (a socket
             section filled across its hole, e.g. a mesh that is not watertight), so the test cannot see blocking
             (rs2 2026-11-23: 00256 0.96, 00471 0.50, 00480 1.0, 00597 0.46, 00755 1.0, 00768 0.73, 00855 0.06,
             01092 0.12; 0 for the other 92)

Category (from the section at mid-depth h = min_z(plug) + L/2):
    multi-pin  >= 2 polygons each with >= 2 % of the section area
    sleeve     a polygon has an interior ring (hole) with >= 5 % of its area (the plug surrounds part of the socket)
    flat       long / short side of the minimum rotated rectangle >= 2.5
    round      symmetry "round";  polygonal  n >= 3;  keyed  n in {1, 2}
(first matching rule in this order). Size: equivalent diameter 2 sqrt(A / pi) and the long side of that rectangle.
"""

from __future__ import annotations

import numpy as np

DZ = 0.0002
TOL = 0.03


def _sections(mesh, heights: np.ndarray) -> list:
    """Shapely geometry (world x, y) of the mesh cut by the planes z = heights; empty Polygon where nothing is cut."""
    import shapely
    from shapely import affinity
    from shapely.geometry import Polygon
    out = []
    paths = mesh.section_multiplane(plane_origin=[0.0, 0.0, 0.0], plane_normal=[0.0, 0.0, 1.0], heights=list(heights))
    for p in paths:
        if p is None:
            out.append(Polygon())
            continue
        try:
            polys = [g for g in p.polygons_full if g is not None and g.area > 0]
        except Exception:
            polys = []
        g = shapely.union_all(polys) if polys else Polygon()
        if not g.is_valid:
            g = g.buffer(0)
        T = np.asarray(p.metadata.get("to_3D", np.eye(4)), dtype=np.float64)   # plane frame -> world
        g = affinity.affine_transform(g, [T[0, 0], T[0, 1], T[1, 0], T[1, 1], T[0, 3], T[1, 3]])
        out.append(g)
    return out


TOL_OVERLAP = 1e-4
O0_MAX = 0.01
PHYS_STEP_DEG = 0.5


def physical_symmetry_sections(P: list, S: list, axis_xy, step_deg: float = PHYS_STEP_DEG,
                               tol: float = TOL_OVERLAP) -> dict:
    """Option 2 of the module docstring on given sections: P[i], S[i] plug / socket shapely geometry on the same
    plane i (seated pose), axis_xy the rotation centre -> {"n_phys", "overlap0", "max_overlap", "blocked_deg" (the
    blocked angles of the scan, at most 50 listed), "n_checks": {n: o(360 / n)}}."""
    import shapely
    from shapely import affinity
    pairs = [(p, s) for p, s in zip(P, S) if p is not None and s is not None and not p.is_empty and not s.is_empty
             and p.area > 0]
    if not pairs:
        return {"n_phys": None, "reason": "no plane with both sections"}
    pa = np.array([p for p, _ in pairs], dtype=object)
    sa = np.array([s for _, s in pairs], dtype=object)
    area = np.array([p.area for p, _ in pairs])
    o = (float(axis_xy[0]), float(axis_xy[1]))

    def ov(phi: float) -> float:
        rot = np.array([affinity.rotate(p, phi, origin=o) for p in pa], dtype=object)
        return float(np.max(shapely.area(shapely.intersection(rot, sa)) / area))

    o0 = ov(0.0)
    angles = np.arange(step_deg, 360.0, step_deg)
    vals = np.array([ov(a) for a in angles])
    blocked = angles[vals > o0 + tol]
    checks = {n: ov(360.0 / n) for n in range(12, 1, -1)}
    if o0 > O0_MAX:
        n_phys = None
    elif len(blocked) == 0:
        n_phys = "round"
    else:
        n_phys = next((n for n in range(12, 1, -1) if checks[n] <= o0 + tol), 1)
    return {"n_phys": n_phys, "reliable": bool(o0 <= O0_MAX), "overlap0": o0, "max_overlap": float(vals.max(initial=0.0)),
            "n_blocked": int(len(blocked)), "blocked_deg": [float(a) for a in blocked[:50]],
            "n_checks": {str(n): v for n, v in checks.items()}, "tol": tol, "step_deg": step_deg,
            "n_planes": len(pairs)}


def physical_symmetry(plug, sock, axis_xy, step_deg: float = PHYS_STEP_DEG, tol: float = TOL_OVERLAP) -> dict:
    """physical_symmetry_sections on the DZ grid planes where both seated meshes have material."""
    lo = max(float(plug.bounds[0, 2]), float(sock.bounds[0, 2]))
    hi = min(float(plug.bounds[1, 2]), float(sock.bounds[1, 2]))
    zs = np.arange(lo + DZ / 2, hi, DZ)
    return physical_symmetry_sections(_sections(plug, zs), _sections(sock, zs), axis_xy, step_deg, tol)


def analyse(plug, sock, c_side: float) -> dict:
    import shapely
    from shapely import affinity
    pz0, pz1 = float(plug.bounds[0, 2]), float(plug.bounds[1, 2])
    sz0, sz1 = float(sock.bounds[0, 2]), float(sock.bounds[1, 2])
    base = sz0 + DZ / 2                                          # common grid z_k = base + k DZ (socket and plug)
    ks = np.arange(0, int(np.floor((sz1 - base) / DZ)) + 1)
    zs = base + ks * DZ
    kp = np.arange(int(np.ceil((pz0 - base) / DZ)), int(np.floor((min(pz1, sz1) - base) / DZ)) + 1)
    S = _sections(sock, zs)
    Pl = _sections(plug, base + kp * DZ)
    P = {int(k): g for k, g in zip(kp, Pl)}
    c_ref = max(c_side, 1e-4)

    def gap(d: float) -> float:
        m = int(round(d / DZ))                                  # lift in grid steps
        a, b = [], []
        for i, zi in enumerate(ks):
            g = P.get(int(zi) - m)
            if g is None or g.is_empty or S[i].is_empty:
                continue
            a.append(g)
            b.append(S[i])
        if not a:
            return float("inf")
        return float(np.min(shapely.distance(np.array(a, dtype=object), np.array(b, dtype=object))))

    g0 = gap(0.0)
    d_max = sz1 - pz0 + DZ
    L, coarse = None, []
    d = 0.0
    while d <= d_max + 1e-9:
        gd = gap(d)
        coarse.append((d, gd))
        if gd > 2 * c_ref:
            lo = max(0.0, d - 0.001)
            dd = lo
            while dd <= d + 1e-9:
                if gap(dd) > 2 * c_ref:
                    L = dd
                    break
                dd += DZ
            break
        d += 0.001
    if L is None:
        L = d_max
    # symmetry and category on the tight-fit band
    def plug_at(h):
        k = int(np.clip(round((h - base) / DZ), kp[0], kp[-1])) if len(kp) else None
        return P[k] if k is not None else shapely.Polygon()
    band = [plug_at(pz0 + L * f) for f in (0.1, 0.3, 0.5, 0.7, 0.9)]
    band = [g for g in band if not g.is_empty and g.area > 0]
    filled = [shapely.union_all([shapely.Polygon(p.exterior) for p in getattr(g, "geoms", [g])]) for g in band]
    area = sum(g.area for g in filled)
    xc = float(sum(g.centroid.x * g.area for g in filled) / area) if band else 0.0
    yc = float(sum(g.centroid.y * g.area for g in filled) / area) if band else 0.0

    def err(phi):
        return max(g.symmetric_difference(affinity.rotate(g, phi, origin=(xc, yc))).area / g.area for g in band)

    def spread(ring, o) -> float:
        pts = np.array([ring.interpolate(t).coords[0] for t in np.arange(0.0, ring.length, 1e-4)])
        rr = np.hypot(pts[:, 0] - o[0], pts[:, 1] - o[1])
        return float((rr.max() - rr.min()) / rr.max())

    def radial_spread(g):
        return max(spread(poly.exterior, (xc, yc)) for poly in getattr(g, "geoms", [g]))

    def bores(g) -> list:
        out = []
        for poly in getattr(g, "geoms", [g]):
            a_f = shapely.Polygon(poly.exterior).area
            for r in poly.interiors:
                h = shapely.Polygon(r)
                if h.area >= 0.05 * a_f:
                    cx, cy = h.centroid.coords[0]
                    out.append({"offset_m": float(np.hypot(cx - xc, cy - yc)), "spread": spread(r, (cx, cy)),
                                "area_fraction": float(h.area / a_f)})
        return out

    sym = {"n": None, "round": False, "radial_spread": None, "axis_xy_m": [xc, yc],
           "axis_rule": "area centroid of the band sections with interior rings filled"}
    if band:
        single = all(len(getattr(g, "geoms", [g])) == 1 for g in band)
        spr = max(radial_spread(g) for g in band)
        holes = [b for g in band for b in bores(g)]
        holes_round = all(b["offset_m"] <= c_ref and b["spread"] < 0.025 for b in holes)
        sym.update(radial_spread=spr, bores=holes[:10], bores_round=holes_round)
        if single and spr < 0.025 and holes_round:
            sym.update(round=True, n="round", period_deg=None)
        else:
            n = 1
            for k in range(12, 1, -1):
                if err(360.0 / k) < TOL:
                    n = k
                    break
            sym.update(n=n, period_deg=360.0 / n)
    mid = plug_at(pz0 + L / 2)
    cat = {"category": "unknown"}
    if not mid.is_empty:
        parts = list(getattr(mid, "geoms", [mid]))
        A = mid.area
        big = [g for g in parts if g.area >= 0.02 * A]
        holes = [r for g in big for r in g.interiors if abs(shapely.Polygon(r).area) >= 0.05 * g.area]
        rect = mid.minimum_rotated_rectangle
        xy = np.asarray(rect.exterior.coords)[:4]
        sides = sorted([np.linalg.norm(xy[1] - xy[0]), np.linalg.norm(xy[2] - xy[1])])
        aspect = sides[1] / max(sides[0], 1e-9)
        if len(big) >= 2:
            c = "multi-pin"
        elif holes:
            c = "sleeve"
        elif aspect >= 2.5:
            c = "flat"
        elif sym.get("round"):
            c = "round"
        elif isinstance(sym.get("n"), int) and sym["n"] >= 3:
            c = "polygonal"
        else:
            c = "keyed"
        cen = mid.centroid
        cat = {"category": c, "n_parts": len(big), "aspect": aspect, "equiv_diameter_m": 2 * np.sqrt(A / np.pi),
               "long_side_m": sides[1], "short_side_m": sides[0], "centroid_offset_m": float(np.hypot(cen.x, cen.y))}
    return {"gap0_m": g0, "c_ref_m": c_ref, "L_tight_m": float(L), "entrance_z_m": pz0 + float(L),
            "sweep_coarse": [(round(a, 4), (None if not np.isfinite(b) else round(b, 6))) for a, b in coarse],
            **sym, **cat}
