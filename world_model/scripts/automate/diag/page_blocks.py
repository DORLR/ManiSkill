"""Per-assembly blocks of the milestone-1b results page (imported by page_text). Numbers from diag/sim, diag/sim/rerun,
diag/sim/rerun2, diag/mesh, diag/logs/lift.log, m1b/batch2, m1b/analysis.json. A record from before 2026-11-22: the
1.25 mm depth rule named here is the old ManiSkill-style rule, replaced on 2026-11-22 by AutoMate's
check_plug_inserted_in_socket as the only success criterion."""


def T(rows, head):
    h = "".join(f"<th>{x}</th>" for x in head)
    b = "".join("<tr>" + "".join(f"<td>{x}</td>" for x in r) + "</tr>" for r in rows)
    return f"<div class='box'><table><tr>{h}</tr>{b}</table></div>"


HEAD = ["run", "depth left mm", "lateral mm", "tilt °", "plug–socket N (peak)", "finger–socket N (peak)", "note"]

SHIFT_ROWS = [
    ["ii_shift +0.534 mm (y)", "29.28", "0.54", "0.08", "7.5", "0", "line moved to the rest offset: still blocked"],
    ["ii_shift +0.3 mm (y)", "29.34", "0.30", "0.09", "7.9", "0", "blocked"],
    ["ii_shift −0.534 mm (y)", "29.39", "0.57", "0.12", "7.8", "0", "blocked"],
]
SHIFT_OPTION = (" (measured: a straight insertion along the axis moved by +0.534 mm is still blocked at the entrance, so the"
                " offset alone does not make it insertable)")

ROWS_00360 = [
    ["i", "0.01", "0.04", "0.01", "0", "0", "pen (trimesh) 0.27, ray parity 0 points"],
    ["ii", "0.01", "0.04", "0.01", "0", "0", "pen (trimesh) 0.24, ray parity 0"],
    ["teleport", "0.00", "0.00", "0.00", "0", "0", "stays seated; ray parity 0.033 mm"],
]

BLOCK_00755 = ("""<p><b>Cause: the approach of check (i) and a limit at the size of the pose error.</b> No overlap in the
CAD at the seated pose (0 points inside by either test). During the 20 mm/s approach of (i) the plug touches the
entrance 2.7 mm before seated (12 N sideways, command 2.7 mm ahead of the hand) and ends 0.058 mm off-centre in a hole
with 0.069 mm clearance per side; 4 plug points are then 0.062 mm inside the socket (signed distance and ray parity
agree), just over the 0.05 mm limit. (ii) at 10 mm/s and the teleport test end with 0 penetration. Across the 89
assemblies that pass (i), the lateral error at the end of (i) is 0.041 mm median, 0.047 mm max, and their
penetration is 0.</p>""" + T([
    ["i (20 mm/s)", "−0.003", "0.058", "0.01", "12 (at 2.7 mm before seated)", "0", "pen 0.062 (both tests)"],
    ["ii (10 mm/s)", "0.01", "0.04", "0.01", "0", "0", "pen 0"],
    ["teleport", "0.00", "0.00", "0.00", "0", "0", "pen 0"],
], HEAD))
OPTIONS_00755 = ["Loosen the (i) penetration limit to 0.07 mm (00755 passes with 0.062 mm; the 89 passing runs have 0).",
                 "Run (i) slower (5 mm/s) or with the (ii) procedure, and keep 0.05 mm.",
                 "Keep and mark as failing (i) only; (ii) passes."]

BLOCK_01026 = ("""<p><b>Cause: both finger boxes land on the socket's outer wall.</b> In the CAD seated pose the box bottoms
are 0.9 mm below the wall top (picture); in the simulator the fingers sit on the wall and the plug stops 0.89 mm
short (passes the old ManiSkill-style rule (depth left &lt; 1.25 mm; replaced on 2026-11-22 by AutoMate's check_plug_inserted_in_socket), fails the 0.5 mm of check (i)). With the fingers opened or the grasp moved by
Δp = −2.0 mm the plug seats to 0.05 mm. The mesh check finds bare clearance at Δp = −1.0 mm and 1 mm at −2.0 mm.
The CAD also has the plug's bottom face on the socket floor (signed distance and ray parity: up to 0.16 mm, 433
points at z = 13.84&ndash;13.87 mm; sections free after a 0.05 mm lift); in the simulator this costs at most the
0.05 mm that remains.</p>""" + T([
    ["i", "0.89", "0.23", "0.07", "7.5 (lateral)", "7.3 (right), 0.4 (left)", ""],
    ["ii", "0.89", "0.22", "0.07", "3.0", "2.5", "done (not blocked)"],
    ["i_open", "0.05", "0.02", "0.01", "2.8", "0.8 (before opening)", ""],
    ["ii_open", "0.05", "0.02", "0.01", "0.4", "2.3 (before opening)", ""],
    ["teleport", "0.89", "0.06", "0.05", "0", "7.3 / 4.8", "pushed 0.89 mm out by the fingers"],
    ["ii_dp-2.0", "0.05", "0.03", "0.01", "0.4", "0", ""],
], HEAD))
OPTIONS_01026 = ["Keep the nominal grasp; (ii) passes, (i) fails (0.89 mm short).",
                 "Re-centre the grasp at Δp₀ = −2.0 mm (−1.0 mm for bare clearance); then (i) would be met (0.05 mm measured with ii_dp-2.0).",
                 "Finer finger collision shape (environment change, not tried)."]

BLOCKS = [
 {"id": "00855", "short": "keyed, c = 0.016 mm, L = 29.65 mm; (i) and (ii) fail, plug stops at the entrance",
  "body": """<p><b>Cause: the CAD meshes overlap.</b> At the seated pose the plug is wider than the socket's collar
by up to 0.53 mm on one side, over the top 6 mm of the socket (z = 23.1&ndash;29.3 mm). Signed distance and ray
parity agree (226 of 23 314 plug points inside, deepest 0.527 mm); cross-sections overlap on 32 planes. The overlap
does not shrink when the plug is lifted along the axis (25.1 mm³ at every lift 0&ndash;2 mm): it is sideways, not a
depth error. Speed and fingers do not matter.</p>""" + T([
      ["i (20 mm/s)", "29.32", "0.31", "1.69", "147 (141 axial)", "0", "stops on the collar"],
      ["i_slow (5 mm/s)", "29.32", "0.30", "1.69", "146", "0", ""],
      ["i_open", "29.56", "7.22", "2.49", "150", "0", "fingers open"],
      ["ii (10 mm/s)", "29.37", "0.01", "0.09", "85", "0", "blocked"],
      ["ii_slow (2 mm/s)", "29.38", "0.03", "0.09", "8", "0", "blocked"],
      ["ii_open", "29.37", "0.00", "0.09", "8", "0", "blocked"],
      ["teleport", "0.07", "0.53", "0.05", "0.9", "0", "placed seated: pushed 0.534 mm sideways (+y socket), then at rest"],
  ] + SHIFT_ROWS, HEAD) + """<p>Placed inside (teleport), the simulator resolves the overlap by pushing the plug 0.534 mm
sideways and it rests 0.07 mm from seated. Starting outside, the plug does not enter along the CAD axis nor along
axes moved by &plusmn;0.534 mm or +0.3 mm. For reference, the socket's SDF collision has resolution 256 over a 40 mm
part (about 0.16 mm per cell) against a 0.016 mm clearance; whether a finer SDF changes the result was not tested.</p>""",
  "figs": [("00855_ii_deepest_zoom.png", "00855 section")],
  "options": ["Mark as infeasible in this setup (no tested straight insertion enters).",
              "Keep it, defining the seated pose as the simulator's rest pose (0.534 mm off the CAD axis)" + SHIFT_OPTION + ".",
              "Test a finer SDF resolution for this socket (environment change, not tried).",
              "Edit the mesh to remove the 0.53 mm overlap (not tried)."]},
 {"id": "00681", "short": "round, c = 0.371 mm, L = 18.39 mm; (i) and (ii) fail about 5 mm short",
  "body": """<p><b>Cause: the left finger's collision box lands on the socket 5 mm before the plug is seated.</b> No
plug-socket overlap in the CAD (0 points inside, no section overlap). With the nominal grasp the finger box
intersects the socket at H* (mesh check); the finger first clears the socket at Δp = −5.5 mm and keeps 1 mm at
Δp = −6.5 mm.</p>""" + T([
      ["i", "5.06", "0.53", "0.40", "37 (lateral)", "43", ""],
      ["ii", "5.08", "0.44", "0.15", "5.4", "5.5", "blocked"],
      ["i_open", "0.02", "0.04", "0.01", "19", "22 (before opening)", "seated"],
      ["ii_open", "0.02", "0.04", "0.01", "5.4", "1.5 (before opening)", "seated"],
      ["teleport", "5.07", "0.55", "0.40", "33", "42", "pushed back out by the finger"],
      ["ii_dp-6.5", "0.02", "0.04", "0.01", "0", "0", "seated"],
  ], HEAD),
  "figs": [("00681_ii_hand_y_zoom.png", "00681 section")],
  "options": ["Keep AutoMate's grasp and mark 00681 as failing (fingers hit the socket).",
              "Re-centre the grasp for this assembly at Δp₀ = −6.5 mm (first grasp with 1 mm clearance; −5.5 mm for bare clearance), then build the range around it.",
              "Use a finer finger collision shape than the bounding box (environment change; would affect all assemblies; not tried)."]},
 {"id": "00271", "short": "round, c = 0.443 mm, L = 21.80 mm; (i) 1.23 mm short, (ii) 1.88 mm short",
  "body": """<p><b>Cause: the left finger's collision box sits on the socket's top plate.</b> In the CAD seated pose
the box bottom is 1.0 mm below the plate top (picture). No plug-socket overlap. Finger clears at Δp = −1.0 mm, 1 mm
clearance at Δp = −2.0 mm.</p>""" + T([
      ["i", "1.23", "0.38", "0.11", "0", "3.6", ""],
      ["ii", "1.88", "0.51", "0.16", "5.7", "5.4", "blocked"],
      ["i_open", "0.01", "0.04", "0.01", "0", "3.4 (before opening)", "seated"],
      ["ii_open", "0.01", "0.04", "0.01", "0.5", "5.1 (before opening)", "seated"],
      ["teleport", "1.04", "0.33", "0.09", "0", "2.9", "pushed back out by the finger"],
      ["ii_dp-2.0", "0.01", "0.04", "0.01", "0", "0", "seated"],
  ], HEAD),
  "figs": [("00271_ii_hand_y_zoom.png", "00271 section")],
  "options": ["Keep and mark as failing.", "Re-centre the grasp at Δp₀ = −2.0 mm (−1.0 mm for bare clearance).",
              "Finer finger collision shape (environment change, not tried)."]},
 {"id": "01092", "short": "round, c = 0.426 mm, L = 12.86 mm; (i) and (ii) fail about 4.4 mm short",
  "body": """<p><b>Two causes.</b> (1) The left finger box hits the socket (41 N); it clears at Δp = −4.5 mm, 1 mm
clearance at −5.5 mm. (2) After removing the finger contact the plug still stops 0.72 mm short: the plug's shoulder
overlaps the socket's top rim (z &asymp; 17.4 mm) in the CAD. Cross-sections on a 0.05 mm grid: the overlap is gone
only after lifting the plug by 0.70 mm; the simulator stops at 0.717 mm, with 1.7&ndash;1.9 N axial on the plug.</p>""" + T([
      ["i", "4.36", "0.50", "0.30", "45", "41", ""],
      ["ii", "4.35", "0.47", "0.16", "16", "16", "blocked"],
      ["i_open", "0.72", "0.27", "0.07", "23 (1.3 at the end)", "23 (before opening)", ""],
      ["ii_open", "0.72", "0.23", "0.06", "6.4 (1.9 at the end)", "2.6 (before opening)", ""],
      ["ii_dp-5.5", "0.72", "0.23", "0.06", "2.1", "0", ""],
  ], HEAD) + """<p>Teleport placed the plug inside the rim overlap and the solver produced 7 600 N; not usable as a
rest test here. The 0.72 mm passes the old ManiSkill-style rule (depth left &lt; 1.25 mm; replaced on 2026-11-22 by AutoMate's check_plug_inserted_in_socket) but not the 0.5 mm of check (i).</p>""",
  "figs": [("01092_ii_open_hand_y_zoom.png", "01092 section")],
  "options": ["Keep and mark as failing.",
              "Re-centre the grasp at Δp₀ = −5.5 mm AND accept a seated pose 0.7 mm above the CAD seat (define seated from the reachable pose, or loosen (i) to |depth left| &lt; 0.75 mm for this check).",
              "Correct L / the seated pose by the measured 0.70 mm (mesh-level change)."]},
 {"id": "01026", "short": "sleeve, round, c = 0.179 mm, L = 28.0 mm; (i) 0.89 mm short, (ii) passes",
  "body": BLOCK_01026,
  "figs": [("01026_ii_hand_y_zoom.png", "01026 section")],
  "options": OPTIONS_01026},
 {"id": "00175", "short": "sleeve, n = 1, c = 0.341 mm, L = 39.96 mm; (i) fails on penetration 0.44 mm, (ii) passes",
  "body": """<p><b>Two separate findings.</b> (1) The 0.44 mm penetration is the false positive of our metric: the
flagged points lie 0.28 mm above a socket face at z = 0.001 mm that has up- and down-facing triangles at the same
height; ray parity: 0 of them inside; no cross-section overlap at any lift. With the depth 0.16 mm, (i) would pass
with a correct metric. (2) The left finger box rests 0.16 mm deep on the socket top (0.6&ndash;2.7 N), which is why
the plug stops 0.16 mm short; it clears at Δp = −0.5 mm, 1 mm clearance at −1.5 mm.</p>""" + T([
      ["i", "0.16", "0.02", "0.01", "0", "3.7", "pen (trimesh) 0.44"],
      ["ii", "0.16", "0.03", "0.01", "0", "2.7", ""],
      ["i_open", "0.01", "0.04", "0.01", "0", "0.6 (before opening)", ""],
      ["teleport", "0.16", "0.05", "0.02", "0", "0.4", "pushed 0.16 mm out by the finger"],
      ["ii_dp-1.5", "0.01", "0.04", "0.01", "0", "0", ""],
  ], HEAD),
  "figs": [("00175_ii_hand_y_zoom.png", "00175 section")],
  "options": ["Fix the penetration metric (bug 1); then (i) passes as it stands (depth 0.16 mm).",
              "Grasp: keep the nominal grasp (fingers touch the socket at 0.6 N) or re-centre at Δp₀ = −1.5 mm so a range exists."]},
 {"id": "00141", "short": "round, c = 0.442 mm, L = 29.45 mm; (i) fails on penetration 13.7 mm, (ii) passes",
  "body": """<p><b>Cause: our penetration metric.</b> The plug is seated (depth left 0.012 mm) with 0 N everywhere.
The 165 "inside" points are at z = 28.8&ndash;42.9 mm, up to 13.7 mm above the socket's top (29.25 mm), which is
impossible. The socket mesh has up- and down-facing triangles at z = 29.246 mm that span the hole; trimesh takes the
sign from the downward one. Ray parity: 0 of 165 inside (0 also at the simulator poses of i, ii); cross-sections:
no overlap at any lift. Separately: no grasp range, because the nominal grasp has 0.88 mm finger-socket margin
(&lt; 1 mm, no contact); 1 mm is reached at Δp = −0.5 mm.</p>""" + T([
      ["i", "0.01", "0.04", "0.01", "0", "0", "pen (trimesh) 13.71, ray parity 0"],
      ["ii", "0.01", "0.03", "0.01", "0", "0", ""],
      ["teleport", "0.00", "0.00", "0.00", "0", "0", "stays seated"],
  ], HEAD),
  "options": ["Fix the penetration metric (bug 1); (i) then passes.",
              "Range: lower the margin to 0.8 mm, or re-centre at Δp₀ = −0.5 mm, or keep without a range."]},
 {"id": "00360", "short": "round, c = 0.389 mm, L = 22.0 mm; (i) fails on penetration 0.27 mm, (ii) passes",
  "body": """<p><b>Cause: our penetration metric.</b> Same mechanism at the socket face z = 0 (up- and down-facing
triangles at one height). Ray parity: none of the 7 flagged CAD points inside; no section overlap. Contact forces 0 N,
depth left 0.01 mm. Range: full base box, (iii) run with 0 N.</p>""" + T(ROWS_00360, HEAD),
  "options": ["Fix the penetration metric (bug 1); (i) then passes."]},
 {"id": "00755", "short": "sleeve, n = 6, c = 0.069 mm, L = 5.4 mm; (i) fails on penetration 0.062 mm, (ii) passes",
  "body": BLOCK_00755,
  "options": OPTIONS_00755},
 {"id": "00597", "short": "round, c = 0 mm, L = 16.2 mm; (i) 4.58 mm short, (ii) passes (0.19 mm)",
  "body": """<p><b>Causes: a small CAD overlap plus the speed of check (i).</b> The plug's shoulder sits inside the socket's
internal step at the seated pose (picture): overlap up to 0.16 mm (signed distance and ray parity agree, 391 points, z = 6.0&ndash;6.1 mm); the ray clearance is 0. The
overlap is gone after a 0.2 mm lift, and every run that seats stops at 0.185&ndash;0.187 mm. Check (i) approaches at
20 mm/s and jams 4.6 mm short with 26 N (24 N sideways, tilt 0.26°); at 5 mm/s the same motion seats. Fingers play no
part (0 N; i_open also jams).</p>""" + T([
      ["i (20 mm/s)", "4.58", "0.11", "0.26", "26", "0", "jammed"],
      ["i_slow (5 mm/s)", "0.19", "0.05", "0.01", "26 (0.4 at the end)", "0", ""],
      ["i_open", "4.60", "0.11", "0.26", "35", "0", "jammed"],
      ["ii (10 mm/s)", "0.19", "0.05", "0.01", "0.7", "0", ""],
      ["ii_slow (2 mm/s)", "0.19", "0.04", "0.01", "0.8", "0", ""],
      ["teleport", "0.19", "0.05", "0.01", "3.9", "0", "pushed 0.185 mm out, then at rest"],
  ], HEAD),
  "figs": [("00597_ii_deepest_zoom.png", "00597 section")],
  "options": ["Run check (i) at 5 mm/s (or with the (ii) procedure); then 00597 passes (i) with 0.19 mm.",
              "Keep (i) as is and mark 00597 as failing.",
              "Note in the task that the reachable seat is 0.19 mm above the CAD seat (interference in the CAD)."]},
 {"id": "01125", "short": "keyed, c = 0.529 mm, L = 7.31 mm; no result in batch2",
  "body": """<p><b>Cause: the Isaac Sim process crashed</b> 46 s into the assembly: carb.tasking assertion "Recursion
not allowed" (Mutex.cpp:103), core dump (log batch2s_1.log, 23:39). Re-run in a fresh process with the unchanged
batch script: (i) pass (depth 0.008 mm, pen 0, 0 N), (ii) pass (0.010 mm), (iii) run at the closest grasp
(−13.3°, 6.7 mm, 3.75°; mesh margin 1.7 mm): 0 N, depth 0.026 mm. The first batch (batch/) also passed it.</p>""",
  "options": ["Use the re-run result (included in the table, marked *).", "Re-run once more to see if the crash repeats."]},
]
