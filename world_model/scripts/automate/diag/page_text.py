"""Hand-written text of the milestone-1b results page (build_page.py). Every number is from the files named in FILES."""

HEADER = """
<h1>AutoMate parts check (milestone 1b)</h1>
<p class="note"><b>Record from before 2026-11-22.</b> Where this page says "success" it means the old ManiSkill-style
depth rule; since 2026-11-22 the only success criterion is AutoMate's check_plug_inserted_in_socket.</p>
<p class="note">2026-11-22. All 100 AutoMate assemblies, table mount, top-down insertion, plug welded to the hand,
joint-position PD at 20 Hz. Batch run 2026-11-21 (batch2) plus diagnosis runs 2026-11-22. Nothing below is decided;
each failing assembly ends with options for you.</p>
"""

SUMMARY = """
<h2>Summary</h2>
<div class="box"><table>
<tr><th></th><th>batch2 as run</th><th>after re-running 01125</th></tr>
<tr><td>assemblies</td><td>100 (1 without result)</td><td>100</td></tr>
<tr><td>(i) H* reached</td><td>89 pass</td><td>90 pass</td></tr>
<tr><td>(ii) aligned straight insertion</td><td>95 pass</td><td>96 pass</td></tr>
<tr><td>(i) and (ii) both pass</td><td>89</td><td>90</td></tr>
<tr><td>(iii) grasp range found</td><td colspan=2>90 (33 equal to the full base box, 57 smaller); 10 have no range</td></tr>
</table></div>
<p>Causes of the 11 assemblies that did not pass, measured (details below):</p>
<div class="box"><table>
<tr><th>cause</th><th>kind</th><th>assemblies</th></tr>
<tr><td>the penetration number of check (i) is wrong: trimesh's signed distance calls points inside the socket that are
outside (they are up to 13.7 mm above it)</td><td>our pipeline</td><td>00141, 00175, 00360</td></tr>
<tr><td>a finger's collision box touches the socket before the plug is seated (AutoMate's grasp, unchanged)</td>
<td>our pipeline (grasp choice) + asset (box-shaped finger collision)</td><td>00175, 00271, 00681, 01092, 01026</td></tr>
<tr><td>plug and socket overlap in the CAD meshes at the seated pose</td><td>mesh</td>
<td>00855 (0.5 mm, blocks entry), 01092 (0.7 mm at the rim), 00597 (0.16 mm), 01026 (&le; 0.16 mm at the floor)</td></tr>
<tr><td>check (i) approaches at 20 mm/s and jams on the 0.16 mm overlap; at 5 mm/s it seats</td><td>our pipeline (check procedure) + mesh</td><td>00597</td></tr>
<tr><td>penetration limit 0.05 mm is at the size of the simulator's pose error</td><td>our pipeline (threshold)</td><td>00755</td></tr>
<tr><td>Isaac Sim process crashed (carb mutex assertion); re-run passes (i), (ii), (iii)</td><td>simulator (not the assembly)</td><td>01125</td></tr>
</table></div>
<p>Only 00855 cannot be inserted along the CAD axis at all. No assembly was found to be physically impossible for a
plug held without the finger contact: with the fingers opened, every finger-blocked assembly seated (depth left
&le; 0.72 mm). An assembly can appear in more than one row.</p>
"""

CHECKS = r"""
<h2>What the three checks measure</h2>
<ul>
<li><b>(i) H* reached.</b> From the reset pose (plug on the axis, lowest point 12 mm above the highest socket point),
<code>move_to(H*)</code> (waypoints of 2 mm, 2 control steps each, i.e. 20 mm/s), hold 5 steps.
Pass: \(|\text{depth left}| &lt; 0.5\) mm and \(\text{pen} &lt; 0.05\) mm, where pen is the deepest plug surface point inside the
socket mesh at the simulator pose.</li>
<li><b>(ii) aligned straight insertion.</b> Start at \(H^*\) moved back by \(L + 10\) mm along \(a\); command a straight
line at 10 mm/s; stop when the command leads the hand by more than 1.5 mm ("blocked"); hold 10 steps.
Pass (in this 2026-11-21 batch) = the old ManiSkill-style rule: depth left \(&lt; 1.25\) mm, tilt \(&lt; 4^\circ\), lateral \(&lt; 2\) mm; replaced on 2026-11-22 by AutoMate's check_plug_inserted_in_socket (the only success criterion now; the current (ii) pass needs |depth left| &lt; 0.5 mm and AutoMate success at the end).</li>
<li><b>(iii) worst grasp in simulation.</b> See the section on (iii) below.</li>
</ul>
<p class="note">Depth left = how far the plug still is from the CAD seated pose along the axis (simulator truth).
The (i) limits (0.5 mm, 0.05 mm) are stricter than the success criterion; they were set for this check only.</p>
"""

TABLE_INTRO = """
<h2>All 100 assemblies</h2>
<p class="note">c = radial clearance per side (mesh rays); L = tight-fit depth; d_dis = AutoMate's disassembly distance;
depth check compares them (lead-in = AutoMate counts a looser upper part of the socket); n / P = rotational symmetry
and period; N_max = episode limit. Δθ, Δp, Δψ = grasp ranges after the hand/finger-socket collision check
(yellow = smaller than the base box Δθ ∈ [−20, 20]°, Δp ∈ [−10, 10] mm, Δψ ∈ [−min(P/4, 15°), +min(P/4, 15°)]).
(i): depth left / pen in mm, then plug-socket and hand/finger-socket force when non-zero. (ii): depth left at the end.
Red rows = not passing (i) and (ii). * = result from the re-run in a fresh process.</p>
"""

TABLE_NOTES = """
<p class="note">Shape category from the tight-fit cross-section (fit.py): round; non-round = polygonal (n &ge; 3) or keyed
(n = 1, 2); flat (aspect &ge; 2.5); multi-pin; sleeve (the plug surrounds part of the socket). "12-gon" = a 12-fold
section with &lt; 4 % radial spread, possibly a tessellated cylinder.</p>
"""

FAIL_INTRO = """
<p>Tests used (scripts/automate/diag/sim_diag.py, environment code unchanged; every run starts with a reset):
<b>i</b> and <b>ii</b> = the batch checks with a record every control step; <b>i_slow</b> = (i) with 0.5 mm waypoints
(5 mm/s); <b>ii_slow</b> = (ii) at 2 mm/s; <b>i_open / ii_open</b> = fingers opened fully (0.04 m) before the motion;
<b>ii_dp&lt;x&gt;</b> = (ii) with the grasp moved by Δp = x mm (Δθ = Δψ = 0); <b>teleport</b> = joints written at
IK(H*) so the plug sits exactly at the CAD seated pose, then held 20 steps. Forces are the contact-sensor forces
(socket on plug, socket on hand / fingers), peak over the run. The re-runs of (i) and (ii) reproduced the batch values
to 0.01 mm. Mesh tests (no simulator): mesh_diag.py, sd_sign_check.py, sd_face_check.py, sheet_check.py,
lift_overlap.py. Pictures: cut through the insertion axis (plot_sections.py), finger and hand shapes = the PhysX
collision shapes.</p>
"""

from page_blocks import BLOCKS  # noqa: E402

III = """
<h2>Check (iii): what "skipped" means</h2>
<p>The grasp ranges are not found in the simulator. They come from a mesh collision check (grasp.py, hpp-fcl) of the
PhysX collision shapes of the hand (convex hull) and the two fingers (their bounding boxes, the "boundingCube"
approximation in AutoMate's franka_mimic.usd) against socket.obj, with the hand at H* and the fingers closed until they
touch the plug. A grasp \\(v = (\\Delta\\theta, \\Delta p, \\Delta\\psi)\\) is free if</p>
<div class="box">$$\\min_{b \\in \\{\\text{hand},\\,\\text{left},\\,\\text{right}\\}} \\operatorname{dist}\\big(b(v),\\ \\text{socket}\\big) \\;\\ge\\; 1\\ \\text{mm}.$$</div>
<p>This was evaluated for all 100 assemblies on a 7 &times; 7 &times; 5 grid of the base box; the range is the largest
sub-box that contains the nominal grasp (0, 0, 0) with all grid points free, then checked on a 9 &times; 9 &times; 9 grid.
Check (iii) in the simulator then tests only <b>one</b> grasp per assembly: the grid point with the smallest mesh
distance. "Skipped: mesh margin &ge; 3 mm" means that even that closest grasp keeps 3 mm or more between hand/fingers
and socket, so no simulator run was made.</p>
<div class="box"><table>
<tr><th></th><th>assemblies</th></tr>
<tr><td>mesh collision check done</td><td>100</td></tr>
<tr><td>no range: the nominal grasp itself fails the 1 mm rule</td><td>10 (00042, 00141, 00175, 00271, 00553, 00648, 00681, 01026, 01079, 01092)</td></tr>
<tr><td>&nbsp;&nbsp;of these, the finger box intersects the socket</td><td>5 (00175, 00271, 00681, 01026, 01092)</td></tr>
<tr><td>&nbsp;&nbsp;of these, no contact but margin 0.28&ndash;0.88 mm &lt; 1 mm</td><td>5 (00042, 00141, 00553, 00648, 01079)</td></tr>
<tr><td>range equal to the full base box</td><td>33</td></tr>
<tr><td>range smaller than the base box</td><td>57</td></tr>
<tr><td>&nbsp;&nbsp;Δp range smaller</td><td>54 (upper end cut in all 54: moving the fingers towards the tip collides first; 29 of them end at Δp = 0; the lower end is also cut in 00340 and 00471)</td></tr>
<tr><td>&nbsp;&nbsp;Δθ range smaller</td><td>18 (00077: zero width, Δθ = 0 only)</td></tr>
<tr><td>&nbsp;&nbsp;Δψ range smaller</td><td>9 (00614, 00741: zero width, Δψ = 0 only)</td></tr>
<tr><td>simulator check (iii) run (closest grasp, margin &lt; 3 mm)</td><td>58 (57 in batch2 + 01125 re-run)</td></tr>
<tr><td>&nbsp;&nbsp;result</td><td>hand/finger-socket force 0.0 N in all 58; plug seated (depth left &le; 0.045 mm) in all 58; one plug-socket force of 0.10 N (00426)</td></tr>
<tr><td>simulator check (iii) skipped, margin &ge; 3 mm</td><td>32</td></tr>
</table></div>
<p>So: the collision check was done for every assembly, but on meshes, not in the simulator; the simulator confirmed
it at one grasp for 58 assemblies and found no contact there. The 10 assemblies without a range have no simulator
check of the range at all. The width fractions per assembly are in diag/table_m1b.json ("width_fraction").</p>
<p class="note">The task definition says "collision check in simulation". What was built is a mesh check of the
simulator's collision shapes plus one simulator confirmation per assembly. Whether that is acceptable is your call.</p>
"""

BUGS = """
<h2>Bugs and pipeline problems found (not fixed)</h2>
<ol>
<li><b>Penetration metric gives false positives.</b> <code>src/wm/sim_automate/checks.py:45</code>
(<code>penetration</code>) and <code>scripts/automate/analyse_assemblies.py:29</code> (<code>obj_overlap_mm</code>) use
<code>trimesh.proximity.signed_distance</code> (trimesh 4.5.1). It takes the sign from the normal of the closest
triangle when the point projects into it. Some socket meshes have up- and down-facing faces at the same height
(00141: z = 29.246 mm, triangles spanning the hole; 00175: z = 0.001 mm; 00360: z = 0), so the closest triangle is a
tie and the downward one is picked: 165 points of 00141 up to 13.7 mm <i>above</i> the socket were called inside; ray
parity (<code>mesh.contains</code>) says 0 of them are inside, and no cross-section overlaps. Proposed fix: inside =
<code>sock.contains(q)</code> (ray parity), depth = unsigned distance of the inside points; evaluate in chunks of
about 1000 points (below).</li>
<li><b>Penetration query memory.</b> The same call on 20 000+ points at once needed more than 14 GB for 00681, 01026
and 00141 (three diagnosis processes were killed by their own 14 GB limit inside this call; the batch ran under a
40 GB limit). Closest-point queries are per point, so chunking gives identical numbers with a fraction of the memory
(implemented in diag/sim_diag.py).</li>
<li><b>Crash marker left behind.</b> <code>scripts/automate/batch_assemblies.py:136-138</code>: when a process died in
an assembly, the next process writes the error JSON and <code>continue</code>s without removing
<code>&lt;id&gt;.started</code> (01125.started is still in batch2/). Proposed fix: <code>mark.unlink()</code> before
<code>continue</code>.</li>
<li><b>Range rule drops assemblies whose nominal grasp is free but within 1 mm.</b> <code>src/wm/sim_automate/grasp.py:158</code>:
the box must contain (0, 0, 0), and (0, 0, 0) needs margin &ge; 1 mm (<code>MARGIN_M</code>, line 45). Five
assemblies have no contact at the nominal grasp (margin 0.28&ndash;0.88 mm) and pass (i) and (ii), yet get no range.
This is a design choice, listed as an option, not a code error.</li>
<li><b>(iii) is a mesh check, verified at one point.</b> See the (iii) section.</li>
</ol>
"""

FILES = """
<h2>Files</h2>
<ul class="note">
<li>Per-assembly table: rs2:/mnt/storage3/qingxin/automate/m1b/diag/table_m1b.json (diag/table_m1b.py)</li>
<li>Diagnosis runs with per-step traces and final poses: rs2:.../m1b/diag/sim/&lt;id&gt;.json, .../diag/sim/rerun/&lt;id&gt;.json; logs .../diag/logs/</li>
<li>Mesh tests: .../diag/mesh/&lt;id&gt;.json; pictures .../diag/figs/; 01125 re-run .../diag/batch_rerun/01125.json</li>
<li>Scripts (local, uncommitted): ManiSkill-wm/world_model/scripts/automate/diag/ (copied to rs2 under code/automate/world_model/scripts/automate/diag/)</li>
</ul>
"""
