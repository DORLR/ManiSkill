"""Print a summary json from scripts/analyze.py as a compact table."""
import json
import sys

d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/summary.json"))
print(f"{'role':7}{'condition':14}{'method':10}{'n':>5}{'insert':>8}{'95% CI':>15}{'inID':>6}"
      f"{'end mm':>8}{'end deg':>8}  MMD p (end, per fold)  MMD p (start)")
for k, v in d.items():
    role, cond, m = k.split("|")
    if m == "field_minus_none":
        continue
    print(f"{role:7}{cond:14}{m:10}{v['n']:5d}{v['push_success']:8.2f}   [{v['ci'][0]:.2f},{v['ci'][1]:.2f}]"
          f"{(v['in_id'] or 0):6.2f}{(v['end_dist_t_med_mm'] or 0):8.2f}{(v['end_dist_r_med_deg'] or 0):8.2f}  "
          f"{[round(x, 3) for x in v['mmd_p_end_per_fold']]}  {[round(x, 3) for x in v['mmd_p_start_per_fold']]}")
print()
for k, v in d.items():
    if k.endswith("field_minus_none"):
        print(f"{k:40} grasps {v['n_grasps']:3d}  field-none {v['mean']:+.2f}  CI [{v['ci'][0]:+.2f},{v['ci'][1]:+.2f}]")
