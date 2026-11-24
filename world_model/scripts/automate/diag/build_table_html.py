"""HTML rows of the milestone-1b per-assembly table from diag/table_m1b.json (output of table_m1b.py).
    python build_table_html.py table_m1b.json > table_rows.html
"""

from __future__ import annotations

import html
import json
import sys


def f(v, d=1):
    return "–" if v is None else f"{v:.{d}f}"


def rng(b, k, d):
    lo, hi = b[k]
    return f"[{lo:.{d}f}, {hi:.{d}f}]"


def main():
    rows = json.load(open(sys.argv[1]))
    out = []
    for r in rows:
        cls = []
        i, ii, iii = r.get("i"), r.get("ii"), r.get("iii")
        ok = bool(i and ii and i["pass"] and ii["pass"])
        if not ok:
            cls.append("bad")
        shape = r["category"] + ("" if r["category"] == r["category_fit"] else f" ({r['category_fit']})")
        if r["twelve_gon"]:
            shape += ", 12-gon"
        n = "round" if r["n"] == "round" else f"{r['n']} / {r['P_deg']:.0f}°"
        if r["box"]:
            b = r["box"]
            th, dp, ps = rng(b, "dtheta_deg", 0), rng(b, "dp_mm", 1), rng(b, "dpsi_deg", 1)
            if r["shrunk"]:
                w = r["width_fraction"]
                mark = lambda k: " class='sh'" if w[k] < 1 - 1e-9 else ""
            else:
                mark = lambda k: ""
            rg = f"<td{mark('dtheta_deg')}>{th}</td><td{mark('dp_m')}>{dp}</td><td{mark('dpsi_deg')}>{ps}</td>"
        else:
            rg = "<td colspan=3 class='bad'>none: nominal grasp fails the 1 mm check</td>"
        if i:
            ti = ("pass" if i["pass"] else "<b>FAIL</b>") + f" {i['depth_left_mm']:.2f} / {i['max_penetration_mm']:.2f}"
            fN = max(i["hand_finger_socket_N"], i["plug_socket_N"])
            if fN > 0.05:
                ti += f" / {i['plug_socket_N']:.1f}, {i['hand_finger_socket_N']:.1f} N"
        else:
            ti = "<b>no result</b>"
        if ii:
            tii = ("pass" if ii["pass"] else "<b>FAIL</b>") + f" {ii['depth_left_mm']:.2f}" + ("" if ii["push_end"] == "done" else f" ({ii['push_end']})")
        else:
            tii = "<b>no result</b>"
        if iii is None:
            tiii = "–"
        elif "skipped" in iii:
            s = iii["skipped"]
            tiii = "no range" if s == "no box" else "not run (margin " + s.split("margin ")[1].split(" mm")[0] + " mm)"
        else:
            tiii = f"run: {iii['hand_finger_socket_N']:.1f} N, depth {iii['depth_left_mm']:.2f}"
        src = "" if r["source"] == "batch2" else " *"
        out.append(f"<tr class='{' '.join(cls)}'><td>{r['id']}{src}</td><td>{html.escape(shape)}</td><td>{n}</td>"
                   f"<td>{f(r['c_mm'], 3)}</td><td>{f(r['L_mm'])}</td><td>{f(r['d_dis_mm'])}</td><td>{html.escape(r['depth_check'])}</td>"
                   f"<td>{r['N_max']}</td>{rg}<td>{ti}</td><td>{tii}</td><td>{tiii}</td></tr>")
    print("\n".join(out))


if __name__ == "__main__":
    main()
