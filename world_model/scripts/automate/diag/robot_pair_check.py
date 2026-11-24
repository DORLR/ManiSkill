"""Post-batch check of the robot-pair rule of the contact report (r9-expert-01; information only, no code change).

The rule (checks.check_contact_report, called on every guarded expert step and every transfer leg) raises
ContactReportError when the hand / finger contact sensors read F_sum = F_hand + F_left + F_right > 1 N but PhysX's
contact report has no robot point. If it fired falsely, those trajectories (or a whole grasp at t00) would be dropped
as contact_report_error. Free evidence comes from the milestone-1b batch: (i) and (iii) record, over the same hold,
    F_max = hand_finger_socket_N   the largest sensor force on any ONE of hand, left finger, right finger
    R     = robot_socket_pairs_N   the report's robot-socket pair sum
Since F_sum <= 3 F_max, a row can reach F_sum > 1 N only if F_max > 1/3 N. Listed here (to be reported before the
collector smoke): every (i) / (iii) row with
    F_max > 1/3 N   and   R == 0
plus the counts n(F_max > 1/3 N), n(of those with R == 0), and rows without R (results recorded before the field).
With --acceptance (the smoke's <smoke out>/smoke/acceptance.json; the draw report of check_expert_data under item
(h) info.draws), also prints per assembly (00537 first; r10-expert-01):
    c_A = sum of per_assembly[A].contact_report_error_by_stage   (trajectories, idx != 0: stages label / placement /
                                                                  run; each loses that one trajectory)
    g_A = per_assembly[A].t00_refused_by_cause["contact_report_error"]   (grasps whose t00 reference was refused by
                                                                  the rule; each loses all of that grasp's planned
                                                                  trajectories, 12 in production)
and c = sum_A c_A, g = sum_A g_A ("all" repeats the per-assembly entries and is not added). When the file has neither
key (item h failed before info.draws was written), it says "no draws section in acceptance.json (item h failed?)"
and prints no total.
With --data <smoke out> (the collection folder: <port>/<grasp>/grasp.json), also splits the contact_report_error
cases by rule from their error texts (checks.check_contact_report): r = texts containing "no robot-socket point"
(the robot-pair rule), p = texts containing "no plug-socket point" (the plug rule), over the grasps with
reference_soft_outcome "contact_report_error" (text: reference_failure) and the trajectories with outcome
"contact_report_error" (text: error); any other text is counted as "other".
The one-line result next to r8-expert-01:
    "rows with sensor max > 1/3 N: a, of which b with no robot force; contact_report_error in the smoke:
     c trajectories + g refused t00 grasps (robot rule r, plug rule p)"
(without --acceptance the part after "no robot force" is left out; without --data, r and p read "?").
Missing inputs never read as clean numbers (r11-expert-01 / -02): a batch folder that does not exist or holds no (i) /
(iii) rows prints "no (i) / (iii) rows in <B> (batch not run yet, or wrong folder)" and the line reads "rows with
sensor max > 1/3 N: ?"; a --data folder without any <port>/<grasp>/grasp.json prints "no grasp.json under <data>" and
r and p read "?"; an unreadable --acceptance file prints "acceptance.json unreadable: <error>" and c and g read "?".
When both are given and r + p + other != c + g (they count the same cases of the same smoke), it prints
"WARNING: --data and --acceptance disagree (...): not the same smoke?".
Reads JSON only (no simulator, no wm import); run anywhere, e.g. on rs2 in a capped scope (qx_run_retry: a caps
refusal, exit 99, is tried again every 60 s for up to QX_RETRY_MAX_S, default 3600 s):
    source <B>/code/scripts/automate/caps.sh
    qx_run_retry prep robotpair 2G 100% 0 /usr/bin/python3 -u <B>/code/scripts/automate/diag/robot_pair_check.py <B> \
        [--acceptance <smoke out>/smoke/acceptance.json] [--data <smoke out>]
with B = /mnt/storage3/qingxin/automate/m1b/batch_r3_cpu (the CPU batch, still to run). Exit 0 always (information;
also for a missing batch folder, --data folder or --acceptance file).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

F_MAX_INFO_N = 1.0 / 3.0          # the rule's 1 N on the sum of three bodies, as a bound on the largest one
FIRST = "00537"                    # most IK-unreachable poses (collector smoke list): read first


def rows(batch: Path) -> list[dict]:
    out = []
    for f in sorted(batch.glob("[0-9]*.json")):
        try:
            r = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(r, dict):
            continue
        for part in ("i", "iii"):
            x = r.get(part)
            if not isinstance(x, dict) or "hand_finger_socket_N" not in x:
                continue                                    # missing, or (iii) skipped
            out.append({"assembly": f.stem, "part": part, "hand_finger_socket_N": x.get("hand_finger_socket_N"),
                        "robot_socket_pairs_N": x.get("robot_socket_pairs_N")})
    return out


def report(rs: list[dict]) -> dict:
    num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)  # noqa: E731
    high = [r for r in rs if num(r["hand_finger_socket_N"]) and r["hand_finger_socket_N"] > F_MAX_INFO_N]
    no_robot = [r for r in high if num(r["robot_socket_pairs_N"]) and r["robot_socket_pairs_N"] == 0]
    unknown = [r for r in rs if not num(r["robot_socket_pairs_N"])]
    return {"rows": len(rs), "threshold_N": F_MAX_INFO_N, "sensor_max_above": len(high),
            "of_which_no_robot_force": len(no_robot), "listed": no_robot,
            "rows_without_robot_socket_pairs_N": [(r["assembly"], r["part"]) for r in unknown]}


def find_key(doc, key: str, path: str = "") -> list[tuple[str, object]]:
    found = []
    if isinstance(doc, dict):
        for k, v in doc.items():
            if k == key:
                found.append((path, v))
            found += find_key(v, key, f"{path}/{k}")
    elif isinstance(doc, list):
        for i, v in enumerate(doc):
            found += find_key(v, key, f"{path}/{i}")
    return found


def acceptance_counts(doc) -> dict | None:
    """Per assembly {"trajectories": c_A, "t00_grasps": g_A} from an acceptance.json (module docstring); None when it
    has neither contact_report_error_by_stage nor t00_refused_by_cause."""
    by_stage = find_key(doc, "contact_report_error_by_stage")
    t00 = find_key(doc, "t00_refused_by_cause")
    if not by_stage and not t00:
        return None
    per: dict[str, dict] = {}
    for found, field, take in ((by_stage, "trajectories", lambda v: sum(int(x) for x in v.values())),
                               (t00, "t00_grasps", lambda v: int(v.get("contact_report_error", 0)))):
        for path, v in found:
            if "/per_assembly/" in path and isinstance(v, dict):          # per assembly only ("all" repeats them)
                aid = path.rsplit("/", 1)[-1]
                per.setdefault(aid, {"trajectories": 0, "t00_grasps": 0})[field] += take(v)
    return dict(sorted(per.items(), key=lambda kv: (kv[0] != FIRST, kv[0])))


def rule_counts(data: Path) -> dict:
    """{"robot": r, "plug": p, "other": o, "grasp_files": n} over <data>/*/*/grasp.json (module docstring)."""
    n = {"robot": 0, "plug": 0, "other": 0, "grasp_files": 0}

    def add(text) -> None:
        s = str(text or "")
        n["robot" if "no robot-socket point" in s else "plug" if "no plug-socket point" in s else "other"] += 1

    for f in sorted(data.glob("*/*/grasp.json")):
        try:
            g = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(g, dict):
            continue
        n["grasp_files"] += 1
        if g.get("reference_soft_outcome") == "contact_report_error":
            add(g.get("reference_failure"))
        for t in g.get("trajectories") or []:
            if isinstance(t, dict) and t.get("outcome") == "contact_report_error":
                add(t.get("error"))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch", help="milestone-1b batch folder (<id>.json results)")
    ap.add_argument("--acceptance", help="the collector smoke's smoke/acceptance.json")
    ap.add_argument("--data", help="the collector smoke's output folder (<port>/<grasp>/grasp.json)")
    a = ap.parse_args()
    rep = report(rows(Path(a.batch)))
    no_batch = not Path(a.batch).is_dir() or rep["rows"] == 0           # r11-expert-02: not a clean "0"
    if no_batch:
        print(f"no (i) / (iii) rows in {a.batch} (batch not run yet, or wrong folder)")
    else:
        print(f"(i) / (iii) rows: {rep['rows']}; hand_finger_socket_N > 1/3 N: {rep['sensor_max_above']}, of which "
              f"robot_socket_pairs_N == 0: {rep['of_which_no_robot_force']}")
    for r in rep["listed"]:
        print(f"REPORT BEFORE THE COLLECTOR SMOKE: {r['assembly']} ({r['part']}) hand_finger_socket_N "
              f"{r['hand_finger_socket_N']:.3f} N, robot_socket_pairs_N 0")
    if rep["rows_without_robot_socket_pairs_N"]:
        print(f"rows without robot_socket_pairs_N (recorded before the field): {rep['rows_without_robot_socket_pairs_N']}")
    line = ("rows with sensor max > 1/3 N: ?" if no_batch else
            f"rows with sensor max > 1/3 N: {rep['sensor_max_above']}, of which {rep['of_which_no_robot_force']} with "
            "no robot force")
    rules = None
    if a.data:
        rules = rule_counts(Path(a.data))
        if rules["grasp_files"] == 0:                                   # r11-expert-01: no grasp.json is not "0"
            print(f"no grasp.json under {a.data}")
            rules = None
        else:
            print(f"contact_report_error by rule over {rules['grasp_files']} grasp.json in {a.data}: robot rule "
                  f"{rules['robot']}, plug rule {rules['plug']}, other text {rules['other']}")
        rep["contact_report_error_by_rule"] = rules
    rr = f"robot rule {rules['robot']}, plug rule {rules['plug']}" if rules else "robot rule ?, plug rule ?"
    if a.acceptance:
        try:                                                            # r11-expert-02: exit 0 always
            doc = json.loads(Path(a.acceptance).read_text())
        except (OSError, ValueError) as exc:
            doc = exc
        per = None if isinstance(doc, Exception) else acceptance_counts(doc)
        if isinstance(doc, Exception):
            print(f"acceptance.json unreadable: {doc}")
            line += f"; contact_report_error in the smoke: ? trajectories + ? refused t00 grasps ({rr})"
        elif per is None:
            print("no draws section in acceptance.json (item h failed?)")
        else:
            for aid, v in per.items():
                print(f"contact_report_error {aid}: {v['trajectories']} trajectories, {v['t00_grasps']} refused t00 grasps")
            c = sum(v["trajectories"] for v in per.values())
            g = sum(v["t00_grasps"] for v in per.values())
            rep["contact_report_error"] = {"trajectories": c, "t00_grasps": g, "per_assembly": per}
            if rules and rules["robot"] + rules["plug"] + rules["other"] != c + g:   # the same cases of one smoke
                print(f"WARNING: --data and --acceptance disagree (r+p+o = "
                      f"{rules['robot'] + rules['plug'] + rules['other']}, c+g = {c + g}): not the same smoke?")
            line += f"; contact_report_error in the smoke: {c} trajectories + {g} refused t00 grasps ({rr})"
    print(line)
    print(json.dumps(rep, default=str))


if __name__ == "__main__":
    main()
