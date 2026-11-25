"""Port plan T2 (CPU, the Isaac Lab python, no app): the AutoMate side of the online attempts imports without ManiSkill.

    /mnt/storage3/qingxin/envs/isaaclab/bin/python world_model/tests/test_online_port_isaac_import.py

  1 wm.online.shared320, attempt320, automate320, agent320 (torch, wm320, encoders320: transformers / torchvision for
    resnet34 / segformer) and scripts/online/run_wm320_automate.py (with run_wm320_attempts and eval_policy320) import;
    afterwards no mani_skill, sapien or mplib module is loaded, and neither is isaaclab / omni (the app is not started:
    the adapter imports Isaac Lab only inside the functions that need a scene)
  2 common320.simulator_of_robot: an instance of the REAL wm.sim_automate.robot.Robot class (object.__new__, no scene)
    is "automate", one of a class from wm.env.robot is "maniskill". wm.sim_automate.robot imports Isaac Lab
    (scene.py) at module level; if that import fails without the app, this part says so and is covered by the smoke
    (T7: run_attempt builds AutomateTester on the live Robot, which refuses any robot that simulator_of_robot does
    not call "automate")
  3 shared320 / attempt320 / automate320 hold no top-level import of wm.collect, wm.env.robot / expert, wm.data.recorder
    or wm.sim.env_adapter (source check, so a later edit cannot slip one in)
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WM = HERE.parent
sys.path.insert(0, str(WM / "src"))
sys.path.insert(0, str(WM / "scripts" / "online"))
FORBIDDEN = ("mani_skill", "sapien", "mplib")


def loaded(prefixes) -> list[str]:
    return sorted(m for m in sys.modules if any(m == p or m.startswith(p + ".") for p in prefixes))


def main():
    import wm.online.shared320  # noqa: F401
    import wm.online.attempt320  # noqa: F401
    import wm.online.automate320  # noqa: F401
    import wm.online.agent320  # noqa: F401
    import run_wm320_automate  # noqa: F401
    bad = loaded(FORBIDDEN)
    assert not bad, f"ManiSkill-side modules loaded: {bad[:10]}"
    isaac = loaded(("isaaclab", "omni", "isaacsim", "pxr"))
    assert not isaac, f"Isaac modules loaded without the app: {isaac[:10]}"
    print("T2.1 OK: shared320, attempt320, automate320, agent320, run_wm320_automate import; no mani_skill / sapien / "
          "mplib / Isaac module loaded", flush=True)
    from wm.online.common320 import simulator_of_robot
    ms = type("Robot", (), {"__module__": "wm.env.robot"})
    assert simulator_of_robot(object.__new__(ms)) == "maniskill"
    try:
        from wm.sim_automate.robot import Robot
    except Exception as exc:                                       # Isaac Lab without the app
        print(f"T2.2 PARTIAL: wm.sim_automate.robot does not import without the app ({type(exc).__name__}: "
              f"{str(exc)[:200]}); the real class is checked in the smoke T7 (AutomateTester refuses anything else); "
              "a wm.env.robot class is 'maniskill'", flush=True)
    else:
        assert simulator_of_robot(object.__new__(Robot)) == "automate"
        print("T2.2 OK: simulator_of_robot(real wm.sim_automate Robot) = automate, (wm.env.robot class) = maniskill", flush=True)
    for f in ("shared320.py", "attempt320.py", "automate320.py"):
        tree = ast.parse((WM / "src" / "wm" / "online" / f).read_text())
        for n in tree.body:
            names = []
            if isinstance(n, ast.ImportFrom):
                names = [f"{'.' * n.level}{n.module or ''}"] + [f"{'.' * n.level}{n.module or ''}.{a.name}" for a in n.names]
            elif isinstance(n, ast.Import):
                names = [a.name for a in n.names]
            for x in names:
                assert not any(k in x for k in ("collect", "env.robot", "env.expert", "data.recorder", "env_adapter",
                                                "mani_skill", "sapien", "mplib", "isaaclab", "sim_automate.robot",
                                                "sim_automate.scene", "sim_automate.expert")), (f, x)
    print("T2.3 OK: no simulator import at module level in shared320 / attempt320 / automate320", flush=True)


if __name__ == "__main__":
    main()
