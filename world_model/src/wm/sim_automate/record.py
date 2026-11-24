"""wm.data.recorder.Recorder, unchanged, on the AutoMate robot.

wm/data/recorder.py imports two ManiSkill-only names at module level: `_copy_pose` from Chongyu's scripted.py (used
only to copy the goal and plug poses before `goal^-1 * plug`) and `Robot` from wm.env.robot (a type annotation).
The Isaac Lab environment has neither ManiSkill nor SAPIEN, so the module is imported here with two stand-ins that
exist only for the duration of the import (sys.modules is restored afterwards):
    _copy_pose -> identity: robot.base.goal_pose / charger.pose return new Pose objects on every access
                  (wm.sim_automate.robot._Base / _Charger), so no copy is needed;
    wm.env.robot.Robot -> wm.sim_automate.robot.Robot.
The recorder's code, keys, dtypes and image schedule are therefore exactly those of wm_v1 (the same check was run
in scripts/automate/smoke_env.py "h").
"""

from __future__ import annotations

import importlib
import sys
import types

_SCRIPTED = "mani_skill.examples.motionplanning.panda.solutions.wminsertion_v2.scripted"
_PARENTS = ("mani_skill", "mani_skill.examples", "mani_skill.examples.motionplanning",
            "mani_skill.examples.motionplanning.panda", "mani_skill.examples.motionplanning.panda.solutions",
            "mani_skill.examples.motionplanning.panda.solutions.wminsertion_v2")


def _load():
    if "wm.data.recorder" in sys.modules:
        return sys.modules["wm.data.recorder"]
    from . import robot as AR
    names = _PARENTS + (_SCRIPTED, "wm.env.robot")
    saved = {n: sys.modules.get(n) for n in names}
    try:
        for n in _PARENTS:
            sys.modules.setdefault(n, types.ModuleType(n))
        scripted = types.ModuleType(_SCRIPTED)
        scripted._copy_pose = lambda pose: pose
        sys.modules[_SCRIPTED] = scripted
        envrobot = types.ModuleType("wm.env.robot")
        envrobot.Robot = AR.Robot
        sys.modules["wm.env.robot"] = envrobot
        mod = importlib.import_module("wm.data.recorder")
    finally:
        for n, m in saved.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m
        env_pkg = sys.modules.get("wm.env")
        if env_pkg is not None and isinstance(getattr(env_pkg, "robot", None), types.ModuleType) \
                and getattr(env_pkg.robot, "__file__", None) is None:
            delattr(env_pkg, "robot")
    return mod


_recorder = _load()
Recorder = _recorder.Recorder
truth_rel = _recorder._truth_rel
