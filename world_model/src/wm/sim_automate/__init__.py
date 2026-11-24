"""Second simulator: AutoMate assemblies (Isaac Lab v2.3.2) behind the same robot interface as wm.sim / wm.env.

    config.py      constants: control (20 Hz joint PD, ManiSkill gains), D405 camera, socket mount (table default)
    assets.py      AutoMate asset files, AssemblySpec (tight-fit depth L, clearance, N_max); no Isaac import
    fit.py         cross-section analysis: tight-fit depth, entrance, rotational symmetry, shape category
    grasp.py       grasp variables (clamp angle, offset and spin along / about the plug axis), per-assembly ranges
                   from a hand/finger-socket collision check at H*, held-out intervals
    frames.py      lateral axes around the insertion direction (approved rule)
    kinematics.py  Pinocchio FK / IK of the Franka (URDF matching franka_mimic.usd); no Isaac import
    scene.py       the Isaac Lab scene (AutoMate assets and PhysX settings, welded plug, joint PD, wrist camera);
                   import only after the Isaac Sim app is running (wm.sim_automate.app.launch)
    robot.py       Robot: the adapter with the attributes wm.collect / wm.online use on wm.env.robot.Robot
    checks.py      simulator-truth helpers for checks (seated criterion, penetration, straight pushes)
Isaac Lab and the AutoMate task source are used unchanged (imported, never edited).
"""
