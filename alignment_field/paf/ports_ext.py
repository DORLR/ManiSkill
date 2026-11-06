"""Procedurally generated plug/socket pairs, registered at runtime.

Chongyu's environment accepts only ports A-D (PORT_TYPES, PORT_SPECS,
ROBOT_UID_BY_PORT and one robot subclass per port). Without editing his files,
`register_generated()` adds G00..G{n-1} in memory: it extends the port list
and spec table, creates one fixed-charger robot subclass per shape (same
construction as his B/C/D subclasses) and adds its uid to the lookup table
and the env's supported robots. Must run in every process before gym.make.

Shape distribution (fixed seed): 1-4 rectangular pins, full cross-section
1.2-12 mm x 1.2-7 mm (randomly flat or tall), pin length 14-20 mm, centres
within |y| <= 9 mm, |z| <= 6 mm, holes (pin + 0.5 mm per side) at least
1.5 mm apart; housing contains the pin footprint with >= 2.5 mm margin;
clearance 0.5 mm per side as A-D. A-D themselves are never generated.
"""

from __future__ import annotations

import json
from types import MappingProxyType

import numpy as np

N_GENERATED = 50      # G00..G49: the shape set of v3-v6 (all default shape lists use these)
N_REGISTERED = 150    # G50..G149 added on 2026-11-06 for more training-shape diversity (prefix-stable)
GEN_SEED = 20261105
_DONE = False


def _sample_spec_kwargs(rng):
    while True:
        n = int(rng.choice([1, 2, 3, 4], p=[0.15, 0.4, 0.3, 0.15]))
        pins = []
        for _ in range(n):
            a, b = rng.uniform(0.0006, 0.006), rng.uniform(0.0006, 0.0035)  # flat: wide in y
            if rng.random() < 0.5:
                a, b = b, a  # tall: narrow in y, long in z
            pins.append(((rng.uniform(-0.009, 0.009), rng.uniform(-0.006, 0.006)), (a, b)))
        ok = True
        for i in range(n):
            (yi, zi), (ai, bi) = pins[i]
            for j in range(i):
                (yj, zj), (aj, bj) = pins[j]
                if abs(yi - yj) < ai + aj + 2 * 0.0005 + 0.0015 and abs(zi - zj) < bi + bj + 2 * 0.0005 + 0.0015:
                    ok = False
        if not ok:
            continue
        ext_y = max(abs(y) + a for (y, _), (a, _) in pins) + 0.0025
        ext_z = max(abs(z) + b for (_, z), (_, b) in pins) + 0.0025
        hy = max(ext_y, rng.uniform(0.015, 0.022))
        hz = max(ext_z, rng.uniform(0.012, 0.015))
        if hy > 0.022 or hz > 0.015:
            continue
        return dict(housing_half_size_m=(rng.uniform(0.020, 0.030), hy, hz),
                    socket_half_size_m=(0.010, 0.055, 0.050),
                    pin_half_length_m=rng.uniform(0.007, 0.010),
                    pins=[dict(center_yz_m=c, half_size_yz_m=s) for c, s in pins])


def generated_specs(n=N_GENERATED, seed=GEN_SEED):
    rng = np.random.default_rng(seed)
    return {f"G{i:02d}": _sample_spec_kwargs(rng) for i in range(n)}


def extra_specs(seed=GEN_SEED):
    """The shapes beyond the v3-v6 set (G50..G149); same generator, so G00..G49 are unchanged."""
    allspec = generated_specs(N_REGISTERED, seed)
    return {k: v for i, (k, v) in enumerate(allspec.items()) if i >= N_GENERATED}


def register_generated(n=N_REGISTERED, seed=GEN_SEED):
    global _DONE
    if _DONE:
        return
    from mani_skill.agents.registration import register_agent
    from mani_skill.envs.tasks.tabletop.wminsertion_v2 import env as wenv
    from mani_skill.envs.tasks.tabletop.wminsertion_v2 import ports, robot

    specs = generated_specs(n, seed)
    ports.PORT_TYPES = tuple(ports.PORT_TYPES) + tuple(specs)
    table = dict(ports.PORT_SPECS)
    for name, kw in specs.items():
        table[name] = ports.PortSpec(name, kw["housing_half_size_m"], kw["socket_half_size_m"],
                                     kw["pin_half_length_m"],
                                     tuple(ports.PinSpec(p["center_yz_m"], p["half_size_yz_m"]) for p in kw["pins"]))
        ports.partition_socket(table[name], (0.02, 0.02))  # board must hold the holes at the extreme offset
    ports.PORT_SPECS = MappingProxyType(table)
    for name in specs:
        cls = type(f"PandaWristCamFixedCharger{name}", (robot.PandaWristCamFixedCharger,),
                   {"uid": f"{robot.ROBOT_UID}_port_{name.lower()}", "port_spec": table[name]})
        register_agent()(cls)
        robot.ROBOT_UID_BY_PORT[name] = cls.uid
        wenv.WMInsertionChargerEnvV2.SUPPORTED_ROBOTS.append(cls.uid)
    _DONE = True


def dump(path):
    specs = generated_specs()
    with open(path, "w") as f:
        json.dump(specs, f, indent=1)
