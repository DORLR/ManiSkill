"""Pure-stdlib fixed-board aperture geometry and episode factor sampling.

No simulator, NumPy, Torch, or package import is needed to run this file's
mathematical checks: ``python -m mani_skill.envs.tasks.tabletop.wminsertion_v2.geometry``.
All dimensions are metres; boxes use board-local XYZ and half sizes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import random
from typing import Iterable

ACTIVE_GRASP_PITCH_RANGE_DEG = (-45.0, -25.0)
NARROW_GRASP_PITCH_RANGE_DEG = (-37.0, -33.0)
HOLE_OFFSET_LIMIT_M = 0.005
RESET_IK_SEED = 9101
RNG_NAMESPACE = "wm_insertion_charger_randomized_v2_20261102"


def _finite_tuple(values: Iterable[float], length: int, name: str) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain {length} finite numbers") from exc
    if len(result) != length or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain {length} finite numbers")
    return result


def validate_pitch_range(values: Iterable[float]) -> tuple[float, float]:
    low, high = _finite_tuple(values, 2, "pitch_range_deg")
    if (
        not ACTIVE_GRASP_PITCH_RANGE_DEG[0]
        <= low
        < high
        <= ACTIVE_GRASP_PITCH_RANGE_DEG[1]
    ):
        raise ValueError("pitch_range_deg must be ordered inside [-45,-25] degrees")
    return low, high


def validate_hole_offset(values: Iterable[float]) -> tuple[float, float]:
    dy, dz = _finite_tuple(values, 2, "hole_offset_yz_m")
    if max(abs(dy), abs(dz)) > HOLE_OFFSET_LIMIT_M:
        raise ValueError("hole_offset_yz_m must lie inside [-.005,.005] m on each axis")
    return dy, dz


def _factor_rng(episode_seed: int, factor: str) -> random.Random:
    # A separate deterministic stream for each factor preserves paired test
    # cases when either the pitch range or hole-enable flag changes.
    return random.Random(f"{RNG_NAMESPACE}:{factor}:{int(episode_seed)}")


def sample_pitch(
    episode_seed: int, pitch_range_deg=ACTIVE_GRASP_PITCH_RANGE_DEG
) -> float:
    low, high = validate_pitch_range(pitch_range_deg)
    return _factor_rng(episode_seed, "pitch").uniform(low, high)


def sample_hole_offset(episode_seed: int, enabled: bool = True) -> tuple[float, float]:
    if not enabled:
        return 0.0, 0.0
    rng = _factor_rng(episode_seed, "hole_offset_yz")
    return (
        rng.uniform(-HOLE_OFFSET_LIMIT_M, HOLE_OFFSET_LIMIT_M),
        rng.uniform(-HOLE_OFFSET_LIMIT_M, HOLE_OFFSET_LIMIT_M),
    )


@dataclass(frozen=True)
class BoardBox:
    name: str
    center_m: tuple[float, float, float]
    half_size_m: tuple[float, float, float]

    def as_dict(self) -> dict:
        return asdict(self)


def partition_receptacle(
    peg_size, receptacle_size, gap: float, hole_offset_yz_m=(0.0, 0.0)
) -> tuple[BoardBox, ...]:
    """Return the native five-box union with translated slots, fixed exterior.

    ``peg_size`` is the native socket-builder argument: Y/Z already include
    clearance. The pair centres are Y=dy±gap, Z=dz. Its X passage remains
    [-2*receptacle_size[0], 0]. Native overlapping top/side corner boxes are
    retained, including their exact zero-offset construction and ordering.
    """
    px, py, pz = _finite_tuple(peg_size, 3, "peg_size")
    rx, ry, rz = _finite_tuple(receptacle_size, 3, "receptacle_size")
    gap = float(gap)
    dy, dz = validate_hole_offset(hole_offset_yz_m)
    if not all(value > 0 for value in (px, py, pz, rx, ry, rz, gap)):
        raise ValueError("All native geometry dimensions must be positive")
    if not math.isfinite(gap) or gap <= py:
        raise ValueError("gap must be finite and exceed the aperture Y half-size")
    if abs(dy) + gap + py >= ry or abs(dz) + pz >= rz:
        raise ValueError("Both apertures must remain strictly inside the fixed board")

    sy = 0.5 * (ry - py - gap)
    sz = 0.5 * (rz - pz)
    native_dy = py + gap + sy
    native_dz = pz + sz
    return (
        BoardBox("top", (-rx, 0.0, native_dz + dz / 2), (rx, ry, sz - dz / 2)),
        BoardBox("bottom", (-rx, 0.0, -native_dz + dz / 2), (rx, ry, sz + dz / 2)),
        BoardBox("positive_y", (-rx, native_dy + dy / 2, 0.0), (rx, sy - dy / 2, rz)),
        BoardBox("negative_y", (-rx, -native_dy + dy / 2, 0.0), (rx, sy + dy / 2, rz)),
        BoardBox("center_fill", (-rx, dy, dz), (rx, gap - py, pz)),
    )


def mathematical_self_check() -> dict:
    """Check native equality, fixed extents, exact two-slot complement and RNG.

    Cell decomposition uses every box/aperture boundary. On each resulting
    open cell the union-membership function is constant, making the check
    complete for the board-plane area rather than a random point sample.
    """
    peg, board, gap = (0.008, 0.00125, 0.0037), (0.01, 0.05, 0.05), 0.007
    rx, ry, rz = board
    _, py, pz = peg
    sy, sz = 0.5 * (ry - py - gap), 0.5 * (rz - pz)
    native = (
        BoardBox("top", (-rx, 0.0, pz + sz), (rx, ry, sz)),
        BoardBox("bottom", (-rx, 0.0, -(pz + sz)), (rx, ry, sz)),
        BoardBox("positive_y", (-rx, py + gap + sy, 0.0), (rx, sy, rz)),
        BoardBox("negative_y", (-rx, -(py + gap + sy), 0.0), (rx, sy, rz)),
        BoardBox("center_fill", (-rx, 0.0, 0.0), (rx, gap - py, pz)),
    )
    assert (
        partition_receptacle(peg, board, gap) == native
    ), "Zero offset changed native boxes"
    checked_cells = 0
    max_area_error = 0.0
    for dy in (-HOLE_OFFSET_LIMIT_M, 0.0, HOLE_OFFSET_LIMIT_M):
        for dz in (-HOLE_OFFSET_LIMIT_M, 0.0, HOLE_OFFSET_LIMIT_M):
            boxes = partition_receptacle(peg, board, gap, (dy, dz))
            for axis, expected in enumerate(((-2 * rx, 0.0), (-ry, ry), (-rz, rz))):
                extent = (
                    min(b.center_m[axis] - b.half_size_m[axis] for b in boxes),
                    max(b.center_m[axis] + b.half_size_m[axis] for b in boxes),
                )
                assert all(
                    math.isclose(a, e, abs_tol=1e-15) for a, e in zip(extent, expected)
                )
            slots = (
                (dy - gap - py, dy - gap + py, dz - pz, dz + pz),
                (dy + gap - py, dy + gap + py, dz - pz, dz + pz),
            )
            ys = sorted(
                {
                    -ry,
                    ry,
                    *(v for s in slots for v in s[:2]),
                    *(
                        b.center_m[1] + sign * b.half_size_m[1]
                        for b in boxes
                        for sign in (-1, 1)
                    ),
                }
            )
            zs = sorted(
                {
                    -rz,
                    rz,
                    *(v for s in slots for v in s[2:]),
                    *(
                        b.center_m[2] + sign * b.half_size_m[2]
                        for b in boxes
                        for sign in (-1, 1)
                    ),
                }
            )
            solid_area = 0.0
            for ya, yb in zip(ys[:-1], ys[1:]):
                for za, zb in zip(zs[:-1], zs[1:]):
                    if yb - ya < 1e-14 or zb - za < 1e-14:
                        continue
                    y, z = (ya + yb) / 2, (za + zb) / 2
                    in_hole = any(a < y < b and c < z < d for a, b, c, d in slots)
                    solid = any(
                        abs(y - b.center_m[1]) < b.half_size_m[1]
                        and abs(z - b.center_m[2]) < b.half_size_m[2]
                        for b in boxes
                    )
                    assert solid == (
                        not in_hole
                    ), f"Incorrect board complement at {(dy,dz,y,z)}"
                    solid_area += (yb - ya) * (zb - za) * solid
                    checked_cells += 1
            error = abs(solid_area - (4 * ry * rz - 8 * py * pz))
            assert error < 1e-15
            max_area_error = max(max_area_error, error)
    for seed in range(1000):
        wide, narrow = sample_pitch(seed), sample_pitch(
            seed, NARROW_GRASP_PITCH_RANGE_DEG
        )
        offset = sample_hole_offset(seed)
        assert wide == sample_pitch(seed) and offset == sample_hole_offset(seed)
        assert -45.0 <= wide <= -25.0 and -37.0 <= narrow <= -33.0
        assert all(
            -HOLE_OFFSET_LIMIT_M <= value <= HOLE_OFFSET_LIMIT_M for value in offset
        )
        assert sample_hole_offset(seed, False) == (0.0, 0.0)
        assert math.isclose((wide + 45.0) / 20.0, (narrow + 37.0) / 4.0, abs_tol=1e-14)
    return {
        "passed": True,
        "zero_offset_exact_native_boxes": True,
        "extreme_and_center_offsets_checked": 9,
        "complete_board_plane_cells_checked": checked_cells,
        "fixed_board_extents_and_exact_two_slot_complement": True,
        "max_solid_area_error_m2": max_area_error,
        "rng_repeat_and_paired_quantile_seeds_checked": 1000,
        "seed_9101": {
            "pitch_deg": sample_pitch(9101),
            "hole_offset_yz_m": sample_hole_offset(9101),
        },
        "native_socket_builder_peg_half_size_m": peg,
        "board_half_size_m": board,
        "hole_center_spacing_m": 2 * gap,
        "rng_namespace": RNG_NAMESPACE,
    }


if __name__ == "__main__":
    print(json.dumps(mathematical_self_check(), indent=2))
