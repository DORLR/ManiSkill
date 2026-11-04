"""Pure geometry tests runnable without the simulator or its dependencies."""

import math
import runpy
import unittest
from itertools import product
from pathlib import Path
from types import SimpleNamespace

TASK = (
    Path(__file__).resolve().parents[1]
    / "mani_skill/envs/tasks/tabletop/wminsertion_v2"
)
PORTS = runpy.run_path(str(TASK / "ports.py"))
GEOMETRY = runpy.run_path(str(TASK / "geometry.py"))
get_port_spec = PORTS["get_port_spec"]
partition_socket = PORTS["partition_socket"]
socket_holes = PORTS["socket_holes"]
geometry_from_poses = runpy.run_path(str(TASK / "metrics.py"))["geometry_from_poses"]

IDENTITY = (0., 0., 0., 1., 0., 0., 0.)
# Historical connector_env.py dimensions in millimetres. Each pin entry is
# (center Y, center Z, half width Y, half height Z) in the plug frame.
HISTORICAL = {
    "A": ((20, 15, 12), (10, 50, 50), 16, ((7, 0, .75, 3.2), (-7, 0, .75, 3.2))),
    "B": ((25, 18, 14), (10, 50, 45), 18, ((0, 0, 6, 1.8),)),
    "C": (
        (22, 18, 12),
        (12, 60, 50),
        20,
        ((-7, -3, 1.6, 1.6), (7, -3, 1.6, 1.6), (0, 5, 1.6, 1.6)),
    ),
    "D": ((30, 22, 15), (10, 55, 45), 14, ((-6, 1, 3.2, 1.2), (6, -1, 1.2, 3.2))),
}


def pose(x=0., y=0., z=0., *, axis=0, angle=0.):
    quaternion = [math.cos(angle / 2), 0., 0., 0.]
    quaternion[axis + 1] = math.sin(angle / 2)
    return (x, y, z, *quaternion)


def yz_rectangle(center, half_size):
    y, z = center
    hy, hz = half_size
    return y - hy, y + hy, z - hz, z + hz


class TestPortGeometry(unittest.TestCase):
    def test_historical_plug_and_corresponding_socket_dimensions(self):
        for name, (housing, socket, length, historical_pins) in HISTORICAL.items():
            with self.subTest(port=name):
                spec = get_port_spec(name)
                self.assertEqual(
                    spec.housing_half_size_m, tuple(x / 1000 for x in housing)
                )
                self.assertEqual(
                    spec.socket_half_size_m, tuple(x / 1000 for x in socket)
                )
                self.assertEqual(spec.pin_length_m, length / 1000)
                self.assertEqual(spec.clearance_m, .0005)
                holes = socket_holes(spec, (.020, -.020))
                self.assertEqual(len(holes), len(historical_pins))
                self.assertEqual(len(spec.pin_boxes), len(historical_pins))
                for index, (y, z, hy, hz) in enumerate(historical_pins):
                    hole, box = holes[index], spec.pin_boxes[index]
                    self.assertEqual(hole.pin_index, index)
                    self.assertEqual(box.center_m, (length / 2000, y / 1000, z / 1000))
                    self.assertEqual(
                        box.half_size_m, (length / 2000, hy / 1000, hz / 1000)
                    )
                    self.assertAlmostEqual(hole.center_yz_m[0], (.020 - y / 1000))
                    self.assertAlmostEqual(hole.center_yz_m[1], (-.020 + z / 1000))
                    self.assertAlmostEqual(hole.half_size_yz_m[0], (hy + .5) / 1000)
                    self.assertAlmostEqual(hole.half_size_yz_m[1], (hz + .5) / 1000)

    def test_a_socket_boxes_exactly_match_native_partition(self):
        spec = get_port_spec("A")
        for offset in product((-.020, 0., .020), repeat=2):
            with self.subTest(offset=offset):
                native = GEOMETRY["partition_receptacle"](
                    (.008, .00125, .0037), (.010, .050, .050), .007, offset
                )
                actual = partition_socket(spec, offset)
                self.assertEqual(
                    [(box.name, box.center_m, box.half_size_m) for box in actual],
                    [(box.name, box.center_m, box.half_size_m) for box in native],
                )

    def test_socket_solid_is_exact_complement_of_its_slots(self):
        offsets = ((0., 0.), (.020, -.020), (-.020, .020))
        for name, offset in product(HISTORICAL, offsets):
            with self.subTest(port=name, offset=offset):
                spec = get_port_spec(name)
                rx, ry, rz = spec.socket_half_size_m
                boxes = partition_socket(spec, offset)
                self.assertTrue(boxes)
                for box in boxes:
                    self.assertEqual(box.center_m[0], -rx)
                    self.assertEqual(box.half_size_m[0], rx)
                    self.assertTrue(all(size > 0. for size in box.half_size_m))
                solids = [
                    yz_rectangle(box.center_m[1:], box.half_size_m[1:])
                    for box in boxes
                ]
                apertures = [
                    yz_rectangle(hole.center_yz_m, hole.half_size_yz_m)
                    for hole in socket_holes(spec, offset)
                ]
                rectangles = solids + apertures
                y_edges = {edge for rect in rectangles for edge in rect[:2]}
                z_edges = {edge for rect in rectangles for edge in rect[2:]}
                ys = sorted(y_edges | {-ry, ry})
                zs = sorted(z_edges | {-rz, rz})
                self.assertAlmostEqual(ys[0], -ry)
                self.assertAlmostEqual(ys[-1], ry)
                self.assertAlmostEqual(zs[0], -rz)
                self.assertAlmostEqual(zs[-1], rz)
                # The midpoint of each open cell tests the entire cell.
                for (lo_y, hi_y), (lo_z, hi_z) in product(
                    zip(ys, ys[1:]), zip(zs, zs[1:])
                ):
                    if min(hi_y - lo_y, hi_z - lo_z) < 1e-12:
                        continue
                    y, z = (lo_y + hi_y) / 2, (lo_z + hi_z) / 2
                    solid = any(a < y < b and c < z < d for a, b, c, d in solids)
                    aperture = any(a < y < b and c < z < d for a, b, c, d in apertures)
                    self.assertEqual(solid, not aperture, (name, offset, y, z))

    def test_a_native_geometry_fields_and_threshold_boundaries(self):
        spec = get_port_spec("A")
        aligned = geometry_from_poses(IDENTITY, IDENTITY, spec)
        self.assertEqual(aligned["goal_relative_pose"], list(IDENTITY))
        self.assertEqual(aligned["minimum_required_tip_goal_x_m"], .015)
        self.assertEqual(aligned["position_error_mm"], 0.)
        self.assertEqual(aligned["angle_error_deg"], 0.)
        for pin, y in zip(aligned["pins"], (-.007, .007)):
            self.assertEqual(pin["center_y_m"], y)
            self.assertEqual(pin["min_goal_xyz_m"], [0., y - .00075, -.0032])
            self.assertEqual(pin["max_goal_xyz_m"], [.016, y + .00075, .0032])
            self.assertEqual(pin["min_tip_goal_x_m"], .016)
            self.assertTrue(pin["slot_cross_section_fits"])
        boundaries = ((-.001001, False), (-.001, True), (.0002, True), (.000201, False))
        for x, expected in boundaries:
            with self.subTest(x=x):
                row = geometry_from_poses(pose(x), IDENTITY, spec)
                self.assertEqual(row["depth_ok"], expected)
                self.assertEqual(row["axial_error_mm"], x * 1000)
        for y, expected in ((.00049, True), (.00051, False)):
            row = geometry_from_poses(pose(y=y), IDENTITY, spec)
            self.assertEqual(row["pin_slot_fit"], expected)
            self.assertTrue(row["depth_ok"])

    def test_a_pitched_corners_match_native_corner_rule(self):
        spec = get_port_spec("A")
        angle = math.radians(1.)
        x0, y0, z0 = -.0003, .0001, .00005
        row = geometry_from_poses(pose(x0, y0, z0, axis=1, angle=angle), IDENTITY, spec)
        self.assertAlmostEqual(row["angle_error_deg"], 1.)
        self.assertAlmostEqual(row["position_error_mm"], math.hypot(x0, y0, z0) * 1000)
        for actual, center_y in zip(row["pins"], (-.007, .007)):
            points = [
                (
                    x0 + x * math.cos(angle) + z * math.sin(angle),
                    y0 + center_y + y,
                    z0 - x * math.sin(angle) + z * math.cos(angle),
                )
                for x, y, z in product((0., .016), (-.00075, .00075), (-.0032, .0032))
            ]
            for axis in range(3):
                expected = [point[axis] for point in points]
                self.assertAlmostEqual(actual["min_goal_xyz_m"][axis], min(expected))
                self.assertAlmostEqual(actual["max_goal_xyz_m"][axis], max(expected))
            self.assertAlmostEqual(
                actual["min_tip_goal_x_m"], min(point[0] for point in points[4:])
            )
        self.assertTrue(row["depth_ok"])
        self.assertTrue(row["pin_slot_fit"])

    def test_shifted_rotated_socket_and_clearance(self):
        # Board yaw is +90 degrees, so after Rz(pi) the goal yaw is -90
        # degrees. Board-local (+20, -20) mm translates world X and Z by -20 mm.
        goal = pose(.20, -.17, .08, axis=2, angle=-math.pi / 2)
        for name, (_, _, length, _) in HISTORICAL.items():
            spec = get_port_spec(name)
            for lateral, fits in ((0., True), (.00049, True), (.00051, False)):
                with self.subTest(port=name, lateral=lateral):
                    plug = pose(.20 + lateral, -.17, .08, axis=2, angle=-math.pi / 2)
                    row = geometry_from_poses(plug, goal, spec)
                    self.assertAlmostEqual(row["goal_relative_pose"][1], lateral)
                    self.assertAlmostEqual(row["lateral_error_mm"], lateral * 1000)
                    self.assertAlmostEqual(
                        row["minimum_required_tip_goal_x_m"], (length - 1) / 1000
                    )
                    self.assertEqual(row["pin_slot_fit"], fits)
                    self.assertEqual(row["pin_in_board_slot_fit"], fits)
                    self.assertTrue(row["depth_ok"])
            unshifted_plug = pose(.22, -.17, .10, axis=2, angle=-math.pi / 2)
            self.assertFalse(
                geometry_from_poses(unshifted_plug, goal, spec)["pin_slot_fit"]
            )

    def test_pitch_distinguishes_b_and_c_from_a_tip_threshold(self):
        for name in ("B", "C"):
            with self.subTest(port=name):
                spec = get_port_spec(name)
                pitch = .1
                lowest_z = min(
                    pin.center_yz_m[1] - pin.half_size_yz_m[1] for pin in spec.pins
                )
                # Every root stays in tolerance, but tilt takes the least
                # advanced leading corner below the family-specific depth.
                tilt_loss = spec.pin_length_m * (1 - math.cos(pitch))
                x = -.001 - lowest_z * math.sin(pitch) + tilt_loss / 2
                row = geometry_from_poses(pose(x, axis=1, angle=pitch), IDENTITY, spec)
                min_x = min(pin["min_goal_xyz_m"][0] for pin in row["pins"])
                self.assertGreaterEqual(min_x, -.001)
                min_tip = min(pin["min_tip_goal_x_m"] for pin in row["pins"])
                self.assertGreater(min_tip, .015)
                self.assertLess(min_tip, row["minimum_required_tip_goal_x_m"])
                self.assertFalse(row["depth_ok"])
        short_blade = geometry_from_poses(IDENTITY, IDENTITY, get_port_spec("D"))
        self.assertTrue(short_blade["depth_ok"])
        max_tip = max(pin["min_tip_goal_x_m"] for pin in short_blade["pins"])
        self.assertLess(max_tip, .015)

    def test_rolled_triangular_and_asymmetric_pins_miss_their_slots(self):
        for name in ("C", "D"):
            with self.subTest(port=name):
                row = geometry_from_poses(
                    pose(angle=math.pi), IDENTITY, get_port_spec(name)
                )
                self.assertAlmostEqual(row["angle_error_deg"], 180.)
                self.assertTrue(row["depth_ok"])
                self.assertFalse(row["pin_slot_fit"])
                self.assertFalse(row["pin_in_board_slot_fit"])
                self.assertTrue(
                    all(not pin["slot_cross_section_fits"] for pin in row["pins"])
                )

    def test_yawed_blade_checks_the_part_crossing_the_board_face(self):
        spec = get_port_spec("B")
        yaw = math.radians(6)
        y = -spec.pin_length_m * math.sin(yaw)
        cases = ((-.018, y, True), (-.018, y - .002, False), (-.030, y, None))
        depth = spec.socket_half_size_m[0] * 2
        for (x, lateral, board_fit), front in product(cases, (True, False)):
            with self.subTest(x=x, lateral=lateral, front=front):
                # Reflect the front case across the board midpoint with a
                # half-turn around Z. B's slot is centered and symmetric.
                if front:
                    plug = pose(x, lateral, axis=2, angle=yaw)
                else:
                    plug = pose(depth - x, -lateral, axis=2, angle=yaw + math.pi)
                row = geometry_from_poses(plug, IDENTITY, spec)
                self.assertFalse(row["depth_ok"])
                self.assertFalse(row["pin_slot_fit"])
                self.assertIs(row["pin_in_board_slot_fit"], board_fit)
                pin = row["pins"][0]
                self.assertEqual(pin["intersects_board_slab"], board_fit is not None)
                margin = pin["minimum_board_slab_slot_margin_mm"]
                if board_fit is None:
                    self.assertIsNone(margin)
                else:
                    self.assertEqual(margin > 0, board_fit)

    def test_pose_wrappers_and_quaternion_sign(self):
        spec = get_port_spec("D")
        plug = pose(-.0012, .0001, -.0002, axis=1, angle=.03)
        wrapper = SimpleNamespace(sp=SimpleNamespace(p=plug[:3], q=plug[3:]))
        goal = SimpleNamespace(p=IDENTITY[:3], q=IDENTITY[3:])
        expected = geometry_from_poses(plug, IDENTITY, spec)
        self.assertEqual(geometry_from_poses(wrapper, goal, spec), expected)
        flipped = (*plug[:3], *(-component for component in plug[3:]))
        actual = geometry_from_poses(flipped, IDENTITY, spec)
        self.assertEqual(actual["pins"], expected["pins"])
        self.assertEqual(actual["angle_error_deg"], expected["angle_error_deg"])

    def test_rejects_nonfinite_or_malformed_poses(self):
        for invalid in ((0.,) * 6, (0.,) * 7, (float("nan"), *IDENTITY[1:])):
            with self.subTest(pose=invalid), self.assertRaises(ValueError):
                geometry_from_poses(invalid, IDENTITY, get_port_spec("A"))


if __name__ == "__main__":
    unittest.main()
