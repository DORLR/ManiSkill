"""Matched historical A/B/C/D port geometry in Charger-v2 coordinates.

Dimensions are metres. The plug origin is at the pin roots; plug +X points
toward the tips. The socket face is board-local X=0 and its solid lies at
negative X. The aligned goal is board * Translate(0, dy, dz) * Rz(pi).
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType


GEOMETRY_SCHEMA = "charger_ports_abcd_geometry_20261103_v1"
PORT_TYPES = ("A", "B", "C", "D")
CLEARANCE_M = 0.0005
GRIP_HEIGHT_OFFSET_M = 0.0015


def _finite_number(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite numeric data")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite numeric data") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite numeric data")
    return number


def _dimensions(values, length, name, *, positive=False):
    try:
        result = tuple(_finite_number(value, name) for value in values)
    except TypeError as error:
        raise ValueError(f"{name} must contain {length} numbers") from error
    if len(result) != length or positive and any(value <= 0. for value in result):
        raise ValueError(f"Invalid {name}: {result}")
    return result


@dataclass(frozen=True)
class PinSpec:
    center_yz_m: tuple[float, float]
    half_size_yz_m: tuple[float, float]

    def __post_init__(self):
        object.__setattr__(self, "center_yz_m", _dimensions(self.center_yz_m, 2, "pin center"))
        object.__setattr__(self, "half_size_yz_m", _dimensions(self.half_size_yz_m, 2, "pin half size", positive=True))

    def as_dict(self):
        return {"center_yz_m": list(self.center_yz_m), "half_size_yz_m": list(self.half_size_yz_m)}


@dataclass(frozen=True)
class BoxSpec:
    name: str
    center_m: tuple[float, float, float]
    half_size_m: tuple[float, float, float]

    def as_dict(self):
        return {"name": self.name, "center_m": list(self.center_m), "half_size_m": list(self.half_size_m)}


@dataclass(frozen=True)
class SocketHole:
    pin_index: int
    center_yz_m: tuple[float, float]
    half_size_yz_m: tuple[float, float]

    def as_dict(self):
        return {"pin_index": self.pin_index, "center_yz_m": list(self.center_yz_m),
                "half_size_yz_m": list(self.half_size_yz_m)}


@dataclass(frozen=True)
class PortSpec:
    port_type: str
    housing_half_size_m: tuple[float, float, float]
    socket_half_size_m: tuple[float, float, float]
    pin_half_length_m: float
    pins: tuple[PinSpec, ...]
    clearance_m: float = CLEARANCE_M

    def __post_init__(self):
        if self.port_type not in PORT_TYPES:
            raise ValueError(f"Unsupported port type {self.port_type!r}")
        for name in ("housing_half_size_m", "socket_half_size_m"):
            object.__setattr__(self, name, _dimensions(getattr(self, name), 3, name, positive=True))
        length = _finite_number(self.pin_half_length_m, "pin half length")
        if length <= 0.:
            raise ValueError("Pin half length must be positive")
        object.__setattr__(self, "pin_half_length_m", length)
        pins = tuple(self.pins)
        if not pins or any(not isinstance(pin, PinSpec) for pin in pins):
            raise ValueError("Expected a nonempty sequence of PinSpec records")
        object.__setattr__(self, "pins", pins)
        clearance = _finite_number(self.clearance_m, "socket clearance")
        if clearance != CLEARANCE_M:
            raise ValueError("Historical A/B/C/D use 0.5 mm clearance per side")
        object.__setattr__(self, "clearance_m", clearance)

    @property
    def pin_length_m(self):
        return 2.*self.pin_half_length_m

    @property
    def pin_boxes(self):
        x = self.pin_half_length_m
        return tuple(BoxSpec(f"pin_{index}", (x, *pin.center_yz_m), (x, *pin.half_size_yz_m))
                     for index, pin in enumerate(self.pins))

    @property
    def housing_box(self):
        return BoxSpec("housing", (-self.housing_half_size_m[0], 0., 0.), self.housing_half_size_m)

    def grip_center(self, grip_fraction=0.5, *, pad_half_extent_x_m=0.):
        """Return the requested pad center from rear (0) toward pin root (1).

        A caller may pass the pad's projected longitudinal half extent in plug
        coordinates. The center and supplied footprint must lie inside the
        housing. A zero extent validates just the center; reset still audits
        actual finger/table clearance and contact for the chosen pitch.
        """
        fraction = _finite_number(grip_fraction, "grip fraction")
        extent = _finite_number(pad_half_extent_x_m, "pad half extent")
        if not 0. < fraction < 1. or extent < 0.:
            raise ValueError("Grip fraction must be inside (0,1) and pad half extent nonnegative")
        rear = -2.*self.housing_half_size_m[0]
        x = rear*(1.-fraction)
        if x-extent <= rear or x+extent >= 0.:
            raise ValueError("Grasp footprint reaches beyond the usable housing interior")
        return (x, 0., GRIP_HEIGHT_OFFSET_M)

    def as_dict(self):
        return {"schema": GEOMETRY_SCHEMA, "port_type": self.port_type,
                "housing_half_size_m": list(self.housing_half_size_m),
                "socket_half_size_m": list(self.socket_half_size_m),
                "pin_half_length_m": self.pin_half_length_m, "pin_length_m": self.pin_length_m,
                "clearance_m": self.clearance_m, "pins": [pin.as_dict() for pin in self.pins]}

    @property
    def sha256(self):
        encoded = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()


# Exact geometry of the entries in connector_env.py CONNECTORS v3.
# Keep pin order so each simulated pin and matching aperture share an index.
PORT_SPECS = MappingProxyType({
    "A": PortSpec("A", (.020, .015, .012), (.010, .050, .050), .008, (
        PinSpec((.007, 0.), (.00075, .0032)),
        PinSpec((-.007, 0.), (.00075, .0032)),
    )),
    "B": PortSpec("B", (.025, .018, .014), (.010, .050, .045), .009, (
        PinSpec((0., 0.), (.006, .0018)),
    )),
    "C": PortSpec("C", (.022, .018, .012), (.012, .060, .050), .010, (
        PinSpec((-.007, -.003), (.0016, .0016)),
        PinSpec((.007, -.003), (.0016, .0016)),
        PinSpec((0., .005), (.0016, .0016)),
    )),
    "D": PortSpec("D", (.030, .022, .015), (.010, .055, .045), .007, (
        PinSpec((-.006, .001), (.0032, .0012)),
        PinSpec((.006, -.001), (.0012, .0032)),
    )),
})


def get_port_spec(port_type):
    if not isinstance(port_type, str) or port_type not in PORT_SPECS:
        raise ValueError(f"Expected one of {PORT_TYPES}, got {port_type!r}")
    return PORT_SPECS[port_type]


def socket_holes(spec, offset_yz_m=(0., 0.)):
    """Return board-local holes in matched pin order, with +0.5 mm per side."""
    dy, dz = _dimensions(offset_yz_m, 2, "hole offset")
    _, ry, rz = spec.socket_half_size_m
    holes = []
    for index, pin in enumerate(spec.pins):
        y, z = pin.center_yz_m
        hy, hz = (size+spec.clearance_m for size in pin.half_size_yz_m)
        hole = SocketHole(index, (dy-y, dz+z), (hy, hz))
        cy, cz = hole.center_yz_m
        if abs(cy)+hy >= ry or abs(cz)+hz >= rz:
            raise ValueError("Hole must lie strictly inside the board exterior")
        for other in holes:
            oy, oz = other.center_yz_m
            ohy, ohz = other.half_size_yz_m
            if abs(cy-oy) < hy+ohy and abs(cz-oz) < hz+ohz:
                raise ValueError("Matched apertures overlap")
        holes.append(hole)
    return tuple(holes)


def _native_a_partition(spec, offset_yz_m):
    # Same construction, arithmetic order, and names as Randomized-v2 A.
    dy, dz = offset_yz_m
    rx, ry, rz = spec.socket_half_size_m
    gap = spec.pins[0].center_yz_m[0]
    py, pz = (size+spec.clearance_m for size in spec.pins[0].half_size_yz_m)
    sy, sz = .5*(ry-py-gap), .5*(rz-pz)
    native_y, native_z = py+gap+sy, pz+sz
    return (
        BoxSpec("top", (-rx, 0., native_z+dz/2), (rx, ry, sz-dz/2)),
        BoxSpec("bottom", (-rx, 0., -native_z+dz/2), (rx, ry, sz+dz/2)),
        BoxSpec("positive_y", (-rx, native_y+dy/2, 0.), (rx, sy-dy/2, rz)),
        BoxSpec("negative_y", (-rx, -native_y+dy/2, 0.), (rx, sy+dy/2, rz)),
        BoxSpec("center_fill", (-rx, dy, dz), (rx, gap-py, pz)),
    )


def _board_edges(half_extent, hole_edges):
    # D's shared upper Z edge can differ by one ULP after translating the
    # two holes. Coalesce only roundoff-sized differences between hole edges.
    edges = []
    for value in sorted(hole_edges):
        if not edges or value-edges[-1] > 1e-15:
            edges.append(value)
    return [-half_extent, *edges, half_extent]


def partition_socket(spec, offset_yz_m=(0., 0.)):
    """Return the full board solid, using the same boxes for collision and RGB."""
    offset = _dimensions(offset_yz_m, 2, "hole offset")
    holes = socket_holes(spec, offset)
    if spec == PORT_SPECS["A"]:
        return _native_a_partition(spec, offset)

    rx, ry, rz = spec.socket_half_size_m
    y_edges, z_edges = [], []
    for hole in holes:
        y, z = hole.center_yz_m
        hy, hz = hole.half_size_yz_m
        y_edges.extend((y-hy, y+hy))
        z_edges.extend((z-hz, z+hz))
    ys, zs = _board_edges(ry, y_edges), _board_edges(rz, z_edges)
    boxes = []
    for iy, (y0, y1) in enumerate(zip(ys, ys[1:])):
        for iz, (z0, z1) in enumerate(zip(zs, zs[1:])):
            y, z = (y0+y1)/2, (z0+z1)/2
            if any(abs(y-hole.center_yz_m[0]) < hole.half_size_yz_m[0]
                   and abs(z-hole.center_yz_m[1]) < hole.half_size_yz_m[1] for hole in holes):
                continue
            boxes.append(BoxSpec(f"tile_y{iy}_z{iz}", (-rx, y, z), (rx, (y1-y0)/2, (z1-z0)/2)))
    return tuple(boxes)
