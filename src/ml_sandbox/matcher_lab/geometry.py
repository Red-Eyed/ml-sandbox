"""Oriented rectangles and exact convex-polygon intersection in Cartesian coordinates."""

from dataclasses import dataclass, replace
from math import cos, isfinite, pi, sin

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Box:
    """A named OBB with width >= height and a counterclockwise long-edge angle.

    Width is the longest side; angle is measured from +x in radians in
    [-pi/2, pi/2). Short-width inputs are re-encoded with swapped dimensions
    and a quarter-turn, preserving their physical rectangle. Squares retain
    their supplied orientation modulo pi. Cartesian coordinates are finite
    and unrestricted by the display viewport.
    """

    id: str
    cx: float
    cy: float
    width: float
    height: float
    angle: float = 0.0

    def __post_init__(self) -> None:
        """Validate geometry and enforce the long-edge representation at construction."""
        if not self.id:
            raise ValueError("Box identifiers must be nonempty")
        if not all(isfinite(v) for v in (self.cx, self.cy, self.width, self.height, self.angle)):
            raise ValueError(f"{self.id}: box parameters must be finite")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"{self.id}: width and height must be positive")

        width, height, angle = self.width, self.height, self.angle
        if width < height:
            width, height = height, width
            angle += pi / 2
        if not -pi / 2 <= angle < pi / 2:
            angle = (angle + pi / 2) % pi - pi / 2
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "angle", angle)

    @property
    def center(self) -> FloatArray:
        """Return the center as a fresh Cartesian vector."""
        return np.array([self.cx, self.cy], dtype=np.float64)

    @property
    def rotation(self) -> FloatArray:
        """Map local column vectors into Cartesian world coordinates."""
        c, s = cos(self.angle), sin(self.angle)
        return np.array([[c, -s], [s, c]], dtype=np.float64)


def corners(box: Box) -> FloatArray:
    """Return four counterclockwise corners, starting at local bottom left."""
    signs = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float64)
    local = signs * np.array([box.width, box.height]) / 2
    return local @ box.rotation.T + box.center


def contains(box: Box, point: FloatArray) -> bool:
    """Test membership in the closed oriented rectangular region."""
    local = (point - box.center) @ box.rotation
    return bool(np.all(np.abs(local) <= np.array([box.width, box.height]) / 2))


def cross(a: FloatArray, b: FloatArray) -> float:
    """Return the signed two-dimensional cross product."""
    return float(a[0] * b[1] - a[1] * b[0])


def clip_polygon(subject: FloatArray, boundary: FloatArray) -> FloatArray:
    """Intersect convex polygons using counterclockwise clipping half-planes."""
    vertices = list(subject)
    for start, end in zip(boundary, np.roll(boundary, -1, axis=0), strict=True):
        if not vertices:
            break
        edge = end - start
        output: list[FloatArray] = []
        previous = vertices[-1]
        previous_distance = cross(edge, previous - start)
        for current in vertices:
            distance = cross(edge, current - start)
            if (distance >= 0) != (previous_distance >= 0):
                fraction = previous_distance / (previous_distance - distance)
                output.append(previous + fraction * (current - previous))
            if distance >= 0:
                output.append(current)
            previous, previous_distance = current, distance
        vertices = output
    return np.asarray(vertices, dtype=np.float64).reshape(-1, 2)


def polygon_area(vertices: FloatArray) -> float:
    """Return unsigned polygon area, including zero for an empty intersection."""
    if len(vertices) < 3:
        return 0.0
    shifted = vertices - vertices[0]
    return abs(sum(cross(a, b) for a, b in zip(shifted, np.roll(shifted, -1, 0)))) / 2


def rotated_iou(a: Box, b: Box) -> float:
    """Compute polygon IoU without substituting axis-aligned bounding boxes.

    Translation and common scaling reduce cancellation. Geometry outside float64
    resolution raises ValueError; dimensions are never silently enlarged.
    """
    scale = max(a.width, a.height, b.width, b.height)
    local_a = replace(a, cx=0.0, cy=0.0, width=a.width / scale, height=a.height / scale)
    local_b = replace(
        b,
        cx=(b.cx - a.cx) / scale,
        cy=(b.cy - a.cy) / scale,
        width=b.width / scale,
        height=b.height / scale,
    )
    va, vb = corners(local_a), corners(local_b)
    if np.any(va.max(axis=0) < vb.min(axis=0)) or np.any(vb.max(axis=0) < va.min(axis=0)):
        return 0.0
    area_a, area_b = local_a.width * local_a.height, local_b.width * local_b.height
    intersection = polygon_area(clip_polygon(va, vb))
    union = area_a + area_b - intersection
    if union <= 0 or not isfinite(union):
        raise ValueError("IoU is outside float64 geometric resolution")
    value = intersection / union
    if not isfinite(value) or value < -1e-10 or value > 1 + 1e-10:
        raise ValueError("IoU is outside float64 geometric resolution")
    return float(np.clip(value, 0.0, 1.0))


def angle_distance(a: float, b: float) -> float:
    """Return the shortest rectangle orientation difference, normalized by pi."""
    return abs((a - b + pi / 2) % pi - pi / 2) / pi


def resize_from_corner(box: Box, index: int, point: FloatArray, minimum: float) -> Box:
    """Move a corner in the original local frame while fixing its opposite corner.

    Crossing the opposite corner stops at the explicit minimum side length.
    If the long side changes, construction swaps dimensions and turns the
    stored angle by a quarter-turn without changing the resized rectangle.
    """
    signs = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float64)[index]
    anchor = corners(box)[(index + 2) % 4]
    local = (point - anchor) @ box.rotation
    dimensions = np.maximum(signs * local, minimum)
    center = anchor + (signs * dimensions / 2) @ box.rotation.T
    return replace(
        box,
        cx=float(center[0]),
        cy=float(center[1]),
        width=float(dimensions[0]),
        height=float(dimensions[1]),
    )
