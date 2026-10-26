"""Validated OBB-to-Gaussian conversion with explicit covariance conventions.

A Gaussian keeps its principal-axis covariance factor, avoiding loss of the thin
axis when a dense rotated covariance is rounded. Dense covariances are display
values, not the input to distance calculations.
"""

from dataclasses import dataclass
from enum import StrEnum
from math import atan2, cos, hypot, isfinite, log, pi, sin, sqrt

import numpy as np

from .geometry import Box, FloatArray


class CovarianceConvention(StrEnum):
    """The shared mapping from rectangular side lengths to Gaussian variances."""

    UNIFORM = "Uniform rectangle: w²/12, h²/12"
    HALF_EXTENT = "Half extent: w²/4, h²/4"

    @property
    def standard_deviation_divisor(self) -> float:
        """Return the side-length divisor used by the covariance factor."""
        match self:
            case CovarianceConvention.UNIFORM:
                return sqrt(12)
            case CovarianceConvention.HALF_EXTENT:
                return 2.0


@dataclass(frozen=True)
class Gaussian2D:
    """A named Gaussian in Cartesian scene units, represented by principal axes.

    Standard deviations are major-axis first. The angle describes that axis in
    radians. Immutable tuples preserve validated values and factor ownership.
    Reject nonfinite values and variances outside float64 representation; never
    add covariance jitter or enlarge a small axis.
    """

    id: str
    mean: tuple[float, float]
    standard_deviations: tuple[float, float]
    angle: float

    def __post_init__(self) -> None:
        """Validate the factor representation independently of a dense determinant."""
        major, minor = self.standard_deviations
        if not self.id or not all(isfinite(v) for v in (*self.mean, major, minor, self.angle)):
            raise ValueError("Gaussian identity and finite parameters are required")
        if not major >= minor > 0:
            raise ValueError("Gaussian standard deviations must be positive and major-axis first")
        if not all(0 < value * value < float("inf") for value in (major, minor)):
            raise ValueError("Gaussian variances are outside float64 representation")

    @property
    def factor(self) -> FloatArray:
        """Return a fresh A=R diag(std) such that covariance=A Aᵀ."""
        c, s = cos(self.angle), sin(self.angle)
        return np.array([[c, -s], [s, c]], dtype=np.float64) * self.standard_deviations

    @property
    def covariance(self) -> FloatArray:
        """Return a dense display covariance; extreme anisotropy can lose its small eigenvalue."""
        factor = self.factor
        return factor @ factor.T

    @property
    def log_determinant(self) -> float:
        """Return log(det(covariance)) from positive axis scales without forming a determinant."""
        return 2 * sum(log(value) for value in self.standard_deviations)


def box_to_gaussian(
    box: Box,
    *,
    convention: CovarianceConvention = CovarianceConvention.UNIFORM,
) -> Gaussian2D:
    """Convert a Cartesian OBB to a Gaussian without modifying its geometry.

    Canonical major-axis ordering preserves width/height-swap equivalence.
    Square boxes yield isotropic Gaussians with no recoverable orientation.
    Input dimensions and centers must use the same physical coordinate system.
    """
    major, minor, angle = box.width, box.height, box.angle
    if major < minor:
        major, minor, angle = minor, major, angle + pi / 2
    angle = 0.0 if major == minor else angle % pi
    divisor = convention.standard_deviation_divisor
    return Gaussian2D(box.id, (box.cx, box.cy), (major / divisor, minor / divisor), angle)


@dataclass(frozen=True)
class NormalizedBox:
    """A named detector record with cx, cy, w, h and shifted cosine/sine in [0,1].

    The angle encoding is ((cos(theta)+1)/2, (sin(theta)+1)/2). Coordinates
    retain the caller's axis convention; the adapter does not flip the y axis.
    """

    id: str
    cx: float
    cy: float
    width: float
    height: float
    cosine: float
    sine: float

    def __post_init__(self) -> None:
        """Validate the external normalized record once, before physical conversion."""
        values = (self.cx, self.cy, self.width, self.height, self.cosine, self.sine)
        if not self.id or not all(isfinite(v) and 0 <= v <= 1 for v in values):
            raise ValueError("Normalized box needs an identity and finite parameters in [0,1]")
        if self.width == 0 or self.height == 0:
            raise ValueError("Normalized box dimensions must be positive")


def normalized_box_to_gaussian(
    box: NormalizedBox,
    *,
    image_width: float,
    image_height: float,
    convention: CovarianceConvention = CovarianceConvention.UNIFORM,
) -> Gaussian2D:
    """Decode a detector box using explicit image dimensions, then convert it.

    Undo the angle shift and normalize the direction. Reject the undefined
    (0.5,0.5) encoding. Denormalize dimensions before rotating so nonsquare
    images preserve physical geometry. Positive dimensions are required.
    """
    if any(not isfinite(v) or v <= 0 for v in (image_width, image_height)):
        raise ValueError("Image dimensions must be finite and positive")
    cosine, sine = 2 * box.cosine - 1, 2 * box.sine - 1
    length = hypot(cosine, sine)
    if length == 0:
        raise ValueError("Normalized cosine/sine direction is undefined")
    angle = atan2(sine / length, cosine / length)
    physical = Box(
        box.id,
        box.cx * image_width,
        box.cy * image_height,
        box.width * image_width,
        box.height * image_height,
        angle,
    )
    return box_to_gaussian(physical, convention=convention)


def relative_rotation(a: Gaussian2D, b: Gaussian2D) -> FloatArray:
    """Express b's principal axes in a's frame, retaining exact equal-angle identity."""
    angle = b.angle - a.angle
    c, s = cos(angle), sin(angle)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def nonnegative_distance(value: float, *, name: str, roundoff_scale: float = 1.0) -> float:
    """Reject invalid distances; remove only negative floating-point roundoff."""
    tolerance = 64 * np.finfo(np.float64).eps * max(1.0, roundoff_scale)
    if not isfinite(value) or value < -tolerance:
        raise ValueError(f"{name} is outside float64 resolution")
    return max(value, 0.0)
