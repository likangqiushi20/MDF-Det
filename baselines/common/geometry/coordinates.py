"""Coordinate conversions with the WPAFB truth pixel-center convention."""

from __future__ import annotations

import math
from typing import Sequence

from affine import Affine
from rasterio.transform import rowcol, xy


def _as_affine(transform: Affine | Sequence[float]) -> Affine:
    if isinstance(transform, Affine):
        return transform
    values = tuple(transform)
    if len(values) != 6:
        raise ValueError("A serialized affine transform must contain six values")
    return Affine(*values)


def pixel_index_to_geo(
    transform: Affine | Sequence[float],
    x: int,
    y: int,
) -> tuple[float, float]:
    """Return ``(longitude, latitude)`` at the center of integer pixel X/Y."""
    affine = _as_affine(transform)
    longitude, latitude = xy(affine, y, x, offset="center")
    return float(longitude), float(latitude)


def geo_to_fractional_pixel_center(
    transform: Affine | Sequence[float],
    longitude: float,
    latitude: float,
) -> tuple[float, float]:
    """Return fractional X/Y where integers denote pixel centers."""
    affine = _as_affine(transform)
    corner_x, corner_y = (~affine) * (longitude, latitude)
    return float(corner_x - 0.5), float(corner_y - 0.5)


def geo_to_pixel_index(
    transform: Affine | Sequence[float],
    longitude: float,
    latitude: float,
) -> tuple[int, int]:
    """Return the containing integer pixel as WPAFB ``(X, Y)``."""
    affine = _as_affine(transform)
    y, x = rowcol(affine, longitude, latitude, op=math.floor)
    return int(x), int(y)
