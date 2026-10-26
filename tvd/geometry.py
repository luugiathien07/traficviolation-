"""Small 2D helpers: which side of a line a point is on, and segment crossings."""
from __future__ import annotations

import cv2
import numpy as np

Point = tuple[float, float]


def side(p: Point, a: Point, b: Point) -> float:
    """Signed area: >0 on one side of line a->b, <0 on the other, 0 on the line."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def crossing(p0: Point, p1: Point, a: Point, b: Point) -> float | None:
    """If the motion p0->p1 crosses segment a-b, return the fraction (0..1] of the
    motion at which it crosses; otherwise None."""
    s0, s1 = side(p0, a, b), side(p1, a, b)
    if s0 == 0 or (s0 > 0) == (s1 > 0):
        return None
    # The segment a-b must also be crossed within its extent.
    t0, t1 = side(a, p0, p1), side(b, p0, p1)
    if (t0 > 0) == (t1 > 0) and t0 != 0 and t1 != 0:
        return None
    return s0 / (s0 - s1)


def in_polygon(p: Point, poly: np.ndarray) -> bool:
    return cv2.pointPolygonTest(poly, (float(p[0]), float(p[1])), False) >= 0
