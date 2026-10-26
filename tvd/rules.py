"""Violation rules. Each rule looks at one tracked vehicle per frame and returns
zero or more violations. A track is flagged at most once per violation type."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .geometry import crossing, in_polygon, side

VEHICLES = ("car", "motorcycle", "bus", "truck")


@dataclass
class Track:
    id: int
    cls: str
    conf: float
    bbox: tuple[float, float, float, float]
    last_seen: float
    # (t, anchor) where anchor is the bottom-centre of the box (road contact point)
    history: deque = field(default_factory=lambda: deque(maxlen=60))
    cross_times: dict[str, float] = field(default_factory=dict)
    lane_frames: dict[str, int] = field(default_factory=dict)
    speed_kmh: float | None = None
    flagged: set[str] = field(default_factory=set)
    plate: str | None = None
    plate_conf: float = 0.0
    # helmet model votes over frames: summed confidence for "helmet" / "no_helmet"
    helmet_votes: list[float] = field(default_factory=lambda: [0.0, 0.0])
    helmet_frames: int = 0

    @property
    def anchor(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)


@dataclass
class Violation:
    type: str
    track: Track
    detail: str
    t: float
    speed_kmh: float | None = None


def _pts(line):
    return tuple(map(float, line[0])), tuple(map(float, line[1]))


def _last_move(tr: Track):
    if len(tr.history) < 2:
        return None
    (t0, p0), (t1, p1) = tr.history[-2], tr.history[-1]
    return t0, p0, t1, p1


class RedLightRule:
    type = "red_light"

    def __init__(self, cfg: dict):
        self.a, self.b = _pts(cfg["stop_line"])
        # A point on the side vehicles approach from; only crossings away from it count.
        self.approach_sign = np.sign(side(tuple(cfg["approach_point"]), self.a, self.b))
        self.classes = set(cfg.get("classes", VEHICLES))

    def update(self, tr: Track, light: str) -> list[Violation]:
        if tr.cls not in self.classes or self.type in tr.flagged:
            return []
        mv = _last_move(tr)
        if not mv:
            return []
        t0, p0, t1, p1 = mv
        frac = crossing(p0, p1, self.a, self.b)
        if frac is None or np.sign(side(p0, self.a, self.b)) != self.approach_sign:
            return []
        tr.cross_times["stop"] = t0 + (t1 - t0) * frac
        if light != "red":
            return []
        tr.flagged.add(self.type)
        return [Violation(self.type, tr, "Qua vạch dừng khi đèn đỏ", t1)]


class SpeedRule:
    type = "speeding"

    def __init__(self, cfg: dict):
        self.la = _pts(cfg["line_a"])
        self.lb = _pts(cfg["line_b"])
        self.distance_m = float(cfg["distance_m"])
        self.limit = float(cfg.get("limit_kmh", 50))
        self.tolerance = float(cfg.get("tolerance_kmh", 5))
        self.limits_by_class = cfg.get("limit_by_class", {})
        self.classes = set(cfg.get("classes", VEHICLES))

    def limit_for(self, cls: str) -> float:
        return float(self.limits_by_class.get(cls, self.limit))

    def update(self, tr: Track, light: str) -> list[Violation]:
        if tr.cls not in self.classes or tr.speed_kmh is not None:
            return []
        mv = _last_move(tr)
        if not mv:
            return []
        t0, p0, t1, p1 = mv
        for key, (a, b) in (("A", self.la), ("B", self.lb)):
            frac = crossing(p0, p1, a, b)
            if frac is not None and key not in tr.cross_times:
                tr.cross_times[key] = t0 + (t1 - t0) * frac
        if "A" not in tr.cross_times or "B" not in tr.cross_times:
            return []
        dt = abs(tr.cross_times["B"] - tr.cross_times["A"])
        if dt < 1e-3:
            return []
        tr.speed_kmh = self.distance_m / dt * 3.6
        limit = self.limit_for(tr.cls)
        excess = tr.speed_kmh - limit
        if excess < self.tolerance:
            return []
        tr.flagged.add(self.type)
        return [Violation(self.type, tr,
                          f"{tr.speed_kmh:.0f} km/h, giới hạn {limit:.0f} km/h (+{excess:.0f})",
                          t1, speed_kmh=tr.speed_kmh)]


class WrongLaneRule:
    type = "wrong_lane"

    def __init__(self, cfg: dict):
        self.lanes = [
            {
                "name": ln["name"],
                "poly": np.array(ln["polygon"], dtype=np.float32).reshape(-1, 1, 2),
                "allowed": set(ln["allowed"]),
            }
            for ln in cfg["lanes"]
        ]
        self.min_frames = int(cfg.get("min_frames", 8))

    def update(self, tr: Track, light: str) -> list[Violation]:
        if tr.cls not in VEHICLES or self.type in tr.flagged:
            return []
        for ln in self.lanes:
            if in_polygon(tr.anchor, ln["poly"]):
                n = tr.lane_frames.get(ln["name"], 0) + 1
                tr.lane_frames[ln["name"]] = n
                if tr.cls not in ln["allowed"] and n >= self.min_frames:
                    tr.flagged.add(self.type)
                    return [Violation(self.type, tr, f"{tr.cls} đi trong {ln['name']}", tr.last_seen)]
            else:
                tr.lane_frames[ln["name"]] = 0
        return []


class NoHelmetRule:
    """Flag a motorcycle once enough frames agree that a rider has no helmet.
    Votes come from HelmetChecker, which runs a separate helmet / no_helmet model."""
    type = "no_helmet"

    def __init__(self, cfg: dict):
        self.min_frames = int(cfg.get("min_frames", 3))
        self.min_share = float(cfg.get("min_share", 0.6))

    def update(self, tr: Track, light: str) -> list[Violation]:
        if tr.cls != "motorcycle" or self.type in tr.flagged or tr.helmet_frames < self.min_frames:
            return []
        yes, no = tr.helmet_votes
        share = no / max(yes + no, 1e-6)
        if share < self.min_share:
            return []
        tr.flagged.add(self.type)
        return [Violation(self.type, tr,
                          f"Có người trên xe không đội mũ bảo hiểm ({share:.0%} của {tr.helmet_frames} khung hình)",
                          tr.last_seen)]


class NoTurnRule:
    """Banned turns at an intersection. A vehicle is armed when it crosses `entry_line` coming from
    `approach_point`'s side, then flagged when it crosses the line of a banned exit
    (e.g. the far side of the left-turn road)."""
    type = "no_turn"

    def __init__(self, cfg: dict):
        self.entry = _pts(cfg["entry_line"])
        self.approach_sign = np.sign(side(tuple(cfg["approach_point"]), *self.entry))
        self.exits = [{"name": e["name"], "line": _pts(e["line"]), "classes": set(e.get("classes", VEHICLES))}
                      for e in cfg["banned_exits"]]

    def update(self, tr: Track, light: str) -> list[Violation]:
        if self.type in tr.flagged:
            return []
        mv = _last_move(tr)
        if not mv:
            return []
        t0, p0, t1, p1 = mv
        if "entry" not in tr.cross_times:
            frac = crossing(p0, p1, *self.entry)
            if frac is not None and np.sign(side(p0, *self.entry)) == self.approach_sign:
                tr.cross_times["entry"] = t0 + (t1 - t0) * frac
            return []
        for ex in self.exits:
            if tr.cls in ex["classes"] and crossing(p0, p1, *ex["line"]) is not None:
                tr.flagged.add(self.type)
                return [Violation(self.type, tr, f"{tr.cls} {ex['name']} tại nơi cấm", t1)]
        return []


def build_rules(cfg: dict) -> list:
    rules = []
    r = cfg.get("rules", {})
    if r.get("red_light", {}).get("enabled"):
        rules.append(RedLightRule(r["red_light"]))
    if r.get("speeding", {}).get("enabled"):
        rules.append(SpeedRule(r["speeding"]))
    if r.get("wrong_lane", {}).get("enabled"):
        rules.append(WrongLaneRule(r["wrong_lane"]))
    if r.get("no_helmet", {}).get("enabled"):
        rules.append(NoHelmetRule(r["no_helmet"]))
    if r.get("no_turn", {}).get("enabled"):
        rules.append(NoTurnRule(r["no_turn"]))
    return rules
