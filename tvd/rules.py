"""Violation rules. Each rule looks at one tracked vehicle per frame and returns
zero or more violations. A track is flagged at most once per violation type."""
from __future__ import annotations

from collections import Counter, deque
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
    # latest helmet-model box of this motorcycle with its riders (the evidence crop must show their heads)
    rider_box: tuple[float, float, float, float] | None = None
    # detected class per frame, counted: the class used by zone rules is the majority
    cls_votes: Counter = field(default_factory=Counter)
    # zone rules: (t, x, y) where the track entered a zone, frames inside, zones it was seen outside of
    zone_entry: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    zone_frames: dict[str, int] = field(default_factory=dict)
    was_outside: set[str] = field(default_factory=set)
    # next video time to try reading this vehicle's plate when plate.read_all is on
    plate_next: float = 0.0
    # overload: frames the riders were counted, frames with too many, rider counts seen, bike + riders box
    rider_frames: int = 0
    overload_frames: int = 0
    rider_counts: Counter = field(default_factory=Counter)
    overload_box: tuple[float, float, float, float] | None = None

    def stable_cls(self, min_share: float) -> str | None:
        """The class seen in at least min_share of at least 5 frames, else None (not sure yet)."""
        total = sum(self.cls_votes.values())
        if total < 5:
            return None
        cls, n = self.cls_votes.most_common(1)[0]
        return cls if n >= min_share * total else None

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


class SignGate:
    """Remembers when each traffic sign was last seen, so a rule part tied to a sign (`require_sign`) is
    only enforced while that sign is in place. Names are the sign model's classes (models/vn_signs.pt).
    A fixed camera sees its signs all the time; `latch_seconds` bridges frames where a passing truck
    hides them.

    Prohibition signs (QCVN 41:2024) the model recognises and the rule each one drives:
    - one way prohibition (P.102 cấm đi ngược chiều) -> wrong_way / no_entry for every vehicle
    - no car entry/turning (P.103a-c cấm ô tô) -> no_entry for car, bus, truck
    - no motorbike entry/turning (P.104, P.111a) -> no_entry for motorcycle
    - no truck entry/turning (P.106a-b cấm xe tải) -> no_entry for truck
    - no turn left / no turn right / no u turn / no u and left turn / no u and right turn
      (P.123a-b, P.124a-f) -> no_turn exits
    - no parking (P.131a-c cấm đỗ) / no stopping and parking (P.130 cấm dừng và đỗ) -> no_parking
    - speed limit (P.127) -> speeding with the limit written on the sign (entered per camera: the model
      has one class for every limit)
    - vehicle permission lane / vehicle and speed permission lane (R.412, R.415) -> wrong_lane
    Not enforced: weight / height limit (P.115, P.117: need a scale or height sensor), "other prohibition"
    (one class for many different signs) and "no more prohibition" (ends a restriction; the zone polygon is
    drawn to stop there).
    """
    # vehicle classes a sign bans when a no_entry zone does not list them itself
    BANS = {
        "one way prohibition": VEHICLES,
        "no car entry/turning": ("car", "bus", "truck"),
        "no motorbike entry/turning": ("motorcycle",),
        "no truck entry/turning": ("truck",),
    }

    def __init__(self, latch_seconds: float = 30):
        self.latch = float(latch_seconds)
        self.seen: dict[str, float] = {}

    def update(self, visible, t: float) -> None:
        for s in visible:
            self.seen[s] = t

    def active(self, required, t: float) -> bool:
        """No sign required, or one of them seen within latch_seconds."""
        return not required or any(s in self.seen and t - self.seen[s] <= self.latch for s in required)


def _pts(line):
    return tuple(map(float, line[0])), tuple(map(float, line[1]))


def _last_move(tr: Track):
    if len(tr.history) < 2:
        return None
    (t0, p0), (t1, p1) = tr.history[-2], tr.history[-1]
    return t0, p0, t1, p1


class RedLightRule:
    type = "red_light"
    fixed_camera = True  # needs lines / zones drawn for a camera that stands still (see motion.py)

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
    fixed_camera = True

    def __init__(self, cfg: dict):
        self.la = _pts(cfg["line_a"])
        self.lb = _pts(cfg["line_b"])
        self.distance_m = float(cfg["distance_m"])
        self.limit = float(cfg.get("limit_kmh", 50))
        self.tolerance = float(cfg.get("tolerance_kmh", 5))
        self.limits_by_class = cfg.get("limit_by_class", {})
        self.classes = set(cfg.get("classes", VEHICLES))
        # limit from a speed-limit sign while it is seen; otherwise default_limit_kmh (TT 38/2024), or no check
        self.signs = list(cfg.get("require_sign", []))
        self.default_limit = cfg.get("default_limit_kmh")
        self.latch = float(cfg.get("latch_seconds", 30))
        self.gate = SignGate(self.latch)
        self.now = 0.0
        # number read from the speed-limit sign ("speed limit 50" from SignDetector) and when it was last seen
        self.sign_kmh: int | None = None
        self.sign_seen_at = 0.0

    def limit_for(self, cls: str) -> float:
        return float(self.limits_by_class.get(cls, self.limit))

    def signs_seen(self, visible, t: float) -> None:
        self.gate.update(visible, t)
        self.now = t
        for s in visible:
            if s.startswith("speed limit ") and s[12:].isdigit():
                self.sign_kmh, self.sign_seen_at = int(s[12:]), t

    def _limit_now(self, cls: str) -> float | None:
        """The limit in force: the number read from the sign; else the configured one while the sign is seen
        (per class); else the default one without a sign; or None for no check."""
        if self.sign_kmh is not None and self.now - self.sign_seen_at <= self.latch:
            return float(self.sign_kmh)
        if self.gate.active(self.signs, self.now):
            return self.limit_for(cls)
        if self.default_limit is None:
            return None
        return float(self.limits_by_class.get("default", {}).get(cls, self.default_limit))

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
        limit = self._limit_now(tr.cls)
        if limit is None:
            return []
        excess = tr.speed_kmh - limit
        if excess < self.tolerance:
            return []
        tr.flagged.add(self.type)
        source = "" if not self.signs else " theo biển" if self.gate.active(self.signs, self.now) else " mặc định TT 38/2024"
        return [Violation(self.type, tr,
                          f"{tr.speed_kmh:.0f} km/h, giới hạn {limit:.0f} km/h{source} (+{excess:.0f})",
                          t1, speed_kmh=tr.speed_kmh)]


class WrongLaneRule:
    type = "wrong_lane"
    fixed_camera = True

    def __init__(self, cfg: dict):
        self.lanes = [
            {
                "name": ln["name"],
                "poly": np.array(ln["polygon"], dtype=np.float32).reshape(-1, 1, 2),
                "allowed": set(ln["allowed"]),
                "signs": list(ln.get("require_sign", [])),
            }
            for ln in cfg["lanes"]
        ]
        self.min_frames = int(cfg.get("min_frames", 8))
        self.gate = SignGate(cfg.get("latch_seconds", 30))

    def signs_seen(self, visible, t: float) -> None:
        self.gate.update(visible, t)

    def update(self, tr: Track, light: str) -> list[Violation]:
        if tr.cls not in VEHICLES or self.type in tr.flagged:
            return []
        for ln in self.lanes:
            if in_polygon(tr.anchor, ln["poly"]):
                n = tr.lane_frames.get(ln["name"], 0) + 1
                tr.lane_frames[ln["name"]] = n
                if tr.cls not in ln["allowed"] and n >= self.min_frames and self.gate.active(ln["signs"], tr.last_seen):
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
        # 3 frames at 60% flagged riders who wore a helmet: ask for more and clearer agreement
        self.min_frames = int(cfg.get("min_frames", 6))
        self.min_share = float(cfg.get("min_share", 0.75))

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
    fixed_camera = True

    def __init__(self, cfg: dict):
        self.entry = _pts(cfg["entry_line"])
        self.approach_sign = np.sign(side(tuple(cfg["approach_point"]), *self.entry))
        self.exits = [{"name": e["name"], "line": _pts(e["line"]), "classes": set(e.get("classes", VEHICLES)),
                       "signs": list(e.get("require_sign", []))}
                      for e in cfg["banned_exits"]]
        self.gate = SignGate(cfg.get("latch_seconds", 30))

    def signs_seen(self, visible, t: float) -> None:
        self.gate.update(visible, t)

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
            if tr.cls in ex["classes"] and self.gate.active(ex["signs"], t1) and crossing(p0, p1, *ex["line"]) is not None:
                tr.flagged.add(self.type)
                return [Violation(self.type, tr, f"{tr.cls} {ex['name']} tại nơi cấm", t1)]
        return []


def _cover(inner, outer) -> float:
    """Share of box `inner` that lies inside box `outer`."""
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    return ix * iy / max(1.0, (inner[2] - inner[0]) * (inner[3] - inner[1]))


def _on_bike(p, moto, min_cover: float, center_frac: float) -> float:
    """How well person box p fits as a rider of motorcycle box moto (0 = not a rider): the person's centre is
    near the bike's centre line (riders sit one behind the other; in a queue the people of the bikes beside it
    reach over its box edges) and their legs end inside its lower part, not beside it like a pedestrian."""
    mx1, my1, mx2, my2 = moto
    mh = my2 - my1
    cx = (p[0] + p[2]) / 2
    if not (abs(cx - (mx1 + mx2) / 2) <= center_frac * (mx2 - mx1) and my1 + 0.25 * mh <= p[3] <= my2 + 0.15 * mh):
        return 0.0
    c = _cover(p, moto)
    return c if c >= min_cover else 0.0


class OverloadRule:
    """Chở quá số người quy định (NĐ 168/2024, Điều 7): a motorcycle carrying more than `max_riders` people.
    People come from the vehicle detector (COCO person); each one is given to the motorcycle it sits on.
    The law allows a child under 12 as a third rider, so only adult-sized riders count: at least
    `adult_ratio` of the tallest rider's box height. A passenger half hidden behind the driver also looks
    small, so this misses some cases rather than flagging a parent with a child."""
    type = "overload"

    def __init__(self, cfg: dict):
        self.max_riders = int(cfg.get("max_riders", 2))
        self.adult_ratio = float(cfg.get("adult_ratio", 0.8))  # 0.7 counted a child in front, half hidden, as an adult
        self.min_cover = float(cfg.get("min_cover", 0.3))       # share of the person box over the bike box
        self.center_frac = float(cfg.get("center_frac", 0.25))  # rider centre within this share of the bike width
        self.min_height = float(cfg.get("min_box_height", 90))  # far bikes: riders merge into one box
        self.min_frames = int(cfg.get("min_frames", 6))
        self.min_share = float(cfg.get("min_share", 0.6))

    def people_seen(self, people, tracks, t: float) -> None:
        motos = [tr for tr in tracks if tr.cls == "motorcycle" and tr.bbox[3] - tr.bbox[1] >= self.min_height]
        riders: dict[int, list] = {tr.id: [] for tr in motos}
        for p in people:
            best = max(motos, key=lambda tr: _on_bike(p, tr.bbox, self.min_cover, self.center_frac), default=None)
            if best is not None and _on_bike(p, best.bbox, self.min_cover, self.center_frac) > 0:
                riders[best.id].append(p)
        for tr in motos:
            rs = riders[tr.id]
            if not rs:
                continue
            tallest = max(p[3] - p[1] for p in rs)
            adults = [p for p in rs if p[3] - p[1] >= self.adult_ratio * tallest]
            tr.rider_frames += 1
            tr.rider_counts[len(adults)] += 1
            if len(adults) > self.max_riders:
                tr.overload_frames += 1
                x1, y1, x2, y2 = tr.bbox
                tr.overload_box = (min([x1] + [p[0] for p in rs]), min([y1] + [p[1] for p in rs]),
                                   max([x2] + [p[2] for p in rs]), max([y2] + [p[3] for p in rs]))

    def update(self, tr: Track, light: str) -> list[Violation]:
        if tr.cls != "motorcycle" or self.type in tr.flagged or tr.rider_frames < self.min_frames:
            return []
        share = tr.overload_frames / tr.rider_frames
        if share < self.min_share:
            return []
        n = max(k for k in tr.rider_counts if k > self.max_riders)
        tr.flagged.add(self.type)
        return [Violation(self.type, tr, f"Xe máy chở {n} người lớn ({share:.0%} của {tr.rider_frames} khung hình)",
                          tr.last_seen)]


class SolidLineRule:
    """Đè / lấn vạch liền (QCVN 41:2024, vạch 1.1 / 1.3: không được đè lên hay vượt qua). Each line is a polyline
    drawn on the solid marking per camera. A vehicle is flagged when its anchor (bottom-centre of the box)
    crosses the line and then stays on the other side for `confirm_frames` frames at least `min_offset` px away,
    so a box that jitters over the line or a wheel touching it does not count."""
    type = "solid_line"
    fixed_camera = True

    def __init__(self, cfg: dict):
        self.lines = [{"name": ln.get("name", "vạch liền"),
                       "segs": [(tuple(map(float, a)), tuple(map(float, b))) for a, b in zip(ln["points"], ln["points"][1:])],
                       "classes": set(ln.get("classes", VEHICLES))}
                      for ln in cfg.get("lines", [])]
        self.confirm = int(cfg.get("confirm_frames", 5))
        self.min_offset = float(cfg.get("min_offset", 15))

    def update(self, tr: Track, light: str) -> list[Violation]:
        if self.type in tr.flagged:
            return []
        mv = _last_move(tr)
        if not mv:
            return []
        _, p0, t1, p1 = mv
        for ln in self.lines:
            if tr.cls not in ln["classes"]:
                continue
            key = f"{self.type}:{ln['name']}"
            for i, (a, b) in enumerate(ln["segs"]):
                if crossing(p0, p1, a, b) is not None:
                    tr.zone_entry[key] = (float(i), float(np.sign(side(p1, a, b))), 0.0)
                    tr.zone_frames[key] = 0
            pending = tr.zone_entry.get(key)
            if pending is None:
                continue
            a, b = ln["segs"][int(pending[0])]
            s = side(p1, a, b)
            if np.sign(s) != pending[1]:  # back on the first side: jitter, not a lane change
                tr.zone_entry.pop(key)
                continue
            tr.zone_frames[key] += 1
            if tr.zone_frames[key] >= self.confirm and abs(s) / max(1e-6, np.hypot(b[0] - a[0], b[1] - a[1])) >= self.min_offset:
                tr.flagged.add(self.type)
                return [Violation(self.type, tr, f"{tr.cls} đè / vượt qua {ln['name']}", t1)]
        return []


class NoYieldRule:
    """Không nhường đường cho người đi bộ trên vạch kẻ đường dành cho người đi bộ (NĐ 168/2024). A crosswalk is a
    polygon drawn per camera. A vehicle is flagged when it moves inside the crosswalk while a pedestrian has
    been on it for `min_ped_frames` frames within `near_px` of the vehicle. Pedestrians are COCO people that
    are not riding a vehicle (their box is not mostly inside a vehicle box). Needs a camera looking down on the
    crosswalk: from the side a pedestrian and the vehicles behind them overlap in the image. With `skip_on_green` (default),
    the rule is off while the vehicles' light is green: pedestrians crossing on red do not have the right of way."""
    type = "no_yield"
    fixed_camera = True

    def __init__(self, cfg: dict):
        self.walks = [{"name": c.get("name", "vạch người đi bộ"),
                       "poly": np.array(c["polygon"], np.float32).reshape(-1, 1, 2),
                       "near": float(c.get("near_px", cfg.get("near_px", 200)))}
                      for c in cfg.get("crosswalks", [])]
        self.classes = set(cfg.get("classes", VEHICLES))
        self.min_ped_frames = int(cfg.get("min_ped_frames", 3))
        self.min_move = float(cfg.get("min_move_px", 20))  # over the last 5 frames: a vehicle waiting does yield
        self.rider_cover = float(cfg.get("rider_cover", 0.3))
        self.skip_on_green = bool(cfg.get("skip_on_green", True))
        self.peds: dict[str, list] = {w["name"]: [] for w in self.walks}   # pedestrian anchors on each crosswalk
        self.ped_frames: dict[str, int] = {w["name"]: 0 for w in self.walks}

    def people_seen(self, people, tracks, t: float) -> None:
        walkers = [p for p in people if not any(_cover(p, tr.bbox) >= self.rider_cover for tr in tracks)]
        for w in self.walks:
            on = [((p[0] + p[2]) / 2, p[3]) for p in walkers if in_polygon(((p[0] + p[2]) / 2, p[3]), w["poly"])]
            self.peds[w["name"]] = on
            self.ped_frames[w["name"]] = self.ped_frames[w["name"]] + 1 if on else 0

    def update(self, tr: Track, light: str) -> list[Violation]:
        if tr.cls not in self.classes or self.type in tr.flagged or len(tr.history) < 5:
            return []
        if self.skip_on_green and light == "green":
            return []
        a = tr.anchor
        (_, first), (_, last) = tr.history[-5], tr.history[-1]
        if np.hypot(last[0] - first[0], last[1] - first[1]) < self.min_move:
            return []
        for w in self.walks:
            if self.ped_frames[w["name"]] < self.min_ped_frames or not in_polygon(a, w["poly"]):
                continue
            if any(np.hypot(a[0] - p[0], a[1] - p[1]) <= w["near"] for p in self.peds[w["name"]]):
                tr.flagged.add(self.type)
                return [Violation(self.type, tr, f"{tr.cls} không nhường đường người đi bộ trên {w['name']}", tr.last_seen)]
        return []


class ZoneRule:
    """Zone rules that only apply where a traffic sign makes them law (Luật TTATGTĐB 2024, Điều 11: signs and
    markings decide what is allowed). Each zone is a polygon on the road drawn once per camera, plus the signs
    that create its restriction (`require_sign`, see SignGate). A zone without `require_sign` is always on
    (e.g. a sidewalk or the junction box). Kinds:
    - wrong_way (đi ngược chiều): moving against the zone's `direction` for at least `min_dist` px
    - no_entry (đi vào đường cấm): a `banned` vehicle class (default: what the sign bans) inside the zone
      for `min_frames` frames
    - sidewalk (đi trên vỉa hè): riding along the sidewalk (`along` axis) for `min_dist` px; crossing it to
      reach a house moves across the axis and is allowed
    - no_parking (dừng / đỗ nơi cấm): staying within `still_px` for `max_stop_seconds`
    - block_junction (dừng trong giao lộ gây ùn tắc): stopped inside the junction box for `max_stop_seconds`
      (default 5) while this camera's light is red, i.e. blocking the cross traffic that now has green
    The vehicle class used is the majority over the track's frames, so one misclassified frame (a motorbike
    seen as a car) does not decide a violation.
    """
    KINDS = ("wrong_way", "no_entry", "sidewalk", "no_parking", "block_junction")
    fixed_camera = True

    def __init__(self, kind: str, cfg: dict):
        self.type = kind
        self.zones = []
        for z in cfg.get("zones", []):
            d = z.get("direction", z.get("along"))
            if d is not None:
                n = float(np.hypot(*d))
                d = (d[0] / n, d[1] / n)
            signs = list(z.get("require_sign", []))
            banned = set(z.get("banned", []))
            if not banned:
                for s in signs:
                    banned.update(SignGate.BANS.get(s, ()))
            self.zones.append({
                "name": z.get("name", kind), "poly": np.array(z["polygon"], np.float32).reshape(-1, 1, 2),
                "signs": signs, "dir": d, "banned": banned, "classes": set(z.get("classes", VEHICLES)),
                "max_stop": float(z.get("max_stop_seconds", cfg.get("max_stop_seconds", 5 if kind == "block_junction" else 60))),
            })
        self.min_dist = float(cfg.get("min_dist", 250))      # px the vehicle must travel (against / along) in the zone
        self.min_frames = int(cfg.get("min_frames", 8))      # no_entry: consecutive frames inside
        self.min_share = float(cfg.get("min_class_share", 0.6))
        self.must_enter = bool(cfg.get("must_enter", False))  # no_entry: only vehicles first seen outside the zone
        self.still_px = float(cfg.get("still_px", 25))       # no_parking: moving less than this counts as stopped
        self.gate = SignGate(cfg.get("latch_seconds", 30))
        self.red_since: float | None = None  # block_junction: video time the light turned red

    def signs_seen(self, visible, t: float) -> None:
        self.gate.update(visible, t)

    def _recent_against(self, tr: Track, d) -> bool:
        """Also moving against the direction over the last 15 frames: a box that shrinks as a pole or tree
        hides the vehicle shifts its centre backwards once, but a vehicle really driving the wrong way keeps going."""
        if len(tr.history) < 15:
            return False
        (_, first), (_, last) = tr.history[-15], tr.history[-1]
        return -((last[0] - first[0]) * d[0] + (last[1] - first[1]) * d[1]) >= self.min_dist / 4

    def update(self, tr: Track, light: str) -> list[Violation]:
        if self.type in tr.flagged:
            return []
        a, t = tr.anchor, tr.last_seen
        if light != "red":
            self.red_since = None
        elif self.red_since is None:
            self.red_since = t
        for z in self.zones:
            key = f"{self.type}:{z['name']}"
            if not in_polygon(a, z["poly"]):
                tr.zone_entry.pop(key, None)
                tr.zone_entry.pop(key + ":still", None)
                tr.zone_frames.pop(key, None)
                tr.was_outside.add(key)
                continue
            entry = tr.zone_entry.setdefault(key, (t, a[0], a[1]))
            still = tr.zone_entry.get(key + ":still")
            if still is None or np.hypot(a[0] - still[1], a[1] - still[2]) > self.still_px:
                still = tr.zone_entry[key + ":still"] = (t, a[0], a[1])
            frames = tr.zone_frames[key] = tr.zone_frames.get(key, 0) + 1
            if not self.gate.active(z["signs"], t):
                continue
            cls = tr.stable_cls(self.min_share)
            if cls is None or cls not in z["classes"]:
                continue
            dx, dy, d = a[0] - entry[1], a[1] - entry[2], z["dir"]
            detail = None
            if self.type == "wrong_way":
                if d and -(dx * d[0] + dy * d[1]) >= self.min_dist and self._recent_against(tr, d):
                    detail = f"{cls} đi ngược chiều trong {z['name']}"
            elif self.type == "no_entry":
                if cls in z["banned"] and frames >= self.min_frames and (not self.must_enter or key in tr.was_outside):
                    detail = f"{cls} đi vào {z['name']} (có biển cấm)"
            elif self.type == "sidewalk":
                if d and abs(dx * d[0] + dy * d[1]) >= self.min_dist:
                    detail = f"{cls} chạy trên {z['name']}"
            elif self.type == "no_parking":
                if t - still[0] >= z["max_stop"]:
                    detail = f"{cls} dừng/đỗ {t - still[0]:.0f} s trong {z['name']}"
            elif self.type == "block_junction":
                stopped = t - max(still[0], self.red_since if self.red_since is not None else t)
                if light == "red" and stopped >= z["max_stop"]:
                    detail = f"{cls} đứng {stopped:.0f} s trong {z['name']} khi đèn đỏ (chắn giao lộ)"
            if detail:
                tr.cls = cls  # record the majority class, not the one of the first frame
                tr.flagged.add(self.type)
                return [Violation(self.type, tr, detail, t)]
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
    if r.get("overload", {}).get("enabled"):
        rules.append(OverloadRule(r["overload"]))
    if r.get("solid_line", {}).get("enabled"):
        rules.append(SolidLineRule(r["solid_line"]))
    if r.get("no_yield", {}).get("enabled"):
        rules.append(NoYieldRule(r["no_yield"]))
    for kind in ZoneRule.KINDS:
        if r.get(kind, {}).get("enabled"):
            rules.append(ZoneRule(kind, r[kind]))
    return rules
