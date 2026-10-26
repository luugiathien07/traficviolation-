import numpy as np

from tvd.enhance import BacklightCompensator
from tvd.geometry import crossing
from tvd.rules import RedLightRule, SpeedRule, Track, WrongLaneRule


def drive(rule, cls, points, light="red", dt=0.1):
    """Feed a box whose bottom-centre follows `points` into a rule."""
    tr = Track(1, cls, 0.9, (0, 0, 0, 0), 0.0)
    out = []
    for i, (x, y) in enumerate(points):
        t = i * dt
        tr.bbox, tr.last_seen = (x - 10, y - 20, x + 10, y), t
        tr.history.append((t, tr.anchor))
        out += rule.update(tr, light if isinstance(light, str) else light[i])
    return tr, out


def test_crossing_fraction():
    assert crossing((0, 0), (0, 10), (-5, 5), (5, 5)) == 0.5
    assert crossing((0, 0), (0, 4), (-5, 5), (5, 5)) is None
    assert crossing((20, 0), (20, 10), (-5, 5), (5, 5)) is None  # outside the segment


STOP = {"stop_line": [[0, 100], [200, 100]], "approach_point": [100, 200]}


def test_red_light_flags_crossing_on_red_only_once():
    _, v = drive(RedLightRule(STOP), "car", [(100, y) for y in range(160, 40, -10)])
    assert [x.type for x in v] == ["red_light"]


def test_red_light_ignores_green_and_wrong_direction():
    _, v = drive(RedLightRule(STOP), "car", [(100, y) for y in range(160, 40, -10)], light="green")
    assert v == []
    _, v = drive(RedLightRule(STOP), "car", [(100, y) for y in range(40, 160, 10)])
    assert v == []  # leaving the intersection towards the approach side


def test_speed_measured_between_lines():
    cfg = {"line_a": [[0, 300], [200, 300]], "line_b": [[0, 100], [200, 100]],
           "distance_m": 20, "limit_kmh": 50, "tolerance_kmh": 5}
    # 10 px per 0.1 s; lines 200 px apart -> 2.0 s for 20 m -> 36 km/h
    tr, v = drive(SpeedRule(cfg), "car", [(100, y) for y in range(350, 50, -10)], light="green")
    assert abs(tr.speed_kmh - 36) < 0.5 and v == []
    # 25 px per 0.1 s -> 0.8 s -> 90 km/h
    tr, v = drive(SpeedRule(cfg), "car", [(100, y) for y in range(350, 50, -25)], light="green")
    assert abs(tr.speed_kmh - 90) < 1 and v[0].type == "speeding"


def test_wrong_lane_needs_consecutive_frames():
    rule = WrongLaneRule({"min_frames": 3, "lanes": [
        {"name": "lan xe may", "polygon": [[0, 0], [100, 0], [100, 500], [0, 500]], "allowed": ["motorcycle"]}]})
    _, v = drive(rule, "motorcycle", [(50, y) for y in range(400, 100, -20)])
    assert v == []
    _, v = drive(rule, "car", [(50, y) for y in range(400, 100, -20)])
    assert [x.type for x in v] == ["wrong_lane"]


def backlit_frame():
    img = np.full((360, 640, 3), 30, np.uint8)       # dark road and vehicles
    img[:120] = 252                                   # blown-out sky
    img[200:260, 200:330] = (55, 50, 45)              # dark "car"
    return img


def test_backlight_auto_detects_and_lifts_shadows():
    blc = BacklightCompensator({"mode": "auto"})
    frame = backlit_frame()
    for _ in range(20):
        out = blc.process(frame)
    assert blc.active and blc.score > 0.45
    assert out[230, 260].mean() > frame[230, 260].mean() + 20   # car got brighter


def test_backlight_auto_leaves_even_scene_alone():
    blc = BacklightCompensator({"mode": "auto"})
    frame = np.full((360, 640, 3), 120, np.uint8)
    for _ in range(20):
        out = blc.process(frame)
    assert not blc.active and np.array_equal(out, frame)


def test_backlight_off_is_passthrough():
    blc = BacklightCompensator({"mode": "off"})
    frame = backlit_frame()
    assert blc.process(frame) is frame


def test_normalize_vn_plates():
    from tvd.plate import normalize_vn
    assert normalize_vn("51A12345") == "51A-123.45"
    assert normalize_vn("29B125662") == "29-B1 256.62"            # only fits the motorbike layout
    assert normalize_vn("30L90860") == "30L-908.60"                   # car layout by default
    assert normalize_vn("30L90860", motorbike=True) == "30-L9 0860"  # motorbike: motorbike layout
    assert normalize_vn("50LD00454") == "50LD-004.54"
    assert normalize_vn("5lA-I23.45") == "51A-123.45"   # look-alikes fixed by position
    assert normalize_vn("29PJ8AW3") is None
    assert normalize_vn("ABC") is None
