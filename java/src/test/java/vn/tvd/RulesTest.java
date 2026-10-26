package vn.tvd;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

import org.junit.jupiter.api.Test;

import vn.tvd.rules.NoHelmetRule;
import vn.tvd.rules.NoTurnRule;
import vn.tvd.rules.RedLightRule;
import vn.tvd.rules.Rule;
import vn.tvd.rules.SpeedRule;
import vn.tvd.rules.Track;
import vn.tvd.rules.Violation;
import vn.tvd.rules.WrongLaneRule;

/** Same cases as tests/test_rules.py of the Python version. */
class RulesTest {
    record Run(Track track, List<Violation> violations) {
        List<String> types() {
            return violations.stream().map(Violation::type).toList();
        }
    }

    /** Feed a box whose bottom-centre follows {@code points} into a rule. */
    static Run drive(Rule rule, String cls, List<double[]> points, String light) {
        Track tr = new Track(1, cls, 0.9, new double[4], 0.0);
        List<Violation> out = new ArrayList<>();
        for (int i = 0; i < points.size(); i++) {
            double t = i * 0.1, x = points.get(i)[0], y = points.get(i)[1];
            tr.bbox = new double[] {x - 10, y - 20, x + 10, y};
            tr.lastSeen = t;
            tr.addHistory(t);
            out.addAll(rule.update(tr, light));
        }
        return new Run(tr, out);
    }

    /** (x, y) for y = from, from+step, ... while y moves towards `to` (exclusive), like Python's range. */
    static List<double[]> column(double x, int from, int to, int step) {
        return IntStream.iterate(from, y -> step > 0 ? y < to : y > to, y -> y + step)
                .mapToObj(y -> new double[] {x, y}).toList();
    }

    static final String STOP = "stop_line: [[0, 100], [200, 100]]\napproach_point: [100, 200]";

    @Test
    void crossingFraction() {
        assertEquals(0.5, Geometry.crossing(new double[] {0, 0}, new double[] {0, 10}, new double[] {-5, 5}, new double[] {5, 5}));
        assertTrue(Double.isNaN(Geometry.crossing(new double[] {0, 0}, new double[] {0, 4}, new double[] {-5, 5}, new double[] {5, 5})));
        // outside the segment
        assertTrue(Double.isNaN(Geometry.crossing(new double[] {20, 0}, new double[] {20, 10}, new double[] {-5, 5}, new double[] {5, 5})));
    }

    @Test
    void redLightFlagsCrossingOnRedOnlyOnce() {
        assertEquals(List.of("red_light"), drive(new RedLightRule(Cfg.parse(STOP)), "car", column(100, 160, 40, -10), "red").types());
    }

    @Test
    void redLightIgnoresGreenAndWrongDirection() {
        assertEquals(List.of(), drive(new RedLightRule(Cfg.parse(STOP)), "car", column(100, 160, 40, -10), "green").types());
        // leaving the intersection towards the approach side
        assertEquals(List.of(), drive(new RedLightRule(Cfg.parse(STOP)), "car", column(100, 40, 160, 10), "red").types());
    }

    @Test
    void speedMeasuredBetweenLines() {
        Cfg cfg = Cfg.parse("""
                line_a: [[0, 300], [200, 300]]
                line_b: [[0, 100], [200, 100]]
                distance_m: 20
                limit_kmh: 50
                tolerance_kmh: 5""");
        // 10 px per 0.1 s; lines 200 px apart -> 2.0 s for 20 m -> 36 km/h
        Run slow = drive(new SpeedRule(cfg), "car", column(100, 350, 50, -10), "green");
        assertEquals(36, slow.track().speedKmh, 0.5);
        assertEquals(List.of(), slow.types());
        // 25 px per 0.1 s -> 0.8 s -> 90 km/h
        Run fast = drive(new SpeedRule(cfg), "car", column(100, 350, 50, -25), "green");
        assertEquals(90, fast.track().speedKmh, 1);
        assertEquals(List.of("speeding"), fast.types());
    }

    @Test
    void wrongLaneNeedsConsecutiveFrames() {
        Cfg cfg = Cfg.parse("""
                min_frames: 3
                lanes:
                  - name: lan xe may
                    polygon: [[0, 0], [100, 0], [100, 500], [0, 500]]
                    allowed: [motorcycle]""");
        assertEquals(List.of(), drive(new WrongLaneRule(cfg), "motorcycle", column(50, 400, 100, -20), "green").types());
        assertEquals(List.of("wrong_lane"), drive(new WrongLaneRule(cfg), "car", column(50, 400, 100, -20), "green").types());
    }

    @Test
    void noHelmetRuleNeedsAgreeingFrames() {
        NoHelmetRule rule = new NoHelmetRule(Cfg.parse("min_frames: 3\nmin_share: 0.6"));
        Track tr = new Track(1, "motorcycle", 0.9, new double[] {0, 0, 10, 10}, 0.0);
        tr.helmetVotes[1] = 1.8;
        tr.helmetFrames = 2;
        assertEquals(List.of(), rule.update(tr, "green"));                  // too few frames
        tr.helmetVotes[0] = 1.6;
        tr.helmetVotes[1] = 1.2;
        tr.helmetFrames = 4;
        assertEquals(List.of(), rule.update(tr, "green"));                  // mostly helmet
        tr.helmetVotes[0] = 0.5;
        tr.helmetVotes[1] = 2.4;
        assertEquals("no_helmet", rule.update(tr, "green").get(0).type());
        assertEquals(List.of(), rule.update(tr, "green"));                  // flagged once
        Track car = new Track(2, "car", 0.9, new double[] {0, 0, 10, 10}, 0.0);
        car.helmetVotes[1] = 3.0;
        car.helmetFrames = 5;
        assertEquals(List.of(), rule.update(car, "green"));
    }

    @Test
    void noTurnFlagsBannedExitOnlyAfterEntry() {
        Cfg cfg = Cfg.parse("""
                entry_line: [[0, 100], [200, 100]]
                approach_point: [100, 200]
                banned_exits:
                  - name: re trai
                    line: [[0, 0], [0, 100]]""");
        // the anchor is the bottom-centre of the box, so drive() gets anchor points directly
        assertEquals(1, drive(new NoTurnRule(cfg), "car",
                List.of(new double[] {100, 150}, new double[] {100, 90}, new double[] {50, 50}, new double[] {-10, 40}), "green")
                .violations().size());                                      // enters, then turns left
        assertEquals(0, drive(new NoTurnRule(cfg), "car",
                List.of(new double[] {100, 150}, new double[] {100, 90}, new double[] {100, 20}), "green")
                .violations().size());                                      // goes straight
        assertEquals(0, drive(new NoTurnRule(cfg), "car",
                List.of(new double[] {50, 50}, new double[] {-10, 40}), "green").violations().size());  // never entered
    }

    @Test
    void polygonContainsEdgesAndInterior() {
        double[][] sq = {{0, 0}, {10, 0}, {10, 10}, {0, 10}};
        assertTrue(Geometry.inPolygon(new double[] {5, 5}, sq));
        assertTrue(Geometry.inPolygon(new double[] {10, 5}, sq));
        assertTrue(!Geometry.inPolygon(new double[] {11, 5}, sq));
    }
}
