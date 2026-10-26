package vn.tvd.rules;

import java.util.List;
import java.util.Locale;
import java.util.Set;

import vn.tvd.Cfg;
import vn.tvd.Geometry;

public final class SpeedRule implements Rule {
    private final double[][] la, lb;
    private final double distanceM, limit, tolerance;
    private final Cfg limitByClass;
    private final Set<String> classes;

    public SpeedRule(Cfg cfg) {
        la = cfg.points("line_a");
        lb = cfg.points("line_b");
        distanceM = cfg.num("distance_m", 0);
        limit = cfg.num("limit_kmh", 50);
        tolerance = cfg.num("tolerance_kmh", 5);
        limitByClass = cfg.sub("limit_by_class");
        classes = Rule.classes(cfg.strings("classes", VEHICLES));
    }

    @Override
    public String type() {
        return "speeding";
    }

    public double limitFor(String cls) {
        return limitByClass.num(cls, limit);
    }

    @Override
    public List<Violation> update(Track tr, String light) {
        if (!classes.contains(tr.cls) || tr.speedKmh != null) {
            return List.of();
        }
        double[] mv = Rule.lastMove(tr);
        if (mv == null) {
            return List.of();
        }
        double[] p0 = {mv[1], mv[2]}, p1 = {mv[4], mv[5]};
        String[] keys = {"A", "B"};
        double[][][] lines = {la, lb};
        for (int i = 0; i < 2; i++) {
            double frac = Geometry.crossing(p0, p1, lines[i][0], lines[i][1]);
            if (!Double.isNaN(frac) && !tr.crossTimes.containsKey(keys[i])) {
                tr.crossTimes.put(keys[i], mv[0] + (mv[3] - mv[0]) * frac);
            }
        }
        if (!tr.crossTimes.containsKey("A") || !tr.crossTimes.containsKey("B")) {
            return List.of();
        }
        double dt = Math.abs(tr.crossTimes.get("B") - tr.crossTimes.get("A"));
        if (dt < 1e-3) {
            return List.of();
        }
        tr.speedKmh = distanceM / dt * 3.6;
        double lim = limitFor(tr.cls), excess = tr.speedKmh - lim;
        if (excess < tolerance) {
            return List.of();
        }
        tr.flagged.add(type());
        String detail = String.format(Locale.ROOT, "%.0f km/h, giới hạn %.0f km/h (+%.0f)", tr.speedKmh, lim, excess);
        return List.of(new Violation(type(), tr, detail, mv[3], tr.speedKmh));
    }
}
