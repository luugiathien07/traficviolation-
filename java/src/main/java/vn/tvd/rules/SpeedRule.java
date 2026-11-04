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
    /** Limit from a speed-limit sign while it is seen; otherwise default_limit_kmh (TT 38/2024), or no check. */
    private final List<String> signs;
    private final Double defaultLimit;
    private final SignGate gate;
    private final double latch;
    private double now;
    /** Number read from the speed-limit sign ("speed limit 50" from SignDetector) and when it was last seen. */
    private Integer signKmh;
    private double signSeenAt;

    public SpeedRule(Cfg cfg) {
        la = cfg.points("line_a");
        lb = cfg.points("line_b");
        distanceM = cfg.num("distance_m", 0);
        limit = cfg.num("limit_kmh", 50);
        tolerance = cfg.num("tolerance_kmh", 5);
        limitByClass = cfg.sub("limit_by_class");
        classes = Rule.classes(cfg.strings("classes", VEHICLES));
        signs = cfg.strings("require_sign", List.of());
        defaultLimit = cfg.has("default_limit_kmh") ? cfg.num("default_limit_kmh", 50) : null;
        latch = cfg.num("latch_seconds", 30);
        gate = new SignGate(latch);
    }

    @Override
    public String type() {
        return "speeding";
    }

    public double limitFor(String cls) {
        return limitByClass.num(cls, limit);
    }

    @Override
    public void signs(Set<String> visible, double t) {
        gate.update(visible, t);
        now = t;
        for (String s : visible) {
            if (s.matches("speed limit \\d+")) {
                signKmh = Integer.parseInt(s.substring(12));
                signSeenAt = t;
            }
        }
    }

    /**
     * The limit in force: the number read from the sign; else the configured one while the sign is seen
     * (per class); else the default one without a sign; or null for no check.
     */
    private Double limitNow(String cls) {
        if (signKmh != null && now - signSeenAt <= latch) {
            return (double) signKmh;
        }
        if (gate.active(signs, now)) {
            return limitFor(cls);
        }
        return defaultLimit == null ? null : limitByClass.sub("default").num(cls, defaultLimit);
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
        Double limNow = limitNow(tr.cls);
        if (limNow == null) {
            return List.of();
        }
        double lim = limNow, excess = tr.speedKmh - lim;
        if (excess < tolerance) {
            return List.of();
        }
        tr.flagged.add(type());
        String source = signs.isEmpty() ? "" : gate.active(signs, now) ? " theo biển" : " mặc định TT 38/2024";
        String detail = String.format(Locale.ROOT, "%.0f km/h, giới hạn %.0f km/h%s (+%.0f)", tr.speedKmh, lim, source, excess);
        return List.of(new Violation(type(), tr, detail, mv[3], tr.speedKmh));
    }
}
