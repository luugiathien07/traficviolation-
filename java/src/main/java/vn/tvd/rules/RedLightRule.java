package vn.tvd.rules;

import java.util.List;
import java.util.Set;

import vn.tvd.Cfg;
import vn.tvd.Geometry;

public final class RedLightRule implements Rule {
    private final double[] a, b;
    private final double approachSign;
    private final Set<String> classes;

    public RedLightRule(Cfg cfg) {
        double[][] line = cfg.points("stop_line");
        a = line[0];
        b = line[1];
        // A point on the side vehicles approach from; only crossings away from it count.
        approachSign = Math.signum(Geometry.side(cfg.point("approach_point"), a, b));
        classes = Rule.classes(cfg.strings("classes", VEHICLES));
    }

    @Override
    public String type() {
        return "red_light";
    }

    @Override
    public List<Violation> update(Track tr, String light) {
        if (!classes.contains(tr.cls) || tr.flagged.contains(type())) {
            return List.of();
        }
        double[] mv = Rule.lastMove(tr);
        if (mv == null) {
            return List.of();
        }
        double[] p0 = {mv[1], mv[2]}, p1 = {mv[4], mv[5]};
        double frac = Geometry.crossing(p0, p1, a, b);
        if (Double.isNaN(frac) || Math.signum(Geometry.side(p0, a, b)) != approachSign) {
            return List.of();
        }
        tr.crossTimes.put("stop", mv[0] + (mv[3] - mv[0]) * frac);
        if (!"red".equals(light)) {
            return List.of();
        }
        tr.flagged.add(type());
        return List.of(new Violation(type(), tr, "Qua vạch dừng khi đèn đỏ", mv[3]));
    }
}
