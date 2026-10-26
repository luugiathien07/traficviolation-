package vn.tvd.rules;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import vn.tvd.Cfg;
import vn.tvd.Geometry;

/**
 * Banned turns at an intersection. A vehicle is armed when it crosses `entry_line` coming from
 * `approach_point`'s side, then flagged when it crosses the line of a banned exit
 * (e.g. the far side of the left-turn road).
 */
public final class NoTurnRule implements Rule {
    private record Exit(String name, double[][] line, Set<String> classes) {
    }

    private final double[][] entry;
    private final double approachSign;
    private final List<Exit> exits = new ArrayList<>();

    public NoTurnRule(Cfg cfg) {
        entry = cfg.points("entry_line");
        approachSign = Math.signum(Geometry.side(cfg.point("approach_point"), entry[0], entry[1]));
        for (Cfg e : cfg.list("banned_exits")) {
            exits.add(new Exit(e.str("name", "re"), e.points("line"), Rule.classes(e.strings("classes", VEHICLES))));
        }
    }

    @Override
    public String type() {
        return "no_turn";
    }

    @Override
    public List<Violation> update(Track tr, String light) {
        if (tr.flagged.contains(type())) {
            return List.of();
        }
        double[] mv = Rule.lastMove(tr);
        if (mv == null) {
            return List.of();
        }
        double[] p0 = {mv[1], mv[2]}, p1 = {mv[4], mv[5]};
        if (!tr.crossTimes.containsKey("entry")) {
            double frac = Geometry.crossing(p0, p1, entry[0], entry[1]);
            if (!Double.isNaN(frac) && Math.signum(Geometry.side(p0, entry[0], entry[1])) == approachSign) {
                tr.crossTimes.put("entry", mv[0] + (mv[3] - mv[0]) * frac);
            }
            return List.of();
        }
        for (Exit ex : exits) {
            if (ex.classes().contains(tr.cls) && !Double.isNaN(Geometry.crossing(p0, p1, ex.line()[0], ex.line()[1]))) {
                tr.flagged.add(type());
                return List.of(new Violation(type(), tr, tr.cls + " " + ex.name() + " tại nơi cấm", mv[3]));
            }
        }
        return List.of();
    }
}
