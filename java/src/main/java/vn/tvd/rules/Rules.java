package vn.tvd.rules;

import java.util.ArrayList;
import java.util.List;

import vn.tvd.Cfg;

public final class Rules {
    private Rules() {
    }

    public static List<Rule> build(Cfg cfg) {
        Cfg r = cfg.sub("rules");
        List<Rule> rules = new ArrayList<>();
        if (r.sub("red_light").enabled()) {
            rules.add(new RedLightRule(r.sub("red_light")));
        }
        if (r.sub("speeding").enabled()) {
            rules.add(new SpeedRule(r.sub("speeding")));
        }
        if (r.sub("wrong_lane").enabled()) {
            rules.add(new WrongLaneRule(r.sub("wrong_lane")));
        }
        if (r.sub("no_helmet").enabled()) {
            rules.add(new NoHelmetRule(r.sub("no_helmet")));
        }
        if (r.sub("no_turn").enabled()) {
            rules.add(new NoTurnRule(r.sub("no_turn")));
        }
        if (r.sub("overload").enabled()) {
            rules.add(new OverloadRule(r.sub("overload")));
        }
        if (r.sub("solid_line").enabled()) {
            rules.add(new SolidLineRule(r.sub("solid_line")));
        }
        if (r.sub("no_yield").enabled()) {
            rules.add(new NoYieldRule(r.sub("no_yield")));
        }
        for (String kind : ZoneRule.KINDS) {
            if (r.sub(kind).enabled()) {
                rules.add(new ZoneRule(kind, r.sub(kind)));
            }
        }
        return rules;
    }
}
