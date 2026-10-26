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
        return rules;
    }
}
