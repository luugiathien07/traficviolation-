package vn.tvd.rules;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import vn.tvd.Cfg;
import vn.tvd.Geometry;

public final class WrongLaneRule implements Rule {
    private record Lane(String name, double[][] poly, Set<String> allowed, List<String> signs) {
    }

    private final List<Lane> lanes = new ArrayList<>();
    private final int minFrames;
    private final SignGate gate;

    public WrongLaneRule(Cfg cfg) {
        for (Cfg ln : cfg.list("lanes")) {
            lanes.add(new Lane(ln.str("name", "lan"), ln.points("polygon"), Rule.classes(ln.strings("allowed", List.of())),
                    ln.strings("require_sign", List.of())));
        }
        gate = new SignGate(cfg.num("latch_seconds", 30));
        minFrames = cfg.integer("min_frames", 8);
    }

    @Override
    public String type() {
        return "wrong_lane";
    }

    @Override
    public void signs(Set<String> visible, double t) {
        gate.update(visible, t);
    }

    @Override
    public List<Violation> update(Track tr, String light) {
        if (!VEHICLES.contains(tr.cls) || tr.flagged.contains(type())) {
            return List.of();
        }
        for (Lane ln : lanes) {
            if (Geometry.inPolygon(tr.anchor(), ln.poly())) {
                int n = tr.laneFrames.getOrDefault(ln.name(), 0) + 1;
                tr.laneFrames.put(ln.name(), n);
                if (!ln.allowed().contains(tr.cls) && n >= minFrames && gate.active(ln.signs(), tr.lastSeen)) {
                    tr.flagged.add(type());
                    return List.of(new Violation(type(), tr, tr.cls + " đi trong " + ln.name(), tr.lastSeen));
                }
            } else {
                tr.laneFrames.put(ln.name(), 0);
            }
        }
        return List.of();
    }
}
