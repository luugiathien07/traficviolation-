package vn.tvd.rules;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import vn.tvd.Cfg;
import vn.tvd.Geometry;

/**
 * Không nhường đường cho người đi bộ trên vạch kẻ đường dành cho người đi bộ (NĐ 168/2024). A crosswalk is a polygon
 * drawn per camera. A vehicle is flagged when it moves inside the crosswalk while a pedestrian has been on it for
 * {@code min_ped_frames} frames within {@code near_px} of the vehicle. Pedestrians are COCO people that are not
 * riding a vehicle (their box is not mostly inside a vehicle box). Needs a camera looking down on the crosswalk:
 * from the side a pedestrian and the vehicles behind them overlap in the image. With {@code skip_on_green} (default), the rule
 * is off while the vehicles' light is green: pedestrians crossing on red do not have the right of way.
 */
public final class NoYieldRule implements Rule {
    private record Walk(String name, double[][] poly, double near) {
    }

    private final List<Walk> walks = new ArrayList<>();
    private final Set<String> classes;
    private final int minPedFrames;
    private final double minMove, riderCover;
    private final boolean skipOnGreen;
    /** Pedestrian anchors on each crosswalk this frame, and how many frames in a row it had any. */
    private final Map<String, List<double[]>> peds = new HashMap<>();
    private final Map<String, Integer> pedFrames = new HashMap<>();

    public NoYieldRule(Cfg cfg) {
        for (Cfg c : cfg.list("crosswalks")) {
            walks.add(new Walk(c.str("name", "vạch người đi bộ"), c.points("polygon"), c.num("near_px", cfg.num("near_px", 200))));
        }
        classes = Rule.classes(cfg.strings("classes", VEHICLES));
        minPedFrames = cfg.integer("min_ped_frames", 3);
        minMove = cfg.num("min_move_px", 20);  // over the last 5 frames: a vehicle waiting does yield
        riderCover = cfg.num("rider_cover", 0.3);
        skipOnGreen = cfg.bool("skip_on_green", true);
    }

    @Override
    public String type() {
        return "no_yield";
    }

    @Override
    public void people(List<double[]> people, List<Track> seen, double t) {
        List<double[]> walkers = new ArrayList<>();
        for (double[] p : people) {
            if (seen.stream().noneMatch(tr -> OverloadRule.cover(p, tr.bbox) >= riderCover)) {
                walkers.add(new double[] {(p[0] + p[2]) / 2, p[3]});
            }
        }
        for (Walk w : walks) {
            List<double[]> on = walkers.stream().filter(a -> Geometry.inPolygon(a, w.poly())).toList();
            peds.put(w.name(), on);
            pedFrames.put(w.name(), on.isEmpty() ? 0 : pedFrames.getOrDefault(w.name(), 0) + 1);
        }
    }

    @Override
    public List<Violation> update(Track tr, String light) {
        if (!classes.contains(tr.cls) || tr.flagged.contains(type()) || tr.history.size() < 5) {
            return List.of();
        }
        if (skipOnGreen && "green".equals(light)) {
            return List.of();
        }
        double[] last = tr.history.peekLast();
        double[] first = tr.history.stream().skip(tr.history.size() - 5).findFirst().get();
        if (Math.hypot(last[1] - first[1], last[2] - first[2]) < minMove) {
            return List.of();
        }
        double[] a = tr.anchor();
        for (Walk w : walks) {
            if (pedFrames.getOrDefault(w.name(), 0) < minPedFrames || !Geometry.inPolygon(a, w.poly())) {
                continue;
            }
            for (double[] p : peds.getOrDefault(w.name(), List.of())) {
                if (Math.hypot(a[0] - p[0], a[1] - p[1]) <= w.near()) {
                    tr.flagged.add(type());
                    return List.of(new Violation(type(), tr, tr.cls + " không nhường đường người đi bộ trên " + w.name(), tr.lastSeen));
                }
            }
        }
        return List.of();
    }
}
