package vn.tvd.rules;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import vn.tvd.Cfg;

/**
 * Chở quá số người quy định (NĐ 168/2024, Điều 7): a motorcycle carrying more than {@code max_riders} people.
 * People come from the vehicle detector (COCO person); each one is given to the motorcycle it sits on.
 * The law allows a child under 12 as a third rider, so only adult-sized riders count: at least
 * {@code adult_ratio} of the tallest rider's box height. A passenger half hidden behind the driver also looks
 * small, so this misses some cases rather than flagging a parent with a child.
 */
public final class OverloadRule implements Rule {
    private final int maxRiders, minFrames;
    private final double adultRatio, minCover, centerFrac, minHeight, minShare;

    public OverloadRule(Cfg cfg) {
        maxRiders = cfg.integer("max_riders", 2);
        adultRatio = cfg.num("adult_ratio", 0.8);  // 0.7 counted a child in front, half hidden, as an adult
        minCover = cfg.num("min_cover", 0.3);        // share of the person box over the bike box
        centerFrac = cfg.num("center_frac", 0.25);   // rider centre within this share of the bike width
        minHeight = cfg.num("min_box_height", 90);   // far bikes: riders merge into one box
        minFrames = cfg.integer("min_frames", 6);
        minShare = cfg.num("min_share", 0.6);
    }

    @Override
    public boolean fixedCamera() {
        return false;
    }

    @Override
    public String type() {
        return "overload";
    }

    /** Share of box {@code inner} that lies inside box {@code outer}. */
    public static double cover(double[] inner, double[] outer) {
        double ix = Math.max(0, Math.min(inner[2], outer[2]) - Math.max(inner[0], outer[0]));
        double iy = Math.max(0, Math.min(inner[3], outer[3]) - Math.max(inner[1], outer[1]));
        return ix * iy / Math.max(1, (inner[2] - inner[0]) * (inner[3] - inner[1]));
    }

    /**
     * How well person box p fits as a rider of motorcycle box m (0 = not a rider): the person's centre is near the
     * bike's centre line (riders sit one behind the other; in a queue the people of the bikes beside it reach over
     * its box edges) and their legs end inside its lower part, not beside it like a pedestrian.
     */
    private double onBike(double[] p, double[] m) {
        double mh = m[3] - m[1], cx = (p[0] + p[2]) / 2;
        if (Math.abs(cx - (m[0] + m[2]) / 2) > centerFrac * (m[2] - m[0]) || p[3] < m[1] + 0.25 * mh || p[3] > m[3] + 0.15 * mh) {
            return 0;
        }
        double c = cover(p, m);
        return c >= minCover ? c : 0;
    }

    @Override
    public void people(List<double[]> people, List<Track> seen, double t) {
        List<Track> motos = new ArrayList<>();
        Map<Integer, List<double[]>> riders = new HashMap<>();
        for (Track tr : seen) {
            if ("motorcycle".equals(tr.cls) && tr.bbox[3] - tr.bbox[1] >= minHeight) {
                motos.add(tr);
                riders.put(tr.id, new ArrayList<>());
            }
        }
        for (double[] p : people) {
            Track best = null;
            double bestScore = 0;
            for (Track tr : motos) {
                double s = onBike(p, tr.bbox);
                if (s > bestScore) {
                    best = tr;
                    bestScore = s;
                }
            }
            if (best != null) {
                riders.get(best.id).add(p);
            }
        }
        for (Track tr : motos) {
            List<double[]> rs = riders.get(tr.id);
            if (rs.isEmpty()) {
                continue;
            }
            double tallest = rs.stream().mapToDouble(p -> p[3] - p[1]).max().getAsDouble();
            int adults = (int) rs.stream().filter(p -> p[3] - p[1] >= adultRatio * tallest).count();
            tr.riderFrames++;
            tr.riderCounts.merge(adults, 1, Integer::sum);
            if (adults > maxRiders) {
                tr.overloadFrames++;
                double[] b = tr.bbox.clone();
                for (double[] p : rs) {
                    b[0] = Math.min(b[0], p[0]);
                    b[1] = Math.min(b[1], p[1]);
                    b[2] = Math.max(b[2], p[2]);
                    b[3] = Math.max(b[3], p[3]);
                }
                tr.overloadBox = b;
            }
        }
    }

    @Override
    public List<Violation> update(Track tr, String light) {
        if (!"motorcycle".equals(tr.cls) || tr.flagged.contains(type()) || tr.riderFrames < minFrames) {
            return List.of();
        }
        double share = (double) tr.overloadFrames / tr.riderFrames;
        if (share < minShare) {
            return List.of();
        }
        int n = tr.riderCounts.keySet().stream().filter(k -> k > maxRiders).max(Integer::compare).get();
        tr.flagged.add(type());
        String detail = String.format(Locale.ROOT, "Xe máy chở %d người lớn (%.0f%% của %d khung hình)",
                n, share * 100, tr.riderFrames);
        return List.of(new Violation(type(), tr, detail, tr.lastSeen));
    }
}
