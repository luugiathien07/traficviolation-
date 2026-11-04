package vn.tvd.rules;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import vn.tvd.Cfg;
import vn.tvd.Geometry;

/**
 * Đè / lấn vạch liền (QCVN 41:2024, vạch 1.1 / 1.3: không được đè lên hay vượt qua). Each line is a polyline drawn
 * on the solid marking per camera. A vehicle is flagged when its anchor (bottom-centre of the box) crosses the
 * line and then stays on the other side for {@code confirm_frames} frames at least {@code min_offset} px away, so
 * a box that jitters over the line or a wheel touching it does not count.
 */
public final class SolidLineRule implements Rule {
    private record Line(String name, double[][] pts, Set<String> classes) {
    }

    private final List<Line> lines = new ArrayList<>();
    private final int confirm;
    private final double minOffset;

    public SolidLineRule(Cfg cfg) {
        for (Cfg ln : cfg.list("lines")) {
            lines.add(new Line(ln.str("name", "vạch liền"), ln.points("points"), Rule.classes(ln.strings("classes", VEHICLES))));
        }
        confirm = cfg.integer("confirm_frames", 5);
        minOffset = cfg.num("min_offset", 15);
    }

    @Override
    public String type() {
        return "solid_line";
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
        for (Line ln : lines) {
            if (!ln.classes().contains(tr.cls)) {
                continue;
            }
            String key = type() + ":" + ln.name();
            for (int i = 0; i + 1 < ln.pts().length; i++) {
                if (!Double.isNaN(Geometry.crossing(p0, p1, ln.pts()[i], ln.pts()[i + 1]))) {
                    tr.zoneEntry.put(key, new double[] {i, Math.signum(Geometry.side(p1, ln.pts()[i], ln.pts()[i + 1])), 0});
                    tr.zoneFrames.put(key, 0);
                }
            }
            double[] pending = tr.zoneEntry.get(key);
            if (pending == null) {
                continue;
            }
            double[] a = ln.pts()[(int) pending[0]], b = ln.pts()[(int) pending[0] + 1];
            double s = Geometry.side(p1, a, b);
            if (Math.signum(s) != pending[1]) {  // back on the first side: jitter, not a lane change
                tr.zoneEntry.remove(key);
                continue;
            }
            int frames = tr.zoneFrames.merge(key, 1, Integer::sum);
            if (frames >= confirm && Math.abs(s) / Math.max(1e-6, Math.hypot(b[0] - a[0], b[1] - a[1])) >= minOffset) {
                tr.flagged.add(type());
                return List.of(new Violation(type(), tr, tr.cls + " đè / vượt qua " + ln.name(), mv[3]));
            }
        }
        return List.of();
    }
}
