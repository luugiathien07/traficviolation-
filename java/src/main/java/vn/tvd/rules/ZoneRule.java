package vn.tvd.rules;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import vn.tvd.Cfg;
import vn.tvd.Geometry;

/**
 * Zone rules that only apply where a traffic sign makes them law (Luật TTATGTĐB 2024, Điều 11: signs and
 * markings decide what is allowed). Each zone is a polygon on the road drawn once per camera, plus the
 * signs that create its restriction ({@code require_sign}, see {@link SignGate}). A zone without
 * {@code require_sign} is always on (e.g. a sidewalk or the junction box). Kinds:
 * <ul>
 * <li>wrong_way (đi ngược chiều): moving against the zone's {@code direction} for at least {@code min_dist} px</li>
 * <li>no_entry (đi vào đường cấm): a {@code banned} vehicle class (default: what the sign bans) inside the
 * zone for {@code min_frames} frames</li>
 * <li>sidewalk (đi trên vỉa hè): riding along the sidewalk ({@code along} axis) for {@code min_dist} px;
 * crossing it to reach a house moves across the axis and is allowed</li>
 * <li>no_parking (dừng / đỗ nơi cấm): staying within {@code still_px} for {@code max_stop_seconds}</li>
 * <li>block_junction (dừng trong giao lộ gây ùn tắc): stopped inside the junction box for {@code max_stop_seconds}
 * (default 5) while this camera's light is red, i.e. blocking the cross traffic that now has green</li>
 * </ul>
 * The vehicle class used is the majority over the track's frames, so one misclassified frame (a motorbike
 * seen as a car) does not decide a violation.
 */
public final class ZoneRule implements Rule {
    public static final List<String> KINDS = List.of("wrong_way", "no_entry", "sidewalk", "no_parking", "block_junction");

    private record Zone(String name, double[][] poly, List<String> signs, double[] dir, Set<String> banned,
                        Set<String> classes, double maxStop) {
    }

    private final String type;
    private final List<Zone> zones = new ArrayList<>();
    private final double minDist, minShare, stillPx;
    private final int minFrames;
    private final boolean mustEnter;
    private final SignGate gate;
    /** block_junction: video time the light turned red, or NaN while it is not red. */
    private double redSince = Double.NaN;

    public ZoneRule(String type, Cfg cfg) {
        this.type = type;
        for (Cfg z : cfg.list("zones")) {
            double[] dir = z.has("direction") ? z.numbers("direction") : z.has("along") ? z.numbers("along") : null;
            if (dir != null) {
                double n = Math.hypot(dir[0], dir[1]);
                dir = new double[] {dir[0] / n, dir[1] / n};
            }
            List<String> signs = z.strings("require_sign", List.of());
            Set<String> banned = new LinkedHashSet<>(z.strings("banned", List.of()));
            if (banned.isEmpty()) {
                signs.forEach(s -> banned.addAll(SignGate.BANS.getOrDefault(s, List.of())));
            }
            zones.add(new Zone(z.str("name", type), z.points("polygon"), signs, dir, banned,
                    Rule.classes(z.strings("classes", VEHICLES)), z.num("max_stop_seconds", cfg.num("max_stop_seconds", "block_junction".equals(type) ? 5 : 60))));
        }
        minDist = cfg.num("min_dist", 250);       // px the vehicle must travel (against / along) inside the zone
        minFrames = cfg.integer("min_frames", 8);  // no_entry: consecutive frames inside
        minShare = cfg.num("min_class_share", 0.6);
        mustEnter = cfg.bool("must_enter", false);  // no_entry: only vehicles first seen outside the zone
        stillPx = cfg.num("still_px", 25);          // no_parking: moving less than this counts as stopped
        gate = new SignGate(cfg.num("latch_seconds", 30));
    }

    @Override
    public String type() {
        return type;
    }

    @Override
    public void signs(Set<String> visible, double t) {
        gate.update(visible, t);
    }

    /**
     * Also moving against the direction over the last 15 frames: a box that shrinks as a pole or tree hides
     * the vehicle shifts its centre backwards once, but a vehicle really driving the wrong way keeps going.
     */
    private boolean recentAgainst(Track tr, double[] dir) {
        if (tr.history.size() < 15) {
            return false;
        }
        double[] last = tr.history.peekLast();
        double[] first = tr.history.stream().skip(tr.history.size() - 15).findFirst().get();
        return -((last[1] - first[1]) * dir[0] + (last[2] - first[2]) * dir[1]) >= minDist / 4;
    }

    @Override
    public List<Violation> update(Track tr, String light) {
        if (tr.flagged.contains(type)) {
            return List.of();
        }
        double[] a = tr.anchor();
        double t = tr.lastSeen;
        if (!"red".equals(light)) {
            redSince = Double.NaN;
        } else if (Double.isNaN(redSince)) {
            redSince = t;
        }
        for (Zone z : zones) {
            String key = type + ":" + z.name();
            if (!Geometry.inPolygon(a, z.poly())) {
                tr.zoneEntry.remove(key);
                tr.zoneEntry.remove(key + ":still");
                tr.zoneFrames.remove(key);
                tr.wasOutside.add(key);
                continue;
            }
            double[] entry = tr.zoneEntry.computeIfAbsent(key, k -> new double[] {t, a[0], a[1]});
            double[] still = tr.zoneEntry.get(key + ":still");
            if (still == null || Math.hypot(a[0] - still[1], a[1] - still[2]) > stillPx) {
                still = new double[] {t, a[0], a[1]};
                tr.zoneEntry.put(key + ":still", still);
            }
            int frames = tr.zoneFrames.merge(key, 1, Integer::sum);
            if (!gate.active(z.signs(), t)) {
                continue;
            }
            String cls = tr.stableCls(minShare);
            if (cls == null || !z.classes().contains(cls)) {
                continue;
            }
            double dx = a[0] - entry[1], dy = a[1] - entry[2];
            double stopped = t - Math.max(still[0], Double.isNaN(redSince) ? t : redSince);
            String detail = switch (type) {
                case "wrong_way" -> z.dir() != null && -(dx * z.dir()[0] + dy * z.dir()[1]) >= minDist
                        && recentAgainst(tr, z.dir()) ? cls + " đi ngược chiều trong " + z.name() : null;
                case "no_entry" -> z.banned().contains(cls) && frames >= minFrames
                        && (!mustEnter || tr.wasOutside.contains(key))
                        ? cls + " đi vào " + z.name() + " (có biển cấm)" : null;
                case "sidewalk" -> z.dir() != null && Math.abs(dx * z.dir()[0] + dy * z.dir()[1]) >= minDist
                        ? cls + " chạy trên " + z.name() : null;
                case "no_parking" -> t - still[0] >= z.maxStop()
                        ? String.format(java.util.Locale.ROOT, "%s dừng/đỗ %.0f s trong %s", cls, t - still[0], z.name()) : null;
                case "block_junction" -> "red".equals(light) && stopped >= z.maxStop()
                        ? String.format(java.util.Locale.ROOT, "%s đứng %.0f s trong %s khi đèn đỏ (chắn giao lộ)", cls, stopped, z.name()) : null;
                default -> null;
            };
            if (detail != null) {
                tr.cls = cls;  // record the majority class, not the one of the first frame
                tr.flagged.add(type);
                return List.of(new Violation(type, tr, detail, t));
            }
        }
        return List.of();
    }
}
