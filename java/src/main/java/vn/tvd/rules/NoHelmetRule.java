package vn.tvd.rules;

import java.util.List;
import java.util.Locale;

import vn.tvd.Cfg;

/**
 * Flag a motorcycle once enough frames agree that a rider has no helmet.
 * Votes come from HelmetChecker, which runs a separate helmet / no_helmet model.
 */
public final class NoHelmetRule implements Rule {
    private final int minFrames;
    private final double minShare;

    public NoHelmetRule(Cfg cfg) {
        // 3 frames at 60% flagged riders who wore a helmet: ask for more and clearer agreement
        minFrames = cfg.integer("min_frames", 6);
        minShare = cfg.num("min_share", 0.75);
    }

    @Override
    public boolean fixedCamera() {
        return false;
    }

    @Override
    public String type() {
        return "no_helmet";
    }

    @Override
    public List<Violation> update(Track tr, String light) {
        if (!"motorcycle".equals(tr.cls) || tr.flagged.contains(type()) || tr.helmetFrames < minFrames) {
            return List.of();
        }
        double yes = tr.helmetVotes[0], no = tr.helmetVotes[1];
        double share = no / Math.max(yes + no, 1e-6);
        if (share < minShare) {
            return List.of();
        }
        tr.flagged.add(type());
        String detail = String.format(Locale.ROOT, "Có người trên xe không đội mũ bảo hiểm (%.0f%% của %d khung hình)",
                share * 100, tr.helmetFrames);
        return List.of(new Violation(type(), tr, detail, tr.lastSeen));
    }
}
