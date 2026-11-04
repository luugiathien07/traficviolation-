package vn.tvd.rules;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Remembers when each traffic sign was last seen, so a rule part tied to a sign ({@code require_sign})
 * is only enforced while that sign is in place. Names are the sign model's classes (models/vn_signs.pt).
 * A fixed camera sees its signs all the time; {@code latch_seconds} bridges frames where a passing
 * truck hides them.
 *
 * <p>Prohibition signs (QCVN 41:2024) the model recognises and the rule each one drives:
 * <ul>
 * <li>one way prohibition (P.102 cấm đi ngược chiều) -> wrong_way / no_entry for every vehicle</li>
 * <li>no car entry/turning (P.103a-c cấm ô tô) -> no_entry for car, bus, truck</li>
 * <li>no motorbike entry/turning (P.104, P.111a) -> no_entry for motorcycle</li>
 * <li>no truck entry/turning (P.106a-b cấm xe tải) -> no_entry for truck</li>
 * <li>no turn left / no turn right / no u turn / no u and left turn / no u and right turn
 * (P.123a-b, P.124a-f) -> no_turn exits</li>
 * <li>no parking (P.131a-c cấm đỗ) / no stopping and parking (P.130 cấm dừng và đỗ) -> no_parking</li>
 * <li>speed limit (P.127) -> speeding with the limit written on the sign (entered per camera: the model
 * has one class for every limit)</li>
 * <li>vehicle permission lane / vehicle and speed permission lane (R.412, R.415) -> wrong_lane</li>
 * </ul>
 * Not enforced: weight / height limit (P.115, P.117: need a scale or height sensor), "other prohibition"
 * (one class for many different signs) and "no more prohibition" (ends a restriction; the zone polygon is
 * drawn to stop there).
 */
public final class SignGate {
    /** Vehicle classes a sign bans when a no_entry zone does not list them itself. */
    static final Map<String, List<String>> BANS = Map.of(
            "one way prohibition", Rule.VEHICLES,
            "no car entry/turning", List.of("car", "bus", "truck"),
            "no motorbike entry/turning", List.of("motorcycle"),
            "no truck entry/turning", List.of("truck"));

    private final Map<String, Double> seen = new HashMap<>();
    private final double latch;

    public SignGate(double latchSeconds) {
        latch = latchSeconds;
    }

    public void update(Set<String> visible, double t) {
        for (String s : visible) {
            seen.put(s, t);
        }
    }

    /** No sign required, or one of them seen within latch_seconds. */
    public boolean active(List<String> required, double t) {
        if (required.isEmpty()) {
            return true;
        }
        for (String s : required) {
            Double last = seen.get(s);
            if (last != null && t - last <= latch) {
                return true;
            }
        }
        return false;
    }
}
