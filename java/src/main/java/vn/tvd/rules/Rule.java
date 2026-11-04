package vn.tvd.rules;

import java.util.Iterator;
import java.util.List;
import java.util.Set;

/** A violation rule looks at one tracked vehicle per frame. A track is flagged at most once per type. */
public interface Rule {
    List<String> VEHICLES = List.of("car", "motorcycle", "bus", "truck");

    String type();

    List<Violation> update(Track tr, String light);

    /**
     * Needs lines / zones drawn for a camera that stands still: switched off while {@link vn.tvd.CameraMotion} says the
     * camera moves (dashcam, handheld), where they land on the wrong part of the road.
     */
    default boolean fixedCamera() {
        return true;
    }

    /** Names of the traffic signs visible in this frame (rules that depend on a sign override this). */
    default void signs(Set<String> visible, double t) {
    }

    /** The people (x1, y1, x2, y2) in this frame and the vehicles seen (rules about riders / pedestrians override this). */
    default void people(List<double[]> people, List<Track> seen, double t) {
    }

    /** Rules that need the people in the frame: the detector then also finds COCO person. */
    List<String> PEOPLE_RULES = List.of("overload", "no_yield");

    /** The last move of a track: {t0, x0, y0, t1, x1, y1}, or null with fewer than 2 points. */
    static double[] lastMove(Track tr) {
        if (tr.history.size() < 2) {
            return null;
        }
        Iterator<double[]> it = tr.history.descendingIterator();
        double[] p1 = it.next(), p0 = it.next();
        return new double[] {p0[0], p0[1], p0[2], p1[0], p1[1], p1[2]};
    }

    static Set<String> classes(List<String> list) {
        return Set.copyOf(list);
    }
}
