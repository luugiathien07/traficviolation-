package vn.tvd.rules;

import java.util.Iterator;
import java.util.List;
import java.util.Set;

/** A violation rule looks at one tracked vehicle per frame. A track is flagged at most once per type. */
public interface Rule {
    List<String> VEHICLES = List.of("car", "motorcycle", "bus", "truck");

    String type();

    List<Violation> update(Track tr, String light);

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
