package vn.tvd.rules;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeSet;

/** A tracked vehicle as the rules see it. */
public final class Track {
    public static final int HISTORY = 60;

    public final int id;
    public String cls;
    public double conf;
    public double[] bbox;
    public double lastSeen;
    /** (t, anchor x, anchor y) where the anchor is the bottom-centre of the box (road contact point). */
    public final Deque<double[]> history = new ArrayDeque<>();
    public final Map<String, Double> crossTimes = new HashMap<>();
    public final Map<String, Integer> laneFrames = new HashMap<>();
    public Double speedKmh;
    public final TreeSet<String> flagged = new TreeSet<>();
    public String plate;
    public double plateConf;
    /** Next video time to try reading this vehicle's plate when plate.read_all is on. */
    public double plateNext;
    /** Helmet model votes over frames: summed confidence for helmet / no_helmet. */
    public final double[] helmetVotes = new double[2];
    public int helmetFrames;

    public Track(int id, String cls, double conf, double[] bbox, double lastSeen) {
        this.id = id;
        this.cls = cls;
        this.conf = conf;
        this.bbox = bbox;
        this.lastSeen = lastSeen;
    }

    public double[] anchor() {
        return new double[] {(bbox[0] + bbox[2]) / 2, bbox[3]};
    }

    public void addHistory(double t) {
        double[] a = anchor();
        history.addLast(new double[] {t, a[0], a[1]});
        while (history.size() > HISTORY) {
            history.removeFirst();
        }
    }
}
