package vn.tvd.track;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import vn.tvd.detect.Detection;
import vn.tvd.detect.YoloDetector;

/**
 * ByteTrack multi-object tracker, ported from Ultralytics (trackers/byte_tracker.py) with the defaults of
 * bytetrack.yaml: high-score detections are matched first (IoU fused with score), then low-score ones keep
 * existing tracks alive; unmatched tracks are kept "lost" for track_buffer frames before being dropped.
 */
public final class ByteTracker {
    private final double highThresh, lowThresh, newTrackThresh, matchThresh;
    private final int maxLost;
    private final KalmanFilter kf = new KalmanFilter();
    private List<STrack> tracked = new ArrayList<>(), lost = new ArrayList<>();
    private int frameId;
    private int nextId;

    public ByteTracker() {
        this(0.25, 0.1, 0.25, 0.8, 30);
    }

    public ByteTracker(double highThresh, double lowThresh, double newTrackThresh, double matchThresh, int trackBuffer) {
        this.highThresh = highThresh;
        this.lowThresh = lowThresh;
        this.newTrackThresh = newTrackThresh;
        this.matchThresh = matchThresh;
        this.maxLost = trackBuffer;
    }

    public void reset() {
        tracked = new ArrayList<>();
        lost = new ArrayList<>();
        frameId = 0;
        nextId = 0;
    }

    /** Returns the confirmed tracks of this frame. */
    public List<STrack> update(List<Detection> dets) {
        frameId++;
        List<STrack> activated = new ArrayList<>(), refind = new ArrayList<>(), lostNow = new ArrayList<>(),
                removed = new ArrayList<>();
        List<STrack> high = new ArrayList<>(), low = new ArrayList<>();
        for (Detection d : dets) {
            if (d.w() <= 0 || d.h() <= 0) {
                continue;
            }
            if (d.conf() >= highThresh) {
                high.add(new STrack(d.xyxy(), d.conf(), d.cls()));
            } else if (d.conf() > lowThresh) {
                low.add(new STrack(d.xyxy(), d.conf(), d.cls()));
            }
        }
        List<STrack> unconfirmed = new ArrayList<>(), confirmed = new ArrayList<>();
        for (STrack t : tracked) {
            (t.activated ? confirmed : unconfirmed).add(t);
        }
        List<STrack> pool = joint(confirmed, lost);
        for (STrack t : pool) {
            t.predict(kf);
        }

        // 1st association: all tracks vs high-score detections
        LinearAssignment.Result r1 = LinearAssignment.solve(fusedIouCost(pool, high), pool.size(), high.size(), matchThresh);
        for (int[] m : r1.matches()) {
            apply(pool.get(m[0]), high.get(m[1]), activated, refind);
        }

        // 2nd association: still-tracked leftovers vs low-score detections
        List<STrack> rTracked = new ArrayList<>();
        for (int i : r1.unmatchedA()) {
            if (pool.get(i).state == STrack.State.TRACKED) {
                rTracked.add(pool.get(i));
            }
        }
        LinearAssignment.Result r2 = LinearAssignment.solve(iouCost(rTracked, low), rTracked.size(), low.size(), 0.5);
        for (int[] m : r2.matches()) {
            apply(rTracked.get(m[0]), low.get(m[1]), activated, refind);
        }
        for (int i : r2.unmatchedA()) {
            STrack t = rTracked.get(i);
            t.state = STrack.State.LOST;
            lostNow.add(t);
        }

        // unconfirmed tracks (seen once) vs remaining high-score detections
        List<STrack> rest = new ArrayList<>();
        for (int j : r1.unmatchedB()) {
            rest.add(high.get(j));
        }
        LinearAssignment.Result r3 = LinearAssignment.solve(fusedIouCost(unconfirmed, rest), unconfirmed.size(), rest.size(), 0.7);
        for (int[] m : r3.matches()) {
            unconfirmed.get(m[0]).update(kf, rest.get(m[1]), frameId);
            activated.add(unconfirmed.get(m[0]));
        }
        for (int i : r3.unmatchedA()) {
            unconfirmed.get(i).state = STrack.State.REMOVED;
            removed.add(unconfirmed.get(i));
        }
        for (int j : r3.unmatchedB()) {
            STrack t = rest.get(j);
            if (t.score >= newTrackThresh) {
                t.activate(kf, frameId, ++nextId);
                activated.add(t);
            }
        }
        for (STrack t : lost) {
            if (frameId - t.frameId > maxLost) {
                t.state = STrack.State.REMOVED;
                removed.add(t);
            }
        }

        List<STrack> newTracked = new ArrayList<>();
        for (STrack t : tracked) {
            if (t.state == STrack.State.TRACKED) {
                newTracked.add(t);
            }
        }
        newTracked = joint(joint(newTracked, activated), refind);
        List<STrack> newLost = sub(lost, newTracked);
        newLost.addAll(lostNow);
        newLost = sub(newLost, removed);
        removeDuplicates(newTracked, newLost);
        tracked = newTracked;
        lost = newLost;

        List<STrack> out = new ArrayList<>();
        for (STrack t : tracked) {
            if (t.activated) {
                out.add(t);
            }
        }
        return out;
    }

    private void apply(STrack track, STrack det, List<STrack> activated, List<STrack> refind) {
        if (track.state == STrack.State.TRACKED) {
            track.update(kf, det, frameId);
            activated.add(track);
        } else {
            track.reActivate(kf, det, frameId);
            refind.add(track);
        }
    }

    private static double[][] iouCost(List<STrack> a, List<STrack> b) {
        double[][] c = new double[a.size()][b.size()];
        for (int i = 0; i < a.size(); i++) {
            double[] ba = a.get(i).xyxy();
            for (int j = 0; j < b.size(); j++) {
                c[i][j] = 1 - YoloDetector.iou(ba, b.get(j).xyxy());
            }
        }
        return c;
    }

    private static double[][] fusedIouCost(List<STrack> a, List<STrack> b) {
        double[][] c = iouCost(a, b);
        for (double[] row : c) {
            for (int j = 0; j < row.length; j++) {
                row[j] = 1 - (1 - row[j]) * b.get(j).score;
            }
        }
        return c;
    }

    private static List<STrack> joint(List<STrack> a, List<STrack> b) {
        Map<Integer, STrack> m = new LinkedHashMap<>();
        for (STrack t : a) {
            m.put(t.trackId, t);
        }
        for (STrack t : b) {
            m.putIfAbsent(t.trackId, t);
        }
        return new ArrayList<>(m.values());
    }

    private static List<STrack> sub(List<STrack> a, List<STrack> b) {
        Set<Integer> ids = new HashSet<>();
        for (STrack t : b) {
            ids.add(t.trackId);
        }
        List<STrack> out = new ArrayList<>();
        for (STrack t : a) {
            if (!ids.contains(t.trackId)) {
                out.add(t);
            }
        }
        return out;
    }

    /** A track and a lost track that overlap almost fully are the same object: keep the older one. */
    private static void removeDuplicates(List<STrack> a, List<STrack> b) {
        double[][] d = iouCost(a, b);
        Set<STrack> dropA = new HashSet<>(), dropB = new HashSet<>();
        for (int i = 0; i < a.size(); i++) {
            for (int j = 0; j < b.size(); j++) {
                if (d[i][j] < 0.15) {
                    STrack p = a.get(i), q = b.get(j);
                    if (p.frameId - p.startFrame > q.frameId - q.startFrame) {
                        dropB.add(q);
                    } else {
                        dropA.add(p);
                    }
                }
            }
        }
        a.removeAll(dropA);
        b.removeAll(dropB);
    }
}
