package vn.tvd.track;

/** One tracklet of ByteTrack: Kalman state plus bookkeeping (port of Ultralytics STrack). */
public final class STrack {
    enum State { NEW, TRACKED, LOST, REMOVED }

    double[] tlwh;          // detection box (top-left x, y, width, height) before activation
    double[] mean;
    double[][] cov;
    double score;
    int cls;
    State state = State.NEW;
    boolean activated;
    int trackId;
    int frameId, startFrame, trackletLen;

    STrack(double[] xyxy, double score, int cls) {
        this.tlwh = new double[] {xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]};
        this.score = score;
        this.cls = cls;
    }

    static double[] toXyah(double[] tlwh) {
        return new double[] {tlwh[0] + tlwh[2] / 2, tlwh[1] + tlwh[3] / 2, tlwh[2] / tlwh[3], tlwh[3]};
    }

    double[] tlwh() {
        if (mean == null) {
            return tlwh.clone();
        }
        double w = mean[2] * mean[3], h = mean[3];
        return new double[] {mean[0] - w / 2, mean[1] - h / 2, w, h};
    }

    public double[] xyxy() {
        double[] t = tlwh();
        return new double[] {t[0], t[1], t[0] + t[2], t[1] + t[3]};
    }

    void activate(KalmanFilter kf, int frame, int id) {
        trackId = id;
        mean = new double[8];
        cov = kf.initiate(toXyah(tlwh), mean);
        trackletLen = 0;
        state = State.TRACKED;
        activated = frame == 1;
        frameId = frame;
        startFrame = frame;
    }

    void reActivate(KalmanFilter kf, STrack det, int frame) {
        cov = kf.update(mean, cov, toXyah(det.tlwh));
        trackletLen = 0;
        state = State.TRACKED;
        activated = true;
        frameId = frame;
        score = det.score;
        cls = det.cls;
    }

    void update(KalmanFilter kf, STrack det, int frame) {
        frameId = frame;
        trackletLen++;
        cov = kf.update(mean, cov, toXyah(det.tlwh));
        state = State.TRACKED;
        activated = true;
        score = det.score;
        cls = det.cls;
    }

    void predict(KalmanFilter kf) {
        if (state != State.TRACKED) {
            mean[7] = 0;
        }
        cov = kf.predict(mean, cov);
    }

    public int id() {
        return trackId;
    }

    public double score() {
        return score;
    }

    public int cls() {
        return cls;
    }
}
