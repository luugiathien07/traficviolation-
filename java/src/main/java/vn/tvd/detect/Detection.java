package vn.tvd.detect;

/** One box in frame pixels: x1, y1, x2, y2, confidence, class id. */
public record Detection(double x1, double y1, double x2, double y2, double conf, int cls) {
    public double w() {
        return x2 - x1;
    }

    public double h() {
        return y2 - y1;
    }

    public double[] xyxy() {
        return new double[] {x1, y1, x2, y2};
    }
}
