package vn.tvd;

/** Small 2D helpers: which side of a line a point is on, segment crossings, point in polygon. */
public final class Geometry {
    private Geometry() {
    }

    /** Signed area: >0 on one side of line a->b, <0 on the other, 0 on the line. */
    public static double side(double[] p, double[] a, double[] b) {
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
    }

    /**
     * If the motion p0->p1 crosses segment a-b, return the fraction (0..1] of the motion at which it crosses;
     * otherwise NaN.
     */
    public static double crossing(double[] p0, double[] p1, double[] a, double[] b) {
        double s0 = side(p0, a, b), s1 = side(p1, a, b);
        if (s0 == 0 || (s0 > 0) == (s1 > 0)) {
            return Double.NaN;
        }
        // The segment a-b must also be crossed within its extent.
        double t0 = side(a, p0, p1), t1 = side(b, p0, p1);
        if ((t0 > 0) == (t1 > 0) && t0 != 0 && t1 != 0) {
            return Double.NaN;
        }
        return s0 / (s0 - s1);
    }

    /** Ray casting; points on the edge count as inside (like cv2.pointPolygonTest >= 0). */
    public static boolean inPolygon(double[] p, double[][] poly) {
        boolean inside = false;
        for (int i = 0, j = poly.length - 1; i < poly.length; j = i++) {
            double[] a = poly[i], b = poly[j];
            if (onSegment(p, a, b)) {
                return true;
            }
            if ((a[1] > p[1]) != (b[1] > p[1])
                    && p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0]) {
                inside = !inside;
            }
        }
        return inside;
    }

    private static boolean onSegment(double[] p, double[] a, double[] b) {
        return Math.abs(side(p, a, b)) < 1e-9
                && p[0] >= Math.min(a[0], b[0]) && p[0] <= Math.max(a[0], b[0])
                && p[1] >= Math.min(a[1], b[1]) && p[1] <= Math.max(a[1], b[1]);
    }
}
