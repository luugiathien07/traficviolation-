package vn.tvd;

import java.util.Arrays;
import java.util.List;

import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.CvType;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.Video;

/**
 * Moving camera check (camera di động). Lines and zones are drawn once for a camera that stands still; on a dashcam
 * or a camera carried by hand they land on the wrong part of the road, and every parked car seems to move because
 * the camera does (a stopped car beside the dashcam "drives along the sidewalk", a car it overtakes "drives the wrong
 * way"). This measures how far the background moves between frames (sparse optical flow on corners outside the
 * vehicle boxes) so the pipeline can switch off the rules that need a fixed camera.
 *
 * <p>Measured on the sample videos: fixed cameras stay under 0.4 px per frame (leaves in the wind), dashcams run at
 * 13-27 px. A fixed camera never starts moving, so with {@code latch} (default) the first sustained movement keeps
 * the camera "moving" for the rest of the video. Modes: "off", "auto" (default).
 */
public final class CameraMotion {
    public final String mode;
    private final int width;
    private final double onPx, offPx, smooth;
    private final boolean latch;
    private Mat prev;
    /** Smoothed median background shift, px per frame at full size. */
    public double shift;
    public boolean moving;

    public CameraMotion(Cfg cfg) {
        mode = cfg.str("mode", "auto");
        if (!List.of("off", "auto").contains(mode)) {
            throw new IllegalArgumentException("camera_motion.mode must be off or auto");
        }
        width = cfg.integer("width", 480);       // frames are measured at this width
        onPx = cfg.num("on_px", 2.0);            // median background shift per frame, full-size px
        offPx = cfg.num("off_px", 0.6);
        latch = cfg.bool("latch", true);
        smooth = cfg.num("smooth", 0.1);         // EMA weight of a new frame
    }

    /** Feed every frame with this frame's vehicle boxes; returns whether the camera is moving. */
    public boolean update(Mat frame, List<double[]> boxes) {
        if ("off".equals(mode)) {
            return false;
        }
        double scale = (double) width / frame.cols();
        Mat small = new Mat(), gray = new Mat();
        Imgproc.resize(frame, small, new Size(width, Math.round(frame.rows() * scale)));
        Imgproc.cvtColor(small, gray, Imgproc.COLOR_BGR2GRAY);
        small.release();
        Mat last = prev;
        prev = gray;
        if (last == null) {
            return moving;
        }
        try {
            Mat mask = new Mat(last.size(), CvType.CV_8UC1, new Scalar(255));
            for (double[] b : boxes) {  // vehicles move on their own: measure only the background
                int x1 = Math.max(0, (int) (b[0] * scale)), y1 = Math.max(0, (int) (b[1] * scale));
                int x2 = Math.min(mask.cols(), (int) (b[2] * scale) + 1), y2 = Math.min(mask.rows(), (int) (b[3] * scale) + 1);
                if (x2 > x1 && y2 > y1) {
                    mask.submat(new Rect(x1, y1, x2 - x1, y2 - y1)).setTo(new Scalar(0));
                }
            }
            MatOfPoint corners = new MatOfPoint();
            Imgproc.goodFeaturesToTrack(last, corners, 200, 0.01, 8, mask);
            mask.release();
            Point[] pts = corners.toArray();
            corners.release();
            if (pts.length < 10) {
                return moving;
            }
            MatOfPoint2f p0 = new MatOfPoint2f(pts), p1 = new MatOfPoint2f();
            MatOfByte status = new MatOfByte();
            MatOfFloat err = new MatOfFloat();
            Video.calcOpticalFlowPyrLK(last, gray, p0, p1, status, err, new Size(15, 15), 2);
            Point[] next = p1.toArray();
            byte[] ok = status.toArray();
            double[] d = new double[pts.length];
            int n = 0;
            for (int i = 0; i < pts.length; i++) {
                if (ok[i] == 1) {
                    d[n++] = Math.hypot(next[i].x - pts[i].x, next[i].y - pts[i].y);
                }
            }
            p0.release();
            p1.release();
            status.release();
            err.release();
            if (n < 10) {
                return moving;
            }
            double[] sorted = Arrays.copyOf(d, n);
            Arrays.sort(sorted);
            double median = n % 2 == 1 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
            shift = (1 - smooth) * shift + smooth * median / scale;
            if (!moving && shift > onPx) {
                moving = true;
            } else if (moving && !latch && shift < offPx) {
                moving = false;
            }
            return moving;
        } finally {
            last.release();
        }
    }
}
