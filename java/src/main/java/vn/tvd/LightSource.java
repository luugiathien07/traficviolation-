package vn.tvd;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

/**
 * Traffic-light state sources.
 * <ul>
 * <li>roi: read the lamp colour from a region of the frame (HSV thresholds)</li>
 * <li>timer: fixed signal plan (green/yellow/red seconds + offset)</li>
 * <li>manual: state set from the dashboard / API (useful when the light is off-camera)</li>
 * </ul>
 * Any source can be overridden at runtime with {@link #override}; null returns to the configured source.
 */
public final class LightSource {
    public static final List<String> STATES = List.of("red", "yellow", "green");

    public final String mode;
    public final double[] roi;
    private final double minRatio;
    private final int smooth;
    private final Deque<String> history = new ArrayDeque<>();
    private final double green, yellow, red, offset;
    private final String manualState;
    private volatile String override;
    public volatile String state = "unknown";
    public Map<String, Double> ratios = new HashMap<>();

    public LightSource(Cfg cfg) {
        mode = cfg.str("mode", "manual");
        roi = cfg.numbers("roi");
        minRatio = cfg.num("min_ratio", 0.04);
        smooth = cfg.integer("smooth_frames", 5);
        Cfg plan = cfg.sub("timer");
        green = plan.num("green", 20);
        yellow = plan.num("yellow", 3);
        red = plan.num("red", 20);
        offset = plan.num("offset", 0);
        manualState = cfg.str("initial", "green");
    }

    public void override(String s) {
        if (s != null && !STATES.contains(s)) {
            throw new IllegalArgumentException("state phải là một trong " + STATES + " hoặc auto");
        }
        override = s;
    }

    public boolean overridden() {
        return override != null;
    }

    public String update(Mat frame, double t) {
        if (override != null) {
            state = override;
        } else if ("roi".equals(mode) && roi != null) {
            history.addLast(classify(frame));
            while (history.size() > smooth) {
                history.removeFirst();
            }
            Map<String, Integer> counts = new LinkedHashMap<>();
            for (String s : history) {
                if (!"unknown".equals(s)) {
                    counts.merge(s, 1, Integer::sum);
                }
            }
            state = counts.entrySet().stream().max(Map.Entry.comparingByValue()).map(Map.Entry::getKey).orElse("unknown");
        } else if ("timer".equals(mode)) {
            double tt = (t + offset) % (green + yellow + red);
            state = tt < green ? "green" : tt < green + yellow ? "yellow" : "red";
        } else {
            state = manualState;
        }
        return state;
    }

    private String classify(Mat frame) {
        int x1 = Math.max(0, (int) roi[0]), y1 = Math.max(0, (int) roi[1]);
        int x2 = Math.min(frame.cols(), (int) roi[2]), y2 = Math.min(frame.rows(), (int) roi[3]);
        if (x2 <= x1 || y2 <= y1) {
            return "unknown";
        }
        Mat hsv = new Mat();
        Imgproc.cvtColor(frame.submat(new Rect(x1, y1, x2 - x1, y2 - y1)), hsv, Imgproc.COLOR_BGR2HSV);
        // lit pixels: saturated and bright; hue ranges per lamp colour (OpenCV hue is 0..180)
        Mat red1 = new Mat(), red2 = new Mat(), redM = new Mat(), yel = new Mat(), grn = new Mat();
        Core.inRange(hsv, new Scalar(0, 91, 141), new Scalar(9, 255, 255), red1);
        Core.inRange(hsv, new Scalar(166, 91, 141), new Scalar(180, 255, 255), red2);
        Core.bitwise_or(red1, red2, redM);
        Core.inRange(hsv, new Scalar(12, 91, 141), new Scalar(35, 255, 255), yel);
        Core.inRange(hsv, new Scalar(40, 91, 141), new Scalar(95, 255, 255), grn);
        double n = (double) hsv.rows() * hsv.cols();
        ratios = new HashMap<>(Map.of("red", Core.countNonZero(redM) / n,
                "yellow", Core.countNonZero(yel) / n, "green", Core.countNonZero(grn) / n));
        for (Mat m : List.of(hsv, red1, red2, redM, yel, grn)) {
            m.release();
        }
        String best = ratios.entrySet().stream().max(Map.Entry.comparingByValue()).get().getKey();
        return ratios.get(best) >= minRatio ? best : "unknown";
    }
}
