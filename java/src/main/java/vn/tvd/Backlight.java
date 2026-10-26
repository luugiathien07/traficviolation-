package vn.tvd;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.CLAHE;
import org.opencv.imgproc.Imgproc;

/**
 * Backlight compensation (bù ngược sáng).
 * Measures how backlit the frame is (share of clipped highlights vs. deep shadows), lifts shadows with a
 * tone curve, compresses highlights, then applies CLAHE on the luminance channel (L in LAB) only.
 * Modes: off, on (fixed strength), auto (strength follows the measured score, with hysteresis).
 * The traffic-light classifier still reads the raw frame, because enhancement shifts lamp colours.
 */
public final class Backlight {
    public static final List<String> MODES = List.of("off", "on", "auto");

    public volatile String mode;
    private final double strength, gamma, onAt, offAt;
    private final int knee;
    private final CLAHE clahe;
    public double score;
    public boolean active;
    public double appliedStrength;
    private final Map<Integer, Mat> lutCache = new HashMap<>();

    public Backlight(Cfg cfg) {
        mode = cfg.str("mode", "auto");
        if (!MODES.contains(mode)) {
            throw new IllegalArgumentException("backlight.mode phải là một trong " + MODES);
        }
        strength = cfg.num("strength", 0.8);
        gamma = cfg.num("shadow_gamma", 0.55);
        knee = cfg.integer("highlight_knee", 210);
        onAt = cfg.num("auto_on", 0.40);
        offAt = cfg.num("auto_off", 0.25);
        int tiles = cfg.integer("clahe_tiles", 8);
        clahe = Imgproc.createCLAHE(cfg.num("clahe_clip", 2.5), new Size(tiles, tiles));
    }

    public void setMode(String m) {
        if (!MODES.contains(m)) {
            throw new IllegalArgumentException("mode phải là một trong " + MODES);
        }
        mode = m;
    }

    /** 0 = evenly lit, 1 = strongly backlit (lots of clipped highlights AND deep shadows). */
    public static double measure(Mat gray) {
        Mat small = new Mat();
        Imgproc.resize(gray, small, new Size(160, 90), 0, 0, Imgproc.INTER_AREA);
        byte[] px = new byte[(int) small.total()];
        small.get(0, 0, px);
        small.release();
        int bright = 0, dark = 0;
        for (byte b : px) {
            int v = b & 0xff;
            if (v > 230) {
                bright++;
            } else if (v < 60) {
                dark++;
            }
        }
        double n = px.length;
        // Need both: a bright scene alone is just daylight, a dark one alone is night.
        return Math.min(Math.min(bright / n / 0.08, 1.0) * Math.min(dark / n / 0.30, 1.0), 1.0);
    }

    private Mat lut(double s) {
        int key = (int) Math.round(s * 20);
        return lutCache.computeIfAbsent(key, k -> {
            double ss = k / 20.0;
            byte[] table = new byte[256];
            for (int x = 0; x < 256; x++) {
                double lifted = 255.0 * Math.pow(x / 255.0, gamma);
                double w = ss * (1.0 - x / 255.0);  // lift dark tones most, leave bright tones alone
                double y = x + (lifted - x) * w;
                if (y > knee) {
                    y = knee + (y - knee) * (1.0 - 0.5 * ss);
                }
                table[x] = (byte) Math.max(0, Math.min(255, (int) y));
            }
            Mat m = new Mat(1, 256, CvType.CV_8UC1);
            m.put(0, 0, table);
            return m;
        });
    }

    /** Returns the enhanced frame (a new Mat), or the same frame when not active. */
    public Mat process(Mat frame) {
        if ("off".equals(mode)) {
            active = false;
            appliedStrength = 0;
            return frame;
        }
        Mat lab = new Mat();
        Imgproc.cvtColor(frame, lab, Imgproc.COLOR_BGR2Lab);
        List<Mat> ch = new ArrayList<>();
        Core.split(lab, ch);
        score = 0.85 * score + 0.15 * measure(ch.get(0));
        double s;
        if ("on".equals(mode)) {
            active = true;
            s = strength;
        } else {
            if (active && score < offAt) {
                active = false;
            } else if (!active && score > onAt) {
                active = true;
            }
            s = active ? Math.max(0.35, Math.min(1.0, score * 1.4)) : 0;
        }
        appliedStrength = s;
        if (!active) {
            lab.release();
            ch.forEach(Mat::release);
            return frame;
        }
        Mat l = new Mat();
        Core.LUT(ch.get(0), lut(s), l);
        clahe.apply(l, ch.get(0));
        l.release();
        Core.merge(ch, lab);
        Mat out = new Mat();
        Imgproc.cvtColor(lab, out, Imgproc.COLOR_Lab2BGR);
        lab.release();
        ch.forEach(Mat::release);
        return out;
    }
}
