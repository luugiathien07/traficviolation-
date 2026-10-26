package vn.tvd;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;

import vn.tvd.plate.PlateNormalizer;

class PlateAndBacklightTest {
    @BeforeAll
    static void opencv() {
        nu.pattern.OpenCV.loadLocally();
    }

    @Test
    void normalizeVietnamesePlates() {
        assertEquals("51A-123.45", PlateNormalizer.normalize("51A12345", false));
        assertEquals("29-B1 256.62", PlateNormalizer.normalize("29B125662", false));   // only fits the motorbike layout
        assertEquals("30L-908.60", PlateNormalizer.normalize("30L90860", false));      // car layout by default
        assertEquals("30-L9 0860", PlateNormalizer.normalize("30L90860", true));       // motorbike: motorbike layout
        assertEquals("50LD-004.54", PlateNormalizer.normalize("50LD00454", false));
        assertEquals("51A-123.45", PlateNormalizer.normalize("5lA-I23.45", false));    // look-alikes fixed by position
        assertNull(PlateNormalizer.normalize("29PJ8AW3", false));
        assertNull(PlateNormalizer.normalize("ABC", false));
    }

    static Mat backlitFrame() {
        Mat img = new Mat(360, 640, CvType.CV_8UC3, new Scalar(30, 30, 30));          // dark road and vehicles
        img.submat(new Rect(0, 0, 640, 120)).setTo(new Scalar(252, 252, 252));         // blown-out sky
        img.submat(new Rect(200, 200, 130, 60)).setTo(new Scalar(55, 50, 45));         // dark "car"
        return img;
    }

    static double meanAt(Mat m, int y, int x) {
        double[] p = m.get(y, x);
        return (p[0] + p[1] + p[2]) / 3;
    }

    @Test
    void backlightAutoDetectsAndLiftsShadows() {
        Backlight blc = new Backlight(Cfg.parse("mode: auto"));
        Mat frame = backlitFrame(), out = frame;
        for (int i = 0; i < 20; i++) {
            out = blc.process(frame);
        }
        assertTrue(blc.active && blc.score > 0.45, "score " + blc.score);
        assertTrue(meanAt(out, 230, 260) > meanAt(frame, 230, 260) + 20);   // car got brighter
    }

    @Test
    void backlightAutoLeavesEvenSceneAlone() {
        Backlight blc = new Backlight(Cfg.parse("mode: auto"));
        Mat frame = new Mat(360, 640, CvType.CV_8UC3, new Scalar(120, 120, 120)), out = null;
        for (int i = 0; i < 20; i++) {
            out = blc.process(frame);
        }
        assertTrue(!blc.active);
        assertSame(frame, out);
    }

    @Test
    void backlightOffIsPassthrough() {
        Backlight blc = new Backlight(Cfg.parse("mode: \"off\""));
        Mat frame = backlitFrame();
        assertSame(frame, blc.process(frame));
        assertEquals(0, Core.countNonZero(new Mat(1, 1, CvType.CV_8UC1, new Scalar(0))));
    }
}
