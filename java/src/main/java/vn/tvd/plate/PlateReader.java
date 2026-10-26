package vn.tvd.plate;

import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import vn.tvd.Cfg;
import vn.tvd.detect.Detection;
import vn.tvd.detect.YoloDetector;

/**
 * Licence-plate reading: find the plate inside a vehicle crop (YOLO plate detector), then OCR it with the
 * fast-plate-ocr global model (ONNX: uint8 RGB [N, 64, 128, 3] -> per-slot character probabilities).
 * Square plates are read whole and split into two rows; the reading that fits the Vietnamese layout
 * (then the more confident one) wins.
 */
public final class PlateReader implements AutoCloseable {
    private static final String ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_";
    private static final int OCR_H = 64, OCR_W = 128;

    public record Result(String text, String raw, double conf, double detConf, Rect box, boolean twoRows, Mat image) {
    }

    private record Ocr(String text, double conf) {
    }

    private final YoloDetector detector;
    private final double detConf;
    private final OrtEnvironment env = OrtEnvironment.getEnvironment();
    private final OrtSession ocr;
    private final String ocrInput;
    private final String twoRowMode;
    private final double twoRowRatio, minConf;
    private final int minWidth;

    public PlateReader(Cfg cfg) throws OrtException {
        detector = new YoloDetector(Path.of(Cfg.onnxPath(cfg.str("detector", "models/plate-v1s.pt"))), 640, true);
        detConf = cfg.num("det_conf", 0.25);
        String ocrPath = cfg.str("ocr_onnx", System.getProperty("user.home")
                + "/.cache/fast-plate-ocr/cct-s-v2-global-model/cct_s_v2_global.onnx");
        ocr = env.createSession(ocrPath, vn.tvd.detect.Onnx.options());
        ocrInput = ocr.getInputNames().iterator().next();
        twoRowMode = cfg.str("two_row_mode", "best");
        twoRowRatio = cfg.num("two_row_ratio", 0.45);
        minWidth = cfg.integer("min_plate_width", 40);
        minConf = cfg.num("min_conf", 0.6);
    }

    private Ocr ocr(Mat bgr) throws OrtException {
        Mat rgb = new Mat();
        Imgproc.resize(bgr, rgb, new Size(OCR_W, OCR_H), 0, 0, Imgproc.INTER_LINEAR);
        Imgproc.cvtColor(rgb, rgb, Imgproc.COLOR_BGR2RGB);
        byte[] px = new byte[OCR_H * OCR_W * 3];
        rgb.get(0, 0, px);
        rgb.release();
        try (OnnxTensor t = OnnxTensor.createTensor(env, ByteBuffer.wrap(px), new long[] {1, OCR_H, OCR_W, 3},
                ai.onnxruntime.OnnxJavaType.UINT8);
             OrtSession.Result res = ocr.run(Map.of(ocrInput, t))) {
            float[][] slots = ((float[][][]) res.get("plate").get().getValue())[0];
            StringBuilder text = new StringBuilder();
            double[] probs = new double[slots.length];
            for (int i = 0; i < slots.length; i++) {
                int best = 0;
                for (int c = 1; c < slots[i].length; c++) {
                    if (slots[i][c] > slots[i][best]) {
                        best = c;
                    }
                }
                probs[i] = slots[i][best];
                char ch = ALPHABET.charAt(best);
                if (ch != '_') {
                    text.append(ch);
                }
            }
            int n = Math.max(1, text.length());
            double sum = 0;
            for (int i = 0; i < n; i++) {
                sum += probs[i];
            }
            return new Ocr(text.toString(), sum / n);
        }
    }

    /** Raw text, confidence, read as two rows. */
    Object[] readPlate(Mat plate) throws OrtException {
        int h = plate.rows(), w = plate.cols();
        Ocr whole = ocr(plate);
        if ((double) h / Math.max(w, 1) <= twoRowRatio || "whole".equals(twoRowMode)) {
            return new Object[] {whole.text(), whole.conf(), false};
        }
        int cut = h / 2, pad = Math.max(1, h / 20);
        Ocr top = ocr(plate.submat(0, Math.min(h, cut + pad), 0, w));
        Ocr bot = ocr(plate.submat(Math.max(0, cut - pad), h, 0, w));
        String split = top.text() + bot.text();
        double cs = (top.conf() + bot.conf()) / 2;
        if ("split".equals(twoRowMode)) {
            return new Object[] {split, cs, true};
        }
        boolean okWhole = PlateNormalizer.normalize(whole.text(), true) != null;
        boolean okSplit = PlateNormalizer.normalize(split, true) != null;
        boolean useSplit = okSplit != okWhole ? okSplit : cs > whole.conf();
        return useSplit ? new Object[] {split, cs, true} : new Object[] {whole.text(), whole.conf(), false};
    }

    /** Every plate in an image (for evaluation on still photos, where there is no vehicle class). */
    public java.util.List<Result> readAll(Mat img, boolean motorbike) throws OrtException {
        java.util.List<Result> out = new java.util.ArrayList<>();
        for (Detection d : detector.detect(img, detConf, null)) {
            out.add(readBox(img, d, motorbike));
        }
        return out;
    }

    public Result read(Mat vehicle, boolean motorbike) throws OrtException {
        if (vehicle.empty()) {
            return null;
        }
        List<Detection> dets = detector.detect(vehicle, detConf, null);
        if (dets.isEmpty()) {
            return null;
        }
        Detection d = dets.get(0);  // sorted by confidence
        if (d.x2() - d.x1() < minWidth) {
            return null;
        }
        return readBox(vehicle, d, motorbike);
    }

    private Result readBox(Mat vehicle, Detection d, boolean motorbike) throws OrtException {
        int x1 = (int) d.x1(), y1 = (int) d.y1(), x2 = (int) d.x2(), y2 = (int) d.y2();
        // small margin so edge characters are not clipped
        int mx = (int) (0.04 * (x2 - x1)), my = (int) (0.08 * (y2 - y1));
        int H = vehicle.rows(), W = vehicle.cols();
        int px1 = Math.max(0, x1 - mx), py1 = Math.max(0, y1 - my), px2 = Math.min(W, x2 + mx), py2 = Math.min(H, y2 + my);
        Mat plate = vehicle.submat(py1, py2, px1, px2).clone();
        Object[] r = readPlate(plate);
        String raw = (String) r[0];
        double conf = (Double) r[1];
        String text = conf >= minConf ? PlateNormalizer.normalize(raw, motorbike) : null;
        return new Result(text, raw, conf, d.conf(), new Rect(x1, y1, x2 - x1, y2 - y1), (Boolean) r[2], plate);
    }

    @Override
    public void close() throws OrtException {
        detector.close();
        ocr.close();
    }
}
