package vn.tvd.detect;

import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * YOLO (v8/11) detector exported to ONNX by Ultralytics (`model.export(format="onnx", imgsz=...)`).
 * Output is [1, 4 + classes, anchors] with boxes as centre x, centre y, width, height in input pixels.
 * Pre/post-processing matches Ultralytics predict(): letterbox with grey (114) padding to the smallest multiple
 * of 32 (needs a model exported with dynamic=True, see tools/export_onnx.py; a fixed-size model is padded to its
 * size), best class per anchor, class-aware NMS at IoU 0.7.
 */
public final class YoloDetector implements AutoCloseable {
    private static final double NMS_IOU = 0.7;
    private static final int MAX_DET = 300;

    private final OrtEnvironment env = OrtEnvironment.getEnvironment();
    private final OrtSession session;
    private final String inputName;
    private final int size;
    private final boolean fixed;

    /** @param imgsz longest side of the network input (ignored when the model has a fixed input size) */
    public YoloDetector(Path model, int imgsz) throws OrtException {
        this(model, imgsz, false);
    }

    /**
     * @param square always pad to imgsz x imgsz. For inputs of changing size (vehicle crops) one fixed shape is much
     *               faster on GPU than re-planning for every new shape.
     */
    public YoloDetector(Path model, int imgsz, boolean square) throws OrtException {
        session = env.createSession(model.toString(), Onnx.options());
        inputName = session.getInputNames().iterator().next();
        long[] shape = ((ai.onnxruntime.TensorInfo) session.getInputInfo().get(inputName).getInfo()).getShape();
        fixed = shape[2] > 0 || square;
        size = shape[2] > 0 ? (int) shape[2] : imgsz;
    }

    /** Class names stored by Ultralytics in the ONNX metadata ("{0: 'person', 1: 'bicycle', ...}"). */
    public java.util.Map<Integer, String> names() throws OrtException {
        java.util.Map<Integer, String> out = new java.util.HashMap<>();
        String s = session.getMetadata().getCustomMetadata().getOrDefault("names", "");
        java.util.regex.Matcher m = java.util.regex.Pattern.compile("(\\d+):\\s*'((?:[^'\\\\]|\\\\.)*)'").matcher(s);
        while (m.find()) {
            out.put(Integer.parseInt(m.group(1)), m.group(2));
        }
        return out;
    }

    public int inputSize() {
        return size;
    }

    public boolean fixedSize() {
        return fixed;
    }

    /**
     * @param classes class ids to keep (null = all)
     */
    public List<Detection> detect(Mat bgr, double conf, Set<Integer> classes) throws OrtException {
        int h = bgr.rows(), w = bgr.cols();
        double r = Math.min((double) size / h, (double) size / w);
        int nw = (int) Math.round(w * r), nh = (int) Math.round(h * r);
        int padW = size - nw, padH = size - nh;
        if (!fixed) {  // minimal padding, like Ultralytics LetterBox(auto=True, stride=32)
            padW %= 32;
            padH %= 32;
        }
        int inW = nw + padW, inH = nh + padH;
        int top = (int) Math.round(padH / 2.0 - 0.1), left = (int) Math.round(padW / 2.0 - 0.1);

        Mat resized = new Mat();
        if (nw != w || nh != h) {
            Imgproc.resize(bgr, resized, new Size(nw, nh), 0, 0, Imgproc.INTER_LINEAR);
        } else {
            bgr.copyTo(resized);
        }
        Mat padded = new Mat();
        Core.copyMakeBorder(resized, padded, top, inH - nh - top, left, inW - nw - left,
                Core.BORDER_CONSTANT, new Scalar(114, 114, 114));
        Imgproc.cvtColor(padded, padded, Imgproc.COLOR_BGR2RGB);
        padded.convertTo(padded, CvType.CV_32FC3, 1.0 / 255);

        // HWC -> CHW
        float[] hwc = new float[inW * inH * 3];
        padded.get(0, 0, hwc);
        float[] chw = new float[hwc.length];
        int plane = inW * inH;
        for (int i = 0; i < plane; i++) {
            chw[i] = hwc[3 * i];
            chw[plane + i] = hwc[3 * i + 1];
            chw[2 * plane + i] = hwc[3 * i + 2];
        }
        resized.release();
        padded.release();

        float[][] out;
        try (OnnxTensor t = OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), new long[] {1, 3, inH, inW});
             OrtSession.Result res = session.run(Map.of(inputName, t))) {
            out = ((float[][][]) res.get(0).getValue())[0];
        }
        int nc = out.length - 4, n = out[0].length;
        List<Detection> cands = new ArrayList<>();
        for (int a = 0; a < n; a++) {
            int best = 0;
            float bc = out[4][a];
            for (int c = 1; c < nc; c++) {
                if (out[4 + c][a] > bc) {
                    bc = out[4 + c][a];
                    best = c;
                }
            }
            if (bc < conf || (classes != null && !classes.contains(best))) {
                continue;
            }
            double cx = (out[0][a] - left) / r, cy = (out[1][a] - top) / r;
            double bw = out[2][a] / r, bh = out[3][a] / r;
            double x1 = clamp(cx - bw / 2, w), y1 = clamp(cy - bh / 2, h);
            double x2 = clamp(cx + bw / 2, w), y2 = clamp(cy + bh / 2, h);
            cands.add(new Detection(x1, y1, x2, y2, bc, best));
        }
        return nms(cands);
    }

    private static double clamp(double v, int max) {
        return Math.max(0, Math.min(max, v));
    }

    static List<Detection> nms(List<Detection> cands) {
        cands.sort(Comparator.comparingDouble(Detection::conf).reversed());
        boolean[] removed = new boolean[cands.size()];
        List<Detection> keep = new ArrayList<>();
        for (int i = 0; i < cands.size() && keep.size() < MAX_DET; i++) {
            if (removed[i]) {
                continue;
            }
            Detection a = cands.get(i);
            keep.add(a);
            for (int j = i + 1; j < cands.size(); j++) {
                Detection b = cands.get(j);
                if (!removed[j] && a.cls() == b.cls() && iou(a.xyxy(), b.xyxy()) > NMS_IOU) {
                    removed[j] = true;
                }
            }
        }
        return keep;
    }

    public static double iou(double[] a, double[] b) {
        double iw = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
        double ih = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
        double inter = iw * ih;
        double union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
        return union <= 0 ? 0 : inter / union;
    }

    @Override
    public void close() throws OrtException {
        session.close();
    }
}
