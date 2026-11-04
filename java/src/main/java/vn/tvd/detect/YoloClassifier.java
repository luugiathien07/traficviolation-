package vn.tvd.detect;

import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * YOLO classification model (`yolo classify`) exported to ONNX: input [1, 3, s, s] RGB 0..1, output [1, classes]
 * probabilities. Pre-processing matches Ultralytics: shorter side resized to s, centre crop s x s.
 */
public final class YoloClassifier implements AutoCloseable {
    public record Result(String name, double conf) {
    }

    private final OrtEnvironment env = OrtEnvironment.getEnvironment();
    private final OrtSession session;
    private final String inputName;
    private final int size;
    private final Map<Integer, String> names = new HashMap<>();

    public YoloClassifier(Path model) throws OrtException {
        session = env.createSession(model.toString(), Onnx.options());
        inputName = session.getInputNames().iterator().next();
        long[] shape = ((ai.onnxruntime.TensorInfo) session.getInputInfo().get(inputName).getInfo()).getShape();
        size = (int) shape[2];
        // "{0: '30', 1: '40'}" in the ONNX metadata
        String s = session.getMetadata().getCustomMetadata().getOrDefault("names", "");
        for (String part : s.replaceAll("[{}]", "").split(",")) {
            String[] kv = part.split(":", 2);
            if (kv.length == 2) {
                names.put(Integer.parseInt(kv[0].trim()), kv[1].trim().replaceAll("^'|'$", ""));
            }
        }
    }

    public Result classify(Mat bgr) throws OrtException {
        double scale = (double) size / Math.min(bgr.cols(), bgr.rows());
        int w = Math.max(size, (int) Math.round(bgr.cols() * scale)), h = Math.max(size, (int) Math.round(bgr.rows() * scale));
        Mat resized = new Mat();
        Imgproc.resize(bgr, resized, new Size(w, h), 0, 0, Imgproc.INTER_LINEAR);
        Mat crop = resized.submat(new Rect((w - size) / 2, (h - size) / 2, size, size));
        byte[] px = new byte[size * size * 3];
        Mat rgb = new Mat();
        Imgproc.cvtColor(crop, rgb, Imgproc.COLOR_BGR2RGB);
        rgb.get(0, 0, px);
        rgb.release();
        resized.release();
        float[] chw = new float[3 * size * size];
        for (int i = 0; i < size * size; i++) {
            for (int c = 0; c < 3; c++) {
                chw[c * size * size + i] = (px[i * 3 + c] & 0xff) / 255f;
            }
        }
        try (OnnxTensor t = OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), new long[] {1, 3, size, size});
             OrtSession.Result res = session.run(Map.of(inputName, t))) {
            float[] p = ((float[][]) res.get(0).getValue())[0];
            int best = 0;
            for (int i = 1; i < p.length; i++) {
                if (p[i] > p[best]) {
                    best = i;
                }
            }
            return new Result(names.getOrDefault(best, String.valueOf(best)), p[best]);
        }
    }

    @Override
    public void close() throws OrtException {
        session.close();
    }
}
