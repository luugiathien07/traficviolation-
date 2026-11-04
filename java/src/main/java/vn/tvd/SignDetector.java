package vn.tvd;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import ai.onnxruntime.OrtException;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import vn.tvd.detect.Detection;
import vn.tvd.detect.YoloClassifier;
import vn.tvd.detect.YoloDetector;

/**
 * Traffic-sign detection (biển báo) with a YOLO model trained on Vietnamese signs (tools/train_signs.py).
 * Runs every every_n_frames frames on the whole frame and keeps a short memory of what is visible, so the
 * dashboard and the video overlay can show the signs at this camera, and the rules that need a sign
 * ({@code require_sign}) know it is there. The model has one "speed limit" class for every limit; a small
 * classifier (speed_model) reads the number, listed as an extra visible sign "speed limit 50".
 */
public final class SignDetector implements AutoCloseable {
    /** Class names of the Vietnamese traffic-sign dataset -> Vietnamese label shown to users. */
    static final Map<String, String> LABELS_VI = Map.ofEntries(
            Map.entry("one way prohibition", "Cấm đi ngược chiều"),
            Map.entry("speed limit", "Giới hạn tốc độ"),
            Map.entry("no parking", "Cấm đỗ xe"),
            Map.entry("intersection danger", "Nguy hiểm: giao nhau"),
            Map.entry("indication", "Biển hiệu lệnh"),
            Map.entry("no stopping and parking", "Cấm dừng và đỗ xe"),
            Map.entry("other prohibition", "Biển cấm khác"),
            Map.entry("vehicle permission lane", "Làn theo loại xe"),
            Map.entry("road danger", "Nguy hiểm trên đường"),
            Map.entry("direction", "Chỉ hướng đi"),
            Map.entry("slow warning", "Đi chậm"),
            Map.entry("no turn left", "Cấm rẽ trái"),
            Map.entry("pedestrian danger", "Nguy hiểm: người đi bộ"),
            Map.entry("no truck entry/turning", "Cấm xe tải"),
            Map.entry("no car entry/turning", "Cấm ô tô"),
            Map.entry("pedestrian crossing", "Đường người đi bộ sang ngang"),
            Map.entry("no motorbike entry/turning", "Cấm xe máy"),
            Map.entry("no turn right", "Cấm rẽ phải"),
            Map.entry("vehicle and speed permission lane", "Làn theo loại xe và tốc độ"),
            Map.entry("other warning", "Biển cảnh báo khác"),
            Map.entry("other", "Biển khác"),
            Map.entry("height limit", "Hạn chế chiều cao"),
            Map.entry("no u turn", "Cấm quay đầu"),
            Map.entry("no more prohibition", "Hết lệnh cấm"),
            Map.entry("construction danger", "Công trường"),
            Map.entry("no u and left turn", "Cấm quay đầu và rẽ trái"),
            Map.entry("weight limit", "Hạn chế tải trọng"),
            Map.entry("no u and right turn", "Cấm quay đầu và rẽ phải"),
            Map.entry("overpass route", "Lối đi cầu vượt"));
    private static final Scalar SIGN_BGR = new Scalar(0, 200, 255);

    private static final class Seen {
        String label;
        double conf, last;
    }

    private final YoloDetector model;
    private final Map<Integer, String> names;
    private final double conf, hold;
    private final int every;
    private int n;
    private final Map<String, Seen> visible = new LinkedHashMap<>();
    private List<Detection> boxes = new ArrayList<>();
    /** Reads the number on speed-limit signs (tools/prepare_speed_signs.py); null when not configured. */
    private final YoloClassifier speed;
    private final double speedMinConf;
    /** (t, km/h) readings of speed-limit signs; the value shown and used is the majority within hold_seconds. */
    private final java.util.ArrayDeque<double[]> speedVotes = new java.util.ArrayDeque<>();
    private final Map<Detection, String> speedText = new HashMap<>();

    public SignDetector(Cfg cfg) throws OrtException {
        model = new YoloDetector(Path.of(Cfg.onnxPath(cfg.str("model", "models/vn_signs.pt"))), cfg.integer("imgsz", 1600));
        names = model.names();
        conf = cfg.num("conf", 0.4);
        every = cfg.integer("every_n_frames", 5);
        hold = cfg.num("hold_seconds", 2.0);  // keep a sign listed this long after last seeing it
        String sm = cfg.str("speed_model", "models/speed_value.pt");
        speed = Files.exists(Path.of(Cfg.onnxPath(sm))) ? new YoloClassifier(Path.of(Cfg.onnxPath(sm))) : null;
        speedMinConf = cfg.num("speed_min_conf", 0.8);  // below this a reading is left out (blurry, far away)
    }

    public static String labelVi(String name) {
        String n = name.toLowerCase(Locale.ROOT).replace('_', ' ');
        if (n.matches("speed limit \\d+")) {
            return "Giới hạn tốc độ " + n.substring(12) + " km/h";
        }
        return LABELS_VI.getOrDefault(n, name);
    }

    /** Read the number of each speed-limit sign in this run's boxes and vote. */
    private void readSpeeds(Mat frame, double t) throws OrtException {
        speedText.clear();
        if (speed == null) {
            return;
        }
        for (Detection d : boxes) {
            if (!"speed limit".equals(name(d.cls())) || d.x2() - d.x1() < 16) {
                continue;
            }
            int x1 = (int) Math.max(0, d.x1()), y1 = (int) Math.max(0, d.y1());
            int x2 = (int) Math.min(frame.cols(), d.x2()), y2 = (int) Math.min(frame.rows(), d.y2());
            YoloClassifier.Result r = speed.classify(frame.submat(y1, y2, x1, x2));
            if (r.conf() >= speedMinConf) {
                speedVotes.addLast(new double[] {t, Double.parseDouble(r.name())});
                speedText.put(d, r.name());
            }
        }
    }

    /** The majority speed-limit reading of the last hold_seconds, or null. */
    private Integer speedLimit(double t) {
        while (!speedVotes.isEmpty() && t - speedVotes.peekFirst()[0] > hold) {
            speedVotes.removeFirst();
        }
        Map<Integer, Integer> count = new HashMap<>();
        for (double[] v : speedVotes) {
            count.merge((int) v[1], 1, Integer::sum);
        }
        return count.entrySet().stream().max(Map.Entry.comparingByValue()).map(Map.Entry::getKey).orElse(null);
    }

    private String name(int cls) {
        return names.getOrDefault(cls, "sign " + cls);
    }

    public void update(Mat frame, double t) throws OrtException {
        n++;
        if (every == 1 || n % every == 1) {
            boxes = model.detect(frame, conf, null);
            for (Detection d : boxes) {
                String nm = name(d.cls());
                Seen s = visible.get(nm);
                if (s == null || t - s.last > hold || d.conf() >= s.conf) {
                    s = new Seen();
                    s.label = labelVi(nm);
                    s.conf = d.conf();
                    visible.put(nm, s);
                }
                s.last = t;
            }
            readSpeeds(frame, t);
        }
        // "speed limit 50": the number read from the sign, listed like a sign of its own for the rules
        visible.keySet().removeIf(k -> k.startsWith("speed limit "));
        Integer kmh = speedLimit(t);
        if (kmh != null && visible.containsKey("speed limit")) {
            Seen s = new Seen();
            s.label = labelVi("speed limit " + kmh);
            s.conf = visible.get("speed limit").conf;
            s.last = visible.get("speed limit").last;
            visible.put("speed limit " + kmh, s);
        }
        visible.entrySet().removeIf(e -> t - e.getValue().last > hold);
        boxes = boxes.stream().filter(d -> visible.containsKey(name(d.cls()))).toList();
    }

    /** Model class names of the signs currently visible (within hold_seconds). */
    public java.util.Set<String> visibleNames() {
        return java.util.Set.copyOf(visible.keySet());
    }

    public List<Map<String, Object>> summary() {
        List<Map<String, Object>> out = new ArrayList<>();
        visible.entrySet().stream().sorted(Comparator.comparingDouble(e -> -e.getValue().conf)).forEach(e ->
                out.add(Map.of("name", e.getKey(), "label", e.getValue().label,
                        "conf", Math.round(e.getValue().conf * 100) / 100.0)));
        return out;
    }

    public void draw(Mat img) {
        for (Detection d : boxes) {
            Point p1 = new Point((int) d.x1(), (int) d.y1()), p2 = new Point((int) d.x2(), (int) d.y2());
            Imgproc.rectangle(img, p1, p2, SIGN_BGR, 2);
            String nm = speedText.containsKey(d) ? "speed limit " + speedText.get(d) : name(d.cls());
            String text = Pipeline.ascii(String.format(Locale.ROOT, "%s %.2f", labelVi(nm), d.conf()));
            int[] base = new int[1];
            Size ts = Imgproc.getTextSize(text, Imgproc.FONT_HERSHEY_SIMPLEX, 0.45, 1, base);
            double ty = p1.y < ts.height + 8 ? p2.y + ts.height + 6 : p1.y;
            Imgproc.rectangle(img, new Point(p1.x, ty - ts.height - 6), new Point(p1.x + ts.width + 6, ty), SIGN_BGR, -1);
            Imgproc.putText(img, text, new Point(p1.x + 3, ty - 4), Imgproc.FONT_HERSHEY_SIMPLEX, 0.45,
                    new Scalar(20, 20, 20), 1, Imgproc.LINE_AA);
        }
    }

    @Override
    public void close() throws OrtException {
        model.close();
        if (speed != null) {
            speed.close();
        }
    }
}
