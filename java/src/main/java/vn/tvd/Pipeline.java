package vn.tvd;

import java.nio.file.Files;
import java.nio.file.Path;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.BiConsumer;
import java.util.logging.Logger;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import vn.tvd.detect.Detection;
import vn.tvd.detect.YoloDetector;
import vn.tvd.plate.PlateReader;
import vn.tvd.rules.Rule;
import vn.tvd.rules.Rules;
import vn.tvd.rules.Track;
import vn.tvd.rules.Violation;
import vn.tvd.store.Store;
import vn.tvd.track.ByteTracker;
import vn.tvd.track.STrack;
import vn.tvd.video.FfmpegReader;
import vn.tvd.video.FfmpegWriter;

/** Video -> detect & track -> rules -> evidence. Runs in its own thread when served. */
public final class Pipeline {
    private static final Logger LOG = Logger.getLogger("tvd");

    static final Map<String, Integer> COCO_IDS = Map.of("bicycle", 1, "car", 2, "motorcycle", 3, "bus", 5, "truck", 7);
    static final Map<Integer, String> COCO_NAMES = Map.of(1, "bicycle", 2, "car", 3, "motorcycle", 5, "bus", 7, "truck");
    static final Map<String, Scalar> LIGHT_BGR = Map.of("red", new Scalar(60, 60, 255), "yellow", new Scalar(40, 190, 250),
            "green", new Scalar(100, 210, 60), "unknown", new Scalar(160, 160, 160));
    static final Map<String, Scalar> TYPE_BGR = Map.of("red_light", new Scalar(60, 60, 255),
            "speeding", new Scalar(40, 150, 255), "wrong_lane", new Scalar(255, 120, 180),
            "no_helmet", new Scalar(255, 190, 60), "no_turn", new Scalar(200, 80, 255));
    private static final Scalar TRACK_BGR = new Scalar(200, 210, 80);

    private final Cfg cfg;
    private final Store store;
    public final String cameraId;
    public final String source;
    private final double conf;
    private final Set<Integer> classIds = new HashSet<>();
    final boolean loop;
    boolean realtime;
    public final LightSource light;
    public final Backlight backlight;
    private final List<Rule> rules;
    private final Cfg plateCfg, helmetCfg;
    private PlateReader plateReader;
    private HelmetChecker helmet;
    private YoloDetector detector;
    private final ByteTracker tracker = new ByteTracker();
    private final Map<Integer, Track> tracks = new LinkedHashMap<>();
    private final Set<Integer> seenIds = new HashSet<>();
    private final Map<Integer, Mat> plateImg = new HashMap<>();

    /** Plate reading retried on later frames: track id -> rows waiting for a plate. */
    private static final class PlateJob {
        final List<Long> rows = new ArrayList<>();
        double until, next;
    }

    private final Map<Integer, PlateJob> plateJobs = new LinkedHashMap<>();

    public volatile byte[] latestJpeg;
    public volatile int[] frameSize;
    public final Map<String, Object> stats = new ConcurrentHashMap<>();
    private final Map<String, Integer> byClass = new ConcurrentHashMap<>();
    private int vehiclesSeen;
    public volatile boolean stop;
    public BiConsumer<Long, Violation> onViolation;
    /** Debug: TVD_DUMP_DETS=file writes every detection (t cls x1 y1 x2 y2 conf), to compare with the Python version. */
    private final java.io.PrintWriter dump = openDump();

    private static java.io.PrintWriter openDump() {
        String f = System.getenv("TVD_DUMP_DETS");
        try {
            return f == null ? null : new java.io.PrintWriter(new java.io.FileWriter(f), true);
        } catch (java.io.IOException e) {
            throw new java.io.UncheckedIOException(e);
        }
    }

    public Pipeline(Cfg cfg, Store store) {
        this.cfg = cfg;
        this.store = store;
        cameraId = cfg.str("camera_id", "cam01");
        source = cfg.str("source", "");
        Cfg det = cfg.sub("detector");
        conf = det.num("conf", 0.3);
        for (String c : det.strings("classes", Rule.VEHICLES)) {
            classIds.add(COCO_IDS.get(c));
        }
        loop = cfg.bool("loop", false);
        realtime = cfg.bool("realtime", true);
        light = new LightSource(cfg.sub("light"));
        backlight = new Backlight(cfg.sub("backlight"));
        rules = Rules.build(cfg);
        plateCfg = cfg.sub("plate");
        helmetCfg = cfg.sub("helmet");
        stats.put("frames", 0);
        stats.put("fps", 0.0);
        stats.put("active_tracks", 0);
        stats.put("vehicles_seen", 0);
        stats.put("by_class", byClass);
        stats.put("light", "unknown");
        stats.put("running", false);
        stats.put("source", source);
    }

    private void load() throws Exception {
        String model = Cfg.onnxPath(cfg.sub("detector").str("model", "models/yolo11n.pt"));
        int imgsz = cfg.sub("detector").integer("imgsz", 960);
        detector = new YoloDetector(Path.of(model), imgsz);
        if (imgsz != detector.inputSize()) {
            LOG.warning(String.format("detector.imgsz=%d nhưng %s xuất ở %d; dùng %d (xuất lại bằng tools/export_onnx.py)",
                    imgsz, model, detector.inputSize(), detector.inputSize()));
        }
        if (helmetCfg.enabled() && helmet == null) {
            String hm = Cfg.onnxPath(helmetCfg.str("model", "models/helmet.pt"));
            if (Files.exists(Path.of(hm))) {
                helmet = new HelmetChecker(helmetCfg);
            } else {
                LOG.warning("Chưa có model mũ bảo hiểm (" + hm + "); bỏ qua luật mũ bảo hiểm");
            }
        }
        if (plateCfg.enabled() && plateReader == null) {
            plateReader = new PlateReader(plateCfg);
        }
    }

    // ---------- main loop ----------
    public void run(Integer maxFrames, String writerPath) throws Exception {
        load();
        boolean isFile = FfmpegReader.isFile(source);
        FfmpegWriter writer = null;
        long frameIdx = 0, tStart = System.nanoTime(), tLast = tStart;
        double fpsStat = 0;
        stats.put("running", true);
        try (FfmpegReader cap = new FfmpegReader(source)) {
            double fps = cap.fps;
            LOG.info(String.format(Locale.ROOT, "Đang chạy %s (fps=%.1f, %dx%d)", source, fps, cap.width, cap.height));
            while (!stop) {
                Mat frame = cap.read();
                if (frame == null) {
                    if (isFile && loop) {
                        cap.start(0);
                        tracker.reset();  // new track ids between loops
                        tracks.clear();
                        continue;
                    }
                    break;
                }
                // Video time: frame clock for files (exact), wall clock for live streams.
                double t = isFile ? frameIdx / fps : (System.nanoTime() - tStart) / 1e9;
                Mat annotated = process(frame, t);
                if (writerPath != null) {
                    if (writer == null) {
                        writer = new FfmpegWriter(writerPath, annotated.cols(), annotated.rows(), fps);
                    }
                    writer.write(annotated);
                }
                annotated.release();
                frame.release();
                frameIdx++;
                long now = System.nanoTime();
                fpsStat = 0.9 * fpsStat + 0.1 / Math.max(1e-6, (now - tLast) / 1e9);
                stats.put("fps", fpsStat);
                tLast = now;
                if (realtime && isFile) {
                    double lag = frameIdx / fps - (now - tStart) / 1e9;
                    if (lag > 0) {
                        Thread.sleep((long) (lag * 1000));
                    }
                }
                if (maxFrames != null && frameIdx >= maxFrames) {
                    break;
                }
            }
        } finally {
            if (writer != null) {
                writer.close();
            }
            stats.put("running", false);
        }
    }

    public int vehiclesSeen() {
        return vehiclesSeen;
    }

    public Mat process(Mat raw, double t) throws Exception {
        frameSize = new int[] {raw.cols(), raw.rows()};
        String lightState = light.update(raw, t);  // raw frame: enhancement would shift lamp colours
        Mat frame = backlight.process(raw);
        List<Detection> dets = detector.detect(frame, conf, classIds);
        if (dump != null) {
            for (Detection d : dets) {
                dump.printf(Locale.ROOT, "%.2f %d %.1f %.1f %.1f %.1f %.3f%n", t, d.cls(), d.x1(), d.y1(), d.x2(), d.y2(), d.conf());
            }
        }
        List<Violation> violations = new ArrayList<>();
        List<Track> seen = new ArrayList<>();
        for (STrack st : tracker.update(dets)) {
            String cname = COCO_NAMES.getOrDefault(st.cls(), "car");
            double[] box = st.xyxy();
            Track tr = tracks.get(st.id());
            if (tr == null) {
                tr = new Track(st.id(), cname, st.score(), box, t);
                tracks.put(st.id(), tr);
                if (seenIds.add(st.id())) {
                    vehiclesSeen++;
                    byClass.merge(cname, 1, Integer::sum);
                }
            }
            tr.bbox = box;
            tr.lastSeen = t;
            tr.conf = Math.max(tr.conf, st.score());
            tr.addHistory(t);
            seen.add(tr);
        }
        // helmet votes need this frame's boxes of every track, so rules run after all tracks moved
        if (helmet != null) {
            helmet.update(frame, seen);
        }
        for (Track tr : seen) {
            for (Rule rule : rules) {
                violations.addAll(rule.update(tr, lightState));
            }
        }
        // forget tracks not seen for 3 s
        for (Iterator<Map.Entry<Integer, Track>> it = tracks.entrySet().iterator(); it.hasNext(); ) {
            Map.Entry<Integer, Track> e = it.next();
            if (t - e.getValue().lastSeen > 3.0) {
                it.remove();
                Mat m = plateImg.remove(e.getKey());
                if (m != null) {
                    m.release();
                }
            }
        }

        Mat annotated = annotate(frame, t, lightState);
        for (Violation v : violations) {
            save(frame, annotated, v, lightState);
        }
        if (plateReader != null) {
            readPlates(frame, t);
            if (plateCfg.bool("read_all", false)) {
                readAllPlates(frame, seen, t);
            }
        }

        stats.merge("frames", 1, (a, b) -> (Integer) a + (Integer) b);
        stats.put("vehicles_seen", vehiclesSeen);
        stats.put("active_tracks", (int) tracks.values().stream().filter(v -> v.lastSeen == t).count());
        stats.put("light", lightState);
        stats.put("backlight", Map.of("mode", backlight.mode, "active", backlight.active,
                "score", Math.round(backlight.score * 1000) / 1000.0,
                "strength", Math.round(backlight.appliedStrength * 100) / 100.0));
        MatOfByte buf = new MatOfByte();
        if (Imgcodecs.imencode(".jpg", annotated, buf, new MatOfInt(Imgcodecs.IMWRITE_JPEG_QUALITY, 80))) {
            latestJpeg = buf.toArray();
        }
        buf.release();
        if (frame != raw) {
            frame.release();
        }
        return annotated;
    }

    private void save(Mat raw, Mat annotated, Violation v, String lightState) throws Exception {
        Track tr = v.track();
        int x1 = (int) tr.bbox[0], y1 = (int) tr.bbox[1], x2 = (int) tr.bbox[2], y2 = (int) tr.bbox[3];
        int h = raw.rows(), w = raw.cols();
        int pad = (int) (0.15 * Math.max(x2 - x1, y2 - y1)) + 8;
        Mat crop = raw.submat(clampRect(x1 - pad, y1 - pad, x2 + pad, y2 + pad, w, h)).clone();
        Mat evidence = annotated.clone();
        Scalar color = TYPE_BGR.getOrDefault(v.type(), new Scalar(0, 0, 255));
        Imgproc.rectangle(evidence, new Point(x1 - 4, y1 - 4), new Point(x2 + 4, y2 + 4), color, 4);
        banner(evidence, v.type().toUpperCase(Locale.ROOT) + "  #" + tr.id + "  " + v.detail(), color);
        long rowId = store.add(cameraId, v.t(), v.type(), tr.id, tr.cls, tr.conf, v.speedKmh(), lightState, v.detail(),
                evidence, crop);
        LOG.info(String.format("VI PHẠM #%d %s track=%d %s (%s)", rowId, v.type(), tr.id, tr.cls, v.detail()));
        if (plateReader != null) {
            if (tr.plate != null) {
                store.setPlateAuto(rowId, tr.plate, tr.plateConf, plateImg.getOrDefault(tr.id, crop));
            } else {
                PlateJob job = plateJobs.computeIfAbsent(tr.id, k -> {
                    PlateJob j = new PlateJob();
                    j.next = v.t();
                    return j;
                });
                job.rows.add(rowId);
                job.until = v.t() + plateCfg.num("retry_seconds", 2.0);
            }
        }
        if (onViolation != null) {
            onViolation.accept(rowId, v);
        }
        evidence.release();
        crop.release();
    }

    /**
     * Try to read the plate of every vehicle with a pending violation; keep the most confident reading and
     * write it to all of that vehicle's violations.
     */
    private void readPlates(Mat frame, double t) throws Exception {
        double step = plateCfg.num("retry_every", 0.1);
        for (Iterator<Map.Entry<Integer, PlateJob>> it = plateJobs.entrySet().iterator(); it.hasNext(); ) {
            Map.Entry<Integer, PlateJob> e = it.next();
            int tid = e.getKey();
            PlateJob job = e.getValue();
            Track tr = tracks.get(tid);
            if (tr == null || t > job.until) {
                if (tr == null || tr.plate == null) {
                    LOG.info("Không đọc được biển số xe #" + tid);
                }
                it.remove();
                continue;
            }
            if (tr.lastSeen != t || t < job.next) {
                continue;
            }
            job.next = t + step;
            Mat vehicle = frame.submat(clampRect((int) tr.bbox[0], (int) tr.bbox[1], (int) tr.bbox[2], (int) tr.bbox[3],
                    frame.cols(), frame.rows()));
            PlateReader.Result res = plateReader.read(vehicle, "motorcycle".equals(tr.cls));
            if (res != null && res.text() != null && res.conf() > tr.plateConf) {
                tr.plate = res.text();
                tr.plateConf = res.conf();
                Mat old = plateImg.put(tid, res.image());
                if (old != null) {
                    old.release();
                }
                for (long rowId : job.rows) {
                    store.setPlateAuto(rowId, res.text(), res.conf(), res.image());
                }
                LOG.info(String.format(Locale.ROOT, "Biển số xe #%d: %s (%.2f)", tid, res.text(), res.conf()));
                if (res.conf() >= 0.95) {
                    it.remove();
                }
            }
        }
    }

    /**
     * plate.read_all: read the plate of every vehicle on screen (not only violators) so it can be shown live,
     * at most every read_all_every seconds per vehicle and only while no confident reading exists yet.
     */
    private void readAllPlates(Mat frame, List<Track> seen, double t) throws Exception {
        double every = plateCfg.num("read_all_every", 0.5);
        double minW = plateCfg.num("read_all_min_width", 80);  // vehicle box width; smaller plates can't be read
        for (Track tr : seen) {
            if (tr.plateConf >= 0.95 || t < tr.plateNext || tr.bbox[2] - tr.bbox[0] < minW) {
                continue;
            }
            tr.plateNext = t + every;
            Mat vehicle = frame.submat(clampRect((int) tr.bbox[0], (int) tr.bbox[1], (int) tr.bbox[2], (int) tr.bbox[3],
                    frame.cols(), frame.rows()));
            PlateReader.Result res = plateReader.read(vehicle, "motorcycle".equals(tr.cls));
            if (res != null && res.text() != null && res.conf() > tr.plateConf) {
                tr.plate = res.text();
                tr.plateConf = res.conf();
                Mat old = plateImg.put(tr.id, res.image());
                if (old != null) {
                    old.release();
                }
            }
        }
    }

    private static Rect clampRect(int x1, int y1, int x2, int y2, int w, int h) {
        x1 = Math.max(0, Math.min(w - 1, x1));
        y1 = Math.max(0, Math.min(h - 1, y1));
        x2 = Math.max(x1 + 1, Math.min(w, x2));
        y2 = Math.max(y1 + 1, Math.min(h, y2));
        return new Rect(x1, y1, x2 - x1, y2 - y1);
    }

    // ---------- drawing ----------
    private static Point pt(double[] p) {
        return new Point((int) p[0], (int) p[1]);
    }

    private static Scalar bgr(double[] c, Scalar def) {
        return c == null ? def : new Scalar(c[0], c[1], c[2]);
    }

    public Mat annotate(Mat frame, double t, String lightState) {
        Mat img = frame.clone();
        Cfg r = cfg.sub("rules");
        Scalar lc = LIGHT_BGR.getOrDefault(lightState, LIGHT_BGR.get("unknown"));
        int font = Imgproc.FONT_HERSHEY_SIMPLEX, aa = Imgproc.LINE_AA;
        if (r.sub("wrong_lane").enabled()) {
            Mat overlay = img.clone();
            for (Cfg ln : r.sub("wrong_lane").list("lanes")) {
                double[][] poly = ln.points("polygon");
                Point[] pts = new Point[poly.length];
                for (int i = 0; i < poly.length; i++) {
                    pts[i] = pt(poly[i]);
                }
                Imgproc.fillPoly(overlay, List.of(new MatOfPoint(pts)), bgr(ln.numbers("color"), new Scalar(200, 160, 60)));
                Imgproc.putText(img, ln.str("name", ""), pts[0], font, 0.45, new Scalar(255, 255, 255), 1, aa);
            }
            Core.addWeighted(overlay, 0.18, img, 0.82, 0, img);
            overlay.release();
        }
        if (r.sub("speeding").enabled()) {
            for (String key : List.of("line_a", "line_b")) {
                double[][] l = r.sub("speeding").points(key);
                Imgproc.line(img, pt(l[0]), pt(l[1]), new Scalar(40, 190, 250), 1, aa);
            }
        }
        if (r.sub("red_light").enabled()) {
            double[][] l = r.sub("red_light").points("stop_line");
            Imgproc.line(img, pt(l[0]), pt(l[1]), lc, 3, aa);
        }
        if (r.sub("no_turn").enabled()) {
            double[][] l = r.sub("no_turn").points("entry_line");
            Imgproc.line(img, pt(l[0]), pt(l[1]), new Scalar(230, 230, 230), 2, aa);
            for (Cfg ex : r.sub("no_turn").list("banned_exits")) {
                double[][] e = ex.points("line");
                Imgproc.line(img, pt(e[0]), pt(e[1]), TYPE_BGR.get("no_turn"), 2, aa);
                Imgproc.putText(img, "cam " + ascii(ex.str("name", "")), pt(e[0]), font, 0.45, TYPE_BGR.get("no_turn"), 1, aa);
            }
        }
        if ("roi".equals(light.mode) && light.roi != null) {
            Imgproc.rectangle(img, new Point(light.roi[0], light.roi[1]), new Point(light.roi[2], light.roi[3]), lc, 2);
        }

        for (Track tr : tracks.values()) {
            if (tr.lastSeen != t) {
                continue;
            }
            int x1 = (int) tr.bbox[0], y1 = (int) tr.bbox[1], x2 = (int) tr.bbox[2], y2 = (int) tr.bbox[3];
            Scalar color = tr.flagged.isEmpty() ? TRACK_BGR : TYPE_BGR.get(tr.flagged.first());
            Imgproc.rectangle(img, new Point(x1, y1), new Point(x2, y2), color, tr.flagged.isEmpty() ? 1 : 2);
            StringBuilder label = new StringBuilder(String.format(Locale.ROOT, "#%d %s %.2f", tr.id, tr.cls, tr.conf));
            if (tr.speedKmh != null) {
                label.append(String.format(Locale.ROOT, " %.0fkm/h", tr.speedKmh));
            }
            if (tr.plate != null) {
                label.append(" [").append(tr.plate).append("]");
            }
            int[] base = new int[1];
            Size ts = Imgproc.getTextSize(label.toString(), font, 0.42, 1, base);
            Imgproc.rectangle(img, new Point(x1, y1 - ts.height - 6), new Point(x1 + ts.width + 6, y1), color, -1);
            Imgproc.putText(img, label.toString(), new Point(x1 + 3, y1 - 4), font, 0.42, new Scalar(20, 20, 20), 1, aa);
            if (tr.history.size() > 1) {
                List<Point> pts = new ArrayList<>();
                for (double[] h : tr.history) {
                    pts.add(new Point((int) h[1], (int) h[2]));
                }
                Imgproc.polylines(img, List.of(new MatOfPoint(pts.toArray(new Point[0]))), false, color, 1, aa);
            }
        }

        String hud = String.format(Locale.ROOT, "%s  t=%6.1fs  den: %s%s  xe: %d", cameraId, t,
                lightState.toUpperCase(Locale.ROOT), light.overridden() ? " (tay)" : "", vehiclesSeen);
        if (backlight.active) {
            hud += String.format(Locale.ROOT, "  BLC %.1f", backlight.appliedStrength);
        }
        Imgproc.rectangle(img, new Point(0, 0), new Point(img.cols(), 26), new Scalar(15, 15, 15), -1);
        Imgproc.circle(img, new Point(14, 13), 7, lc, -1);
        Imgproc.putText(img, hud, new Point(28, 18), font, 0.5, new Scalar(235, 235, 235), 1, aa);
        return img;
    }

    /** OpenCV's Hershey fonts have no Vietnamese glyphs; keep drawn text ASCII. */
    static String ascii(String s) {
        String n = Normalizer.normalize(s.replace('đ', 'd').replace('Đ', 'D'), Normalizer.Form.NFKD);
        return n.replaceAll("[^\\x00-\\x7F]", "");
    }

    private static void banner(Mat img, String text, Scalar color) {
        int h = img.rows();
        Imgproc.rectangle(img, new Point(0, h - 30), new Point(img.cols(), h), color, -1);
        Imgproc.putText(img, ascii(text), new Point(10, h - 9), Imgproc.FONT_HERSHEY_SIMPLEX, 0.55,
                new Scalar(255, 255, 255), 1, Imgproc.LINE_AA);
    }

    /** Draw the configured lines/zones on one frame with a coordinate grid, to help place them. */
    public Mat preview(int frameNo) throws Exception {
        try (FfmpegReader cap = new FfmpegReader(source)) {
            if (frameNo > 0) {
                cap.start(frameNo);
            }
            Mat frame = cap.read();
            if (frame == null) {
                throw new IllegalStateException("Không đọc được khung hình từ " + source);
            }
            Mat img = annotate(backlight.process(frame), 0.0, light.update(frame, 0.0));
            int w = img.cols(), h = img.rows();
            for (int x = 0; x < w; x += 50) {
                Imgproc.line(img, new Point(x, 0), new Point(x, h), new Scalar(90, 90, 90), 1);
                Imgproc.putText(img, String.valueOf(x), new Point(x + 2, h - 4), Imgproc.FONT_HERSHEY_PLAIN, 0.8,
                        new Scalar(255, 255, 255), 1);
            }
            for (int y = 0; y < h; y += 50) {
                Imgproc.line(img, new Point(0, y), new Point(w, y), new Scalar(90, 90, 90), 1);
                Imgproc.putText(img, String.valueOf(y), new Point(2, y - 2), Imgproc.FONT_HERSHEY_PLAIN, 0.8,
                        new Scalar(255, 255, 255), 1);
            }
            return img;
        }
    }
}
