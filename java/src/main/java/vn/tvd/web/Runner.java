package vn.tvd.web;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import vn.tvd.Cfg;
import vn.tvd.Pipeline;
import vn.tvd.store.Store;

/**
 * Runs one pipeline at a time for the dashboard and switches between demos (one button per violation type,
 * listed in a demos file) or an uploaded video. All demos share one Store; the dashboard shows the violations
 * of the running demo by its camera_id.
 *
 * <pre>
 * demos:
 *   - key: den_do
 *     title: Vượt đèn đỏ (đèn thật)
 *     config: configs/hcm_redlight_real.yaml
 * upload_templates:            # rule sets offered for an uploaded video
 *   - key: chung
 *     title: Không cần vẽ vùng (biển báo, mũ bảo hiểm, biển số)
 *     config: configs/upload_generic.yaml
 * </pre>
 */
public final class Runner {
    private static final Logger LOG = Logger.getLogger("tvd");

    public record Entry(String key, String title, String config, String note) {
    }

    public final List<Entry> demos = new ArrayList<>();
    public final List<Entry> templates = new ArrayList<>();
    private final Store store;
    private final boolean fast;
    private volatile Pipeline current;
    private volatile String currentKey;
    private Thread thread;

    public Runner(Store store, Cfg demosCfg, boolean fast) {
        this.store = store;
        this.fast = fast;
        for (Cfg d : demosCfg.list("demos")) {
            demos.add(new Entry(d.str("key", ""), d.str("title", ""), d.str("config", ""), d.str("note", "")));
        }
        for (Cfg d : demosCfg.list("upload_templates")) {
            templates.add(new Entry(d.str("key", ""), d.str("title", ""), d.str("config", ""), d.str("note", "")));
        }
    }

    public Pipeline current() {
        return current;
    }

    public String currentKey() {
        return currentKey;
    }

    private static Entry find(List<Entry> list, String key) {
        return list.stream().filter(e -> e.key().equals(key)).findFirst().orElse(null);
    }

    /** Start the demo with this key; false if there is no such demo. */
    public synchronized boolean startDemo(String key) throws Exception {
        Entry e = find(demos, key);
        if (e == null) {
            return false;
        }
        Cfg cfg = Cfg.load(Path.of(e.config()));
        if (fast) {
            cfg.put("realtime", false);
        }
        start(key, cfg);
        return true;
    }

    /**
     * Run an uploaded video once with the rules of a template config. Every frame is processed (no realtime
     * skipping), so no violation is missed even when processing is slower than the video.
     */
    public synchronized boolean startUpload(Path video, String templateKey) throws Exception {
        Entry e = find(templates, templateKey);
        if (e == null) {
            e = find(demos, templateKey);  // a demo's zones fit a video from the same camera
        }
        if (e == null || !Files.exists(video)) {
            return false;
        }
        Cfg cfg = Cfg.load(Path.of(e.config()));
        String name = video.getFileName().toString();
        cfg.put("source", video.toString());
        cfg.put("camera_id", "upload-" + name.replaceAll("\\.[^.]*$", ""));
        cfg.put("loop", false);
        cfg.put("realtime", false);
        start("upload:" + name, cfg);
        return true;
    }

    public synchronized void start(String key, Cfg cfg) throws Exception {
        stop();
        Pipeline p = new Pipeline(cfg, store);
        Thread t = new Thread(() -> {
            try {
                p.run(null, null);
            } catch (Exception ex) {
                LOG.log(Level.SEVERE, "Pipeline dừng do lỗi", ex);
            }
        }, "pipeline-" + key);
        t.setDaemon(true);
        current = p;
        currentKey = key;
        thread = t;
        t.start();
        LOG.info("Đang chạy " + key + " (" + p.source + ")");
    }

    public synchronized void stop() throws Exception {
        if (current == null) {
            return;
        }
        current.stop = true;
        thread.join(15000);
        current.close();
    }
}
