package vn.tvd;

import java.nio.file.Path;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.opencv.imgcodecs.Imgcodecs;

import vn.tvd.store.Store;
import vn.tvd.web.Server;

/**
 * Entry point.
 * <pre>
 *   java -jar java/target/tvd.jar --config configs/sample.yaml --serve        # dashboard at http://127.0.0.1:8000
 *   java -jar java/target/tvd.jar --config configs/sample.yaml --out out.mp4  # headless, write annotated video
 * </pre>
 */
public final class Main {
    private static final String USAGE = """
            Phát hiện vi phạm giao thông từ video

              --config FILE        file cấu hình YAML (bắt buộc)
              --source SRC         ghi đè nguồn video (file, rtsp://..., hoặc số webcam)
              --data DIR           thư mục lưu CSDL và ảnh bằng chứng (mặc định: data)
              --serve              chạy dashboard web
              --host HOST          mặc định 127.0.0.1
              --port PORT          mặc định 8000
              --out FILE.mp4       ghi video đã chú thích ra file .mp4
              --max-frames N
              --fast               không chờ theo FPS gốc khi đọc file
              --preview FILE.jpg   vẽ vạch/vùng cấu hình lên một khung hình rồi lưu ra file ảnh
              --preview-frame N    khung hình dùng cho --preview (mặc định 75)
            """;

    private Main() {
    }

    public static void main(String[] args) throws Exception {
        System.setProperty("java.util.logging.SimpleFormatter.format", "%1$tF %1$tT %4$s %5$s%6$s%n");
        String config = null, source = null, data = "data", host = "127.0.0.1", out = null, preview = null;
        int port = 8000, previewFrame = 75;
        Integer maxFrames = null;
        boolean serve = false, fast = false;
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--config" -> config = args[++i];
                case "--source" -> source = args[++i];
                case "--data" -> data = args[++i];
                case "--serve" -> serve = true;
                case "--host" -> host = args[++i];
                case "--port" -> port = Integer.parseInt(args[++i]);
                case "--out" -> out = args[++i];
                case "--max-frames" -> maxFrames = Integer.parseInt(args[++i]);
                case "--fast" -> fast = true;
                case "--preview" -> preview = args[++i];
                case "--preview-frame" -> previewFrame = Integer.parseInt(args[++i]);
                case "-h", "--help" -> {
                    System.out.print(USAGE);
                    return;
                }
                default -> {
                    System.err.println("Tham số không hợp lệ: " + args[i] + "\n\n" + USAGE);
                    System.exit(2);
                }
            }
        }
        if (config == null) {
            System.err.print(USAGE);
            System.exit(2);
        }
        nu.pattern.OpenCV.loadLocally();
        org.opencv.core.Core.setNumThreads(2);
        Logger.getLogger("").setLevel(Level.INFO);

        Cfg cfg = Cfg.load(Path.of(config));
        if (source != null) {
            cfg.put("source", source);
        }
        if (fast || !serve) {
            cfg.put("realtime", false);
        }
        Store store = new Store(Path.of(data));
        Pipeline pipe = new Pipeline(cfg, store);

        if (preview != null) {
            Imgcodecs.imwrite(preview, pipe.preview(previewFrame));
            System.out.println("Đã lưu " + preview);
            return;
        }
        if (!serve) {
            pipe.run(maxFrames, out);
            System.out.println("Tổng xe: " + pipe.vehiclesSeen() + " | Vi phạm: " + store.counts());
            return;
        }
        final Integer mf = maxFrames;
        final String o = out;
        Thread worker = new Thread(() -> {
            try {
                pipe.run(mf, o);
            } catch (Exception e) {
                Logger.getLogger("tvd").log(Level.SEVERE, "Pipeline dừng do lỗi", e);
            }
        }, "pipeline");
        worker.setDaemon(true);
        worker.start();
        Server.create(pipe, store).start(host, port);
        System.out.println("Dashboard: http://" + host + ":" + port);
    }
}
