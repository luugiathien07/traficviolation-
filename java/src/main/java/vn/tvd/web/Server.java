package vn.tvd.web;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

import io.javalin.Javalin;
import io.javalin.http.Context;
import io.javalin.http.HttpStatus;
import io.javalin.http.UploadedFile;
import io.javalin.http.staticfiles.Location;

import vn.tvd.Pipeline;
import vn.tvd.store.Store;

/**
 * Dashboard: live MJPEG stream, violation list with review, light and backlight override (same API as Python),
 * plus one button per demo and video upload (Runner).
 */
public final class Server {
    private static final Set<String> VIDEO_EXT = Set.of("mp4", "mov", "avi", "mkv", "webm", "m4v");

    private Server() {
    }

    public static Javalin create(Runner runner, Store store) throws IOException {
        byte[] index;
        try (InputStream in = Server.class.getResourceAsStream("/static/index.html")) {
            index = in.readAllBytes();
        }
        Path uploads = store.root.resolve("uploads");
        Files.createDirectories(uploads);
        Javalin app = Javalin.create(config -> {
            config.useVirtualThreads = true;
            config.showJavalinBanner = false;
            config.http.maxRequestSize = 2_000_000_000L;  // uploaded videos
            config.staticFiles.add(sf -> {
                sf.hostedPath = "/evidence";
                sf.directory = store.evidence.toAbsolutePath().toString();
                sf.location = Location.EXTERNAL;
            });
        });

        app.get("/", ctx -> ctx.contentType("text/html; charset=utf-8").result(index));

        app.get("/stream.mjpg", ctx -> stream(ctx, runner));

        app.get("/snapshot.jpg", ctx -> {
            Pipeline p = runner.current();
            byte[] jpg = p == null ? null : p.latestJpeg;
            if (jpg == null) {
                ctx.status(HttpStatus.SERVICE_UNAVAILABLE).result("Chưa có khung hình");
                return;
            }
            ctx.contentType("image/jpeg").result(jpg);
        });

        app.get("/api/stats", ctx -> {
            Pipeline p = runner.current();
            if (p == null) {
                ctx.status(HttpStatus.SERVICE_UNAVAILABLE).json(Map.of("detail", "Chưa chạy demo nào"));
                return;
            }
            Map<String, Object> s = new LinkedHashMap<>(p.stats);
            s.put("violations", store.counts(p.cameraId));
            s.put("light_mode", p.light.mode);
            s.put("light_override", p.light.overridden());
            s.put("frame_size", p.frameSize);
            s.put("camera_id", p.cameraId);
            s.put("demo", runner.currentKey());
            ctx.json(s);
        });

        app.get("/api/demos", ctx -> ctx.json(Map.of("demos", runner.demos, "templates", runner.templates,
                "current", String.valueOf(runner.currentKey()))));

        app.post("/api/demos/{key}", ctx -> {
            if (!runner.startDemo(ctx.pathParam("key"))) {
                ctx.status(404).json(Map.of("detail", "Không có demo này"));
                return;
            }
            ctx.json(Map.of("ok", true, "current", runner.currentKey()));
        });

        app.post("/api/upload", ctx -> {
            UploadedFile f = ctx.uploadedFile("video");
            String ext = f == null ? "" : f.extension().replace(".", "").toLowerCase(Locale.ROOT);
            if (f == null || !VIDEO_EXT.contains(ext)) {
                ctx.status(422).json(Map.of("detail", "Cần một file video (" + String.join(", ", VIDEO_EXT) + ")"));
                return;
            }
            // own file name: never trust the client's path
            String stem = f.filename().replaceAll("\\.[^.]*$", "").replaceAll("[^A-Za-z0-9_-]", "_");
            stem = stem.substring(0, Math.min(40, stem.length()));
            Path dst = uploads.resolve(LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"))
                    + "_" + stem + "." + ext);
            try (InputStream in = f.content()) {
                Files.copy(in, dst, StandardCopyOption.REPLACE_EXISTING);
            }
            String template = ctx.formParamAsClass("template", String.class).getOrDefault("chung");
            if (!runner.startUpload(dst, template)) {
                ctx.status(422).json(Map.of("detail", "Không có bộ luật " + template));
                return;
            }
            ctx.json(Map.of("ok", true, "current", runner.currentKey()));
        });

        app.get("/api/violations", ctx -> {
            int limit = Math.min(Integer.parseInt(ctx.queryParamAsClass("limit", String.class).getOrDefault("100")), 1000);
            String camera = ctx.queryParam("camera");
            if (camera == null && runner.current() != null) {
                camera = runner.current().cameraId;  // the running demo's violations
            }
            ctx.json(store.list(ctx.queryParam("type"), ctx.queryParam("status"), limit, "all".equals(camera) ? null : camera));
        });

        app.patch("/api/violations/{id}", ctx -> {
            long id = Long.parseLong(ctx.pathParam("id"));
            Map<?, ?> body = ctx.bodyAsClass(Map.class);
            Object status = body.get("status"), plate = body.get("plate");
            if (status != null) {
                if (!Store.STATUSES.contains(status.toString())) {
                    ctx.status(422).json(Map.of("detail", "status phải là một trong " + Store.STATUSES));
                    return;
                }
                if (!store.setStatus(id, status.toString())) {
                    ctx.status(404).json(Map.of("detail", "Không tìm thấy vi phạm"));
                    return;
                }
            }
            if (plate != null && !store.setPlate(id, plate.toString())) {
                ctx.status(404).json(Map.of("detail", "Không tìm thấy vi phạm"));
                return;
            }
            ctx.json(Map.of("ok", true));
        });

        app.get("/api/violations.csv", ctx -> {
            List<Map<String, Object>> rows = store.list(null, null, 100000);
            StringBuilder b = new StringBuilder("﻿");
            if (!rows.isEmpty()) {
                b.append(String.join(",", rows.get(0).keySet())).append("\r\n");
                for (Map<String, Object> row : rows) {
                    List<String> cells = new ArrayList<>();
                    for (Object v : row.values()) {
                        cells.add(csv(v));
                    }
                    b.append(String.join(",", cells)).append("\r\n");
                }
            }
            ctx.header("Content-Disposition", "attachment; filename=violations.csv")
                    .contentType("text/csv; charset=utf-8").result(b.toString().getBytes(StandardCharsets.UTF_8));
        });

        app.post("/api/light", ctx -> {
            String state = String.valueOf(ctx.bodyAsClass(Map.class).get("state"));
            try {
                runner.current().light.override("auto".equals(state) ? null : state);
            } catch (IllegalArgumentException e) {
                ctx.status(422).json(Map.of("detail", e.getMessage()));
                return;
            }
            ctx.json(Map.of("ok", true, "state", state));
        });

        app.post("/api/backlight", ctx -> {
            String mode = String.valueOf(ctx.bodyAsClass(Map.class).get("mode"));
            try {
                runner.current().backlight.setMode(mode);
            } catch (IllegalArgumentException e) {
                ctx.status(422).json(Map.of("detail", e.getMessage()));
                return;
            }
            ctx.json(Map.of("ok", true, "mode", mode));
        });
        return app;
    }

    private static void stream(Context ctx, Runner runner) throws InterruptedException {
        ctx.res().setContentType("multipart/x-mixed-replace; boundary=frame");
        byte[] last = null;
        try {
            OutputStream out = ctx.res().getOutputStream();
            while (true) {
                Pipeline p = runner.current();
                byte[] jpg = p == null ? null : p.latestJpeg;
                if (jpg != null && jpg != last) {
                    last = jpg;
                    out.write("--frame\r\nContent-Type: image/jpeg\r\n\r\n".getBytes(StandardCharsets.US_ASCII));
                    out.write(jpg);
                    out.write("\r\n".getBytes(StandardCharsets.US_ASCII));
                    out.flush();
                }
                Thread.sleep(40);
            }
        } catch (IOException clientGone) {
            // viewer closed the page
        }
    }

    private static String csv(Object v) {
        if (v == null) {
            return "";
        }
        String s = v.toString();
        return s.matches(".*[,\"\r\n].*") ? "\"" + s.replace("\"", "\"\"") + "\"" : s;
    }
}
