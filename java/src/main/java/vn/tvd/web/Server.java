package vn.tvd.web;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import io.javalin.Javalin;
import io.javalin.http.Context;
import io.javalin.http.HttpStatus;
import io.javalin.http.staticfiles.Location;

import vn.tvd.Pipeline;
import vn.tvd.store.Store;

/** Dashboard: live MJPEG stream, violation list with review, light and backlight override (same API as Python). */
public final class Server {
    private Server() {
    }

    public static Javalin create(Pipeline pipeline, Store store) throws IOException {
        byte[] index;
        try (InputStream in = Server.class.getResourceAsStream("/static/index.html")) {
            index = in.readAllBytes();
        }
        Javalin app = Javalin.create(config -> {
            config.useVirtualThreads = true;
            config.showJavalinBanner = false;
            config.staticFiles.add(sf -> {
                sf.hostedPath = "/evidence";
                sf.directory = store.evidence.toAbsolutePath().toString();
                sf.location = Location.EXTERNAL;
            });
        });

        app.get("/", ctx -> ctx.contentType("text/html; charset=utf-8").result(index));

        app.get("/stream.mjpg", ctx -> stream(ctx, pipeline));

        app.get("/snapshot.jpg", ctx -> {
            byte[] jpg = pipeline.latestJpeg;
            if (jpg == null) {
                ctx.status(HttpStatus.SERVICE_UNAVAILABLE).result("Chưa có khung hình");
                return;
            }
            ctx.contentType("image/jpeg").result(jpg);
        });

        app.get("/api/stats", ctx -> {
            Map<String, Object> s = new LinkedHashMap<>(pipeline.stats);
            s.put("violations", store.counts());
            s.put("light_mode", pipeline.light.mode);
            s.put("light_override", pipeline.light.overridden());
            s.put("frame_size", pipeline.frameSize);
            s.put("camera_id", pipeline.cameraId);
            ctx.json(s);
        });

        app.get("/api/violations", ctx -> {
            int limit = Math.min(Integer.parseInt(ctx.queryParamAsClass("limit", String.class).getOrDefault("100")), 1000);
            ctx.json(store.list(ctx.queryParam("type"), ctx.queryParam("status"), limit));
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
                    StringBuilder line = new StringBuilder();
                    for (Object v : row.values()) {
                        if (!line.isEmpty()) {
                            line.append(',');
                        }
                        line.append(csv(v));
                    }
                    b.append(line).append("\r\n");
                }
            }
            ctx.header("Content-Disposition", "attachment; filename=violations.csv")
                    .contentType("text/csv; charset=utf-8").result(b.toString().getBytes(StandardCharsets.UTF_8));
        });

        app.post("/api/light", ctx -> {
            String state = String.valueOf(ctx.bodyAsClass(Map.class).get("state"));
            try {
                pipeline.light.override("auto".equals(state) ? null : state);
            } catch (IllegalArgumentException e) {
                ctx.status(422).json(Map.of("detail", e.getMessage()));
                return;
            }
            ctx.json(Map.of("ok", true, "state", state));
        });

        app.post("/api/backlight", ctx -> {
            String mode = String.valueOf(ctx.bodyAsClass(Map.class).get("mode"));
            try {
                pipeline.backlight.setMode(mode);
            } catch (IllegalArgumentException e) {
                ctx.status(422).json(Map.of("detail", e.getMessage()));
                return;
            }
            ctx.json(Map.of("ok", true, "mode", mode));
        });
        return app;
    }

    private static void stream(Context ctx, Pipeline pipeline) throws InterruptedException {
        ctx.res().setContentType("multipart/x-mixed-replace; boundary=frame");
        byte[] last = null;
        try {
            OutputStream out = ctx.res().getOutputStream();
            while (true) {
                byte[] jpg = pipeline.latestJpeg;
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
