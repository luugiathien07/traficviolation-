package vn.tvd.video;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

import org.opencv.core.CvType;
import org.opencv.core.Mat;

/**
 * Decodes a video file, RTSP/HTTP stream or webcam with an ffmpeg subprocess into BGR frames.
 * (The OpenCV Java build from Maven has no video decoders, so ffmpeg does the I/O.)
 */
public final class FfmpegReader implements AutoCloseable {
    public final int width, height;
    public final double fps;
    public final long frameCount;
    private final String source;
    private Process proc;
    private DataInputStream in;
    private final byte[] buf;

    public FfmpegReader(String source) throws IOException {
        this.source = source;
        String[] info = probe(source);
        this.width = Integer.parseInt(info[0]);
        this.height = Integer.parseInt(info[1]);
        this.fps = parseRate(info[2]);
        this.frameCount = info.length > 3 && info[3].matches("\\d+") ? Long.parseLong(info[3]) : -1;
        this.buf = new byte[width * height * 3];
        start(0);
    }

    public static boolean isFile(String source) {
        return !source.matches("^(rtsp|rtmp|http|https)://.*") && !source.matches("\\d+");
    }

    /** Restart decoding at a given frame (files only), e.g. to loop or to grab a preview frame. */
    public void start(long frame) throws IOException {
        close();
        List<String> cmd = new ArrayList<>(List.of("ffmpeg", "-hide_banner", "-loglevel", "error"));
        if (source.startsWith("rtsp://")) {
            cmd.addAll(List.of("-rtsp_transport", "tcp"));
        }
        if (frame > 0) {
            cmd.addAll(List.of("-ss", String.format(java.util.Locale.ROOT, "%.3f", frame / fps)));
        }
        if (source.matches("\\d+")) {
            cmd.addAll(List.of("-f", "v4l2", "-i", "/dev/video" + source));
        } else {
            cmd.addAll(List.of("-i", source));
        }
        cmd.addAll(List.of("-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"));
        proc = new ProcessBuilder(cmd).redirectError(ProcessBuilder.Redirect.INHERIT).start();
        in = new DataInputStream(new java.io.BufferedInputStream(proc.getInputStream(), buf.length * 2));
    }

    /** Next frame, or null at the end of the stream. */
    public Mat read() throws IOException {
        try {
            in.readFully(buf);
        } catch (EOFException e) {
            return null;
        }
        Mat m = new Mat(height, width, CvType.CV_8UC3);
        m.put(0, 0, buf);
        return m;
    }

    @Override
    public void close() {
        if (proc != null) {
            proc.destroy();
            try {
                in.close();
            } catch (IOException ignored) {
            }
            proc = null;
        }
    }

    private static String[] probe(String source) throws IOException {
        List<String> cmd = new ArrayList<>(List.of("ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,avg_frame_rate,nb_frames", "-of", "csv=p=0"));
        if (source.matches("\\d+")) {
            cmd.addAll(List.of("-f", "v4l2", "/dev/video" + source));
        } else {
            cmd.add(source);
        }
        Process p = new ProcessBuilder(cmd).start();
        try (BufferedReader r = new BufferedReader(new InputStreamReader(p.getInputStream(), StandardCharsets.UTF_8))) {
            String line = r.readLine();
            if (line == null || line.isBlank()) {
                throw new IOException("Không mở được nguồn video: " + source);
            }
            return line.trim().split(",");
        }
    }

    private static double parseRate(String s) {
        String[] p = s.split("/");
        double v = p.length == 2 ? Double.parseDouble(p[0]) / Math.max(1, Double.parseDouble(p[1])) : Double.parseDouble(s);
        return v > 0 && v < 1000 ? v : 25.0;
    }
}
