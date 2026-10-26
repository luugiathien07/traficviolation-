package vn.tvd.video;

import java.io.IOException;
import java.io.OutputStream;
import java.util.List;
import java.util.Locale;

import org.opencv.core.Mat;

/** Encodes BGR frames to an H.264 .mp4 (plays in browsers and phones) with an ffmpeg subprocess. */
public final class FfmpegWriter implements AutoCloseable {
    private final Process proc;
    private final OutputStream out;
    private final byte[] buf;

    public FfmpegWriter(String path, int width, int height, double fps) throws IOException {
        List<String> cmd = List.of("ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", width + "x" + height,
                "-r", String.format(Locale.ROOT, "%.3f", fps), "-i", "-",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", path);
        proc = new ProcessBuilder(cmd).redirectError(ProcessBuilder.Redirect.INHERIT).start();
        out = proc.getOutputStream();
        buf = new byte[width * height * 3];
    }

    public void write(Mat frame) throws IOException {
        frame.get(0, 0, buf);
        out.write(buf);
    }

    @Override
    public void close() throws IOException {
        out.close();
        try {
            proc.waitFor();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}
