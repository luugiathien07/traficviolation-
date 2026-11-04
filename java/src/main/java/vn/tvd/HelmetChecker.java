package vn.tvd;

import java.nio.file.Path;
import java.util.List;

import ai.onnxruntime.OrtException;

import org.opencv.core.Mat;

import vn.tvd.detect.Detection;
import vn.tvd.detect.YoloDetector;
import vn.tvd.rules.Track;

/**
 * Helmet check for tracked motorcycles. A detector trained on the HELMET dataset finds motorcycles with their
 * riders and says whether everyone wears a helmet (class 0) or not (class 1). Each detection is matched to the
 * tracked motorcycle it covers and added as a vote, weighted by confidence; NoHelmetRule decides once enough
 * frames agree.
 */
public final class HelmetChecker implements AutoCloseable {
    private final YoloDetector model;
    private final double conf, minCover, minHeight;
    private final int every;
    private int n;

    public HelmetChecker(Cfg cfg) throws OrtException {
        model = new YoloDetector(Path.of(Cfg.onnxPath(cfg.str("model", "models/helmet.pt"))), cfg.integer("imgsz", 1280));
        conf = cfg.num("conf", 0.35);
        every = cfg.integer("every_n_frames", 3);
        minCover = cfg.num("min_cover", 0.5);
        // a far-away motorcycle shows its riders' heads a few pixels tall: no vote, the model would be guessing
        minHeight = cfg.num("min_box_height", 90);
    }

    /** Share of the tracked motorcycle box that lies inside the helmet-model box. */
    static double cover(double[] moto, Detection d) {
        double iw = Math.max(0, Math.min(moto[2], d.x2()) - Math.max(moto[0], d.x1()));
        double ih = Math.max(0, Math.min(moto[3], d.y2()) - Math.max(moto[1], d.y1()));
        return iw * ih / Math.max(1.0, (moto[2] - moto[0]) * (moto[3] - moto[1]));
    }

    public void update(Mat frame, List<Track> tracks) throws OrtException {
        n++;
        List<Track> motos = tracks.stream()
                .filter(t -> "motorcycle".equals(t.cls) && t.bbox[3] - t.bbox[1] >= minHeight).toList();
        if (motos.isEmpty() || n % every != 0) {
            return;
        }
        List<Detection> dets = model.detect(frame, conf, null);
        for (Track tr : motos) {
            Detection best = null;
            double bestCover = 0;
            for (Detection d : dets) {
                double c = cover(tr.bbox, d);
                if (c > bestCover) {
                    bestCover = c;
                    best = d;
                }
            }
            if (best == null || bestCover < minCover) {
                continue;
            }
            tr.helmetVotes[best.cls()] += best.conf();
            tr.helmetFrames++;
            tr.riderBox = new double[] {best.x1(), best.y1(), best.x2(), best.y2()};  // motorcycle + riders
        }
    }

    @Override
    public void close() throws OrtException {
        model.close();
    }
}
