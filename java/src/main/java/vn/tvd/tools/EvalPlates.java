package vn.tvd.tools;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.stream.Stream;

import com.fasterxml.jackson.databind.ObjectMapper;

import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import vn.tvd.Cfg;
import vn.tvd.plate.PlateReader;

/**
 * Plate reading accuracy on a folder of photos (Java counterpart of eval/run_plates.py).
 * <pre>
 *   java -cp "java/target/tvd.jar:java/target/lib/*" vn.tvd.tools.EvalPlates eval/vn eval/vn/labels.json
 * </pre>
 * labels.json maps image file name -> expected plates. Only characters are compared: the car/motorbike layout
 * comes from the vehicle class in the pipeline, which still images don't have.
 */
public final class EvalPlates {
    private EvalPlates() {
    }

    private static String key(String s) {
        return s == null ? "" : s.toUpperCase(Locale.ROOT).replaceAll("[^A-Z0-9]", "");
    }

    private static int same(String a, String b) {
        int n = 0;
        for (int i = 0; i < Math.min(a.length(), b.length()); i++) {
            n += a.charAt(i) == b.charAt(i) ? 1 : 0;
        }
        return n;
    }

    @SuppressWarnings("unchecked")
    public static void main(String[] args) throws Exception {
        nu.pattern.OpenCV.loadLocally();
        Path folder = Path.of(args[0]);
        Map<String, List<String>> labels = args.length > 1
                ? new ObjectMapper().readValue(Files.readString(Path.of(args[1])), Map.class) : Map.of();
        PlateReader reader = new PlateReader(Cfg.parse("det_conf: 0.25"));
        int n = 0, hit = 0, chars = 0, charsOk = 0;
        List<Path> imgs;
        try (Stream<Path> s = Files.list(folder)) {
            imgs = s.filter(p -> p.toString().toLowerCase(Locale.ROOT).matches(".*\\.(jpg|jpeg|png)$")).sorted().toList();
        }
        for (Path img : imgs) {
            Mat m = Imgcodecs.imread(img.toString());
            List<String> got = new ArrayList<>();
            int j = 0;
            for (PlateReader.Result r : reader.readAll(m, false)) {
                got.add(r.text() == null ? "" : r.text());
                System.out.printf(Locale.ROOT, "%-14s #%d raw=%-12s -> %-12s conf=%.2f %s %dx%d%n", img.getFileName(), j++,
                        r.raw(), r.text(), r.conf(), r.twoRows() ? "2 dong" : "1 dong", r.box().width, r.box().height);
            }
            for (String exp : labels.getOrDefault(img.getFileName().toString(), List.of())) {
                n++;
                String e = key(exp), best = "";
                for (String g : got) {
                    if (same(key(g), e) > same(best, e)) {
                        best = key(g);
                    }
                }
                hit += best.equals(e) ? 1 : 0;
                chars += e.length();
                charsOk += same(best, e);
            }
        }
        if (n > 0) {
            System.out.printf(Locale.ROOT, "%nDung ca bien: %d/%d = %.0f%% | dung ky tu: %d/%d = %.0f%%%n",
                    hit, n, 100.0 * hit / n, charsOk, chars, 100.0 * charsOk / chars);
        }
    }
}
