package vn.tvd;

import java.io.IOException;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.yaml.snakeyaml.Yaml;

/** Read-only view over a YAML mapping, with typed getters and defaults (same files as the Python version). */
public final class Cfg {
    private final Map<String, Object> map;

    public Cfg(Map<String, Object> map) {
        this.map = map == null ? new LinkedHashMap<>() : map;
    }

    public static Cfg load(Path file) throws IOException {
        try (Reader r = Files.newBufferedReader(file, StandardCharsets.UTF_8)) {
            Map<String, Object> m = new Yaml().load(r);
            return new Cfg(m);
        }
    }

    /** Parse YAML text (used by tests and for inline configs). */
    public static Cfg parse(String yaml) {
        Map<String, Object> m = new Yaml().load(yaml);
        return new Cfg(m);
    }

    /** models/yolo11n.pt -> models/onnx/yolo11n.onnx (exported with tools/export_onnx.py); .onnx paths stay. */
    public static String onnxPath(String model) {
        if (model.endsWith(".onnx")) {
            return model;
        }
        Path p = Path.of(model);
        String name = p.getFileName().toString().replaceAll("\\.pt$", ".onnx");
        Path parent = p.getParent();
        return (parent == null ? Path.of("onnx") : parent.resolve("onnx")).resolve(name).toString();
    }

    public Map<String, Object> raw() {
        return map;
    }

    public boolean has(String key) {
        return map.get(key) != null;
    }

    public Object get(String key) {
        return map.get(key);
    }

    public void put(String key, Object value) {
        map.put(key, value);
    }

    @SuppressWarnings("unchecked")
    public Cfg sub(String key) {
        Object v = map.get(key);
        return new Cfg(v instanceof Map ? (Map<String, Object>) v : null);
    }

    public String str(String key, String def) {
        Object v = map.get(key);
        return v == null ? def : v.toString();
    }

    public double num(String key, double def) {
        Object v = map.get(key);
        return v instanceof Number n ? n.doubleValue() : v == null ? def : Double.parseDouble(v.toString());
    }

    public int integer(String key, int def) {
        return (int) num(key, def);
    }

    public boolean bool(String key, boolean def) {
        Object v = map.get(key);
        return v instanceof Boolean b ? b : v == null ? def : Boolean.parseBoolean(v.toString());
    }

    public boolean enabled() {
        return bool("enabled", false);
    }

    @SuppressWarnings("unchecked")
    public List<Cfg> list(String key) {
        List<Cfg> out = new ArrayList<>();
        Object v = map.get(key);
        if (v instanceof List<?> l) {
            for (Object o : l) {
                out.add(new Cfg((Map<String, Object>) o));
            }
        }
        return out;
    }

    public List<String> strings(String key, List<String> def) {
        Object v = map.get(key);
        if (!(v instanceof List<?> l)) {
            return def;
        }
        List<String> out = new ArrayList<>();
        for (Object o : l) {
            out.add(o.toString());
        }
        return out;
    }

    /** [x, y] */
    public double[] point(String key) {
        return toPoint(map.get(key));
    }

    /** [[x1, y1], [x2, y2]] or a polygon: list of [x, y]. */
    public double[][] points(String key) {
        Object v = map.get(key);
        if (!(v instanceof List<?> l)) {
            return null;
        }
        double[][] out = new double[l.size()][];
        for (int i = 0; i < l.size(); i++) {
            out[i] = toPoint(l.get(i));
        }
        return out;
    }

    /** Flat list of numbers, e.g. roi [x1, y1, x2, y2] or a BGR colour. */
    public double[] numbers(String key) {
        return toPoint(map.get(key));
    }

    private static double[] toPoint(Object v) {
        if (!(v instanceof List<?> l)) {
            return null;
        }
        double[] p = new double[l.size()];
        for (int i = 0; i < p.length; i++) {
            p[i] = ((Number) l.get(i)).doubleValue();
        }
        return p;
    }
}
