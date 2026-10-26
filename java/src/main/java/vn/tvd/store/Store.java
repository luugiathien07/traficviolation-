package vn.tvd.store;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import java.sql.Statement;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.imgcodecs.Imgcodecs;

/** SQLite store for violations plus evidence images on disk (same schema as the Python version). */
public final class Store implements AutoCloseable {
    public static final List<String> STATUSES = List.of("pending", "confirmed", "rejected");
    private static final String SCHEMA = """
            CREATE TABLE IF NOT EXISTS violations (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at  REAL NOT NULL,
                camera_id   TEXT NOT NULL,
                video_t     REAL NOT NULL,
                type        TEXT NOT NULL,
                track_id    INTEGER NOT NULL,
                vclass      TEXT NOT NULL,
                conf        REAL NOT NULL,
                speed_kmh   REAL,
                light       TEXT,
                detail      TEXT,
                plate       TEXT,
                frame_path  TEXT,
                crop_path   TEXT,
                status      TEXT NOT NULL DEFAULT 'pending'
            )""";

    public final Path root, evidence;
    private final Connection db;

    public Store(Path dataDir) throws IOException, SQLException {
        root = dataDir;
        evidence = dataDir.resolve("evidence");
        Files.createDirectories(evidence);
        db = DriverManager.getConnection("jdbc:sqlite:" + dataDir.resolve("violations.db"));
        try (Statement st = db.createStatement()) {
            st.execute(SCHEMA);
            st.execute("CREATE INDEX IF NOT EXISTS idx_v_created ON violations(created_at)");
            Set<String> cols = new HashSet<>();
            try (ResultSet rs = st.executeQuery("PRAGMA table_info(violations)")) {
                while (rs.next()) {
                    cols.add(rs.getString(2));
                }
            }
            // migrate databases created before plate reading existed
            if (!cols.contains("plate_conf")) {
                st.execute("ALTER TABLE violations ADD COLUMN plate_conf REAL");
            }
            if (!cols.contains("plate_path")) {
                st.execute("ALTER TABLE violations ADD COLUMN plate_path TEXT");
            }
        }
    }

    public synchronized long add(String cameraId, double videoT, String type, int trackId, String vclass, double conf,
                                 Double speedKmh, String light, String detail, Mat frame, Mat crop) throws SQLException, IOException {
        LocalDateTime now = LocalDateTime.now();
        String day = now.format(DateTimeFormatter.ofPattern("yyyyMMdd"));
        Files.createDirectories(evidence.resolve(day));
        String stem = now.format(DateTimeFormatter.ofPattern("HHmmss")) + "_" + cameraId + "_" + trackId + "_" + type;
        String frameRel = day + "/" + stem + "_frame.jpg", cropRel = day + "/" + stem + "_crop.jpg";
        Imgcodecs.imwrite(evidence.resolve(frameRel).toString(), frame, new MatOfInt(Imgcodecs.IMWRITE_JPEG_QUALITY, 85));
        boolean hasCrop = !crop.empty();
        if (hasCrop) {
            Imgcodecs.imwrite(evidence.resolve(cropRel).toString(), crop, new MatOfInt(Imgcodecs.IMWRITE_JPEG_QUALITY, 92));
        }
        try (PreparedStatement ps = db.prepareStatement(
                "INSERT INTO violations (created_at, camera_id, video_t, type, track_id, vclass, conf, speed_kmh, light,"
                        + " detail, plate, frame_path, crop_path) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                Statement.RETURN_GENERATED_KEYS)) {
            ps.setDouble(1, System.currentTimeMillis() / 1000.0);
            ps.setString(2, cameraId);
            ps.setDouble(3, videoT);
            ps.setString(4, type);
            ps.setInt(5, trackId);
            ps.setString(6, vclass);
            ps.setDouble(7, conf);
            ps.setObject(8, speedKmh);
            ps.setString(9, light);
            ps.setString(10, detail);
            ps.setString(11, null);
            ps.setString(12, frameRel);
            ps.setString(13, hasCrop ? cropRel : null);
            ps.executeUpdate();
            try (ResultSet k = ps.getGeneratedKeys()) {
                k.next();
                return k.getLong(1);
            }
        }
    }

    public synchronized List<Map<String, Object>> list(String type, String status, int limit) throws SQLException {
        StringBuilder q = new StringBuilder("SELECT * FROM violations WHERE 1=1");
        List<Object> args = new ArrayList<>();
        if (type != null && !type.isEmpty()) {
            q.append(" AND type = ?");
            args.add(type);
        }
        if (status != null && !status.isEmpty()) {
            q.append(" AND status = ?");
            args.add(status);
        }
        q.append(" ORDER BY id DESC LIMIT ?");
        args.add(limit);
        try (PreparedStatement ps = db.prepareStatement(q.toString())) {
            for (int i = 0; i < args.size(); i++) {
                ps.setObject(i + 1, args.get(i));
            }
            List<Map<String, Object>> rows = new ArrayList<>();
            try (ResultSet rs = ps.executeQuery()) {
                ResultSetMetaData md = rs.getMetaData();
                while (rs.next()) {
                    Map<String, Object> row = new LinkedHashMap<>();
                    for (int c = 1; c <= md.getColumnCount(); c++) {
                        row.put(md.getColumnName(c), rs.getObject(c));
                    }
                    rows.add(row);
                }
            }
            return rows;
        }
    }

    public synchronized Map<String, Integer> counts() throws SQLException {
        Map<String, Integer> out = new LinkedHashMap<>();
        try (Statement st = db.createStatement();
             ResultSet rs = st.executeQuery("SELECT type, COUNT(*) n FROM violations GROUP BY type")) {
            while (rs.next()) {
                out.put(rs.getString(1), rs.getInt(2));
            }
        }
        return out;
    }

    public synchronized boolean setStatus(long id, String status) throws SQLException {
        if (!STATUSES.contains(status)) {
            throw new IllegalArgumentException("status phải là một trong " + STATUSES);
        }
        try (PreparedStatement ps = db.prepareStatement("UPDATE violations SET status = ? WHERE id = ?")) {
            ps.setString(1, status);
            ps.setLong(2, id);
            return ps.executeUpdate() > 0;
        }
    }

    public synchronized boolean setPlate(long id, String plate) throws SQLException {
        try (PreparedStatement ps = db.prepareStatement("UPDATE violations SET plate = ? WHERE id = ?")) {
            ps.setString(1, plate.strip().toUpperCase());
            ps.setLong(2, id);
            return ps.executeUpdate() > 0;
        }
    }

    /** Store an automatically read plate, unless a reviewer already typed one. */
    public synchronized void setPlateAuto(long id, String plate, double conf, Mat image) throws SQLException {
        String framePath;
        try (PreparedStatement ps = db.prepareStatement("SELECT frame_path, plate FROM violations WHERE id = ?")) {
            ps.setLong(1, id);
            try (ResultSet rs = ps.executeQuery()) {
                if (!rs.next() || rs.getString(2) != null && !rs.getString(2).isEmpty()) {
                    return;
                }
                framePath = rs.getString(1);
            }
        }
        String rel = framePath.replace("_frame.jpg", "_plate.jpg");
        Imgcodecs.imwrite(evidence.resolve(rel).toString(), image, new MatOfInt(Imgcodecs.IMWRITE_JPEG_QUALITY, 95));
        try (PreparedStatement ps = db.prepareStatement(
                "UPDATE violations SET plate = ?, plate_conf = ?, plate_path = ? WHERE id = ?")) {
            ps.setString(1, plate);
            ps.setDouble(2, conf);
            ps.setString(3, rel);
            ps.setLong(4, id);
            ps.executeUpdate();
        }
    }

    @Override
    public synchronized void close() throws SQLException {
        db.close();
    }
}
