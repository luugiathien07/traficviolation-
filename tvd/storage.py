"""SQLite store for violations plus evidence images on disk."""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import cv2
import numpy as np

SCHEMA = """
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
);
CREATE INDEX IF NOT EXISTS idx_v_created ON violations(created_at);
"""

STATUSES = ("pending", "confirmed", "rejected")


class Store:
    def __init__(self, data_dir: str | Path):
        self.root = Path(data_dir)
        self.evidence = self.root / "evidence"
        self.evidence.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(self.root / "violations.db", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(violations)")}
        for col, typ in (("plate_conf", "REAL"), ("plate_path", "TEXT")):
            if col not in cols:  # migrate databases created before plate reading existed
                self.db.execute(f"ALTER TABLE violations ADD COLUMN {col} {typ}")
        self.db.commit()

    def add(self, *, camera_id: str, video_t: float, type: str, track_id: int, vclass: str,
            conf: float, speed_kmh: float | None, light: str, detail: str,
            frame: np.ndarray, crop: np.ndarray, plate: str | None = None) -> int:
        now = time.time()
        day = time.strftime("%Y%m%d", time.localtime(now))
        folder = self.evidence / day
        folder.mkdir(exist_ok=True)
        stem = f"{time.strftime('%H%M%S', time.localtime(now))}_{camera_id}_{track_id}_{type}"
        frame_rel = f"{day}/{stem}_frame.jpg"
        crop_rel = f"{day}/{stem}_crop.jpg"
        cv2.imwrite(str(self.evidence / frame_rel), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if crop.size:
            cv2.imwrite(str(self.evidence / crop_rel), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        with self.lock:
            cur = self.db.execute(
                "INSERT INTO violations (created_at, camera_id, video_t, type, track_id, vclass, conf,"
                " speed_kmh, light, detail, plate, frame_path, crop_path) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (now, camera_id, video_t, type, track_id, vclass, conf, speed_kmh, light, detail,
                 plate, frame_rel, crop_rel if crop.size else None),
            )
            self.db.commit()
            return cur.lastrowid

    def list(self, type: str | None = None, status: str | None = None, limit: int = 100) -> list[dict]:
        q, args = "SELECT * FROM violations WHERE 1=1", []
        if type:
            q += " AND type = ?"; args.append(type)
        if status:
            q += " AND status = ?"; args.append(status)
        q += " ORDER BY id DESC LIMIT ?"; args.append(limit)
        with self.lock:
            return [dict(r) for r in self.db.execute(q, args)]

    def counts(self) -> dict[str, int]:
        with self.lock:
            rows = self.db.execute("SELECT type, COUNT(*) n FROM violations GROUP BY type").fetchall()
        return {r["type"]: r["n"] for r in rows}

    def set_status(self, vid: int, status: str) -> bool:
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        with self.lock:
            cur = self.db.execute("UPDATE violations SET status = ? WHERE id = ?", (status, vid))
            self.db.commit()
            return cur.rowcount > 0

    def set_plate_auto(self, vid: int, plate: str, conf: float, image: np.ndarray) -> None:
        """Store an automatically read plate, unless a reviewer already typed one."""
        with self.lock:
            row = self.db.execute("SELECT frame_path, plate FROM violations WHERE id = ?", (vid,)).fetchone()
        if row is None or row["plate"]:
            return
        rel = row["frame_path"].replace("_frame.jpg", "_plate.jpg")
        cv2.imwrite(str(self.evidence / rel), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with self.lock:
            self.db.execute("UPDATE violations SET plate = ?, plate_conf = ?, plate_path = ? WHERE id = ?",
                            (plate, conf, rel, vid))
            self.db.commit()

    def set_plate(self, vid: int, plate: str) -> bool:
        with self.lock:
            cur = self.db.execute("UPDATE violations SET plate = ? WHERE id = ?", (plate.strip().upper(), vid))
            self.db.commit()
            return cur.rowcount > 0
