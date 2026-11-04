"""FastAPI dashboard: live MJPEG stream, violation list with review, light override, one button per demo and
video upload (Runner)."""
from __future__ import annotations

import asyncio
import csv
import io
import re
import shutil
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .runner import Runner
from .storage import STATUSES, Store

STATIC = Path(__file__).parent / "static"
VIDEO_EXT = {"mp4", "mov", "avi", "mkv", "webm", "m4v"}


class StatusIn(BaseModel):
    status: str | None = None
    plate: str | None = None


class BacklightIn(BaseModel):
    mode: str  # off | on | auto


class LightIn(BaseModel):
    state: str  # red | yellow | green | auto


def create_app(runner: Runner, store: Store) -> FastAPI:
    app = FastAPI(title="Traffic Violation MVP")
    app.mount("/evidence", StaticFiles(directory=store.evidence), name="evidence")
    uploads = store.root / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)

    def pipe():
        if runner.current is None:
            raise HTTPException(503, "Chưa chạy demo nào")
        return runner.current

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/stream.mjpg")
    async def stream():
        async def gen():
            last = None
            while True:
                p = runner.current
                jpg = None
                if p is not None:
                    with p.frame_lock:
                        jpg = p.latest_jpeg
                if jpg is not None and jpg is not last:
                    last = jpg
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                await asyncio.sleep(0.04)
        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/snapshot.jpg")
    def snapshot():
        p = pipe()
        with p.frame_lock:
            jpg = p.latest_jpeg
        if jpg is None:
            raise HTTPException(503, "Chưa có khung hình")
        return StreamingResponse(io.BytesIO(jpg), media_type="image/jpeg")

    @app.get("/api/stats")
    def stats():
        p = pipe()
        s = dict(p.stats)
        s["by_class"] = dict(s["by_class"])
        s["violations"] = store.counts(p.camera_id)
        s["light_mode"] = p.light.mode
        s["light_override"] = p.light.overridden
        s["frame_size"] = p.frame_size
        s["camera_id"] = p.camera_id
        s["demo"] = runner.current_key
        return s

    @app.get("/api/demos")
    def demos():
        return {"demos": runner.demos, "templates": runner.templates, "current": str(runner.current_key)}

    @app.post("/api/demos/{key}")
    def start_demo(key: str):
        if not runner.start_demo(key):
            raise HTTPException(404, "Không có demo này")
        return {"ok": True, "current": runner.current_key}

    @app.post("/api/upload")
    def upload(video: UploadFile = File(...), template: str = Form("chung")):
        ext = Path(video.filename or "").suffix.lstrip(".").lower()
        if ext not in VIDEO_EXT:
            raise HTTPException(422, f"Cần một file video ({', '.join(sorted(VIDEO_EXT))})")
        # own file name: never trust the client's path
        stem = re.sub(r"[^A-Za-z0-9_-]", "_", Path(video.filename).stem)[:40]
        dst = uploads / f"{time.strftime('%Y%m%d_%H%M%S')}_{stem}.{ext}"
        with dst.open("wb") as f:
            shutil.copyfileobj(video.file, f)
        if not runner.start_upload(dst, template):
            raise HTTPException(422, f"Không có bộ luật {template}")
        return {"ok": True, "current": runner.current_key}

    @app.get("/api/violations")
    def violations(type: str | None = None, status: str | None = None, limit: int = 100, camera: str | None = None):
        if camera is None and runner.current is not None:
            camera = runner.current.camera_id  # the running demo's violations
        return store.list(type=type, status=status, limit=min(limit, 1000), camera_id=None if camera == "all" else camera)

    @app.patch("/api/violations/{vid}")
    def review(vid: int, body: StatusIn):
        if body.status is not None:
            if body.status not in STATUSES:
                raise HTTPException(422, f"status phải là một trong {STATUSES}")
            if not store.set_status(vid, body.status):
                raise HTTPException(404, "Không tìm thấy vi phạm")
        if body.plate is not None and not store.set_plate(vid, body.plate):
            raise HTTPException(404, "Không tìm thấy vi phạm")
        return {"ok": True}

    @app.get("/api/violations.csv")
    def export_csv():
        rows = store.list(limit=100000)
        buf = io.StringIO()
        if rows:
            w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        return StreamingResponse(iter(["﻿" + buf.getvalue()]), media_type="text/csv",
                                 headers={"Content-Disposition": "attachment; filename=violations.csv"})

    @app.post("/api/light")
    def set_light(body: LightIn):
        try:
            pipe().light.override(None if body.state == "auto" else body.state)
        except ValueError as e:
            raise HTTPException(422, str(e))
        return {"ok": True, "state": body.state}

    @app.post("/api/backlight")
    def set_backlight(body: BacklightIn):
        try:
            pipe().backlight.set_mode(body.mode)
        except ValueError as e:
            raise HTTPException(422, str(e))
        return {"ok": True, "mode": body.mode}

    return app
