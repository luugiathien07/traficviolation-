"""FastAPI dashboard: live MJPEG stream, violation list with review, light override."""
from __future__ import annotations

import asyncio
import csv
import io
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .pipeline import Pipeline
from .storage import STATUSES, Store

STATIC = Path(__file__).parent / "static"


class StatusIn(BaseModel):
    status: str | None = None
    plate: str | None = None


class BacklightIn(BaseModel):
    mode: str  # off | on | auto


class LightIn(BaseModel):
    state: str  # red | yellow | green | auto


def create_app(pipeline: Pipeline, store: Store) -> FastAPI:
    app = FastAPI(title="Traffic Violation MVP")
    app.mount("/evidence", StaticFiles(directory=store.evidence), name="evidence")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/stream.mjpg")
    async def stream():
        async def gen():
            last = None
            while True:
                with pipeline.frame_lock:
                    jpg = pipeline.latest_jpeg
                if jpg is not None and jpg is not last:
                    last = jpg
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                await asyncio.sleep(0.04)
        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/snapshot.jpg")
    def snapshot():
        with pipeline.frame_lock:
            jpg = pipeline.latest_jpeg
        if jpg is None:
            raise HTTPException(503, "Chưa có khung hình")
        return StreamingResponse(io.BytesIO(jpg), media_type="image/jpeg")

    @app.get("/api/stats")
    def stats():
        s = dict(pipeline.stats)
        s["by_class"] = dict(s["by_class"])
        s["violations"] = store.counts()
        s["light_mode"] = pipeline.light.mode
        s["light_override"] = pipeline.light.overridden
        s["frame_size"] = pipeline.frame_size
        s["camera_id"] = pipeline.camera_id
        return s

    @app.get("/api/violations")
    def violations(type: str | None = None, status: str | None = None, limit: int = 100):
        return store.list(type=type, status=status, limit=min(limit, 1000))

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
            pipeline.light.override(None if body.state == "auto" else body.state)
        except ValueError as e:
            raise HTTPException(422, str(e))
        return {"ok": True, "state": body.state}

    @app.post("/api/backlight")
    def set_backlight(body: BacklightIn):
        try:
            pipeline.backlight.set_mode(body.mode)
        except ValueError as e:
            raise HTTPException(422, str(e))
        return {"ok": True, "mode": body.mode}

    return app
