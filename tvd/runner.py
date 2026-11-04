"""Runs one pipeline at a time for the dashboard and switches between demos (one button per violation type,
listed in a demos file) or an uploaded video. All demos share one Store; the dashboard shows the violations of
the running demo by its camera_id.

    demos:
      - key: den_do
        title: Vượt đèn đỏ (đèn thật)
        config: configs/hcm_redlight_real.yaml
    upload_templates:            # rule sets offered for an uploaded video
      - key: chung
        title: Không cần vẽ vùng (biển báo, mũ bảo hiểm, biển số)
        config: configs/upload_generic.yaml
"""
from __future__ import annotations

import gc
import logging
import re
import threading
from pathlib import Path

import yaml

from .pipeline import Pipeline
from .storage import Store

log = logging.getLogger("tvd")


def _entries(items) -> list[dict]:
    return [{"key": d.get("key", ""), "title": d.get("title", ""), "config": d.get("config", ""),
             "note": d.get("note", "")} for d in items or []]


def load_cfg(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class Runner:
    def __init__(self, store: Store, demos_cfg: dict | None = None, fast: bool = False):
        self.store = store
        self.fast = fast
        demos_cfg = demos_cfg or {}
        self.demos = _entries(demos_cfg.get("demos"))
        self.templates = _entries(demos_cfg.get("upload_templates"))
        self.current: Pipeline | None = None
        self.current_key: str | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @staticmethod
    def _find(entries: list[dict], key: str) -> dict | None:
        return next((e for e in entries if e["key"] == key), None)

    def start_demo(self, key: str) -> bool:
        """Start the demo with this key; False if there is no such demo."""
        e = self._find(self.demos, key)
        if e is None:
            return False
        cfg = load_cfg(e["config"])
        if self.fast:
            cfg["realtime"] = False
        self.start(key, cfg)
        return True

    def start_upload(self, video: Path, template_key: str) -> bool:
        """Run an uploaded video once with the rules of a template config. Every frame is processed (no realtime
        skipping), so no violation is missed even when processing is slower than the video."""
        e = self._find(self.templates, template_key) or self._find(self.demos, template_key)
        if e is None or not video.exists():
            return False  # a demo's zones fit a video from the same camera, otherwise use a template
        cfg = load_cfg(e["config"])
        cfg.update(source=str(video), camera_id="upload-" + re.sub(r"\.[^.]*$", "", video.name),
                   loop=False, realtime=False)
        self.start("upload:" + video.name, cfg)
        return True

    def start(self, key: str, cfg: dict) -> None:
        with self._lock:
            self.stop()
            p = Pipeline(cfg, self.store)

            def worker():
                try:
                    p.run()
                except Exception:
                    log.exception("Pipeline dừng do lỗi")

            self.current, self.current_key = p, key
            self._thread = threading.Thread(target=worker, name=f"pipeline-{key}", daemon=True)
            self._thread.start()
            log.info("Đang chạy %s (%s)", key, p.source)

    def stop(self) -> None:
        if self.current is None:
            return
        self.current.stop_event.set()
        self._thread.join(timeout=15)
        self.current = None
        gc.collect()  # drop the old models before the next demo loads its own
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
