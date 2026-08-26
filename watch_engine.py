"""
Background watch-folder daemon for AutoLoop Render.

Uses a module-level thread so it survives across Streamlit rerenders
(Streamlit reimports nothing — the module stays cached in sys.modules).
Only one watcher per process; this is a local single-user tool.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

import core_engine as engine
import job_history


@dataclass
class WatchConfig:
    watch_dir: str
    output_dir: str
    target_seconds: int
    audio_path: str | None = None
    volume: float = 1.0
    fade_in: float = 0.0
    fade_out: float = 0.0
    poll_interval: float = 5.0
    ar_id: str = "source"
    watermark: "engine.WatermarkConfig | None" = None
    preset_id: str = "copy"
    enc_crf: int | None = None


@dataclass
class LogEntry:
    timestamp: str
    filename: str
    status: str  # "info" | "processing" | "done" | "error" | "skipped"
    detail: str = ""


# ── Module-level singletons — survive Streamlit rerenders ────────────────────
_thread: threading.Thread | None = None
_stop_event: threading.Event = threading.Event()
_log: list[LogEntry] = []
_log_lock: threading.Lock = threading.Lock()
_files_done: int = 0
_active_config: WatchConfig | None = None


# ── Public API ────────────────────────────────────────────────────────────────

def is_watching() -> bool:
    return _thread is not None and _thread.is_alive()


def active_config() -> WatchConfig | None:
    return _active_config


def files_processed() -> int:
    return _files_done


def get_log() -> list[LogEntry]:
    with _log_lock:
        return list(_log)


def clear_log() -> None:
    global _files_done
    with _log_lock:
        _log.clear()
        _files_done = 0


def start_watcher(config: WatchConfig) -> None:
    global _thread, _stop_event, _files_done, _active_config
    if is_watching():
        return
    _stop_event = threading.Event()
    _files_done = 0
    _active_config = config
    _thread = threading.Thread(target=_watcher_loop, args=(config,), daemon=True)
    _thread.start()


def stop_watcher() -> None:
    global _thread, _active_config
    _stop_event.set()
    if _thread is not None:
        _thread.join(timeout=5)
        _thread = None
    _active_config = None


# ── Internal helpers ──────────────────────────────────────────────────────────

def _add_log(filename: str, status: str, detail: str = "") -> None:
    ts = time.strftime("%H:%M:%S")
    with _log_lock:
        _log.append(LogEntry(timestamp=ts, filename=filename, status=status, detail=detail))


def _is_video(fname: str) -> bool:
    return os.path.splitext(fname)[1].lower() in engine.ALLOWED_EXTENSIONS


def _wait_stable(path: str, stable_secs: float = 2.0, timeout: float = 60.0) -> bool:
    """Block until the file stops growing (copy/download finished). Returns False on timeout."""
    prev_size = -1
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            size = os.path.getsize(path)
        except OSError:
            time.sleep(0.5)
            continue
        if size > 0 and size == prev_size:
            return True
        prev_size = size
        time.sleep(stable_secs)
    return False


def _process(path: str, config: WatchConfig) -> None:
    global _files_done
    fname = os.path.basename(path)
    _add_log(fname, "processing", "rendering…")
    _tmp: list[str] = []  # temp files to clean up regardless of success/failure

    def _mk_tmp(suffix: str) -> str:
        stem = os.path.splitext(fname)[0]
        p = os.path.join(config.output_dir, f"_tmp_{stem}_{suffix}")
        _tmp.append(p)
        return p

    try:
        engine.validate_extension(fname)
        video_info = engine.probe_video(path)

        preset = engine.ENCODE_PRESETS.get(config.preset_id, engine.ENCODE_PRESETS["copy"])
        output_ext = None if config.preset_id == "copy" else preset.ext

        est = engine.estimate_output_size_bytes(video_info, config.target_seconds)
        if config.audio_path:
            est += int(24 * 1024 * config.target_seconds)
        engine.ensure_disk_space(config.output_dir, est)

        out_path = engine.build_output_path(config.output_dir, fname, output_ext=output_ext)

        # Build loop unit: AR crop → watermark → encode preset
        render_path = path

        if config.ar_id != "source":
            p = _mk_tmp("ar.mp4")
            engine.create_aspect_ratio_unit(render_path, p, config.ar_id)
            render_path = p

        if config.watermark and config.watermark.is_active:
            p = _mk_tmp("wm.mp4")
            engine.create_watermark_unit(render_path, p, config.watermark)
            render_path = p

        if config.preset_id != "copy":
            p = _mk_tmp(f"enc{preset.ext}")
            engine.create_encode_unit(render_path, p, config.preset_id, config.enc_crf)
            render_path = p

        if config.audio_path:
            engine.render_looped_video_with_looped_audio(
                render_path, config.audio_path, out_path, config.target_seconds,
                volume=config.volume, fade_in=config.fade_in, fade_out=config.fade_out,
            )
        else:
            engine.render_looped_video(render_path, out_path, config.target_seconds)

        out_size = os.path.getsize(out_path)
        sz = engine.format_bytes(out_size)
        _add_log(fname, "done", f"{sz}  →  {os.path.basename(out_path)}")
        with _log_lock:
            _files_done += 1
        try:
            job_history.append_job(job_history.make_record(
                label=fname,
                source_file=fname,
                output_path=out_path,
                duration_seconds=config.target_seconds,
                size_bytes=out_size,
                preset_id=config.preset_id,
                ar_id=config.ar_id,
                tab="watch",
            ))
        except Exception:
            pass

    except engine.AutoLoopError as exc:
        _add_log(fname, "error", str(exc))
    except Exception as exc:
        _add_log(fname, "error", f"Unexpected: {exc}")
    finally:
        for p in _tmp:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


def _watcher_loop(config: WatchConfig) -> None:
    os.makedirs(config.watch_dir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)

    # Snapshot files that already exist — skip them, only handle new arrivals
    try:
        seen = {f for f in os.listdir(config.watch_dir) if _is_video(f)}
    except OSError:
        seen = set()

    _add_log(
        "daemon", "info",
        f"Watching {config.watch_dir!r} — "
        f"{len(seen)} pre-existing file(s) skipped — "
        f"polling every {config.poll_interval:.0f}s",
    )

    while not _stop_event.is_set():
        try:
            current = {
                f for f in os.listdir(config.watch_dir)
                if _is_video(f) and not f.startswith(".")
            }
        except OSError as exc:
            _add_log("daemon", "error", f"Cannot read watch folder: {exc}")
            _stop_event.wait(config.poll_interval)
            continue

        for fname in sorted(current - seen):
            seen.add(fname)
            fpath = os.path.join(config.watch_dir, fname)
            if _wait_stable(fpath):
                _process(fpath, config)
            else:
                _add_log(fname, "skipped", "File did not stabilize in time (still being written?)")

        _stop_event.wait(config.poll_interval)

    _add_log("daemon", "info", "Watcher stopped.")
