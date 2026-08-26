"""
Persistent render-job history for AutoLoop Render.
Thread-safe; writes to outputs/job_history.json so records survive restarts.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field

_HISTORY_FILE = os.path.join("outputs", "job_history.json")
_lock = threading.Lock()

_FIELDS = (
    "id", "timestamp", "timestamp_epoch", "label", "source_file",
    "output_path", "duration_seconds", "size_bytes", "status",
    "error", "preset_id", "ar_id", "tab",
)


@dataclass
class JobRecord:
    id: str
    timestamp: str           # "YYYY-MM-DD HH:MM:SS"
    timestamp_epoch: float
    label: str
    source_file: str
    output_path: str
    duration_seconds: float
    size_bytes: int
    status: str              # "done" | "error"
    error: str = ""
    preset_id: str = "copy"
    ar_id: str = "source"
    tab: str = "single"      # "single" | "matrix" | "watch"


def _read_raw() -> list[dict]:
    if not os.path.exists(_HISTORY_FILE):
        return []
    try:
        with open(_HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _write_raw(records: list[dict]) -> None:
    os.makedirs(os.path.dirname(_HISTORY_FILE) or ".", exist_ok=True)
    tmp = _HISTORY_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    os.replace(tmp, _HISTORY_FILE)


def _to_record(raw: dict) -> JobRecord | None:
    try:
        return JobRecord(
            id=str(raw.get("id", "")),
            timestamp=str(raw.get("timestamp", "")),
            timestamp_epoch=float(raw.get("timestamp_epoch", 0)),
            label=str(raw.get("label", "")),
            source_file=str(raw.get("source_file", "")),
            output_path=str(raw.get("output_path", "")),
            duration_seconds=float(raw.get("duration_seconds", 0)),
            size_bytes=int(raw.get("size_bytes", 0)),
            status=str(raw.get("status", "done")),
            error=str(raw.get("error", "")),
            preset_id=str(raw.get("preset_id", "copy")),
            ar_id=str(raw.get("ar_id", "source")),
            tab=str(raw.get("tab", "single")),
        )
    except Exception:
        return None


def load_history() -> list[JobRecord]:
    with _lock:
        raw = _read_raw()
    records = [r for r in (_to_record(d) for d in raw) if r is not None]
    return sorted(records, key=lambda r: r.timestamp_epoch, reverse=True)


def append_job(record: JobRecord) -> None:
    with _lock:
        records = _read_raw()
        records.append(asdict(record))
        _write_raw(records)


def remove_job(job_id: str) -> None:
    with _lock:
        records = [r for r in _read_raw() if r.get("id") != job_id]
        _write_raw(records)


def remove_missing() -> int:
    """Remove history entries whose output file no longer exists. Returns count removed."""
    with _lock:
        raw = _read_raw()
        kept = [r for r in raw if os.path.exists(str(r.get("output_path", "")))]
        removed = len(raw) - len(kept)
        if removed:
            _write_raw(kept)
    return removed


def clear_history() -> None:
    with _lock:
        _write_raw([])


def make_record(
    *,
    label: str,
    source_file: str,
    output_path: str,
    duration_seconds: float,
    size_bytes: int,
    status: str = "done",
    error: str = "",
    preset_id: str = "copy",
    ar_id: str = "source",
    tab: str = "single",
) -> JobRecord:
    return JobRecord(
        id=uuid.uuid4().hex,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        timestamp_epoch=time.time(),
        label=label,
        source_file=source_file,
        output_path=output_path,
        duration_seconds=duration_seconds,
        size_bytes=size_bytes,
        status=status,
        error=error,
        preset_id=preset_id,
        ar_id=ar_id,
        tab=tab,
    )
