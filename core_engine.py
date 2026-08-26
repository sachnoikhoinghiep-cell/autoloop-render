"""
Core rendering engine for AutoLoop Render.

Wraps ffmpeg/ffprobe via subprocess (no shell=True, list-form args only) to
losslessly stream-loop a short input video out to an arbitrary target
duration using `-stream_loop -1 -t <seconds> -c copy`.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma"}
MAX_TARGET_HOURS = 48  # sanity cap to prevent fat-fingered durations from filling the disk


class AutoLoopError(Exception):
    """Base class for all expected/user-facing errors."""


class FFmpegNotFoundError(AutoLoopError):
    pass


class InvalidVideoError(AutoLoopError):
    pass


class InvalidAudioError(AutoLoopError):
    pass


class InvalidDurationError(AutoLoopError):
    pass


class InsufficientDiskSpaceError(AutoLoopError):
    pass


class RenderError(AutoLoopError):
    pass


@dataclass
class VideoInfo:
    duration_seconds: float
    size_bytes: int
    video_codec: str | None
    width: int | None
    height: int | None
    has_audio: bool = False


@dataclass
class AudioInfo:
    duration_seconds: float
    size_bytes: int
    audio_codec: str | None


def get_ffmpeg_path() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise FFmpegNotFoundError(
            "ffmpeg was not found on PATH. Install it and make sure 'ffmpeg' "
            "is runnable from a terminal (e.g. `choco install ffmpeg` on Windows)."
        )
    return path


def get_ffprobe_path() -> str:
    path = shutil.which("ffprobe")
    if not path:
        raise FFmpegNotFoundError(
            "ffprobe was not found on PATH. It ships alongside ffmpeg; "
            "reinstall ffmpeg if only one of the two is present."
        )
    return path


def validate_extension(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise InvalidVideoError(
            f"Unsupported file type '{ext or 'unknown'}'. "
            f"Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    return ext


def validate_audio_extension(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise InvalidAudioError(
            f"Unsupported audio file type '{ext or 'unknown'}'. "
            f"Allowed types: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}"
        )
    return ext


def validate_duration_seconds(total_seconds: float) -> float:
    if total_seconds <= 0:
        raise InvalidDurationError("Target duration must be greater than 0 seconds.")
    if total_seconds > MAX_TARGET_HOURS * 3600:
        raise InvalidDurationError(
            f"Target duration exceeds the safety cap of {MAX_TARGET_HOURS} hours. "
            "Raise MAX_TARGET_HOURS in core_engine.py if you really need more."
        )
    return total_seconds


def compute_total_seconds(hours: int, minutes: int, seconds: int) -> int:
    total = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    return int(validate_duration_seconds(total))


def _ffprobe_json(input_path: str, error_cls: type[AutoLoopError] = InvalidVideoError) -> dict:
    if not os.path.isfile(input_path):
        raise error_cls("Uploaded file could not be found on disk.")

    ffprobe = get_ffprobe_path()
    cmd = [
        ffprobe,
        "-v", "error",
        "-show_entries", "format=duration:stream=codec_type,codec_name,width,height",
        "-of", "json",
        input_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired as exc:
        raise error_cls("ffprobe timed out reading the uploaded file.") from exc

    if result.returncode != 0:
        raise error_cls(
            f"File does not look like a valid media file (ffprobe failed): {result.stderr.strip()}"
        )

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise error_cls("Could not parse ffprobe output for the uploaded file.") from exc


def _read_duration(data: dict, error_cls: type[AutoLoopError]) -> float:
    duration_raw = data.get("format", {}).get("duration")
    if duration_raw is None:
        raise error_cls("Uploaded file has no readable duration; it may be corrupted.")
    try:
        duration = float(duration_raw)
    except ValueError as exc:
        raise error_cls("Uploaded file reported a non-numeric duration.") from exc
    if duration <= 0:
        raise error_cls("Uploaded file has zero duration; it may be corrupted.")
    return duration


def probe_video(input_path: str) -> VideoInfo:
    """Run ffprobe and validate the file actually contains a readable video stream."""
    data = _ffprobe_json(input_path, InvalidVideoError)
    duration = _read_duration(data, InvalidVideoError)

    streams = data.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video_stream is None:
        raise InvalidVideoError("Uploaded file does not contain a video stream.")

    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    return VideoInfo(
        duration_seconds=duration,
        size_bytes=os.path.getsize(input_path),
        video_codec=video_stream.get("codec_name"),
        width=video_stream.get("width"),
        height=video_stream.get("height"),
        has_audio=has_audio,
    )


def probe_audio(input_path: str) -> AudioInfo:
    """Run ffprobe and validate the file actually contains a readable audio stream."""
    data = _ffprobe_json(input_path, InvalidAudioError)
    duration = _read_duration(data, InvalidAudioError)

    audio_stream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None
    )
    if audio_stream is None:
        raise InvalidAudioError("Uploaded file does not contain an audio stream.")

    return AudioInfo(
        duration_seconds=duration,
        size_bytes=os.path.getsize(input_path),
        audio_codec=audio_stream.get("codec_name"),
    )


def estimate_output_size_bytes(video_info: VideoInfo, target_seconds: float) -> int:
    """Rough estimate based on the source's average bitrate; -c copy keeps bitrate constant."""
    bitrate_bytes_per_second = video_info.size_bytes / video_info.duration_seconds
    return int(bitrate_bytes_per_second * target_seconds)


def ensure_disk_space(output_dir: str, required_bytes: int, safety_margin: float = 1.1) -> None:
    os.makedirs(output_dir, exist_ok=True)
    free = shutil.disk_usage(output_dir).free
    needed = int(required_bytes * safety_margin)
    if needed > free:
        raise InsufficientDiskSpaceError(
            f"Estimated output is {needed / (1024**3):.2f} GB but only "
            f"{free / (1024**3):.2f} GB is free on disk. Free up space or pick a "
            "shorter target duration."
        )


def format_bytes(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} PB"


def _safe_stem(name: str, max_len: int = 40) -> str:
    stem = os.path.splitext(name)[0]
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    return cleaned[:max_len].strip("_")


def build_output_path(
    output_dir: str,
    original_filename: str,
    name_hint: str | None = None,
    output_ext: str | None = None,
) -> str:
    ext = output_ext if output_ext else os.path.splitext(original_filename)[1].lower()
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"{_safe_stem(name_hint)}_" if name_hint else ""
    return os.path.join(output_dir, f"autoloop_{prefix}{uuid.uuid4().hex[:8]}{ext}")


def create_trim_unit(
    input_path: str, output_path: str,
    trim_start: float, trim_end: float,
    has_audio: bool = False, crf: int = 18,
) -> str:
    """
    Frame-accurate trim via the FFmpeg trim filter (re-encodes the short segment).
    The result is cleanly stream-copyable in every downstream loop/boomerang/crossfade step.
    Audio is preserved when has_audio=True and the source file contains an audio track.
    """
    info = probe_video(input_path)
    D = info.duration_seconds

    if trim_start < 0 or trim_end > D + 0.1 or trim_start >= trim_end:
        raise InvalidDurationError(
            f"Invalid trim range [{trim_start:.2f}s, {trim_end:.2f}s] for a {D:.2f}s clip."
        )
    if trim_end - trim_start < 0.5:
        raise InvalidDurationError(
            f"Trimmed segment is too short ({trim_end - trim_start:.2f}s) — minimum 0.5s."
        )

    ffmpeg = get_ffmpeg_path()
    s, e = f"{trim_start:.6f}", f"{trim_end:.6f}"

    if has_audio:
        fc = (
            f"[0:v]trim=start={s}:end={e},setpts=PTS-STARTPTS[vout];"
            f"[0:a]atrim=start={s}:end={e},asetpts=PTS-STARTPTS[aout]"
        )
        cmd = [
            ffmpeg, "-y", "-i", input_path,
            "-filter_complex", fc,
            "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-crf", str(crf), "-preset", "fast",
            "-c:a", "aac", "-b:a", "192k",
            "-pix_fmt", "yuv420p", "-loglevel", "error",
            output_path,
        ]
    else:
        fc = f"[0:v]trim=start={s}:end={e},setpts=PTS-STARTPTS[vout]"
        cmd = [
            ffmpeg, "-y", "-i", input_path, "-an",
            "-filter_complex", fc, "-map", "[vout]",
            "-c:v", "libx264", "-crf", str(crf), "-preset", "fast",
            "-pix_fmt", "yuv420p", "-loglevel", "error",
            output_path,
        ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise FFmpegNotFoundError("ffmpeg disappeared from PATH during execution.") from exc

    if result.returncode != 0:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise RenderError(f"Trim failed:\n{result.stderr.strip()}")

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RenderError("ffmpeg produced no output for the trim unit.")

    return output_path


def create_crossfade_unit(input_path: str, output_path: str, fade_duration: float, crf: int = 18) -> str:
    """
    Build a crossfade loop unit from input_path.

    Output duration = D - C  (where D = source duration, C = fade_duration).
    Structure:  [clip[C : D-C]]  +  [blend(clip[D-C:D] fade-out  over  clip[0:C] fade-in)]

    When stream-looped the last frame of each cycle (clip[C]) matches exactly the
    first frame of the next cycle, so every junction is a smooth dissolve instead
    of a hard cut.  Audio is stripped; add music via render_looped_video_with_audio.
    """
    info = probe_video(input_path)
    D = info.duration_seconds
    C = float(fade_duration)
    min_middle = 0.5

    if C <= 0:
        raise InvalidDurationError("Crossfade duration must be greater than 0.")
    if 2 * C + min_middle > D:
        raise InvalidDurationError(
            f"Crossfade duration {C:.2f}s is too long for a {D:.2f}s clip. "
            f"Maximum allowed: {(D - min_middle) / 2:.2f}s"
        )

    ffmpeg = get_ffmpeg_path()
    fc = (
        f"[0:v]trim=start={C:.6f}:end={D - C:.6f},setpts=PTS-STARTPTS[vmid];"
        f"[0:v]trim=start={D - C:.6f}:end={D:.6f},setpts=PTS-STARTPTS,"
        f"format=yuva420p,fade=t=out:st=0:d={C:.6f}:alpha=1[vfadeout];"
        f"[0:v]trim=start=0:end={C:.6f},setpts=PTS-STARTPTS[vfadein];"
        f"[vfadein][vfadeout]overlay=format=yuv420[vblend];"
        f"[vmid][vblend]concat=n=2:v=1:a=0[vout]"
    )
    cmd = [
        ffmpeg, "-y",
        "-i", input_path,
        "-an",
        "-filter_complex", fc,
        "-map", "[vout]",
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-loglevel", "error",
        output_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise FFmpegNotFoundError("ffmpeg disappeared from PATH during execution.") from exc

    if result.returncode != 0:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise RenderError(f"Failed to create crossfade unit:\n{result.stderr.strip()}")

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RenderError("ffmpeg produced no output for the crossfade unit.")

    return output_path


def create_boomerang_unit(input_path: str, output_path: str, crf: int = 18) -> str:
    """
    Build an A + reverse(A) pre-render unit (video-only, audio stripped).
    The reverse filter buffers all decoded frames in RAM, so it should only
    be called on short source clips. The result is a normal H.264/MP4 file
    that the existing stream-copy loop functions can handle without re-encoding.
    """
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg,
        "-y",
        "-i", input_path,
        "-an",
        "-filter_complex",
        "[0:v]split[fwd][rev];[rev]reverse[rvid];[fwd][rvid]concat=n=2:v=1:a=0[out]",
        "-map", "[out]",
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-loglevel", "error",
        output_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise FFmpegNotFoundError("ffmpeg disappeared from PATH during execution.") from exc

    if result.returncode != 0:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise RenderError(
            f"Failed to create boomerang unit (A + reverse(A)):\n{result.stderr.strip()}"
        )

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RenderError("ffmpeg produced no output for the boomerang unit.")

    return output_path


def generate_waveform_png(
    audio_path: str,
    output_path: str,
    width: int = 900,
    height: int = 130,
) -> str:
    """
    Render the audio waveform to a PNG using FFmpeg's showwavespic filter.
    The caller overlays loop-boundary markers with Pillow at display time
    so the same PNG can be reused for different loop intervals without re-encoding.
    """
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-y",
        "-i", audio_path,
        "-filter_complex",
        f"showwavespic=s={width}x{height}:colors=#3D9BE9:draw=full:filter=average",
        "-frames:v", "1",
        "-loglevel", "error",
        output_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise FFmpegNotFoundError("ffmpeg disappeared from PATH during execution.") from exc

    if result.returncode != 0:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise RenderError(
            f"ffmpeg failed to generate the waveform image.\n\nffmpeg said:\n{result.stderr.strip()}"
        )

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RenderError("ffmpeg produced no waveform image.")

    return output_path


def create_junction_preview(
    input_path: str,
    output_path: str,
    clip_duration: float,
    window: float = 2.0,
    crf: int = 23,
) -> str:
    """
    Render a short preview that simulates the loop junction:
    the last `window` seconds of the clip immediately followed by the first
    `window` seconds — exactly what the viewer sees at every loop boundary.

    Video-only output (no audio); re-encoded to H.264 so every browser can
    play it inline. `clip_duration` is passed in to avoid a redundant probe.
    """
    ffmpeg = get_ffmpeg_path()

    # Clamp window to fit inside the clip with a small guard on each side
    window = min(window, (clip_duration - 0.1) / 2)
    if window < 0.1:
        raise InvalidVideoError(
            f"Clip is too short ({clip_duration:.2f}s) for a junction preview — need at least 0.3s."
        )

    tail_start = max(0.0, clip_duration - window)
    s = f"{tail_start:.6f}"
    d = f"{clip_duration:.6f}"
    w = f"{window:.6f}"

    fc = (
        f"[0:v]trim=start={s}:end={d},setpts=PTS-STARTPTS[tail];"
        f"[0:v]trim=start=0:end={w},setpts=PTS-STARTPTS[head];"
        f"[tail][head]concat=n=2:v=1:a=0[vout]"
    )
    cmd = [
        ffmpeg, "-y",
        "-i", input_path,
        "-filter_complex", fc,
        "-map", "[vout]",
        "-c:v", "libx264", "-crf", str(crf), "-preset", "fast", "-pix_fmt", "yuv420p",
        "-an",
        "-loglevel", "error",
        output_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise FFmpegNotFoundError("ffmpeg disappeared from PATH during execution.") from exc

    if result.returncode != 0:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise RenderError(
            f"ffmpeg failed to build the junction preview.\n\nffmpeg said:\n{result.stderr.strip()}"
        )

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RenderError("ffmpeg produced no output for the junction preview.")

    return output_path


@dataclass
class AspectRatio:
    id: str
    label: str
    ratio_w: int
    ratio_h: int
    description: str


ASPECT_RATIOS: dict[str, AspectRatio] = {
    "source": AspectRatio(
        id="source",
        label="Source (no change)",
        ratio_w=0, ratio_h=0,
        description="Keep the original frame dimensions.",
    ),
    "16:9": AspectRatio(
        id="16:9",
        label="16:9 — Landscape",
        ratio_w=16, ratio_h=9,
        description="YouTube, Vimeo, most desktop screens.",
    ),
    "9:16": AspectRatio(
        id="9:16",
        label="9:16 — Portrait",
        ratio_w=9, ratio_h=16,
        description="TikTok, Instagram Reels, YouTube Shorts.",
    ),
    "1:1": AspectRatio(
        id="1:1",
        label="1:1 — Square",
        ratio_w=1, ratio_h=1,
        description="Instagram posts, Facebook feed.",
    ),
    "4:3": AspectRatio(
        id="4:3",
        label="4:3 — Classic",
        ratio_w=4, ratio_h=3,
        description="Classic TV ratio, older digital cameras.",
    ),
}


def compute_crop_dims(src_w: int, src_h: int, ratio_id: str) -> tuple[int, int]:
    """Return output (width, height) for a center crop to ratio_id, or source dims for 'source'."""
    ar = ASPECT_RATIOS.get(ratio_id)
    if ar is None or ar.ratio_w == 0:
        return src_w, src_h
    source_ar = src_w / src_h
    target_ar = ar.ratio_w / ar.ratio_h
    if source_ar > target_ar:
        crop_h = src_h
        crop_w = int(src_h * ar.ratio_w / ar.ratio_h)
    else:
        crop_w = src_w
        crop_h = int(src_w * ar.ratio_h / ar.ratio_w)
    return crop_w - (crop_w % 2), crop_h - (crop_h % 2)


def create_aspect_ratio_unit(
    input_path: str,
    output_path: str,
    ratio_id: str,
    crf: int = 18,
) -> str:
    """
    Center-crop the input clip to the target aspect ratio (no upscaling).
    Re-encodes as H.264/yuv420p so the result is always stream-copyable.
    Audio is stripped; add music via render_looped_video_with_audio.
    Returns input_path unchanged when ratio_id == 'source'.
    """
    if ratio_id == "source":
        return input_path
    if ratio_id not in ASPECT_RATIOS:
        raise RenderError(f"Unknown aspect ratio: {ratio_id!r}")

    info = probe_video(input_path)
    sw, sh = info.width, info.height
    if not sw or not sh:
        raise RenderError("Cannot determine source dimensions for aspect ratio crop.")

    crop_w, crop_h = compute_crop_dims(sw, sh, ratio_id)
    x = (sw - crop_w) // 2
    y = (sh - crop_h) // 2

    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-y",
        "-i", input_path,
        "-an",
        "-vf", f"crop={crop_w}:{crop_h}:{x}:{y}",
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-loglevel", "error",
        output_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise FFmpegNotFoundError("ffmpeg disappeared from PATH during execution.") from exc

    if result.returncode != 0:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise RenderError(
            f"Crop to {ratio_id} failed:\n{result.stderr.strip()}"
        )

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RenderError("ffmpeg produced no output for the aspect ratio crop.")

    return output_path


OVERLAY_POSITIONS: dict[str, str] = {
    "bottom_right": "Bottom right",
    "bottom_left":  "Bottom left",
    "top_right":    "Top right",
    "top_left":     "Top left",
    "center":       "Center",
}


@dataclass
class WatermarkConfig:
    text: str = ""
    text_position: str = "bottom_right"
    text_size: int = 36
    text_color: str = "#FFFFFF"
    text_opacity: float = 0.9
    text_box: bool = True
    text_box_color: str = "#000000"
    text_box_opacity: float = 0.4
    logo_path: str | None = None
    logo_position: str = "top_right"
    logo_width_pct: int = 15
    logo_opacity: float = 0.9
    margin: int = 20

    @property
    def is_active(self) -> bool:
        return bool(self.text.strip()) or (
            bool(self.logo_path) and os.path.isfile(self.logo_path)
        )


def _escape_drawtext(text: str) -> str:
    text = text.replace("\\", "\\\\")
    text = text.replace("'", "\\'")
    text = text.replace(":", "\\:")
    return text


def _drawtext_pos_expr(position: str, margin: int) -> tuple[str, str]:
    m = str(margin)
    return {
        "top_left":     (m, m),
        "top_right":    (f"w-tw-{m}", m),
        "bottom_left":  (m, f"h-th-{m}"),
        "bottom_right": (f"w-tw-{m}", f"h-th-{m}"),
        "center":       ("(w-tw)/2", "(h-th)/2"),
    }.get(position, (f"w-tw-{m}", f"h-th-{m}"))


def _overlay_pos_expr(position: str, margin: int) -> tuple[str, str]:
    m = str(margin)
    return {
        "top_left":     (m, m),
        "top_right":    (f"W-w-{m}", m),
        "bottom_left":  (m, f"H-h-{m}"),
        "bottom_right": (f"W-w-{m}", f"H-h-{m}"),
        "center":       ("(W-w)/2", "(H-h)/2"),
    }.get(position, (f"W-w-{m}", m))


def create_watermark_unit(
    input_path: str,
    output_path: str,
    config: "WatermarkConfig",
    crf: int = 18,
) -> str:
    """
    Burn text and/or logo overlay into the short loop unit (video-only, audio stripped).
    Re-encodes as H.264/yuv420p so the result is always stream-copyable.
    Returns input_path unchanged when config.is_active is False.
    """
    has_text = bool(config.text.strip())
    has_logo = bool(config.logo_path) and os.path.isfile(config.logo_path)

    if not has_text and not has_logo:
        return input_path

    ffmpeg = get_ffmpeg_path()

    logo_w_px: int | None = None
    if has_logo:
        info = probe_video(input_path)
        logo_w_px = max(20, int((info.width or 1920) * config.logo_width_pct / 100))

    def _drawtext_filter() -> str:
        tx, ty = _drawtext_pos_expr(config.text_position, config.margin)
        escaped = _escape_drawtext(config.text.strip())
        box_bw = max(4, config.text_size // 5)
        box_part = (
            f":box=1:boxcolor={config.text_box_color}@{config.text_box_opacity:.2f}"
            f":boxborderw={box_bw}"
        ) if config.text_box else ""
        return (
            f"drawtext=text='{escaped}'"
            f":fontsize={config.text_size}"
            f":fontcolor={config.text_color}@{config.text_opacity:.2f}"
            f":x={tx}:y={ty}{box_part}"
        )

    cmd = [ffmpeg, "-y", "-i", input_path]
    if has_logo:
        cmd += ["-i", config.logo_path]
    cmd += ["-an"]

    if has_text and has_logo:
        lx, ly = _overlay_pos_expr(config.logo_position, config.margin)
        fc = (
            f"[1:v]scale={logo_w_px}:-2,format=rgba,"
            f"colorchannelmixer=aa={config.logo_opacity:.2f}[logo];"
            f"[0:v][logo]overlay={lx}:{ly},{_drawtext_filter()}[out]"
        )
        cmd += ["-filter_complex", fc, "-map", "[out]"]
    elif has_logo:
        lx, ly = _overlay_pos_expr(config.logo_position, config.margin)
        fc = (
            f"[1:v]scale={logo_w_px}:-2,format=rgba,"
            f"colorchannelmixer=aa={config.logo_opacity:.2f}[logo];"
            f"[0:v][logo]overlay={lx}:{ly}[out]"
        )
        cmd += ["-filter_complex", fc, "-map", "[out]"]
    else:
        cmd += ["-vf", _drawtext_filter()]

    cmd += [
        "-c:v", "libx264", "-crf", str(crf), "-preset", "fast",
        "-pix_fmt", "yuv420p", "-loglevel", "error",
        output_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise FFmpegNotFoundError("ffmpeg disappeared from PATH during execution.") from exc

    if result.returncode != 0:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise RenderError(f"Failed to burn in overlay:\n{result.stderr.strip()}")

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RenderError("ffmpeg produced no output for the watermark unit.")

    return output_path


def render_looped_video(input_path: str, output_path: str, target_seconds: int) -> str:
    """
    Stream-loop input_path to target_seconds using lossless -c copy remuxing.
    Raises RenderError on any ffmpeg failure (e.g. container/codec incompatibility).
    """
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg,
        "-y",
        "-stream_loop", "-1",
        "-i", input_path,
        "-t", str(target_seconds),
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        "-loglevel", "error",
        output_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise FFmpegNotFoundError("ffmpeg disappeared from PATH during execution.") from exc

    if result.returncode != 0:
        if os.path.exists(output_path):
            os.remove(output_path)
        stderr = result.stderr.strip()
        raise RenderError(
            "ffmpeg failed to render the looped video. This usually means the "
            "container/codec combination doesn't support stream copy.\n\n"
            f"ffmpeg said:\n{stderr}"
        )

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RenderError("ffmpeg reported success but produced no output file.")

    return output_path


@dataclass
class AudioBatchItem:
    display_name: str
    audio_path: str
    audio_info: AudioInfo
    target_seconds: float


@dataclass
class BatchRenderResult:
    display_name: str
    success: bool
    output_path: str | None = None
    size_bytes: int | None = None
    error: str | None = None


@dataclass
class MatrixVideoItem:
    display_name: str
    video_path: str
    video_info: "VideoInfo"


@dataclass
class EncodePreset:
    id: str
    label: str
    description: str
    speed_label: str
    size_label: str
    compat_label: str
    codec: str | None  # None = stream copy (no re-encode unit needed)
    ext: str
    default_crf: int | None = None
    crf_min: int | None = None
    crf_max: int | None = None


ENCODE_PRESETS: dict[str, EncodePreset] = {
    "copy": EncodePreset(
        id="copy",
        label="Stream copy (default)",
        description="Packets remuxed without re-encoding — zero quality loss, instant output.",
        speed_label="Instant",
        size_label="Same as source",
        compat_label="Source codec",
        codec=None,
        ext=".mp4",
    ),
    "h264": EncodePreset(
        id="h264",
        label="H.264 — web-safe",
        description="libx264/yuv420p MP4 — plays everywhere: browsers, phones, smart TVs.",
        speed_label="Fast",
        size_label="Similar to source",
        compat_label="Universal",
        codec="libx264",
        ext=".mp4",
        default_crf=23,
        crf_min=18,
        crf_max=28,
    ),
    "h265": EncodePreset(
        id="h265",
        label="H.265 — ~50% smaller",
        description="libx265/yuv420p MP4 — half the file size of H.264 at the same quality. Slower to encode.",
        speed_label="Slow",
        size_label="~50% of H.264",
        compat_label="Modern devices",
        codec="libx265",
        ext=".mp4",
        default_crf=28,
        crf_min=24,
        crf_max=32,
    ),
    "prores": EncodePreset(
        id="prores",
        label="ProRes 422 — editor-ready",
        description="Apple ProRes 422 HQ MOV — near-lossless, editing handles on every frame. Very large files.",
        speed_label="Fast",
        size_label="Very large (~10×)",
        compat_label="macOS / editors",
        codec="prores_ks",
        ext=".mov",
    ),
}


def _build_video_encode_args(preset_id: str, crf: int | None = None) -> list[str]:
    preset = ENCODE_PRESETS[preset_id]
    if preset.codec is None:
        return ["-c:v", "copy"]
    if preset_id == "h264":
        c = crf if crf is not None else preset.default_crf
        return ["-c:v", "libx264", "-crf", str(c), "-preset", "fast", "-pix_fmt", "yuv420p"]
    if preset_id == "h265":
        c = crf if crf is not None else preset.default_crf
        # -tag:v hvc1 makes H.265 MP4 compatible with Apple QuickTime and iOS
        return ["-c:v", "libx265", "-crf", str(c), "-preset", "fast", "-pix_fmt", "yuv420p", "-tag:v", "hvc1"]
    if preset_id == "prores":
        return ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"]
    raise RenderError(f"Unknown preset id: {preset_id!r}")


def create_encode_unit(
    input_path: str,
    output_path: str,
    preset_id: str,
    crf: int | None = None,
) -> str:
    """
    Re-encode the short loop unit to the target codec (video-only, audio stripped).
    The encoded unit can then be stream-looped at full speed via render_looped_video*.
    Returns input_path unchanged when preset_id == 'copy'.
    """
    if preset_id == "copy":
        return input_path
    if preset_id not in ENCODE_PRESETS:
        raise RenderError(f"Unknown preset: {preset_id!r}")

    ffmpeg = get_ffmpeg_path()
    video_args = _build_video_encode_args(preset_id, crf)
    cmd = [
        ffmpeg, "-y",
        "-i", input_path,
        "-an",
        *video_args,
        "-loglevel", "error",
        output_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise FFmpegNotFoundError("ffmpeg disappeared from PATH during execution.") from exc

    if result.returncode != 0:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise RenderError(
            f"Re-encoding to {ENCODE_PRESETS[preset_id].label} failed:\n{result.stderr.strip()}"
        )

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RenderError("ffmpeg produced no output for the encode unit.")

    return output_path


def build_matrix_output_path(
    output_dir: str,
    video_name: str,
    audio_name: str,
    output_ext: str | None = None,
) -> str:
    v = _safe_stem(os.path.splitext(video_name)[0])
    a = _safe_stem(os.path.splitext(audio_name)[0])
    ext = output_ext if output_ext else ".mp4"
    return os.path.join(output_dir, f"{v}_x_{a}{ext}")


def _build_audio_filter(
    volume: float, fade_in: float, fade_out: float, target_seconds: float
) -> str | None:
    parts = []
    if abs(volume - 1.0) > 0.001:
        parts.append(f"volume={volume:.3f}")
    if fade_in > 0:
        parts.append(f"afade=t=in:st=0:d={fade_in:.3f}")
    if fade_out > 0:
        fade_start = max(0.0, target_seconds - fade_out)
        parts.append(f"afade=t=out:st={fade_start:.3f}:d={fade_out:.3f}")
    return ",".join(parts) if parts else None


def render_audio_batch(
    video_path: str,
    video_filename: str,
    items: list[AudioBatchItem],
    output_dir: str,
    volume: float = 1.0,
    fade_in: float = 0.0,
    fade_out: float = 0.0,
    output_ext: str | None = None,
):
    """
    Render one looped-video-with-audio output per AudioBatchItem, yielding a
    BatchRenderResult as each one finishes. A failure on one item does not abort
    the rest of the batch — errors are captured per item instead of raised.
    output_ext overrides the container extension (e.g. '.mov' for ProRes); if None,
    the extension comes from video_filename.
    """
    for item in items:
        output_path = build_output_path(
            output_dir, video_filename, name_hint=item.display_name, output_ext=output_ext
        )
        try:
            render_looped_video_with_audio(
                video_path, item.audio_path, output_path, item.target_seconds,
                volume=volume, fade_in=fade_in, fade_out=fade_out,
            )
            yield BatchRenderResult(
                display_name=item.display_name,
                success=True,
                output_path=output_path,
                size_bytes=os.path.getsize(output_path),
            )
        except AutoLoopError as exc:
            yield BatchRenderResult(display_name=item.display_name, success=False, error=str(exc))


def render_looped_video_with_audio(
    video_path: str,
    audio_path: str,
    output_path: str,
    target_seconds: float,
    volume: float = 1.0,
    fade_in: float = 0.0,
    fade_out: float = 0.0,
) -> str:
    """
    Stream-loop video_path (video stream only, lossless -c copy) and pair it with
    audio_path as the soundtrack, replacing the source video's own audio (if any).
    target_seconds should be audio_path's own duration; both -t and -shortest are
    used as a belt-and-suspenders pair so the output stops exactly when the audio
    ends, since -shortest alone is unreliable against an infinitely -stream_loop'd input.
    """
    ffmpeg = get_ffmpeg_path()
    af = _build_audio_filter(volume, fade_in, fade_out, target_seconds)
    cmd = [
        ffmpeg,
        "-y",
        "-stream_loop", "-1",
        "-i", video_path,
        "-i", audio_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-t", str(target_seconds),
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        *(["-af", af] if af else []),
        "-shortest",
        "-avoid_negative_ts", "make_zero",
        "-loglevel", "error",
        output_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise FFmpegNotFoundError("ffmpeg disappeared from PATH during execution.") from exc

    if result.returncode != 0:
        if os.path.exists(output_path):
            os.remove(output_path)
        stderr = result.stderr.strip()
        raise RenderError(
            "ffmpeg failed to render the looped video with audio. This usually means "
            "the video's codec doesn't support stream copy into the chosen container.\n\n"
            f"ffmpeg said:\n{stderr}"
        )

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RenderError("ffmpeg reported success but produced no output file.")

    return output_path


def render_looped_video_with_looped_audio(
    video_path: str,
    audio_path: str,
    output_path: str,
    target_seconds: float,
    volume: float = 1.0,
    fade_in: float = 0.0,
    fade_out: float = 0.0,
) -> str:
    """
    Stream-loop BOTH video and audio independently to target_seconds.
    Use when the audio is shorter than the target duration — both streams
    loop at their own cycle lengths and are cut together by -t.
    """
    ffmpeg = get_ffmpeg_path()
    af = _build_audio_filter(volume, fade_in, fade_out, target_seconds)
    cmd = [
        ffmpeg,
        "-y",
        "-stream_loop", "-1",
        "-i", video_path,
        "-stream_loop", "-1",
        "-i", audio_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-t", str(target_seconds),
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        *(["-af", af] if af else []),
        "-avoid_negative_ts", "make_zero",
        "-loglevel", "error",
        output_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise FFmpegNotFoundError("ffmpeg disappeared from PATH during execution.") from exc

    if result.returncode != 0:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise RenderError(
            "ffmpeg failed to render the looped video with looped audio.\n\n"
            f"ffmpeg said:\n{result.stderr.strip()}"
        )

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RenderError("ffmpeg reported success but produced no output file.")

    return output_path
