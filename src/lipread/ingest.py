"""Ingest: fetch a source video, probe it, and stream it as 25 fps BGR frames.

Everything heavy runs in ffmpeg subprocesses. Frames are yielded one at a time,
so a long clip never sits in memory as a whole.
"""
from __future__ import annotations

import glob
import json
import logging
import os
import re
import subprocess
import tempfile
from typing import Any, Iterator, Optional

import numpy as np

from .errors import LipReadError
from .types import FPS, VideoInfo

log = logging.getLogger("lipread")

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)
_VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov")
YTDLP_FORMAT = "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080]/b"


# --------------------------------------------------------------------------- fetch

def fetch(source: str, work_dir: str) -> str:
    """Return a local path for ``source``.

    An http(s) URL is downloaded with yt-dlp into ``work_dir`` (merged to mp4, at
    most 1080p, single video only). Anything else must be an existing file.
    """
    if not _URL_RE.match(source):
        if not os.path.isfile(source):
            raise LipReadError(f"input file not found: {source}")
        return os.path.abspath(source)

    try:
        import yt_dlp
    except ImportError as exc:
        raise LipReadError('URL input needs yt-dlp: pip install "lipread[url]"') from exc

    os.makedirs(work_dir, exist_ok=True)
    opts: dict[str, Any] = {
        "format": YTDLP_FORMAT,
        "merge_output_format": "mp4",
        "outtmpl": os.path.join(work_dir, "%(id)s.%(ext)s"),
        "noplaylist": True,
        "restrictfilenames": True,
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(source, download=False)
            if info is None or info.get("_type") == "playlist" or "entries" in info:
                raise LipReadError("URL must point to a single video, not a playlist")
            video_id = str(info["id"])
            ydl.download([source])
    except yt_dlp.utils.YoutubeDLError as exc:
        raise LipReadError(f"download failed: {exc}") from exc

    exact = os.path.join(work_dir, f"{video_id}.mp4")
    if os.path.isfile(exact):
        return os.path.abspath(exact)
    for cand in sorted(glob.glob(os.path.join(work_dir, glob.escape(video_id) + ".*"))):
        if cand.endswith(_VIDEO_EXTS):
            return os.path.abspath(cand)
    raise LipReadError(f"download finished but no video file was found in {work_dir}")


# --------------------------------------------------------------------------- probe

def probe(path: str) -> VideoInfo:
    """Read stream and container metadata with ffprobe."""
    if not os.path.isfile(path):
        raise LipReadError(f"input file not found: {path}")
    data = _ffprobe_json(path)
    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise LipReadError(f"no video stream in {path}")

    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    if width <= 0 or height <= 0:
        raise LipReadError(f"video has no usable frame size: {path}")

    rotation = _rotation(video)
    if rotation in (90, 270):
        # ffmpeg applies the rotation when decoding, so the frames are swapped.
        width, height = height, width

    fps = _parse_rate(video.get("avg_frame_rate")) or _parse_rate(video.get("r_frame_rate"))
    if fps is None:
        log.warning("no usable frame rate in %s; assuming %d fps", path, FPS)
        fps = float(FPS)

    duration = _duration(data, video, fps)
    if duration <= 0:
        log.warning("could not determine duration of %s", path)

    field_order = video.get("field_order")
    return VideoInfo(
        path=os.path.abspath(path),
        width=width,
        height=height,
        src_fps=fps,
        duration=duration,
        interlaced=field_order not in ("progressive", "unknown", None),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
        rotation=rotation,
    )


def _ffprobe_json(path: str) -> dict[str, Any]:
    cmd = ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise LipReadError("ffprobe not found; install ffmpeg") from exc
    if proc.returncode != 0:
        raise LipReadError(f"ffprobe failed on {path}: {proc.stderr.strip()[-300:]}")
    try:
        return json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise LipReadError(f"ffprobe returned invalid JSON for {path}") from exc


def _parse_rate(value: Optional[str]) -> Optional[float]:
    """Parse '30000/1001' or '25' into a positive float; None when unusable."""
    if not value:
        return None
    try:
        if "/" in value:
            num, den = value.split("/", 1)
            denom = float(den)
            if denom == 0:
                return None
            rate = float(num) / denom
        else:
            rate = float(value)
    except ValueError:
        return None
    return rate if rate > 0 else None


def _rotation(stream: dict[str, Any]) -> int:
    """Display rotation in degrees, normalised to 0, 90, 180 or 270."""
    raw: Any = None
    for side in stream.get("side_data_list") or []:
        if "rotation" in side:
            raw = side["rotation"]
            break
    if raw is None:
        raw = (stream.get("tags") or {}).get("rotate")
    if raw is None:
        return 0
    try:
        degrees = float(raw)
    except (TypeError, ValueError):
        return 0
    return (int(round(degrees / 90.0)) * 90) % 360


def _duration(data: dict[str, Any], stream: dict[str, Any], fps: float) -> float:
    for source in (data.get("format") or {}, stream):
        try:
            value = float(source.get("duration"))
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    try:
        frames = int(stream.get("nb_frames"))
    except (TypeError, ValueError):
        return 0.0
    return frames / fps if frames > 0 else 0.0


# --------------------------------------------------------------------------- frames

def iter_frames(
    info: VideoInfo,
    start: float = 0.0,
    end: Optional[float] = None,
    max_height: int = 1080,
) -> Iterator[np.ndarray]:
    """Yield BGR frames (uint8, H x W x 3) at 25 fps for ``[start, end)`` seconds.

    Frame 0 is the frame at ``start``. Interlaced input is deinterlaced first.
    Frames taller than ``max_height`` are scaled down, keeping the aspect ratio.
    Closing the generator early stops the ffmpeg process.
    """
    start = max(0.0, float(start))
    if end is not None and end <= start:
        return
    if info.width <= 0 or info.height <= 0:
        raise LipReadError("cannot decode a video with zero size")

    out_w, out_h = _output_size(info.width, info.height, max_height)
    filters: list[str] = []
    if info.interlaced:
        filters.append("yadif")
    filters.append(f"fps={FPS}")
    if (out_w, out_h) != (info.width, info.height):
        filters.append(f"scale={out_w}:{out_h}")

    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error"]
    if start > 0:
        cmd += ["-ss", f"{start:.6f}"]          # input seek: fast, and frame-accurate
    cmd += ["-i", info.path]
    if end is not None:
        cmd += ["-t", f"{end - start:.6f}"]
    cmd += [
        "-map", "0:v:0", "-an", "-sn",
        "-vf", ",".join(filters),
        "-pix_fmt", "bgr24",
        "-f", "rawvideo", "pipe:1",
    ]
    frame_shape = (out_h, out_w, 3)

    err = tempfile.TemporaryFile()
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, bufsize=0)
    except FileNotFoundError as exc:
        err.close()
        raise LipReadError("ffmpeg not found; install ffmpeg") from exc

    yielded = 0
    finished = False
    try:
        stream = proc.stdout
        assert stream is not None
        while True:
            frame = np.empty(frame_shape, dtype=np.uint8)
            if not _read_exact(stream, frame):
                break
            yielded += 1
            yield frame
        finished = True
    finally:
        if proc.poll() is None:
            proc.kill()
        if proc.stdout is not None:
            proc.stdout.close()
        rc = proc.wait()
        if finished and rc != 0:
            tail = _tail(err)
            if yielded == 0:
                err.close()
                raise LipReadError(f"ffmpeg failed to decode {info.path}: {tail}")
            log.warning("ffmpeg exited with %d after %d frames: %s", rc, yielded, tail)
        err.close()


def _output_size(width: int, height: int, max_height: int) -> tuple[int, int]:
    """Output (width, height) for a frame: even width, height capped at max_height."""
    if height > max_height:
        out_h = max_height - (max_height % 2)
        out_w = int(round(width * out_h / height))
    else:
        out_h = height
        out_w = width
    out_w -= out_w % 2
    return max(2, out_w), out_h


def _read_exact(stream: Any, frame: np.ndarray) -> bool:
    """Fill ``frame`` from a raw pipe. False on end of stream (partial frames are dropped)."""
    view = memoryview(frame.reshape(-1))
    total = view.nbytes
    got = 0
    while got < total:
        n = stream.readinto(view[got:])
        if not n:
            return False
        got += n
    return True


def _tail(err: Any, limit: int = 400) -> str:
    try:
        err.seek(0)
        return err.read().decode("utf-8", "replace").strip()[-limit:]
    except (OSError, ValueError):
        return ""


def count_frames(info: VideoInfo, start: float = 0.0, end: Optional[float] = None) -> int:
    """Expected number of 25 fps frames in ``[start, end)``, clipped to the duration."""
    stop = info.duration if end is None else min(end, info.duration)
    return max(0, int(round((stop - start) * FPS)))


# --------------------------------------------------------------------------- audio

def extract_audio(
    info: VideoInfo,
    start: float = 0.0,
    end: Optional[float] = None,
    sr: int = 16000,
) -> Optional[np.ndarray]:
    """Mono float32 samples at ``sr`` Hz for ``[start, end)``; None if there is no audio."""
    if not info.has_audio:
        return None
    start = max(0.0, float(start))
    if end is not None and end <= start:
        return np.zeros(0, dtype=np.float32)

    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error"]
    if start > 0:
        cmd += ["-ss", f"{start:.6f}"]
    cmd += ["-i", info.path]
    if end is not None:
        cmd += ["-t", f"{end - start:.6f}"]
    cmd += [
        "-map", "0:a:0", "-vn", "-sn",
        "-ac", "1", "-ar", str(int(sr)),
        "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False)
    except FileNotFoundError as exc:
        raise LipReadError("ffmpeg not found; install ffmpeg") from exc
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace").strip()[-300:]
        raise LipReadError(f"audio extraction failed on {info.path}: {tail}")
    return np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32)
