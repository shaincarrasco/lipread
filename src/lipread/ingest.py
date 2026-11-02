"""Ingest: fetch a source video, probe it, and stream it as 25 fps BGR frames.

Everything heavy runs in ffmpeg subprocesses. Frames are yielded one at a time,
so a long clip never sits in memory as a whole.
"""
from __future__ import annotations

import glob
import ipaddress
import json
import logging
import os
import re
import socket
import subprocess
import tempfile
import urllib.parse
import warnings
from typing import Any, Iterator, Optional

import numpy as np

from .errors import LipReadError
from .types import FPS, VideoInfo

log = logging.getLogger("lipread")

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)
_VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov")
YTDLP_FORMAT = "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080]/b"
DEFAULT_MAX_SIDE = 1920


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        log.warning("%s is not an integer; using %d", name, default)
        return default


MAX_URL_SECONDS = _env_int("LIPREAD_MAX_URL_SECONDS", 1800)
MAX_URL_FILESIZE = 2_000_000_000


# --------------------------------------------------------------------------- fetch

def _check_public_url(url: str) -> None:
    """Refuse URLs that could reach the local machine or a private network (SSRF guard).

    Only http and https are allowed. The host is resolved and every address it maps to
    must be a public unicast address.
    """
    parts = urllib.parse.urlsplit(url)
    if parts.scheme.lower() not in ("http", "https"):
        raise LipReadError(f"only http and https URLs are allowed: {url}")
    host = parts.hostname
    if not host:
        raise LipReadError(f"URL has no host name: {url}")
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError) as exc:
        raise LipReadError(f"cannot resolve host {host}: {exc}") from exc
    for entry in infos:
        raw = str(entry[4][0]).split("%", 1)[0]     # drop an IPv6 scope id
        try:
            ip = ipaddress.ip_address(raw)
        except ValueError as exc:
            raise LipReadError(f"unrecognised address for {host}: {raw}") from exc
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        if (ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_multicast
                or ip.is_reserved or ip.is_unspecified):
            raise LipReadError(f"refusing to fetch a non-public address ({host} -> {ip})")


def _reject_unsuitable(info: Optional[dict[str, Any]]) -> None:
    if info is None or info.get("_type") == "playlist" or "entries" in info:
        raise LipReadError("URL must point to a single video, not a playlist")
    if info.get("is_live"):
        raise LipReadError("live streams are not supported; give a finished video")
    duration = info.get("duration") or 0
    if duration > MAX_URL_SECONDS:
        raise LipReadError(
            f"video is {duration:.0f} s long; the limit is {MAX_URL_SECONDS} s "
            "(set LIPREAD_MAX_URL_SECONDS to change it)"
        )


def _downloaded_path(ydl: Any, info: dict[str, Any], work_dir: str) -> Optional[str]:
    """Locate the file yt-dlp just wrote, without relying on the video id."""
    for entry in info.get("requested_downloads") or []:
        path = entry.get("filepath")
        if path and os.path.isfile(path):
            return os.path.abspath(path)
    try:
        guess = os.path.splitext(ydl.prepare_filename(info))[0] + ".mp4"
    except Exception:  # noqa: BLE001 - fall through to the directory scan
        guess = ""
    if guess and os.path.isfile(guess):
        return os.path.abspath(guess)
    candidates = [
        p for p in glob.glob(os.path.join(work_dir, "*"))
        if p.lower().endswith(_VIDEO_EXTS) and os.path.isfile(p)
    ]
    if candidates:
        return os.path.abspath(max(candidates, key=os.path.getmtime))
    return None


def fetch(source: str, work_dir: str) -> str:
    """Return a local path for ``source``.

    An http(s) URL is downloaded with yt-dlp into ``work_dir`` (merged to mp4, at
    most 1080p, single finished video under the length limit). Anything else must be
    an existing file.
    """
    if not _URL_RE.match(source):
        if not os.path.isfile(source):
            raise LipReadError(f"input file not found: {source}")
        return os.path.abspath(source)

    _check_public_url(source)

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
        "max_downloads": 1,
        "max_filesize": MAX_URL_FILESIZE,
        "socket_timeout": 30,
        "match_filter": yt_dlp.utils.match_filter_func("!is_live"),
        "restrictfilenames": True,
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            _reject_unsuitable(ydl.extract_info(source, download=False))
            info = ydl.extract_info(source, download=True)
            _reject_unsuitable(info)
            path = _downloaded_path(ydl, info, work_dir)
    except yt_dlp.utils.YoutubeDLError as exc:
        raise LipReadError(f"download failed: {exc}") from exc

    if path is None:
        raise LipReadError(f"download finished but no video file was found in {work_dir}")
    return path


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
    cmd = ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json",
           "-protocol_whitelist", "file,pipe", path]
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
    max_side: int = DEFAULT_MAX_SIDE,
    *,
    max_height: Optional[int] = None,
) -> Iterator[np.ndarray]:
    """Yield BGR frames (uint8, H x W x 3) at 25 fps for ``[start, end)`` seconds.

    Frame 0 is the frame at ``start``. Interlaced input is deinterlaced first.
    Frames whose longer side exceeds ``max_side`` are scaled down, keeping the aspect
    ratio. ``max_height`` is a deprecated alias for ``max_side``.
    Closing the generator early stops the ffmpeg process.
    """
    if max_height is not None:
        warnings.warn("max_height is deprecated; use max_side", DeprecationWarning, stacklevel=2)
        max_side = max_height
    start = max(0.0, float(start))
    if end is not None and end <= start:
        return
    if info.width <= 0 or info.height <= 0:
        raise LipReadError("cannot decode a video with zero size")

    out_w, out_h = _output_size(info.width, info.height, max_side)
    filters: list[str] = []
    if info.interlaced:
        filters.append("yadif")
    filters.append(f"fps={FPS}")
    if (out_w, out_h) != (info.width, info.height):
        filters.append(f"scale={out_w}:{out_h}")

    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error"]
    if start > 0:
        cmd += ["-ss", f"{start:.6f}"]          # input seek: fast, and frame-accurate
    cmd += ["-protocol_whitelist", "file,pipe", "-i", info.path]
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
    completed = False           # True only when the stream reached its end
    rc = 0
    tail = ""
    try:
        stream = proc.stdout
        assert stream is not None
        while True:
            frame = np.empty(frame_shape, dtype=np.uint8)
            if not _read_exact(stream, frame):
                break
            yielded += 1
            yield frame
        completed = True
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        rc = _reap(proc, after_eof=completed)
        if completed and rc != 0:
            tail = _tail(err)
        err.close()

    # Reached only after a normal end of stream (a closed or failed generator never gets here).
    if rc != 0:
        if yielded == 0:
            raise LipReadError(f"ffmpeg failed to decode {info.path}: {tail}")
        log.warning("ffmpeg exited with %d after %d frames: %s", rc, yielded, tail)
    if yielded == 0:
        raise LipReadError("no frames in the requested time range")


def _reap(proc: subprocess.Popen, after_eof: bool) -> int:
    """Wait for ffmpeg to exit and return its code.

    After a normal end of stream ffmpeg is given a few seconds to exit on its own; it is
    only killed if it hangs. Otherwise (the generator was closed early or failed) it is
    stopped at once.
    """
    if after_eof:
        try:
            return proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            log.warning("ffmpeg did not exit after the end of the stream; killing it")
            proc.kill()
            proc.wait()
            return 0                # every frame was already read; nothing to report
    if proc.poll() is None:
        proc.kill()
    return proc.wait()


def _output_size(width: int, height: int, max_side: int = DEFAULT_MAX_SIDE) -> tuple[int, int]:
    """Output (width, height) for a frame: even sides, longer side capped at ``max_side``."""
    long_side = max(width, height)
    if long_side > max_side:
        scale = max_side / long_side
        out_w = int(round(width * scale))
        out_h = int(round(height * scale))
    else:
        out_w, out_h = width, height
    out_w -= out_w % 2
    out_h -= out_h % 2
    return max(2, out_w), max(2, out_h)


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
    cmd += ["-protocol_whitelist", "file,pipe", "-i", info.path]
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
