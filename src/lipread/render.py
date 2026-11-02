"""Annotated neon video: face boxes with tier labels, a live caption bar and a provenance badge.

Frames are drawn in place and piped as raw BGR into ffmpeg, so the whole clip is never
held in memory. Sizes are relative to a 1280 px wide frame, so 640 px and 1920 px
frames keep the same proportions.
"""
from __future__ import annotations

import itertools
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Iterable, Iterator

import cv2
import numpy as np

from .errors import LipReadError
from .types import (FPS, Confidence, Detection, FaceTrack, Job, Readability, Result,
                    TrackResult, Utterance)

log = logging.getLogger("lipread")

BADGE_TEXT = "AI LIP-READ ESTIMATE | MAY BE WRONG"
FONT = cv2.FONT_HERSHEY_DUPLEX
REF_WIDTH = 1280.0
MAX_CAPTION_LINES = 2
CAPTION_WIDTH_FRACTION = 0.9

# Colours are BGR. Neon palette from the design brief.
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREY_TEXT = (200, 200, 200)
DIM_GREY = (107, 107, 107)
NEON_GREEN = (20, 255, 57)                       # boosted words
TIER_COLOR = {
    Readability.READABLE: (255, 240, 0),         # #00F0FF cyan
    Readability.MARGINAL: (0, 176, 255),         # #FFB000 amber
    Readability.UNREADABLE: (149, 45, 255),      # #FF2D95 magenta
}
WORD_COLOR = {
    Confidence.HIGH: WHITE,
    Confidence.MEDIUM: (255, 240, 0),            # cyan
    Confidence.LOW: DIM_GREY,
}


@dataclass(frozen=True)
class _Style:
    """Pixel sizes derived from the frame width."""

    width: int
    height: int
    box_px: int
    glow_px: int
    pad: int
    margin: int
    label_scale: float
    label_thick: int
    cap_scale: float
    cap_thick: int
    badge_scale: float
    badge_thick: int

    @classmethod
    def for_size(cls, width: int, height: int) -> "_Style":
        s = width / REF_WIDTH
        return cls(
            width=width,
            height=height,
            box_px=int(min(4, max(2, round(2 * s)))),
            glow_px=max(3, int(round(5 * s))),
            pad=max(4, int(round(8 * s))),
            margin=max(6, int(round(14 * s))),
            label_scale=max(0.45, 0.6 * s),
            label_thick=max(1, int(round(1.6 * s))),
            cap_scale=max(0.5, 1.0 * s),
            cap_thick=max(1, int(round(2 * s + 0.4))),
            badge_scale=max(0.4, 0.55 * s),
            badge_thick=max(1, int(round(1.4 * s))),
        )

    @property
    def panel_w(self) -> int:
        return int(self.width * CAPTION_WIDTH_FRACTION)

    @property
    def inner_w(self) -> int:
        return self.panel_w - 2 * self.pad

    @property
    def bottom(self) -> int:
        return self.margin * 2


@dataclass
class _Token:
    """One drawable piece of the caption. ``start is None`` means always shown."""

    text: str
    color: tuple[int, int, int]
    start: float | None
    end: float | None
    width: int = 0
    x: int = 0
    line: int = 0


@dataclass
class _Caption:
    """Caption layout for one utterance, computed once per utterance."""

    tokens: list[_Token]
    lines: int
    scale: float
    thick: int
    line_h: int
    text_h: int


def _measure(text: str, scale: float, thick: int) -> tuple[int, int]:
    """Return (width, height) of text drawn with the render font."""
    (w, h), _ = cv2.getTextSize(text, FONT, scale, thick)
    return w, h


def _fit_scale(text: str, scale: float, thick: int, max_w: int, floor: float) -> float:
    """Shrink ``scale`` until ``text`` fits in ``max_w`` pixels (never below ``floor``)."""
    while scale > floor and _measure(text, scale, thick)[0] > max_w:
        scale *= 0.93
    return scale


def _put(img: np.ndarray, text: str, org: tuple[int, int], scale: float, thick: int,
         color: tuple[int, int, int], outline: bool = True) -> None:
    """Draw anti-aliased text, with a dark outline for legibility over video."""
    if outline:
        cv2.putText(img, text, org, FONT, scale, BLACK, thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def _rounded_rect(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, r: int,
                  color: tuple[int, int, int], thick: int) -> None:
    """Stroke a rounded rectangle with corner radius ``r``."""
    cv2.line(img, (x0 + r, y0), (x1 - r, y0), color, thick, cv2.LINE_AA)
    cv2.line(img, (x0 + r, y1), (x1 - r, y1), color, thick, cv2.LINE_AA)
    cv2.line(img, (x0, y0 + r), (x0, y1 - r), color, thick, cv2.LINE_AA)
    cv2.line(img, (x1, y0 + r), (x1, y1 - r), color, thick, cv2.LINE_AA)
    cv2.ellipse(img, (x0 + r, y0 + r), (r, r), 180, 0, 90, color, thick, cv2.LINE_AA)
    cv2.ellipse(img, (x1 - r, y0 + r), (r, r), 270, 0, 90, color, thick, cv2.LINE_AA)
    cv2.ellipse(img, (x1 - r, y1 - r), (r, r), 0, 0, 90, color, thick, cv2.LINE_AA)
    cv2.ellipse(img, (x0 + r, y1 - r), (r, r), 90, 0, 90, color, thick, cv2.LINE_AA)


def _draw_face(frame: np.ndarray, det: Detection, color: tuple[int, int, int],
               label: str, style: _Style) -> None:
    """Draw a glowing rounded box, the mouth marker and the tier tag for one detection."""
    H, W = frame.shape[:2]
    x, y, w, h = det.box
    if not np.all(np.isfinite([x, y, w, h])) or w <= 0 or h <= 0:
        return
    x0, y0 = int(round(x)), int(round(y))
    x1, y1 = int(round(x + w)), int(round(y + h))
    r = max(3, min(min(x1 - x0, y1 - y0) // 6, (x1 - x0) // 2, (y1 - y0) // 2))

    # Glow: blur a thick copy of the outline inside a padded ROI and add it back.
    pad = style.glow_px * 4
    rx0, ry0 = max(0, x0 - pad), max(0, y0 - pad)
    rx1, ry1 = min(W, x1 + pad), min(H, y1 + pad)
    if rx1 > rx0 and ry1 > ry0:
        roi = frame[ry0:ry1, rx0:rx1]
        layer = np.zeros(roi.shape, np.uint8)
        _rounded_rect(layer, x0 - rx0, y0 - ry0, x1 - rx0, y1 - ry0, r, color, style.glow_px)
        # Blur at half resolution (4x fewer pixels), then scale back up.
        small = cv2.resize(layer, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), sigmaX=style.glow_px * 0.75)
        glow = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_LINEAR)
        glow = cv2.convertScaleAbs(glow, alpha=2.0, beta=0)
        cv2.add(roi, glow, dst=roi)

    _rounded_rect(frame, x0, y0, x1, y1, r, color, style.box_px)

    if det.keypoints4 is not None:
        mx, my = det.keypoints4[3]
        if np.isfinite(mx) and np.isfinite(my):
            cv2.circle(frame, (int(round(mx)), int(round(my))), style.box_px + 1, color, -1, cv2.LINE_AA)

    _draw_tag(frame, label, color, x0, y0, style)


def _draw_tag(frame: np.ndarray, text: str, color: tuple[int, int, int],
              x: int, y: int, style: _Style) -> None:
    """Solid neon tag with black text, placed above the box or inside it when there is no room."""
    H, W = frame.shape[:2]
    scale, thick, pad = style.label_scale, style.label_thick, max(3, style.pad // 2)
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
    tag_h = th + base + 2 * pad
    tag_w = tw + 2 * pad
    lx = int(np.clip(x, 0, max(0, W - tag_w)))
    ty = y - tag_h - 2
    if ty < 0:
        ty = y + 2
    ty = int(np.clip(ty, 0, max(0, H - tag_h)))
    cv2.rectangle(frame, (lx, ty), (lx + tag_w, ty + tag_h), color, -1)
    cv2.putText(frame, text, (lx + pad, ty + pad + th), FONT, scale, BLACK, thick, cv2.LINE_AA)


def _draw_badge(frame: np.ndarray, style: _Style) -> None:
    """Top-right provenance badge, drawn on every frame."""
    H, W = frame.shape[:2]
    max_w = int(W * 0.6)
    scale = _fit_scale(BADGE_TEXT, style.badge_scale, style.badge_thick, max_w, 0.35)
    tw, th = _measure(BADGE_TEXT, scale, style.badge_thick)
    _, base = cv2.getTextSize(BADGE_TEXT, FONT, scale, style.badge_thick)
    pad = style.pad
    x1 = W - style.margin
    x0 = x1 - tw - 2 * pad
    y0 = style.margin
    y1 = y0 + th + base + 2 * pad
    roi = frame[y0:y1, x0:x1]
    roi[:] = cv2.convertScaleAbs(roi, alpha=0.4, beta=0)
    cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), GREY_TEXT, 1, cv2.LINE_AA)
    _put(frame, BADGE_TEXT, (x0 + pad, y0 + pad + th), scale, style.badge_thick, WHITE, outline=False)


def _wrap(tokens: list[tuple[str, tuple[int, int, int], float | None, float | None]],
          scale: float, thick: int, max_w: int) -> list[list[_Token]]:
    """Greedy word-wrap. Returns the lines as token lists with x offsets set."""
    space_w, _ = _measure(" ", scale, thick)
    lines: list[list[_Token]] = [[]]
    x = 0
    for text, color, start, end in tokens:
        tok = _Token(text, color, start, end, width=_measure(text, scale, thick)[0])
        if lines[-1] and x + space_w + tok.width > max_w:
            lines.append([])
            x = 0
        if lines[-1]:
            x += space_w
        tok.x = x
        tok.line = len(lines) - 1
        lines[-1].append(tok)
        x += tok.width
    return lines


def _build_caption(track_id: int, track_color: tuple[int, int, int], utt: Utterance,
                   style: _Style) -> _Caption:
    """Lay out ``FACE id:`` plus the utterance words in at most two lines."""
    head = (f"FACE {track_id}:", track_color, None, None)
    words = []
    for w in utt.words:
        text = w.text + ("?" if Confidence(w.tier) == Confidence.LOW else "")
        color = NEON_GREEN if w.boosted else WORD_COLOR[Confidence(w.tier)]
        words.append((text, color, w.start, w.end))

    # Shrink a little to fit two lines, but never below 75% of the base size.
    scale = style.cap_scale
    floor = max(0.45, style.cap_scale * 0.75)
    lines = _wrap([head] + words, scale, style.cap_thick, style.inner_w)
    while len(lines) > MAX_CAPTION_LINES and scale > floor:
        scale = max(floor, scale * 0.9)
        lines = _wrap([head] + words, scale, style.cap_thick, style.inner_w)
    if len(lines) > MAX_CAPTION_LINES:
        # Still too long: keep the latest words and mark the cut with "...".
        for k in range(1, len(words) + 1):
            ellipsis = ("...", GREY_TEXT, None, None)
            lines = _wrap([head, ellipsis] + words[k:], scale, style.cap_thick, style.inner_w)
            if len(lines) <= MAX_CAPTION_LINES:
                break

    _, text_h = _measure("Ag", scale, style.cap_thick)
    tokens = [tok for line in lines for tok in line]
    return _Caption(tokens=tokens, lines=len(lines), scale=scale, thick=style.cap_thick,
                    line_h=int(text_h * 1.55), text_h=text_h)


def _draw_caption(frame: np.ndarray, cap: _Caption, track_color: tuple[int, int, int],
                  t: float, style: _Style, lift: int = 0) -> int:
    """Draw the caption panel ``lift`` px above the default bottom position and return its height.
    Words appear as they are spoken; the current word is underlined."""
    H, W = frame.shape[:2]
    panel_h = cap.lines * cap.line_h + 2 * style.pad
    x0 = (W - style.panel_w) // 2
    x1 = x0 + style.panel_w
    y1 = H - style.bottom - lift
    y0 = y1 - panel_h
    if y0 < 0:
        return panel_h
    roi = frame[y0:y1, x0:x1]
    roi[:] = cv2.convertScaleAbs(roi, alpha=0.35, beta=0)
    cv2.line(frame, (x0, y0), (x1 - 1, y0), track_color, max(2, cap.thick), cv2.LINE_AA)

    for tok in cap.tokens:
        if tok.start is not None and t < tok.start:
            continue
        ox = x0 + style.pad + tok.x
        oy = y0 + style.pad + tok.line * cap.line_h + cap.text_h
        _put(frame, tok.text, (ox, oy), cap.scale, cap.thick, tok.color)
        if tok.start is not None and tok.end is not None and tok.start <= t < tok.end:
            uy = oy + max(3, cap.thick * 2)
            cv2.line(frame, (ox, uy), (ox + tok.width, uy), WHITE, max(2, cap.thick), cv2.LINE_AA)
    return panel_h


_TIER_RANK = {Readability.READABLE: 0, Readability.MARGINAL: 1, Readability.UNREADABLE: 2}
MAX_CAPTIONS = 2   # simultaneous speakers shown at once


def _active_utterances(t: float, tracks: list[TrackResult]) -> list[tuple[TrackResult, Utterance]]:
    """Decoded utterances spanning ``t``, most trustworthy first (readability tier, then
    confidence), at most MAX_CAPTIONS. A clear speaker is never hidden behind a guess."""
    found: list[tuple[TrackResult, Utterance]] = []
    for tr in tracks:
        if tr.skipped:
            continue
        for utt in tr.utterances:
            if utt.start <= t <= utt.end:
                found.append((tr, utt))
    found.sort(key=lambda p: (_TIER_RANK[Readability(p[0].quality.tier)], -p[1].confidence))
    return found[:MAX_CAPTIONS]


def _normalize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Return a 3-channel uint8 BGR frame of the expected size."""
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.shape[1] != width or frame.shape[0] != height:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(frame)


def _ffmpeg_command(out_path: str, width: int, height: int, job: Job,
                    audio_src: str | None) -> list[str]:
    """Build the ffmpeg command: raw BGR on stdin, H.264 out, optional trimmed audio."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise LipReadError("ffmpeg was not found on PATH; it is needed to write the annotated video")
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(FPS),
           "-i", "pipe:0"]
    if audio_src:
        cmd += ["-ss", f"{job.start:.3f}"]
        if job.end is not None:
            cmd += ["-t", f"{max(0.0, job.end - job.start):.3f}"]
        cmd += ["-i", audio_src]
    cmd += ["-map", "0:v"]
    if audio_src:
        cmd += ["-map", "1:a?", "-c:a", "aac", "-b:a", "128k", "-shortest"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
            "-r", str(FPS), "-movflags", "+faststart", out_path]
    return cmd


def render_video(result: Result, frames: Iterable[np.ndarray], tracks_by_id: dict[int, FaceTrack],
                 out_path: str, audio_src: str | None = None) -> str:
    """Write the annotated MP4 and return ``out_path``.

    ``frames`` yields BGR frames at 25 fps starting at ``result.job.start``; frame ``i`` is at
    ``job.start + i / 25`` seconds. Only tracks present in both ``tracks_by_id`` and
    ``result.tracks`` are drawn. Boxes come from the track detections, so the frames must be the
    same size as the frames the detector saw. If ``audio_src`` is given, its audio is trimmed to
    the job's window and muxed in.
    """
    job = result.job
    iterator: Iterator[np.ndarray] = iter(frames)
    first = next(iterator, None)
    if first is None:
        raise LipReadError("no frames to render")
    if first.ndim == 2:
        first = cv2.cvtColor(first, cv2.COLOR_GRAY2BGR)
    height, width = first.shape[:2]
    # yuv420p needs even dimensions; crop one pixel if needed.
    out_w, out_h = width - width % 2, height - height % 2
    style = _Style.for_size(width, height)

    tr_by_id = {tr.track_id: tr for tr in result.tracks}
    drawn = {tid: track for tid, track in tracks_by_id.items() if tid in tr_by_id}
    det_maps = {tid: {d.frame: d for d in track.detections} for tid, track in drawn.items()}
    captions: dict[int, _Caption] = {}

    parent = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(parent, exist_ok=True)
    cmd = _ffmpeg_command(out_path, out_w, out_h, job, audio_src)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    count = 0
    warned_size = False
    try:
        assert proc.stdin is not None
        for i, frame in enumerate(itertools.chain([first], iterator)):
            if frame.shape[:2] != (height, width) and not warned_size:
                log.warning("frame size changed mid-stream; resizing to %dx%d", width, height)
                warned_size = True
            frame = _normalize_frame(frame, width, height)
            t = job.start + i / FPS

            for tid, det_map in det_maps.items():
                det = next((det_map[i + k] for k in (0, -1, 1, -2, 2) if i + k in det_map), None)
                if det is None:   # detection may run on every Nth frame: use the nearest one
                    continue
                tier = Readability(tr_by_id[tid].quality.tier)
                color = TIER_COLOR[tier]
                _draw_face(frame, det, color, f"FACE {tid} | {tier.value.upper()}", style)

            lift = 0
            for tr, utt in _active_utterances(t, result.tracks):
                cap = captions.get(id(utt))
                if cap is None:
                    cap = _build_caption(tr.track_id, TIER_COLOR[Readability(tr.quality.tier)], utt, style)
                    captions[id(utt)] = cap
                lift += _draw_caption(frame, cap, TIER_COLOR[Readability(tr.quality.tier)], t, style,
                                      lift) + style.pad

            _draw_badge(frame, style)

            out = frame[:out_h, :out_w] if (out_w, out_h) != (width, height) else frame
            try:
                proc.stdin.write(np.ascontiguousarray(out).tobytes())
            except BrokenPipeError:
                break
            count += 1
        try:
            proc.stdin.close()
        except BrokenPipeError:
            pass  # ffmpeg exited early; the exit code check below reports it
        stderr = proc.stderr.read() if proc.stderr else b""
        rc = proc.wait()
    except BaseException:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        raise

    if rc != 0:
        msg = stderr.decode("utf-8", "replace").strip()[-2000:]
        raise LipReadError(f"ffmpeg failed while writing {out_path} (exit {rc}): {msg}")
    log.info("rendered %d frames to %s", count, out_path)
    return out_path
