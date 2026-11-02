"""Pipeline orchestration: video in, transcripts and output files out.

Stages (each timed and reported through ``progress``):

1. ``fetch`` / ``probe``: download or locate the source, read its metadata.
2. ``detect`` (pass 1, streaming): shot cuts, face detection, keypoint refinement.
3. ``shots`` and ``track``: shot list, face tracks, drop tracks shorter than ``min_track_frames``.
4. ``crops`` (pass 2, streaming): per-track mouth crops, aligned with the same
   smoothing as ``align.mouth_crops``, plus thumbnails. No full-resolution frame is kept.
5. ``quality``: readability tier per track.
6. ``infer`` / ``decode``: engine log-probs (cached per track) and CTC decoding of the
   selected tracks.
7. ``export`` and ``render`` (pass 3, streaming): JSON/SRT/VTT/TXT and the annotated MP4.

Stage modules are imported inside ``run`` so that the CLI and web server can start
without every builder's module being present.
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional

import cv2
import numpy as np

from . import align, models
from .engine import Engine
from .errors import LipReadError
from .types import (CROP, FPS, Detection, FaceTrack, Job, ProgressFn, QualityReport, Readability,
                    Result, TrackResult, VideoInfo)

log = logging.getLogger("lipread")

MAX_FACES_PER_FRAME = 20
THUMB_PX = 112
STRIP_CROPS = 8
STRIP_GAP = 4
_PROGRESS_EVERY = 25
_FACE_MODES = ("none", "all", "readable", "largest")


# --------------------------------------------------------------------------- helpers

# Face detection is the slowest stage (~50 ms/frame); detecting every 2nd frame halves it.
# Keypoints on skipped frames are linearly interpolated and then smoothed over +-6 frames anyway.
DETECT_STRIDE = max(1, int(os.environ.get("LIPREAD_DETECT_STRIDE", "2")))


def parse_faces(spec: Optional[str]) -> str | list[int]:
    """Validate ``Job.faces``. Returns a mode name ("none", "all", "readable", "largest") or a list of track ids."""
    s = (spec or "readable").strip().lower()
    if s in _FACE_MODES:
        return s
    try:
        ids = [int(x) for x in s.split(",") if x.strip()]
    except ValueError:
        raise LipReadError(f"invalid faces selection {spec!r}: use all, readable, largest, none or ids like 1,3") from None
    if not ids:
        raise LipReadError(f"invalid faces selection {spec!r}: no track ids given")
    return ids


def select_track_ids(cands: list[tuple[int, Readability, float]], faces: str | list[int],
                     force_unreadable: bool) -> set[int]:
    """Choose which tracks to decode.

    cands: (track_id, tier, median face height in px) for every kept track.
    UNREADABLE tracks are only eligible when ``force_unreadable`` is set.
    """
    if faces == "none":
        return set()
    allowed = [c for c in cands if c[1] != Readability.UNREADABLE or force_unreadable]
    if faces == "all":
        return {c[0] for c in allowed}
    if faces == "readable":
        return {c[0] for c in allowed if c[1] in (Readability.READABLE, Readability.MARGINAL)}
    if faces == "largest":
        return {max(allowed, key=lambda c: c[2])[0]} if allowed else set()
    known = {c[0]: c for c in cands}
    chosen: set[int] = set()
    for tid in faces:  # explicit ids
        if tid not in known:
            log.warning("no track with id %d; available: %s", tid, sorted(known) or "none")
            continue
        if known[tid][1] == Readability.UNREADABLE and not force_unreadable:
            log.warning("track %d is UNREADABLE; not decoded (pass force_unreadable to decode it anyway)", tid)
            continue
        chosen.add(tid)
    return chosen


def default_engine(accept_license: bool = False) -> Engine:
    """The default lip-reading engine: Auto-AVSR int8 ONNX, downloaded on first use."""
    from .engine import AutoAvsrOnnx

    return AutoAvsrOnnx(models.ensure("vsr", accept_license=accept_license))


@dataclass
class _Plan:
    """Everything needed to cut one track's mouth crops in pass 2."""
    track: FaceTrack
    kp: np.ndarray            # (T,4,2) interpolated alignment points
    Ms: np.ndarray            # (T,2,3) per-frame warp, smoothed as in align.mouth_crops
    origins: np.ndarray       # (T,2) int (x0, y0) of the 96 px patch on the warped canvas
    crops: np.ndarray         # (T,88,88) uint8, filled during pass 2
    thumb_frame: int
    thumb_box: tuple[float, float, float, float]
    thumb: Optional[np.ndarray] = None


def _track_keypoints(track: FaceTrack) -> Optional[np.ndarray]:
    """(T,4,2) keypoints over the track's full frame span, linearly interpolated over gaps."""
    by_frame = {d.frame: d.keypoints4 for d in track.detections if d.keypoints4 is not None}
    seq = [by_frame.get(f) for f in range(track.start_frame, track.end_frame + 1)]
    return align.interpolate_keypoints(seq)


def _affines(kp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame warp and patch origin. Mirrors the smoothing in align.mouth_crops exactly.

    Keeping this here lets pass 2 stream frames one at a time while the smoothing still
    sees the whole keypoint track. align.py is not modified.
    """
    T = len(kp)
    Ms = np.empty((T, 2, 3), np.float64)
    origins = np.empty((T, 2), np.int64)
    ref = align.STABLE_REFERENCE.astype(np.float32)
    prev: Optional[np.ndarray] = None
    limit = align.TARGET - align.PATCH
    for i in range(T):
        m = min(align.SMOOTH_HALF, i, T - 1 - i)
        sm = kp[i - m:i + m + 1].mean(0)
        sm = sm + (kp[i].mean(0) - sm.mean(0))
        M = cv2.estimateAffinePartial2D(sm.astype(np.float32), ref, method=cv2.LMEDS)[0]
        if M is None:  # degenerate points: reuse the previous transform
            M = prev if prev is not None else np.array([[1, 0, 0], [0, 1, 0]], np.float64)
        prev = M
        cx, cy = sm[3] @ M[:, :2].T + M[:, 2]
        x0 = int(round(np.clip(cx - align.PATCH / 2, 0, limit)))
        y0 = int(round(np.clip(cy - align.PATCH / 2, 0, limit)))
        Ms[i] = M
        origins[i] = (x0, y0)
    return Ms, origins


def _crop_one(gray: np.ndarray, M: np.ndarray, x0: int, y0: int) -> np.ndarray:
    """One 88x88 mouth crop from a grayscale frame, identical to one row of align.mouth_crops."""
    warped = cv2.warpAffine(gray, M, (align.TARGET, align.TARGET), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    patch = warped[y0:y0 + align.PATCH, x0:x0 + align.PATCH]
    off = (align.PATCH - CROP) // 2
    return patch[off:off + CROP, off:off + CROP]


def _thumbnail(frame_bgr: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    """Square face thumbnail around a detection box, padded with black where it leaves the frame."""
    x, y, w, h = box
    side = max(int(round(max(w, h) * 1.4)), 1)
    x0 = int(round(x + w / 2 - side / 2))
    y0 = int(round(y + h / 2 - side / 2))
    H, W = frame_bgr.shape[:2]
    canvas = np.zeros((side, side, 3), np.uint8)
    xs, ys = max(x0, 0), max(y0, 0)
    xe, ye = min(x0 + side, W), min(y0 + side, H)
    if xe > xs and ye > ys:
        canvas[ys - y0:ye - y0, xs - x0:xe - x0] = frame_bgr[ys:ye, xs:xe]
    return cv2.resize(canvas, (THUMB_PX, THUMB_PX), interpolation=cv2.INTER_AREA)


def _strip(crops: np.ndarray, n: int = STRIP_CROPS) -> np.ndarray:
    """Evenly spaced mouth crops side by side, separated by black gaps."""
    idx = np.linspace(0, len(crops) - 1, min(n, len(crops))).round().astype(int)
    gap = np.zeros((CROP, STRIP_GAP), np.uint8)
    tiles: list[np.ndarray] = []
    for i in idx:
        tiles.append(crops[i])
        tiles.append(gap)
    return np.hstack(tiles[:-1])


def _logprobs(engine: Engine, crops: np.ndarray, track_id: int, cache_dir: str) -> np.ndarray:
    """Engine log-probs for a track, cached on disk. The cache key covers the crop bytes and engine name."""
    name = str(engine.info.get("name", "engine"))
    digest = hashlib.sha1(name.encode() + crops.tobytes()).hexdigest()[:16]
    path = os.path.join(cache_dir, f"track_{track_id}_{digest}.npy")
    if os.path.exists(path):
        try:
            lp = np.load(path)
            if lp.ndim == 2 and lp.shape == (len(crops), len(engine.vocab)):
                log.debug("track %d: using cached log-probs", track_id)
                return lp
        except (OSError, ValueError):
            log.warning("cached log-probs for track %d are unreadable; recomputing", track_id)
    lp = np.asarray(engine.logprobs(crops), dtype=np.float32)
    tmp = path + ".part.npy"
    np.save(tmp, lp)
    os.replace(tmp, path)
    return lp


def _report(progress: Optional[ProgressFn], stage: str, frac: float, msg: str = "") -> None:
    if progress is not None:
        progress(stage, float(min(max(frac, 0.0), 1.0)), msg)


class _Timer:
    """Accumulates wall-clock seconds per stage."""

    def __init__(self, progress: Optional[ProgressFn] = None) -> None:
        self.t: dict[str, float] = {}
        self._t0 = time.perf_counter()
        self._progress = progress
        self._announced: set[str] = set()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if name not in self._announced:     # every stage shows up in the UI / CLI progress
            self._announced.add(name)
            _report(self._progress, name, 0.0, name)
        start = time.perf_counter()
        try:
            yield
        finally:
            self.t[name] = self.t.get(name, 0.0) + (time.perf_counter() - start)
            log.debug("stage %-8s %6.1fs", name, self.t[name])

    def finish(self) -> dict[str, float]:
        self.t["total"] = time.perf_counter() - self._t0
        return self.t


def _area(d: Detection) -> float:
    return float(d.box[2] * d.box[3])


# --------------------------------------------------------------------------- run

def run(job: Job, engine: Engine | None = None, progress: ProgressFn | None = None) -> Result:
    """Run the full pipeline for ``job``. ``engine=None`` uses the default Auto-AVSR engine."""
    from . import decode, export, ingest, quality, render, tracker
    from . import shots as shots_mod
    from .faces import FaceDetector

    faces_sel = parse_faces(job.faces)
    decoding = faces_sel != "none"
    if job.start < 0:
        raise LipReadError("start must be >= 0")
    if job.end is not None and job.end <= job.start:
        raise LipReadError("end must be after start")
    if decoding and engine is None:
        engine = default_engine()  # raises LipReadError early if the licence is not accepted

    out_dir = os.path.abspath(job.out_dir)
    for sub in ("source", "cache", "thumbs", "crops"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    timer = _Timer(progress)

    with timer.stage("models"):
        for name in ("yunet", "blazeface"):
            models.ensure(name)

    with timer.stage("fetch"):
        _report(progress, "fetch", 0.0, "fetching source")
        local = ingest.fetch(job.source, os.path.join(out_dir, "source"))
        _report(progress, "fetch", 1.0, "source ready")
    with timer.stage("probe"):
        info: VideoInfo = ingest.probe(local)
        log.info("video %dx%d @ %.2f fps, %.1f s, audio=%s", info.width, info.height,
                 info.src_fps, info.duration, info.has_audio)

    # ---- pass 1: shots + detection + refinement (streaming)
    total = max(ingest.count_frames(info, job.start, job.end), 1)
    shot_det = shots_mod.ShotDetector()
    detector = FaceDetector(models.models_dir())
    dets_per_frame: list[list[Detection]] = []
    with timer.stage("detect"):
        for f, frame in enumerate(ingest.iter_frames(info, start=job.start, end=job.end)):
            shot_det.update(frame)
            if f % DETECT_STRIDE:
                dets_per_frame.append([])     # skipped frame: keypoints are interpolated later
                continue
            dets = sorted(detector.detect(frame, f), key=_area, reverse=True)[:MAX_FACES_PER_FRAME]
            kept: list[Detection] = []
            for d in dets:
                kp = detector.refine(frame, d)
                if kp is None:
                    continue
                d.keypoints4 = kp
                kept.append(d)
            dets_per_frame.append(kept)
            if f % _PROGRESS_EVERY == 0:
                _report(progress, "detect", f / total, f"frame {f}/{total}")
        _report(progress, "detect", 1.0, f"{len(dets_per_frame)} frames")
    detector.close()  # release the MediaPipe graph before the heavier passes
    del detector
    n_frames = len(dets_per_frame)
    if n_frames == 0:
        raise LipReadError("no frames in the requested time range")

    with timer.stage("shots"):
        shot_list = shot_det.shots(n_frames)
    with timer.stage("track"):
        all_tracks = tracker.track_faces(dets_per_frame, shot_list)
        tracks = [t for t in all_tracks if t.num_frames >= job.min_track_frames]
        log.info("%d face tracks, %d kept (>= %d frames)", len(all_tracks), len(tracks), job.min_track_frames)
    del dets_per_frame  # pass 2 only needs the tracks

    # ---- pass 2: mouth crops + thumbnails (streaming)
    plans: dict[int, _Plan] = {}
    with timer.stage("crops"):
        for t in tracks:
            kp = _track_keypoints(t)
            if kp is None:
                log.warning("track %d has no usable keypoints; skipped", t.track_id)
                continue
            Ms, origins = _affines(kp)
            best = max(t.detections, key=lambda d: d.box[3])
            plans[t.track_id] = _Plan(
                track=t, kp=kp, Ms=Ms, origins=origins,
                crops=np.zeros((t.num_frames, CROP, CROP), np.uint8),
                thumb_frame=best.frame, thumb_box=best.box)
        active = list(plans.values())
        seen = 0
        for f, frame in enumerate(ingest.iter_frames(info, start=job.start, end=job.end)):
            seen = f + 1
            live = [p for p in active if p.track.start_frame <= f <= p.track.end_frame]
            if live:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                for p in live:
                    i = f - p.track.start_frame
                    x0, y0 = p.origins[i]
                    p.crops[i] = _crop_one(gray, p.Ms[i], int(x0), int(y0))
                    if f == p.thumb_frame:
                        p.thumb = _thumbnail(frame, p.thumb_box)
            if f % _PROGRESS_EVERY == 0:
                _report(progress, "crops", f / total, f"frame {f}/{total}, {len(live)} live tracks")
        if seen != n_frames:
            log.warning("pass 2 read %d frames, pass 1 read %d", seen, n_frames)
        _report(progress, "crops", 1.0, f"{len(plans)} tracks")

    # ---- quality gate
    with timer.stage("quality"):
        reports: dict[int, QualityReport] = {
            tid: quality.assess(p.track, p.crops, p.kp) for tid, p in plans.items()}

    # ---- choose tracks, infer, decode
    cands = [(tid, reports[tid].tier, p.track.median_face_height()) for tid, p in plans.items()]
    selected = select_track_ids(cands, faces_sel, job.force_unreadable) if decoding else set()
    log.info("selected %d of %d tracks for decoding", len(selected), len(plans))

    track_results: list[TrackResult] = []
    model_info = dict(engine.info) if (decoding and engine is not None) else {"name": "none", "license": "n/a"}
    with timer.stage("thumbs"):
        for tid in sorted(plans):
            p = plans[tid]
            thumb_path = os.path.join(out_dir, "thumbs", f"track_{tid}.jpg")
            strip_path = os.path.join(out_dir, "crops", f"track_{tid}.png")
            if p.thumb is not None:
                cv2.imwrite(thumb_path, p.thumb, [cv2.IMWRITE_JPEG_QUALITY, 90])
            cv2.imwrite(strip_path, _strip(p.crops))
            track_results.append(TrackResult(
                track_id=tid, shot_id=p.track.shot_id,
                start=job.start + p.track.start_frame / FPS,
                end=job.start + (p.track.end_frame + 1) / FPS,
                quality=reports[tid], utterances=[],
                thumbnail=thumb_path if p.thumb is not None else None,
                crops_preview=strip_path, skipped=tid not in selected))

    by_id = {r.track_id: r for r in track_results}
    for tid in sorted(selected):
        p = plans[tid]
        with timer.stage("infer"):
            lp = _logprobs(engine, p.crops, tid, os.path.join(out_dir, "cache"))
        with timer.stage("decode"):
            t0 = job.start + p.track.start_frame / FPS
            by_id[tid].utterances = decode.decode_track(lp, engine.vocab, engine.blank, t0, names=job.names)
        log.info("track %d (%s): %s", tid, reports[tid].tier.value,
                 " ".join(u.text for u in by_id[tid].utterances) or "-")

    result = Result(job=job, video=info, tracks=track_results, shots=shot_list,
                    model=model_info, timings=timer.t)

    with timer.stage("export"):
        result.outputs.update(export.write_all(result, out_dir))

    if job.render_video and decoding:
        with timer.stage("render"):
            tracks_by_id = {t.track_id: t for t in tracks}
            audio_src = info.path if info.has_audio else None
            frames = ingest.iter_frames(info, start=job.start, end=job.end)
            mp4 = render.render_video(result, frames, tracks_by_id,
                                      os.path.join(out_dir, "annotated.mp4"), audio_src=audio_src)
            result.outputs["mp4"] = mp4

    result.timings = timer.finish()
    _report(progress, "done", 1.0, "finished")
    log.info("done in %.1fs", result.timings["total"])
    return result
