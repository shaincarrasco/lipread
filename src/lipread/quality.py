"""Readability gate: decide whether a face track is worth decoding.

Most faces in sports crowds are too small, turned away or too blurry for lip
reading. Rather than guess words from them, every face track gets one of three
tiers with plain-English reasons:

* READABLE   - all checks pass; decoding is expected to be useful.
* MARGINAL   - usable but weaker (small face, partial turn, blur, little motion).
* UNREADABLE - a hard limit is hit (face too small, turned away, or not tracked
               for enough frames). Decoding would mostly invent words.

The thresholds below are starting points. They are module constants so they can
be retuned from measured studies without touching the logic.
"""
from __future__ import annotations

import cv2
import numpy as np

from .types import FaceTrack, QualityReport, Readability

# --- Tunable thresholds -------------------------------------------------------
FACE_PX_READABLE = 56      # face box height (px); below this the tier drops to MARGINAL
# (face-size study 2026-11-01: WER roughly flat from 200 px down to ~40 px, worse at 28, collapses at 20;
#  thresholds set conservatively because real broadcast footage is blurrier/compressed than the study)
FACE_PX_MARGINAL = 32      # face box height (px); below this the tier is UNREADABLE
YAW_READABLE = 30.0        # absolute yaw (deg); above this the tier drops to MARGINAL
YAW_MARGINAL = 50.0        # absolute yaw (deg); above this the tier is UNREADABLE
SHARP_MARGINAL = 15.0      # median Laplacian variance of the mouth crop; below = blurry
COVERAGE_MIN = 0.5         # fraction of track frames with a real detection; below = UNREADABLE
MOTION_MIN = 0.01          # mouth-motion energy (see motion_energy); below = barely moving

_MOTION_BORDER = 8         # px ignored at the crop's sides and bottom in motion_energy


def estimate_yaw(kp4: np.ndarray) -> float:
    """Rough absolute head yaw in degrees from the 4 alignment keypoints.

    ``kp4`` is (4, 2): image-left eye, image-right eye, nose tip, mouth centre.
    The nose's horizontal offset from the eye midpoint, divided by half the
    inter-eye distance, is mapped through arcsin: a centred nose gives 0 degrees
    and a nose on top of one eye gives 90 degrees. This is a rough estimate. It
    ignores head roll, pitch and perspective, and is only meant to separate
    frontal faces from strongly turned ones.

    Returns 90.0 when the eyes coincide (degenerate input, treated as turned away).
    """
    kp = np.asarray(kp4, dtype=np.float64).reshape(4, 2)
    left_eye, right_eye, nose = kp[0], kp[1], kp[2]
    half_iod = 0.5 * float(np.linalg.norm(right_eye - left_eye))
    if half_iod < 1e-6:
        return 90.0
    eye_mid_x = 0.5 * (left_eye[0] + right_eye[0])
    r = (nose[0] - eye_mid_x) / half_iod
    return float(abs(np.degrees(np.arcsin(np.clip(r, -1.0, 1.0)))))


def motion_energy(crops: np.ndarray) -> float:
    """Visual voice activity over a track of mouth crops, roughly 0..0.05 in practice.

    ``crops`` is uint8 (T, H, W), normally (T, 88, 88). Each frame is first
    registered to the previous one by integer translation (phase correlation),
    which cancels whole-crop camera shake. The mean absolute difference of the
    registered frames is then taken over the mouth region (the lower two thirds
    of the crop, minus an 8 px border so registration edge effects do not count).
    The per-pair values are divided by 255 and summarised with the 75th
    percentile over time, so a few noisy frames do not dominate.

    A still face gives 0. A still face with camera shake also gives about 0.
    Talking gives a clearly higher value. Returns 0.0 for fewer than two frames.

    This replaces an earlier idea of subtracting the top sixth of the crop as a
    shake reference. In real talking crops that strip moves nearly as much as the
    mouth, so the subtraction removed most of the signal.
    """
    crops = np.asarray(crops)
    if crops.ndim != 3 or crops.shape[0] < 2:
        return 0.0
    _, h, w = crops.shape
    b = _MOTION_BORDER
    r0, r1 = h // 3, h - b
    c0, c1 = b, w - b
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    values = np.empty(crops.shape[0] - 1, dtype=np.float32)
    prev = crops[0].astype(np.float32)
    for t in range(1, crops.shape[0]):
        cur = crops[t].astype(np.float32)
        (dx, dy), _ = cv2.phaseCorrelate(prev, cur, win)
        shift = np.array([[1.0, 0.0, -round(dx)], [0.0, 1.0, -round(dy)]], dtype=np.float32)
        aligned = cv2.warpAffine(cur, shift, (w, h), flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_REPLICATE)
        values[t - 1] = np.abs(aligned[r0:r1, c0:c1] - prev[r0:r1, c0:c1]).mean() / 255.0
        prev = aligned
    return float(np.percentile(values, 75))


def sharpness(crops: np.ndarray) -> float:
    """Median over frames of the Laplacian variance of each mouth crop (higher = sharper)."""
    crops = np.asarray(crops)
    if crops.ndim != 3 or crops.shape[0] == 0:
        return 0.0
    values = [float(cv2.Laplacian(np.ascontiguousarray(c), cv2.CV_64F).var()) for c in crops]
    return float(np.median(values))


def _kp4_from_points5(points5: np.ndarray) -> np.ndarray:
    """Build the 4 alignment points from the 5-point landmarks (mouth centre = mean of corners)."""
    p = np.asarray(points5, dtype=np.float64)
    return np.stack([p[0], p[1], p[2], 0.5 * (p[3] + p[4])])


def _track_yaw(track: FaceTrack, keypoints: np.ndarray) -> float | None:
    """Median absolute yaw over the track, or None if no finite pose could be estimated.

    Uses YuNet's ``points5`` (eye centres + nose) first: measured on 2026-11-01, MediaPipe
    BlazeFace's nose keypoint inflates yaw badly (e.g. 81 deg vs 43 deg on the same face,
    55 deg on a frontal 28 px face), while ``keypoints4`` stays the right choice for
    alignment. Falls back to ``keypoints4``, then to the per-frame alignment ``keypoints``.
    """
    yaws: list[float] = []
    for d in track.detections:
        kp = _kp4_from_points5(d.points5) if d.points5 is not None else d.keypoints4
        if np.all(np.isfinite(kp)):
            yaws.append(estimate_yaw(kp))
    if not yaws and keypoints is not None:
        for kp in np.asarray(keypoints, dtype=np.float64).reshape(-1, 4, 2):
            if np.all(np.isfinite(kp)):
                yaws.append(estimate_yaw(kp))
    return float(np.median(yaws)) if yaws else None


def assess(track: FaceTrack, crops: np.ndarray, keypoints: np.ndarray) -> QualityReport:
    """Score one face track and assign its readability tier.

    ``crops`` are the track's uint8 mouth crops (T, 88, 88). ``keypoints`` are the
    interpolated alignment points (T, 4, 2) and are used only as a last-resort
    source for head pose.

    Tier rules. A failed hard check makes the track UNREADABLE. Otherwise a failed
    soft check makes it MARGINAL. Otherwise it is READABLE. Every failed check adds
    a short reason. When the face is too small or turned too far, only the more
    severe of the two face/yaw checks is reported.
    """
    if not track.detections:
        return QualityReport(
            tier=Readability.UNREADABLE, face_px=0.0, mouth_px=0.0, yaw_deg=0.0,
            sharpness=0.0, coverage=0.0, motion=0.0, reasons=["no face detected"],
        )

    face_px = track.median_face_height()
    mouth_px = float(np.median([
        np.linalg.norm(np.asarray(d.points5[3], dtype=np.float64) - np.asarray(d.points5[4], dtype=np.float64))
        for d in track.detections
    ]))
    yaw = _track_yaw(track, keypoints)
    frames = [d.frame for d in track.detections]
    stride = int(np.min(np.diff(frames))) if len(frames) > 1 else 1   # pipeline may detect every Nth frame
    expected = (track.num_frames - 1) // max(stride, 1) + 1
    coverage = min(1.0, len(track.detections) / expected)
    motion = motion_energy(crops)
    sharp = sharpness(crops)

    hard: list[str] = []
    soft: list[str] = []

    if face_px < FACE_PX_MARGINAL:
        hard.append(f"face only {face_px:.0f} px tall (need {FACE_PX_MARGINAL}+)")
    elif face_px < FACE_PX_READABLE:
        soft.append(f"face {face_px:.0f} px tall (best results need {FACE_PX_READABLE}+)")

    if yaw is None:
        hard.append("head pose could not be estimated")
    elif yaw > YAW_MARGINAL:
        hard.append(f"turned ~{yaw:.0f}° away from camera")
    elif yaw > YAW_READABLE:
        soft.append(f"turned ~{yaw:.0f}° from camera (best results under {YAW_READABLE:.0f}°)")

    if coverage < COVERAGE_MIN:
        hard.append(f"face found in only {coverage * 100:.0f}% of frames (need {COVERAGE_MIN * 100:.0f}%+)")

    if sharp < SHARP_MARGINAL:
        soft.append("blurry mouth region")
    if motion < MOTION_MIN:
        soft.append("mouth barely moving (may not be speaking)")

    if hard:
        tier = Readability.UNREADABLE
    elif soft:
        tier = Readability.MARGINAL
    else:
        tier = Readability.READABLE

    return QualityReport(
        tier=tier,
        face_px=float(face_px),
        mouth_px=mouth_px,
        yaw_deg=float(yaw) if yaw is not None else 90.0,
        sharpness=sharp,
        coverage=float(coverage),
        motion=motion,
        reasons=hard + soft,
    )
