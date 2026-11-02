"""Mouth-crop alignment, ported from Auto-AVSR's MediaPipe preprocessing.

Original: https://github.com/mpc001/auto_avsr  preparation/detectors/mediapipe/video_process.py
Copyright 2023 Imperial College London (Pingchuan Ma), Apache License 2.0.
Changes: works on a FaceTrack's keypoints, NaN-aware interpolation, no skimage dependency,
patch is resized instead of raising when it falls off the frame.

The model was trained on exactly this preprocessing, so do not change the
constants without re-running ``eval/lrs3_bench.py``.
"""
from __future__ import annotations

from importlib import resources

import cv2
import numpy as np

from .types import CROP

_MEAN_FACE = np.load(str(resources.files("lipread.assets").joinpath("20words_mean_face.npy")))
# right eye, left eye, nose tip, mouth centre of the 68-point mean face (subject's right = image left)
STABLE_REFERENCE = np.vstack([
    _MEAN_FACE[36:42].mean(0),
    _MEAN_FACE[42:48].mean(0),
    _MEAN_FACE[31:36].mean(0),
    _MEAN_FACE[48:68].mean(0),
])
TARGET = 256          # aligned face canvas
PATCH = 96            # mouth patch before centre crop
SMOOTH_HALF = 6       # +-6 frame landmark smoothing (window_margin=12 in the original)
MEAN, STD = 0.421, 0.165


def interpolate_keypoints(kps: list[np.ndarray | None]) -> np.ndarray | None:
    """Fill missing (None) frames by linear interpolation; edges are held. Returns (T,4,2) or None."""
    if not any(k is not None for k in kps):
        return None
    arr = np.array([k if k is not None else np.full((4, 2), np.nan) for k in kps], dtype=np.float64)
    t = np.arange(len(arr))
    for j in range(4):
        for c in range(2):
            v = arr[:, j, c]
            ok = ~np.isnan(v)
            arr[:, j, c] = np.interp(t, np.flatnonzero(ok), v[ok])
    return arr


def mouth_crops(frames_gray: np.ndarray | list[np.ndarray], keypoints: np.ndarray) -> np.ndarray:
    """Return uint8 mouth crops (T, 88, 88).

    frames_gray: T grayscale frames (H, W) uint8 (full frames at 25 fps).
    keypoints:   (T, 4, 2) alignment points (already interpolated, see interpolate_keypoints).
    """
    T = len(keypoints)
    out = np.empty((T, CROP, CROP), np.uint8)
    off = (PATCH - CROP) // 2
    for i in range(T):
        m = min(SMOOTH_HALF, i, T - 1 - i)
        sm = keypoints[i - m:i + m + 1].mean(0)
        sm = sm + (keypoints[i].mean(0) - sm.mean(0))
        M = cv2.estimateAffinePartial2D(sm.astype(np.float32), STABLE_REFERENCE.astype(np.float32),
                                        method=cv2.LMEDS)[0]
        if M is None:  # degenerate points: reuse previous transform or identity-ish crop
            M = prev if i > 0 else np.array([[1, 0, 0], [0, 1, 0]], np.float64)
        prev = M
        warped = cv2.warpAffine(frames_gray[i], M, (TARGET, TARGET), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        cx, cy = sm[3] @ M[:, :2].T + M[:, 2]
        y0 = int(round(np.clip(cy - PATCH / 2, 0, TARGET - PATCH)))
        x0 = int(round(np.clip(cx - PATCH / 2, 0, TARGET - PATCH)))
        patch = warped[y0:y0 + PATCH, x0:x0 + PATCH]
        out[i] = patch[off:off + CROP, off:off + CROP]
    return out


def normalize(crops: np.ndarray) -> np.ndarray:
    """uint8 (T,88,88) -> float32 (T,88,88) normalised as in training."""
    return ((crops.astype(np.float32) / 255.0) - MEAN) / STD
