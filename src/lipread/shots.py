"""Shot-cut detection on the 25 fps frame stream.

A cut is declared where the frame's HSV histogram stops correlating with the
previous frame's. Histograms are computed on a 160 px wide copy, so the cost is
small next to decoding.

Limitation: only hue and saturation are compared. Cuts between two
low-saturation (grey or black-and-white) shots are not seen.
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("lipread")

_WORK_WIDTH = 160
_H_BINS = 32
_S_BINS = 32
_V_BINS = 32


def _hsv_hist(frame_bgr: np.ndarray) -> np.ndarray:
    """Normalised hue x saturation histogram and value (brightness) histogram of a downscaled frame."""
    h, w = frame_bgr.shape[:2]
    small_h = max(1, round(h * _WORK_WIDTH / w))
    small = cv2.resize(frame_bgr, (_WORK_WIDTH, small_h), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [_H_BINS, _S_BINS], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
    # brightness histogram too, so cuts between grey / dark shots are caught
    val = cv2.calcHist([hsv], [2], None, [_V_BINS], [0, 256])
    cv2.normalize(val, val, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
    return hist, val


class ShotDetector:
    """Feed frames in order with :meth:`update`, then ask for :meth:`shots`.

    ``threshold`` is the cut score: a frame starts a new shot when
    ``1 - correlation(previous, current) > threshold``.
    ``min_len`` is the shortest allowed shot in frames.
    """

    def __init__(self, threshold: float = 0.45, min_len: int = 10) -> None:
        if min_len < 1:
            raise ValueError("min_len must be at least 1")
        self.threshold = float(threshold)
        self.min_len = int(min_len)
        self._prev: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._count = 0
        self._cuts: list[int] = []

    @property
    def cuts(self) -> list[int]:
        """Frame indices where a new shot starts (before merging short shots)."""
        return list(self._cuts)

    def update(self, frame_bgr: np.ndarray) -> None:
        """Consume the next frame (BGR, uint8)."""
        hist = _hsv_hist(frame_bgr)
        idx = self._count
        if self._prev is not None:
            corr_hs = float(cv2.compareHist(self._prev[0], hist[0], cv2.HISTCMP_CORREL))
            corr_v = float(cv2.compareHist(self._prev[1], hist[1], cv2.HISTCMP_CORREL))
            score = 1.0 - min(corr_hs, corr_v)
            if score > self.threshold:
                self._cuts.append(idx)
                log.debug("shot cut at frame %d (score %.3f)", idx, score)
        self._prev = hist
        self._count += 1

    def shots(self, total_frames: int) -> list[tuple[int, int]]:
        """Inclusive ``(start, end)`` ranges covering ``0..total_frames-1`` with no gaps.

        A shot shorter than ``min_len`` is merged into the previous shot. The first
        shot has no predecessor, so a short first shot is merged into the next one.
        """
        if total_frames <= 0:
            return []
        starts = sorted({0, *(c for c in self._cuts if 0 < c < total_frames)})
        bounds: list[list[int]] = []
        for i, start in enumerate(starts):
            end = starts[i + 1] - 1 if i + 1 < len(starts) else total_frames - 1
            bounds.append([start, end])

        if len(bounds) > 1 and bounds[0][1] - bounds[0][0] + 1 < self.min_len:
            bounds[1][0] = bounds[0][0]
            del bounds[0]

        merged: list[list[int]] = []
        for start, end in bounds:
            if merged and end - start + 1 < self.min_len:
                merged[-1][1] = end
            else:
                merged.append([start, end])
        return [(s, e) for s, e in merged]
