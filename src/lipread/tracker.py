"""Greedy IoU tracker that links per-frame face detections into per-person tracks.

Tracks never cross a shot boundary: every shot starts with an empty set of tracks.
Within a shot, each frame's detections are matched to the live tracks by a greedy
assignment on cost. IoU matches are preferred. When IoU is low (a fast camera pan),
a detection may still match a track if its centre lies within half the track's box
height and its height is within a factor of two of the track's.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .types import Detection, FaceTrack

log = logging.getLogger("lipread")

CENTRE_DIST_FRAC = 0.5      # centre-distance fallback radius, as a fraction of box height
SIZE_RATIO_RANGE = (0.5, 2.0)


@dataclass
class _Live:
    """A track that is still open within the current shot."""
    track: FaceTrack
    box: np.ndarray          # (4,) x, y, w, h of the last attached detection
    missed: int = 0          # consecutive frames without a match


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in ``a`` (T,4) and ``b`` (D,4), both as x, y, w, h."""
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None, :]) - np.maximum(a[:, 0][:, None], b[:, 0][None, :]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None, :]) - np.maximum(a[:, 1][:, None], b[:, 1][None, :]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def _candidates(live: list[_Live], dets: list[Detection], iou_thresh: float) -> list[tuple[float, int, int]]:
    """All admissible (cost, track index, detection index) pairs. Lower cost is better.

    IoU matches cost ``1 - IoU`` (at most ``1 - iou_thresh``). Centre-distance matches
    cost ``1 + dist / height``, so they always rank after any IoU match.
    """
    if not live or not dets:
        return []
    tb = np.array([t.box for t in live], dtype=np.float64)
    db = np.array([d.box for d in dets], dtype=np.float64)
    iou = _iou_matrix(tb, db)

    t_centre = tb[:, :2] + tb[:, 2:] / 2
    d_centre = db[:, :2] + db[:, 2:] / 2
    dist = np.linalg.norm(t_centre[:, None, :] - d_centre[None, :, :], axis=2)
    t_h = tb[:, 3][:, None]
    ratio = db[:, 3][None, :] / np.maximum(t_h, 1e-9)

    iou_ok = iou >= iou_thresh
    lo, hi = SIZE_RATIO_RANGE
    dist_ok = (dist < CENTRE_DIST_FRAC * t_h) & (ratio >= lo) & (ratio <= hi)

    out: list[tuple[float, int, int]] = []
    for i, j in zip(*np.nonzero(iou_ok | dist_ok)):
        if iou_ok[i, j]:
            cost = 1.0 - float(iou[i, j])
        else:
            cost = 1.0 + float(dist[i, j] / max(t_h[i, 0], 1e-9))
        out.append((cost, int(i), int(j)))
    return out


def track_faces(
    dets_per_frame: list[list[Detection]],
    shots: list[tuple[int, int]],
    iou_thresh: float = 0.3,
    max_gap: int = 8,
) -> list[FaceTrack]:
    """Link detections into tracks, one set of tracks per shot.

    Args:
        dets_per_frame: ``dets_per_frame[i]`` holds the detections of frame ``i``.
            Frames beyond the end of the list count as empty.
        shots: ``(start_frame, end_frame)`` per shot, both inclusive.
        iou_thresh: minimum IoU for an IoU match.
        max_gap: a track is closed once it has gone more than this many frames
            without a match.

    Returns:
        Tracks sorted by ``(start_frame, track_id)``. ``track_id`` counts from 1 in
        creation order. ``shot_id`` is the index into ``shots``. Detections keep
        their real frame numbers, so gaps inside a track are allowed.
    """
    closed: list[FaceTrack] = []
    next_id = 1

    for shot_id, (start, end) in enumerate(shots):
        live: list[_Live] = []
        for f in range(start, end + 1):
            dets = dets_per_frame[f] if f < len(dets_per_frame) else []

            pairs = sorted(_candidates(live, dets, iou_thresh))
            used_tracks: set[int] = set()
            used_dets: set[int] = set()
            for _, i, j in pairs:
                if i in used_tracks or j in used_dets:
                    continue
                used_tracks.add(i)
                used_dets.add(j)
                trk = live[i]
                trk.track.detections.append(dets[j])
                trk.box = np.asarray(dets[j].box, dtype=np.float64)
                trk.missed = 0

            still_live: list[_Live] = []
            for i, trk in enumerate(live):
                if i not in used_tracks:
                    trk.missed += 1
                if trk.missed > max_gap:
                    closed.append(trk.track)
                else:
                    still_live.append(trk)
            live = still_live

            for j, det in enumerate(dets):
                if j in used_dets:
                    continue
                track = FaceTrack(track_id=next_id, shot_id=shot_id, detections=[det])
                next_id += 1
                live.append(_Live(track=track, box=np.asarray(det.box, dtype=np.float64)))

        closed.extend(trk.track for trk in live)

    closed.sort(key=lambda t: (t.start_frame, t.track_id))
    log.debug("tracker: %d tracks over %d shots", len(closed), len(shots))
    return closed
