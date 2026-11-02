"""Face detection and alignment-keypoint refinement for small faces in wide shots.

Two stages:

1. ``FaceDetector.detect`` runs OpenCV YuNet on the full frame. YuNet finds faces
   down to about 12 px, far below what MediaPipe can see.
2. ``FaceDetector.refine`` crops each face (1.8x its box), upsamples the crop to
   256x256 and runs MediaPipe BlazeFace (short range) on that patch. This matches
   the scale at which the Auto-AVSR alignment was trained. The four alignment points
   are mapped back to frame coordinates. If MediaPipe finds nothing plausible, the
   YuNet landmarks are used instead.
"""
from __future__ import annotations

import logging
import math
import os

os.environ.setdefault("GLOG_minloglevel", "2")      # quiet MediaPipe / LiteRT native INFO logs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from .errors import LipReadError
from .types import Detection

log = logging.getLogger("lipread")

YUNET_FILE = "face_detection_yunet_2023mar.onnx"
BLAZE_FILE = "blaze_face_short_range.tflite"
MAX_DETECT_WIDTH = 1920     # frames wider than this are detected on a downscaled copy
NMS_THRESHOLD = 0.3
TOP_K = 5000
CROP_SCALE = 1.8            # refinement crop side = CROP_SCALE * max(box w, h)
ALIGN_SIZE = 256            # upsampled crop side fed to MediaPipe
BLAZE_MIN_CONF = 0.5


def _sorted_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Order two points so the one with smaller x (image-left) comes first."""
    return (a, b) if a[0] <= b[0] else (b, a)


def _plausible(kp4: np.ndarray) -> bool:
    """Eyes above nose above mouth, in image coordinates (y grows downwards)."""
    eye_y = (kp4[0, 1] + kp4[1, 1]) / 2
    return bool(eye_y < kp4[2, 1] < kp4[3, 1])


class FaceDetector:
    """YuNet face finder plus MediaPipe keypoint refiner.

    Args:
        models_dir: directory holding ``face_detection_yunet_2023mar.onnx`` and
            ``blaze_face_short_range.tflite``.
        score_threshold: minimum YuNet face score.
        min_face_px: faces whose smaller box side is below this are dropped.
        max_faces: keep at most this many faces per frame (largest first).

    Coordinates: boxes and points are in the pixel grid of the frame passed in.
    Pass full-resolution frames to ``refine`` for the boxes ``detect`` returned on
    the same frame.
    """

    def __init__(
        self,
        models_dir: str,
        score_threshold: float = 0.5,
        min_face_px: int = 12,
        max_faces: int = 30,
    ) -> None:
        yunet_path = os.path.join(models_dir, YUNET_FILE)
        blaze_path = os.path.join(models_dir, BLAZE_FILE)
        for path in (yunet_path, blaze_path):
            if not os.path.isfile(path):
                raise LipReadError(f"face model not found: {path}")
        self.score_threshold = float(score_threshold)
        self.min_face_px = float(min_face_px)
        self.max_faces = int(max_faces)
        self._yunet_path = yunet_path
        self._yn: cv2.FaceDetectorYN | None = None
        self._yn_size: tuple[int, int] | None = None
        self._blaze = vision.FaceDetector.create_from_options(
            vision.FaceDetectorOptions(
                base_options=BaseOptions(model_asset_path=blaze_path),
                min_detection_confidence=BLAZE_MIN_CONF,
            )
        )

    def _yunet(self, w: int, h: int) -> cv2.FaceDetectorYN:
        """Return the YuNet detector configured for a (w, h) input, creating it once."""
        if self._yn is None:
            self._yn = cv2.FaceDetectorYN.create(
                self._yunet_path, "", (w, h), self.score_threshold, NMS_THRESHOLD, TOP_K
            )
            self._yn_size = (w, h)
        elif self._yn_size != (w, h):
            self._yn.setInputSize((w, h))
            self._yn_size = (w, h)
        return self._yn

    def close(self) -> None:
        """Release the MediaPipe graph explicitly (avoids noisy errors at interpreter exit)."""
        blaze, self._blaze = getattr(self, "_blaze", None), None
        if blaze is not None:
            try:
                blaze.close()
            except Exception:  # pragma: no cover - best effort
                pass

    def __enter__(self) -> "FaceDetector":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def detect(self, frame_bgr: np.ndarray, frame_idx: int) -> list[Detection]:
        """Find faces in one BGR frame.

        Returns detections sorted by box area (largest first), at most ``max_faces``.
        ``points5`` is ordered image-left eye, image-right eye, nose, image-left
        mouth corner, image-right mouth corner.
        """
        height, width = frame_bgr.shape[:2]
        img = frame_bgr
        dw, dh = width, height
        if width > MAX_DETECT_WIDTH:
            dw = MAX_DETECT_WIDTH
            dh = max(1, round(height * dw / width))
            img = cv2.resize(frame_bgr, (dw, dh), interpolation=cv2.INTER_AREA)
        sx, sy = width / dw, height / dh

        _, raw = self._yunet(dw, dh).detect(img)
        if raw is None:
            return []

        dets: list[Detection] = []
        scale = np.array([sx, sy])
        for row in raw:
            bx, by, bw, bh = float(row[0]) * sx, float(row[1]) * sy, float(row[2]) * sx, float(row[3]) * sy
            if min(bw, bh) < self.min_face_px:
                continue
            pts = row[4:14].reshape(5, 2).astype(np.float64) * scale
            # YuNet's eye and mouth order follows the subject's side, so sort by image x.
            pts[0], pts[1] = _sorted_pair(pts[0], pts[1])
            pts[3], pts[4] = _sorted_pair(pts[3], pts[4])
            dets.append(
                Detection(
                    frame=frame_idx,
                    box=(bx, by, bw, bh),
                    score=float(row[14]),
                    points5=pts,
                )
            )

        dets.sort(key=lambda d: d.box[2] * d.box[3], reverse=True)
        return dets[: self.max_faces]

    def refine(self, frame_bgr: np.ndarray, det: Detection) -> np.ndarray:
        """Return (4, 2) alignment points for one detection, in frame coordinates.

        Points are: image-left eye, image-right eye, nose tip, mouth centre.
        Also stores them in ``det.keypoints4``.
        """
        height, width = frame_bgr.shape[:2]
        x, y, bw, bh = det.box
        cx, cy = x + bw / 2, y + bh / 2
        side = max(bw, bh) * CROP_SCALE
        S = max(1, int(side))
        x0, y0 = math.floor(cx - side / 2), math.floor(cy - side / 2)

        # Crop the S x S square at (x0, y0); pixels outside the frame stay black.
        crop = np.zeros((S, S) + frame_bgr.shape[2:], frame_bgr.dtype)
        fx0, fy0 = max(x0, 0), max(y0, 0)
        fx1, fy1 = min(x0 + S, width), min(y0 + S, height)
        if fx1 > fx0 and fy1 > fy0:
            crop[fy0 - y0 : fy1 - y0, fx0 - x0 : fx1 - x0] = frame_bgr[fy0:fy1, fx0:fx1]

        sc = ALIGN_SIZE / S
        up = cv2.resize(crop, (ALIGN_SIZE, ALIGN_SIZE), interpolation=cv2.INTER_CUBIC)
        offset = np.array([x0, y0], dtype=np.float64)

        H, W = frame_bgr.shape[:2]
        bx, by, bw, bh = det.box
        at_edge = bx < 1 or by < 1 or bx + bw > W - 1 or by + bh > H - 1
        # A face cut by the frame edge makes BlazeFace see black padding and misplace the
        # eyes (measured: frontal face read as 31-52 deg yaw). YuNet's points are safer there.
        kp4 = None if at_edge else self._mediapipe_points(up, sc, offset)
        if kp4 is None or not _plausible(kp4):
            kp4 = self._yunet_points(det)
            log.debug("frame %d: MediaPipe refinement unavailable, using YuNet points", det.frame)

        det.keypoints4 = kp4
        return kp4

    def _mediapipe_points(self, up_bgr: np.ndarray, sc: float, offset: np.ndarray) -> np.ndarray | None:
        """Run BlazeFace on the upsampled crop and map its 4 points to frame coordinates."""
        rgb = np.ascontiguousarray(cv2.cvtColor(up_bgr, cv2.COLOR_BGR2RGB))
        result = self._blaze.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        best = None
        best_dist = math.inf
        for d in result.detections:
            bb = d.bounding_box
            dist = math.hypot(bb.origin_x + bb.width / 2 - ALIGN_SIZE / 2,
                              bb.origin_y + bb.height / 2 - ALIGN_SIZE / 2)
            if dist < best_dist:
                best, best_dist = d, dist
        if best is None or len(best.keypoints) < 4:
            return None
        k = [(p.x * ALIGN_SIZE, p.y * ALIGN_SIZE) for p in best.keypoints[:4]]
        eyes = sorted(k[:2])
        pts = np.array([eyes[0], eyes[1], k[2], k[3]], dtype=np.float64)
        return pts / sc + offset

    @staticmethod
    def _yunet_points(det: Detection) -> np.ndarray:
        """Fallback alignment points from the YuNet landmarks (already in frame coordinates)."""
        p = det.points5
        eyes = sorted([p[0], p[1]], key=lambda q: q[0])
        mouth = (p[3] + p[4]) / 2
        return np.array([eyes[0], eyes[1], p[2], mouth], dtype=np.float64)
