"""Readability gate: yaw, motion, sharpness and the three tiers."""
from __future__ import annotations

import numpy as np
import pytest

from lipread import quality
from lipread.types import Detection, FaceTrack, Readability


def _kp4_for_yaw(deg: float, eye_gap: float = 40.0) -> np.ndarray:
    """Alignment points whose nose sits at the given yaw (from the eye midpoint)."""
    half = eye_gap / 2
    left = np.array([100.0 - half, 100.0])
    right = np.array([100.0 + half, 100.0])
    nose = np.array([100.0 + half * np.sin(np.radians(deg)), 130.0])
    mouth = np.array([100.0, 160.0])
    return np.stack([left, right, nose, mouth])


# --------------------------------------------------------------------------- yaw

@pytest.mark.parametrize("deg", [0.0, 15.0, 30.0, 60.0, 90.0])
def test_estimate_yaw_recovers_angle(deg):
    assert quality.estimate_yaw(_kp4_for_yaw(deg)) == pytest.approx(deg, abs=1e-6)


def test_estimate_yaw_is_symmetric_in_direction():
    assert quality.estimate_yaw(_kp4_for_yaw(-40.0)) == pytest.approx(40.0, abs=1e-6)


def test_estimate_yaw_degenerate_eyes_is_turned_away():
    kp = _kp4_for_yaw(0.0, eye_gap=0.0)
    assert quality.estimate_yaw(kp) == 90.0


# --------------------------------------------------------------------------- motion

def test_motion_energy_still_face_is_near_zero():
    rng = np.random.default_rng(0)
    still = rng.integers(0, 256, size=(88, 88), dtype=np.uint8)
    crops = np.repeat(still[None], 20, axis=0)
    assert quality.motion_energy(crops) == pytest.approx(0.0, abs=1e-6)


def test_motion_energy_moving_mouth_is_clearly_positive():
    rng = np.random.default_rng(1)
    crops = rng.integers(0, 256, size=(20, 88, 88), dtype=np.uint8)   # mouth region changes every frame
    assert quality.motion_energy(crops) > 0.1


def test_motion_energy_cancels_camera_shake():
    """Shifting the whole crop (a shaky camera, no mouth movement) must not look like speech."""
    rng = np.random.default_rng(2)
    base = rng.integers(0, 256, size=(88 + 12, 88 + 12), dtype=np.uint8)
    crops = np.stack([base[2 + dy:2 + dy + 88, 2 + dx:2 + dx + 88]
                      for dy, dx in [(0, 0), (1, 0), (1, 2), (0, 3), (2, 2), (0, 0), (1, 1), (2, 0)] * 2])
    shaky = quality.motion_energy(crops)
    talking = quality.motion_energy(rng.integers(0, 256, size=(16, 88, 88), dtype=np.uint8))
    assert shaky < talking / 5
    assert shaky < quality.MOTION_MIN


def test_motion_energy_needs_two_frames():
    assert quality.motion_energy(np.zeros((1, 88, 88), np.uint8)) == 0.0
    assert quality.motion_energy(np.zeros((0, 88, 88), np.uint8)) == 0.0


def test_sharpness_orders_sharp_above_blurred():
    rng = np.random.default_rng(3)
    sharp = rng.integers(0, 256, size=(5, 88, 88), dtype=np.uint8)
    flat = np.full((5, 88, 88), 120, np.uint8)
    assert quality.sharpness(sharp) > quality.SHARP_MARGINAL
    assert quality.sharpness(flat) == pytest.approx(0.0)


# --------------------------------------------------------------------------- assess

def _track(face_px: float, yaw_deg: float, n: int = 40, mouth_px: float = 24.0) -> FaceTrack:
    """A track of frontal-ish detections with the given face height and head yaw."""
    dets = []
    half = mouth_px / 2
    for f in range(n):
        x, y = 100.0, 50.0
        w = h = face_px
        cx = x + w / 2
        eye_y = y + 0.4 * h
        kp = _kp4_for_yaw(yaw_deg, eye_gap=0.3 * w)
        kp[:, 0] += cx - 100.0
        kp[:, 1] += eye_y - 100.0
        pts5 = np.array([kp[0], kp[1], kp[2], [cx - half, y + 0.8 * h], [cx + half, y + 0.8 * h]])
        dets.append(Detection(frame=f, box=(x, y, w, h), score=0.95, points5=pts5, keypoints4=kp))
    return FaceTrack(track_id=1, shot_id=0, detections=dets)


def _moving_crops(n: int = 40, seed: int = 7) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(n, 88, 88), dtype=np.uint8)


def _kp_track(n: int) -> np.ndarray:
    return np.repeat(_kp4_for_yaw(0.0)[None], n, axis=0)


def test_assess_big_frontal_face_is_readable():
    track = _track(face_px=200, yaw_deg=5)
    rep = quality.assess(track, _moving_crops(), _kp_track(40))
    assert rep.tier == Readability.READABLE
    assert rep.reasons == []
    assert rep.face_px == pytest.approx(200)
    assert rep.coverage == pytest.approx(1.0)
    assert rep.yaw_deg == pytest.approx(5, abs=0.5)
    assert rep.motion > quality.MOTION_MIN


def test_assess_25_px_face_is_unreadable():
    track = _track(face_px=25, yaw_deg=0)
    rep = quality.assess(track, _moving_crops(), _kp_track(40))
    assert rep.tier == Readability.UNREADABLE
    assert any("25 px" in r for r in rep.reasons)


def test_assess_60_degree_yaw_is_unreadable():
    track = _track(face_px=200, yaw_deg=60)
    rep = quality.assess(track, _moving_crops(), _kp_track(40))
    assert rep.tier == Readability.UNREADABLE
    assert any("turned" in r for r in rep.reasons)


def test_assess_small_face_is_marginal_not_unreadable():
    track = _track(face_px=40, yaw_deg=0)
    rep = quality.assess(track, _moving_crops(), _kp_track(40))
    assert rep.tier == Readability.MARGINAL
    assert any("40 px" in r or "best results" in r for r in rep.reasons)


def test_assess_still_mouth_is_marginal():
    still = np.repeat(_moving_crops(1)[None, 0], 40, axis=0)
    rep = quality.assess(_track(200, 0), still, _kp_track(40))
    assert rep.tier == Readability.MARGINAL
    assert any("barely moving" in r for r in rep.reasons)


def test_assess_low_coverage_is_unreadable():
    track = _track(200, 0)
    track.detections = [d for d in track.detections if not 5 <= d.frame <= 26]   # 18 of 40 frames found
    rep = quality.assess(track, _moving_crops(), _kp_track(40))
    assert rep.tier == Readability.UNREADABLE
    assert rep.coverage < quality.COVERAGE_MIN


def test_assess_empty_track_is_unreadable():
    empty = FaceTrack(track_id=9, shot_id=0, detections=[])
    rep = quality.assess(empty, np.zeros((0, 88, 88), np.uint8), np.zeros((0, 4, 2)))
    assert rep.tier == Readability.UNREADABLE
    assert rep.reasons == ["no face detected"]
