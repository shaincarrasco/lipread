"""Ingest (probe, frame streaming, trimming, audio) and shot-cut detection."""
from __future__ import annotations

import numpy as np
import pytest

from lipread import ingest
from lipread.errors import LipReadError
from lipread.shots import ShotDetector
from lipread.types import FPS

# BGR means of the solid shots used by conftest.make_clip
GREY_BGR = np.array([128, 128, 128], float)
RED_BGR = np.array([0, 0, 255], float)


def _mean_bgr(frame: np.ndarray) -> np.ndarray:
    return frame.reshape(-1, 3).mean(axis=0)


# --------------------------------------------------------------------------- probe

def test_probe_reports_stream_fields(two_shot_clip):
    info = ingest.probe(str(two_shot_clip))
    assert (info.width, info.height) == (320, 240)
    assert info.src_fps == pytest.approx(30000 / 1001, rel=1e-6)
    assert info.duration == pytest.approx(4.0, abs=0.1)
    assert info.has_audio is True
    assert info.interlaced is False
    assert info.rotation == 0


def test_probe_reports_no_audio(silent_clip):
    info = ingest.probe(str(silent_clip))
    assert info.has_audio is False


def test_probe_missing_file_raises(tmp_path):
    with pytest.raises(LipReadError, match="not found"):
        ingest.probe(str(tmp_path / "nope.mp4"))


def test_fetch_local_file_returns_absolute_path(two_shot_clip, tmp_path):
    got = ingest.fetch(str(two_shot_clip), str(tmp_path))
    assert got == str(two_shot_clip)


def test_fetch_missing_local_file_raises(tmp_path):
    with pytest.raises(LipReadError, match="not found"):
        ingest.fetch(str(tmp_path / "missing.mp4"), str(tmp_path))


def test_probe_swaps_size_for_a_quarter_turn(monkeypatch, tmp_path):
    """A 90 degree display rotation swaps width and height; ffmpeg already rotates the decoded frames."""
    clip = tmp_path / "rot.mp4"
    clip.write_bytes(b"x")
    data = {
        "streams": [{"codec_type": "video", "width": 320, "height": 240,
                     "avg_frame_rate": "25/1", "side_data_list": [{"rotation": -90}]}],
        "format": {"duration": "2.0"},
    }
    monkeypatch.setattr(ingest, "_ffprobe_json", lambda path: data)
    info = ingest.probe(str(clip))
    assert info.rotation == 270
    assert (info.width, info.height) == (240, 320)


@pytest.mark.parametrize("value,expected", [
    ("30000/1001", 30000 / 1001),
    ("25", 25.0),
    ("0/0", None),
    ("", None),
    (None, None),
    ("abc", None),
])
def test_parse_rate(value, expected):
    got = ingest._parse_rate(value)
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(expected)


@pytest.mark.parametrize("raw,expected", [(90, 90), (-90, 270), (270, 270), (180, 180), ("junk", 0)])
def test_rotation_is_normalised(raw, expected):
    stream = {"side_data_list": [{"rotation": raw}]}
    assert ingest._rotation(stream) == expected


# --------------------------------------------------------------------------- frames

def test_iter_frames_count_and_shape_for_29_97_source(two_shot_clip):
    info = ingest.probe(str(two_shot_clip))
    frames = list(ingest.iter_frames(info))
    expected = ingest.count_frames(info)
    assert expected == 100                              # 4 s at 25 fps
    assert abs(len(frames) - expected) <= 1
    assert frames[0].shape == (240, 320, 3)
    assert frames[0].dtype == np.uint8


def test_iter_frames_trim_returns_the_right_shot(two_shot_clip):
    """Trimming inside each shot must return only that shot's colour: checks the seek is frame-accurate."""
    info = ingest.probe(str(two_shot_clip))
    grey = list(ingest.iter_frames(info, start=0.5, end=1.5))
    red = list(ingest.iter_frames(info, start=2.5, end=3.5))
    assert abs(len(grey) - 25) <= 1
    assert abs(len(red) - 25) <= 1
    for f in grey[::5]:
        np.testing.assert_allclose(_mean_bgr(f), GREY_BGR, atol=12)
    for f in red[::5]:
        np.testing.assert_allclose(_mean_bgr(f), RED_BGR, atol=20)


def test_iter_frames_empty_range_yields_nothing(two_shot_clip):
    info = ingest.probe(str(two_shot_clip))
    assert list(ingest.iter_frames(info, start=2.0, end=2.0)) == []
    assert list(ingest.iter_frames(info, start=3.0, end=1.0)) == []


def test_count_frames_clips_to_duration(two_shot_clip):
    info = ingest.probe(str(two_shot_clip))
    assert ingest.count_frames(info, 0.0, None) == round(info.duration * FPS)
    assert ingest.count_frames(info, 1.0, 2.0) == 25
    assert ingest.count_frames(info, 3.9, 100.0) == round((info.duration - 3.9) * FPS)
    assert ingest.count_frames(info, 5.0, 6.0) == 0


def test_output_size_caps_height_and_keeps_even_width():
    assert ingest._output_size(1920, 1080, 1080) == (1920, 1080)
    assert ingest._output_size(3840, 2160, 1080) == (1920, 1080)
    w, h = ingest._output_size(1001, 2000, 1080)
    assert h == 1080 and w % 2 == 0


# --------------------------------------------------------------------------- audio

def test_extract_audio_length_and_trim(two_shot_clip):
    info = ingest.probe(str(two_shot_clip))
    full = ingest.extract_audio(info)
    assert full is not None and full.dtype == np.float32
    assert len(full) == pytest.approx(4 * 16000, abs=1600)
    part = ingest.extract_audio(info, start=1.0, end=3.0)
    assert len(part) == pytest.approx(2 * 16000, abs=800)
    assert np.abs(part).max() > 0.05                     # the sine tone is really there


def test_extract_audio_is_none_without_audio_track(silent_clip):
    info = ingest.probe(str(silent_clip))
    assert ingest.extract_audio(info) is None


def test_extract_audio_empty_window_is_empty(two_shot_clip):
    info = ingest.probe(str(two_shot_clip))
    assert len(ingest.extract_audio(info, start=2.0, end=1.0)) == 0


# --------------------------------------------------------------------------- shots

def test_shot_detector_finds_grey_to_bright_cut_in_memory():
    det = ShotDetector()
    grey = np.full((120, 160, 3), 128, np.uint8)
    white = np.full((120, 160, 3), 255, np.uint8)
    for _ in range(20):
        det.update(grey)
    for _ in range(20):
        det.update(white)
    assert det.cuts == [20]
    assert det.shots(40) == [(0, 19), (20, 39)]


def test_shot_detector_static_clip_has_one_shot():
    det = ShotDetector()
    frame = np.full((120, 160, 3), (100, 120, 140), np.uint8)
    for _ in range(30):
        det.update(frame)
    assert det.shots(30) == [(0, 29)]


def test_shot_detector_merges_short_shots():
    det = ShotDetector(min_len=10)
    a = np.full((60, 80, 3), 128, np.uint8)
    b = np.full((60, 80, 3), (0, 0, 255), np.uint8)
    for _ in range(30):
        det.update(a)
    for _ in range(3):                                   # 3-frame flash: too short to be a shot
        det.update(b)
    for _ in range(30):
        det.update(a)
    assert det.cuts == [30, 33]
    # the 3-frame flash (frames 30..32) is folded into the shot before it
    assert det.shots(63) == [(0, 32), (33, 62)]


def test_shot_detector_empty_input():
    assert ShotDetector().shots(0) == []


def test_shot_detector_rejects_bad_min_len():
    with pytest.raises(ValueError):
        ShotDetector(min_len=0)


def test_shot_detector_on_decoded_two_shot_clip(two_shot_clip):
    info = ingest.probe(str(two_shot_clip))
    det = ShotDetector()
    n = 0
    for frame in ingest.iter_frames(info):
        det.update(frame)
        n += 1
    shots = det.shots(n)
    assert len(shots) == 2
    assert shots[0][0] == 0 and shots[-1][1] == n - 1
    assert abs(shots[1][0] - 50) <= 2                    # the cut is at 2 s = frame 50 at 25 fps
