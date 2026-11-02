"""Pipeline orchestration on real (synthetic or cropped) clips, with the stub engine.

No licensed model is needed: ``StubEngine`` stands in for the lip reader. Tests that need
the face models skip when they are absent, and the clip-based tests skip when the
frontal source clip is missing.
"""
from __future__ import annotations

import json
import subprocess

import numpy as np
import pytest

from lipread import align, pipeline
from lipread.engine import StubEngine
from lipread.errors import LipReadError
from lipread.types import CROP, Job, Readability

# --------------------------------------------------------------------------- pure helpers

@pytest.mark.parametrize("spec,expected", [
    (None, "readable"), ("", "readable"), ("ALL", "all"), (" largest ", "largest"),
    ("none", "none"), ("1,3", [1, 3]), ("7", [7]),
])
def test_parse_faces_accepts_modes_and_id_lists(spec, expected):
    assert pipeline.parse_faces(spec) == expected


@pytest.mark.parametrize("spec", ["bogus", "1,x", ",", "  ,  "])
def test_parse_faces_rejects_junk(spec):
    with pytest.raises(LipReadError):
        pipeline.parse_faces(spec)


CANDS = [(1, Readability.READABLE, 150.0), (2, Readability.MARGINAL, 90.0),
         (3, Readability.UNREADABLE, 300.0)]


def test_select_readable_keeps_readable_and_marginal_only():
    assert pipeline.select_track_ids(CANDS, "readable", False) == {1, 2}


def test_select_all_still_respects_the_gate_unless_forced():
    assert pipeline.select_track_ids(CANDS, "all", False) == {1, 2}
    assert pipeline.select_track_ids(CANDS, "all", True) == {1, 2, 3}


def test_select_largest_picks_the_biggest_eligible_face():
    assert pipeline.select_track_ids(CANDS, "largest", False) == {1}      # track 3 is biggest but unreadable
    assert pipeline.select_track_ids(CANDS, "largest", True) == {3}


def test_select_explicit_ids_skip_unknown_and_gated_tracks():
    assert pipeline.select_track_ids(CANDS, [1, 3, 99], False) == {1}
    assert pipeline.select_track_ids(CANDS, [1, 3, 99], True) == {1, 3}


def test_select_none_and_empty():
    assert pipeline.select_track_ids(CANDS, "none", True) == set()
    assert pipeline.select_track_ids([], "all", True) == set()
    assert pipeline.select_track_ids([], "largest", False) == set()


def test_streaming_affines_and_crops_match_align_mouth_crops():
    """Pass 2 cuts crops one frame at a time; they must equal align.mouth_crops exactly."""
    rng = np.random.default_rng(11)
    T = 30
    base = np.array([[120, 150], [180, 150], [150, 190], [150, 220]], np.float64)
    drift = np.linspace(0, 6, T)[:, None, None] * np.array([1.0, -0.5])
    kp = base[None] + drift + rng.normal(0, 0.8, size=(T, 4, 2))
    frames = rng.integers(0, 256, size=(T, 300, 320), dtype=np.uint8)

    expected = align.mouth_crops(frames, kp)
    Ms, origins = pipeline._affines(kp)
    got = np.stack([pipeline._crop_one(frames[i], Ms[i], int(origins[i][0]), int(origins[i][1]))
                    for i in range(T)])
    assert got.shape == (T, CROP, CROP)
    np.testing.assert_array_equal(got, expected)


def test_track_keypoints_interpolates_over_missing_frames():
    from lipread.types import Detection, FaceTrack

    def det(f, val):
        return Detection(frame=f, box=(0, 0, 10, 10), score=1.0, points5=np.zeros((5, 2)),
                         keypoints4=np.full((4, 2), float(val)))

    track = FaceTrack(track_id=1, shot_id=0, detections=[det(0, 0), det(2, 4), det(4, 8)])
    kp = pipeline._track_keypoints(track)
    assert kp.shape == (5, 4, 2)
    np.testing.assert_allclose(kp[:, 0, 0], [0, 2, 4, 6, 8])


def test_strip_tiles_crops_side_by_side():
    crops = np.full((20, CROP, CROP), 200, np.uint8)
    strip = pipeline._strip(crops, n=8)
    assert strip.shape == (CROP, 8 * CROP + 7 * pipeline.STRIP_GAP)


def test_logprobs_are_cached_per_track_and_engine(tmp_path):
    class Counting(StubEngine):
        def __init__(self):
            super().__init__("HELLO", 0.9)
            self.calls = 0

        def logprobs(self, crops):
            self.calls += 1
            return super().logprobs(crops)

    eng = Counting()
    crops = np.random.default_rng(0).integers(0, 256, size=(40, CROP, CROP), dtype=np.uint8)
    first = pipeline._logprobs(eng, crops, 4, str(tmp_path))
    second = pipeline._logprobs(eng, crops, 4, str(tmp_path))
    assert eng.calls == 1
    np.testing.assert_array_equal(first, second)


def test_default_engine_refuses_without_licence(tmp_path, monkeypatch):
    from lipread import models

    monkeypatch.setenv(models.ENV_HOME, str(tmp_path / "empty"))
    monkeypatch.delenv(models.ENV_ACCEPT, raising=False)
    with pytest.raises(LipReadError, match="licen"):
        pipeline.default_engine()


def test_run_without_engine_or_licence_fails_before_any_work(tmp_path, monkeypatch):
    """Decoding needs the lip reader: the licence check happens before the source is even opened."""
    from lipread import models

    monkeypatch.setenv(models.ENV_HOME, str(tmp_path / "empty"))
    monkeypatch.delenv(models.ENV_ACCEPT, raising=False)
    job = Job(source=str(tmp_path / "does-not-exist.mp4"), out_dir=str(tmp_path / "out"))
    with pytest.raises(LipReadError, match="licen"):
        pipeline.run(job)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("start,end,msg", [(-1.0, None, "start must be"), (5.0, 2.0, "end must be after")])
def test_run_rejects_bad_time_range(tmp_path, start, end, msg):
    job = Job(source="x.mp4", out_dir=str(tmp_path), start=start, end=end, faces="none")
    with pytest.raises(LipReadError, match=msg):
        pipeline.run(job)


# --------------------------------------------------------------------------- full runs

def _probe_streams(path: str) -> list[str]:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
                          "-of", "csv=p=0", path], capture_output=True, text=True, check=True)
    return out.stdout.split()


def _assert_outputs(result, out_dir):
    for kind in ("json", "srt", "vtt", "txt"):
        path = result.outputs[kind]
        assert path.startswith(str(out_dir)), path
        assert (out_dir / f"lipread.{kind}").is_file()
    data = json.loads((out_dir / "lipread.json").read_text(encoding="utf-8"))
    assert data["disclaimer"] == result.disclaimer
    assert len(data["tracks"]) == len(result.tracks)


def test_frontal_clip_end_to_end_with_stub_engine(frontal_clip, face_models_dir, tmp_path):
    out_dir = tmp_path / "run"
    job = Job(source=str(frontal_clip), out_dir=str(out_dir), faces="all", force_unreadable=True,
              render_video=False)
    stages: list[str] = []

    def progress(stage, frac, msg):
        assert 0.0 <= frac <= 1.0
        stages.append(stage)

    result = pipeline.run(job, engine=StubEngine("HELLO WORLD", confidence=0.95), progress=progress)

    assert len(result.tracks) == 1
    track = result.tracks[0]
    assert track.skipped is False
    assert [u.text for u in track.utterances] == ["HELLO WORLD"]
    assert track.start < track.end <= result.video.duration + 0.01
    assert track.thumbnail and (out_dir / "thumbs" / "track_1.jpg").is_file()
    assert track.crops_preview and (out_dir / "crops" / "track_1.png").is_file()
    assert result.shots and result.shots[0][0] == 0
    assert result.model["name"] == "stub"
    assert "mp4" not in result.outputs
    _assert_outputs(result, out_dir)
    assert "fetch" in stages and stages[-1] == "done"
    assert {"detect", "crops", "infer", "decode"} <= set(stages) | set(result.timings)

    srt = (out_dir / "lipread.srt").read_text(encoding="utf-8")
    assert "[Face 1] HELLO WORLD" in srt


def test_frontal_clip_renders_annotated_mp4(frontal_clip, face_models_dir, tmp_path):
    out_dir = tmp_path / "render"
    job = Job(source=str(frontal_clip), out_dir=str(out_dir), faces="all", force_unreadable=True,
              render_video=True)
    result = pipeline.run(job, engine=StubEngine("HELLO WORLD", confidence=0.95))
    mp4 = result.outputs["mp4"]
    assert (out_dir / "annotated.mp4").is_file()
    assert "video" in _probe_streams(mp4)
    assert "audio" not in _probe_streams(mp4)           # the frontal fixture has no audio track


def test_faces_none_decodes_nothing_and_never_loads_the_model(frontal_clip, face_models_dir,
                                                               tmp_path, monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("faces=none must not load a lip-reading engine")

    monkeypatch.setattr(pipeline, "default_engine", refuse)
    out_dir = tmp_path / "scan"
    job = Job(source=str(frontal_clip), out_dir=str(out_dir), faces="none", render_video=True)
    result = pipeline.run(job)

    assert result.model["name"] == "none"
    assert result.tracks, "the scan still finds and tracks the face"
    assert all(t.skipped for t in result.tracks)
    assert all(t.utterances == [] for t in result.tracks)
    assert "infer" not in result.timings and "decode" not in result.timings
    assert "mp4" not in result.outputs                  # nothing decoded, nothing to annotate
    _assert_outputs(result, out_dir)


def test_clip_without_faces_gives_shots_and_an_empty_transcript(two_shot_clip, face_models_dir, tmp_path):
    out_dir = tmp_path / "shots"
    job = Job(source=str(two_shot_clip), out_dir=str(out_dir), faces="readable", render_video=True)
    result = pipeline.run(job, engine=StubEngine("HELLO WORLD"))

    assert result.tracks == []
    assert len(result.shots) == 2
    assert result.shots[0][0] == 0 and result.shots[-1][1] >= 98
    assert result.shots[1][0] == pytest.approx(50, abs=2)
    assert "video" in _probe_streams(result.outputs["mp4"])   # rendered with the badge only
    srt = (out_dir / "lipread.srt").read_text(encoding="utf-8")
    assert srt.strip().endswith("AI lip-read estimate — may be wrong")
    _assert_outputs(result, out_dir)
