"""Export: JSON, SRT, WebVTT and TXT built from a hand-made Result."""
from __future__ import annotations

import html
import json
import re

import pytest

from lipread import export
from lipread.types import (Confidence, Job, QualityReport, Readability, Result, TrackResult,
                           Utterance, VideoInfo, Word)


def _word(text: str, start: float, end: float, conf: float = 0.95, boosted: bool = False,
          original: str | None = None) -> Word:
    tier = Confidence.HIGH if conf >= 0.9 else Confidence.MEDIUM if conf >= 0.7 else Confidence.LOW
    return Word(text=text, start=start, end=end, confidence=conf, tier=tier,
                boosted=boosted, original=original)


def _quality(tier: Readability = Readability.READABLE) -> QualityReport:
    return QualityReport(tier=tier, face_px=120.0, mouth_px=40.0, yaw_deg=5.0, sharpness=40.0,
                         coverage=1.0, motion=0.05, reasons=[] if tier == Readability.READABLE else ["blurry"])


@pytest.fixture
def result(tmp_path) -> Result:
    job = Job(source="/videos/game.mp4", out_dir=str(tmp_path), start=0.0)
    video = VideoInfo(path="/videos/game.mp4", width=640, height=360, src_fps=25.0, duration=12.0,
                      has_audio=True)
    utt1 = Utterance(start=1.0, end=2.4, words=[
        _word("WE", 1.0, 1.2),
        _word("GO", 1.3, 1.6, conf=0.5),                                     # LOW -> "?"
        _word("KELCE", 1.7, 2.4, conf=0.8, boosted=True, original="KEL SO"),  # boosted
    ])
    utt2 = Utterance(start=6.0, end=7.0, words=[_word("NOW", 6.0, 7.0)])
    track1 = TrackResult(track_id=1, shot_id=0, start=0.4, end=9.0, quality=_quality(),
                         utterances=[utt1, utt2])
    track2 = TrackResult(track_id=2, shot_id=0, start=0.4, end=9.0,
                         quality=_quality(Readability.UNREADABLE), utterances=[], skipped=True)
    track3 = TrackResult(track_id=3, shot_id=1, start=8.0, end=11.0, quality=_quality(Readability.MARGINAL),
                         utterances=[Utterance(9.0, 9.5, [_word("A & B <x>", 9.0, 9.5)])])
    return Result(job=job, video=video, tracks=[track1, track2, track3],
                  shots=[(0, 149), (150, 299)], model={"name": "stub", "version": "1"},
                  timings={"total": 1.0})


TIMING = re.compile(r"^(\d{2}):(\d{2}):(\d{2})[,.](\d{3}) --> (\d{2}):(\d{2}):(\d{2})[,.](\d{3})$")


def _secs(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def _parse_srt(text: str) -> list[tuple[int, float, float, str]]:
    cues = []
    for block in text.strip().split("\n\n"):
        lines = block.split("\n")
        m = TIMING.match(lines[1])
        assert m, lines[1]
        start = _secs(*m.group(1, 2, 3, 4))
        end = _secs(*m.group(5, 6, 7, 8))
        cues.append((int(lines[0]), start, end, "\n".join(lines[2:])))
    return cues


def _parse_vtt(text: str) -> list[tuple[float, float, str]]:
    lines = text.split("\n")
    assert lines[0] == "WEBVTT"
    cues = []
    i = 1
    while i < len(lines):
        if "-->" in lines[i]:
            m = re.match(r"^(\d{2}):(\d{2}):(\d{2})\.(\d{3}) --> (\d{2}):(\d{2}):(\d{2})\.(\d{3})$", lines[i])
            assert m, lines[i]
            body = []
            i += 1
            while i < len(lines) and lines[i].strip():
                body.append(lines[i])
                i += 1
            cues.append((_secs(*m.group(1, 2, 3, 4)), _secs(*m.group(5, 6, 7, 8)), "\n".join(body)))
        i += 1
    return cues


# --------------------------------------------------------------------------- timestamps

@pytest.mark.parametrize("seconds,expected", [
    (0.0, "00:00:00,000"),
    (1.5, "00:00:01,500"),
    (61.042, "00:01:01,042"),
    (3723.9996, "01:02:04,000"),        # rounds up to the next second, not "...,1000"
    (-1.0, "00:00:00,000"),
])
def test_srt_time_format(seconds, expected):
    assert export.srt_time(seconds) == expected


def test_vtt_time_uses_a_dot():
    assert export.vtt_time(61.5) == "00:01:01.500"


# --------------------------------------------------------------------------- cues

def test_build_cues_disclaimer_first_then_utterances_in_order(result):
    cues = export.build_cues(result)
    assert cues[0].text == export.DISCLAIMER_CUE
    # the disclaimer always lasts DISCLAIMER_SECONDS from the window start (it may overlap speech)
    assert cues[0].start == 0.0 and cues[0].end == pytest.approx(export.DISCLAIMER_SECONDS)
    times = [c.start for c in cues]
    assert times == sorted(times)
    texts = [c.text for c in cues[1:]]
    assert texts[0].startswith("[Face 1] WE GO? *KELCE*")
    assert any(t.startswith("[Face 3]") for t in texts)
    assert not any("[Face 2]" in t for t in texts)          # skipped track is ignored


def test_build_cues_without_speech_gives_only_the_disclaimer(tmp_path):
    job = Job(source="x.mp4", out_dir=str(tmp_path))
    res = Result(job=job, video=VideoInfo(path="x.mp4", width=2, height=2, src_fps=25, duration=3),
                 tracks=[], shots=[(0, 74)], model={}, timings={})
    cues = export.build_cues(res)
    assert len(cues) == 1 and cues[0].text == export.DISCLAIMER_CUE


def test_subtitles_always_carry_the_disclaimer_when_speech_starts_at_zero(tmp_path):
    job = Job(source="x.mp4", out_dir=str(tmp_path))
    track = TrackResult(track_id=1, shot_id=0, start=0.0, end=2.0, quality=_quality(),
                        utterances=[Utterance(0.0, 1.0, [_word("HI", 0.0, 1.0)])])
    res = Result(job=job, video=VideoInfo(path="x.mp4", width=2, height=2, src_fps=25, duration=3),
                 tracks=[track], shots=[(0, 74)], model={}, timings={})
    cues = export.build_cues(res)
    assert any(c.text == export.DISCLAIMER_CUE for c in cues)


def test_short_gap_before_first_utterance_still_has_disclaimer_cue(tmp_path):
    job = Job(source="x.mp4", out_dir=str(tmp_path))
    track = TrackResult(track_id=1, shot_id=0, start=0.0, end=2.0, quality=_quality(),
                        utterances=[Utterance(0.2, 1.0, [_word("HI", 0.2, 1.0)])])
    res = Result(job=job, video=VideoInfo(path="x.mp4", width=2, height=2, src_fps=25, duration=3),
                 tracks=[track], shots=[(0, 74)], model={}, timings={})
    cues = export.build_cues(res)
    assert [c.text for c in cues] == [export.DISCLAIMER_CUE, "[Face 1] HI"]


def test_disclaimer_starts_at_trimmed_window(tmp_path):
    job = Job(source="x.mp4", out_dir=str(tmp_path), start=12.0)
    track = TrackResult(track_id=1, shot_id=0, start=12.0, end=14.0, quality=_quality(),
                        utterances=[Utterance(12.1, 13.0, [_word("HI", 12.1, 13.0)])])
    res = Result(job=job, video=VideoInfo(path="x.mp4", width=2, height=2, src_fps=25, duration=30),
                 tracks=[track], shots=[(0, 49)], model={}, timings={})
    cues = export.build_cues(res)
    assert cues[0].text == export.DISCLAIMER_CUE and cues[0].start == pytest.approx(12.0)


def test_utterance_cues_have_minimum_length():
    job = Job(source="x", out_dir=".")
    track = TrackResult(track_id=1, shot_id=0, start=0, end=1, quality=_quality(),
                        utterances=[Utterance(5.0, 5.05, [_word("OK", 5.0, 5.05)])])
    res = Result(job=job, video=VideoInfo(path="x", width=2, height=2, src_fps=25, duration=9),
                 tracks=[track], shots=[], model={}, timings={})
    cue = [c for c in export.build_cues(res) if "OK" in c.text][0]
    assert cue.end - cue.start >= export.MIN_CUE_SECONDS - 1e-9


# --------------------------------------------------------------------------- markers

def test_word_markup_flags_low_confidence_and_boosted():
    assert export.word_markup(_word("GO", 0, 1, conf=0.5)) == "GO?"
    assert export.word_markup(_word("KELCE", 0, 1, boosted=True, original="KEL SO")) == "*KELCE*"
    assert export.word_markup(_word("WE", 0, 1)) == "WE"


# --------------------------------------------------------------------------- SRT / VTT

def test_srt_parses_back_with_disclaimer_and_markers(result):
    srt = export.format_srt(export.build_cues(result))
    cues = _parse_srt(srt)
    assert [n for n, *_ in cues] == list(range(1, len(cues) + 1))
    assert cues[0][3] == export.DISCLAIMER_CUE
    body = " | ".join(text for *_, text in cues)
    assert "WE GO? *KELCE*" in body
    assert "NOW" in body
    for _, start, end, _ in cues:
        assert end > start


def test_vtt_parses_back_and_escapes_markup(result):
    vtt = export.format_vtt(export.build_cues(result))
    assert vtt.startswith("WEBVTT\n")
    cues = _parse_vtt(vtt)
    assert cues[0][2] == export.DISCLAIMER_CUE
    joined = "\n".join(text for *_, text in cues)
    assert "WE GO? *KELCE*" in joined
    # the track-3 utterance contains & and < > which must be escaped in WebVTT
    assert "A &amp; B &lt;x&gt;" in vtt
    assert html.unescape("A &amp; B &lt;x&gt;") == "A & B <x>"


def test_srt_and_vtt_agree_on_times(result):
    cues = export.build_cues(result)
    srt = _parse_srt(export.format_srt(cues))
    vtt = _parse_vtt(export.format_vtt(cues))
    assert len(srt) == len(vtt) == len(cues)
    for (_, s1, e1, _), (s2, e2, _) in zip(srt, vtt):
        assert s1 == pytest.approx(s2, abs=1e-6) and e1 == pytest.approx(e2, abs=1e-6)


# --------------------------------------------------------------------------- txt / json / write_all

def test_txt_has_disclaimer_tiers_and_low_confidence_marker(result):
    txt = export.format_txt(result)
    assert txt.startswith("LipRead transcript\n" + result.disclaimer)
    assert "Model: stub 1" in txt
    assert "=== FACE 1 | READABLE | shot 0 |" in txt
    assert "=== FACE 2 | UNREADABLE" in txt
    assert "Not decoded" in txt
    assert "=== FACE 3 | MARGINAL" in txt
    assert "WE GO? *KELCE* (conf" in txt


def test_write_all_writes_four_files_and_json_round_trips(result, tmp_path):
    out = tmp_path / "out"
    paths = export.write_all(result, str(out))
    assert set(paths) == {"json", "srt", "vtt", "txt"}
    for kind, path in paths.items():
        assert (out / f"lipread.{kind}").is_file(), kind
    data = json.loads((out / "lipread.json").read_text(encoding="utf-8"))
    assert data["disclaimer"] == result.disclaimer
    assert [t["track_id"] for t in data["tracks"]] == [1, 2, 3]
    assert data["tracks"][0]["quality"]["tier"] == "readable"       # enum serialised as its value
    assert data["tracks"][0]["utterances"][0]["words"][2]["boosted"] is True
    assert data["tracks"][0]["utterances"][0]["words"][2]["original"] == "KEL SO"
    assert (out / "lipread.srt").read_text(encoding="utf-8").startswith("1\n00:00:00,000 --> ")
