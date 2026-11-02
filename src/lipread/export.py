"""Write a pipeline ``Result`` to JSON, SRT, WebVTT and plain-text files.

Every subtitle and text line is an estimate. The cue text carries markers so a
reader can see what is uncertain at a glance:

* a word with LOW confidence is suffixed with ``?``;
* a word the roster booster replaced is wrapped in asterisks, e.g. ``*KELCE*``.

Times are seconds on the source timeline (``t = job.start + frame / FPS``), as
documented in ``types.py``.
"""
from __future__ import annotations

import html
import json
import os
from dataclasses import dataclass
from typing import Iterable

from .types import Confidence, Result, TrackResult, Utterance, Word

DISCLAIMER_CUE = "AI lip-read estimate — may be wrong"
DISCLAIMER_SECONDS = 3.0      # the disclaimer cue lasts this long, or until the first utterance
MIN_CUE_SECONDS = 0.3         # keep very short utterances readable
MIN_DISCLAIMER_SECONDS = 0.5  # skip the disclaimer cue if there is no real gap for it


@dataclass(frozen=True)
class Cue:
    """One subtitle cue on the source timeline."""

    start: float
    end: float
    text: str


def word_markup(word: Word) -> str:
    """Text of one word with the uncertainty and boost markers applied."""
    text = f"*{word.text}*" if word.boosted else word.text
    if word.tier == Confidence.LOW:
        text += "?"
    return text


def utterance_markup(utt: Utterance) -> str:
    """Text of an utterance with markers applied to each word."""
    return " ".join(word_markup(w) for w in utt.words)


def srt_time(seconds: float) -> str:
    """Format seconds as an SRT timestamp ``HH:MM:SS,mmm``."""
    ms_total = max(0, round(seconds * 1000))
    h, rem = divmod(ms_total, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def vtt_time(seconds: float) -> str:
    """Format seconds as a WebVTT timestamp ``HH:MM:SS.mmm``."""
    return srt_time(seconds).replace(",", ".")


def _tenths_clock(seconds: float) -> str:
    """Format seconds as ``mm:ss.s`` for the plain-text report."""
    tenths = max(0, round(seconds * 10))
    m, rem = divmod(tenths, 600)
    return f"{m:02d}:{rem / 10:04.1f}"


def build_cues(result: Result) -> list[Cue]:
    """One cue per utterance from every decoded track, sorted by time, with the disclaimer first.

    Skipped tracks are ignored. Ties in start time are ordered by face id.
    """
    cues: list[Cue] = []
    for track in sorted(result.tracks, key=lambda t: t.track_id):
        if track.skipped:
            continue
        for utt in track.utterances:
            end = max(utt.end, utt.start + MIN_CUE_SECONDS)
            text = f"[Face {track.track_id}] {utterance_markup(utt)}"
            cues.append(Cue(utt.start, end, text))
    cues.sort(key=lambda c: c.start)

    # The AI notice is always present (players stack overlapping cues), starting where the
    # analysed window starts so it also shows on trimmed clips.
    t0 = float(result.job.start or 0.0)
    cues.insert(0, Cue(t0, t0 + DISCLAIMER_SECONDS, DISCLAIMER_CUE))
    return cues


def format_srt(cues: Iterable[Cue]) -> str:
    """Render cues as an SRT document."""
    blocks = []
    for i, cue in enumerate(cues, start=1):
        blocks.append(f"{i}\n{srt_time(cue.start)} --> {srt_time(cue.end)}\n{cue.text}\n")
    return "\n".join(blocks)


def format_vtt(cues: Iterable[Cue]) -> str:
    """Render cues as a WebVTT document. Text is HTML-escaped as WebVTT requires."""
    lines = ["WEBVTT", ""]
    for cue in cues:
        lines.append(f"{vtt_time(cue.start)} --> {vtt_time(cue.end)}")
        lines.append(html.escape(cue.text, quote=False))
        lines.append("")
    return "\n".join(lines)


def _track_header(track: TrackResult) -> str:
    """Heading line for one face track in the text report."""
    tier = track.quality.tier.value.upper()
    span = f"{_tenths_clock(track.start)}–{_tenths_clock(track.end)}"
    return f"=== FACE {track.track_id} | {tier} | shot {track.shot_id} | {span} ==="


def format_txt(result: Result) -> str:
    """Plain-text report: disclaimer header, then each face track with its tier and reasons."""
    model = result.model or {}
    model_name = " ".join(str(model[k]) for k in ("name", "version") if k in model) or "unknown"
    lines = [
        "LipRead transcript",
        result.disclaimer,
        f"Source: {result.job.source}",
        f"Model: {model_name}",
        "Times are on the source video timeline.",
        "",
    ]
    for track in sorted(result.tracks, key=lambda t: t.track_id):
        lines.append(_track_header(track))
        if track.skipped:
            lines.append("Not decoded (skipped by the face filter or the readability gate).")
        else:
            for reason in track.quality.reasons:
                lines.append(f"- {reason}")
            if not track.utterances:
                lines.append("(no speech decoded)")
            for utt in track.utterances:
                lines.append(
                    f"[{_tenths_clock(utt.start)}–{_tenths_clock(utt.end)}] "
                    f"{utterance_markup(utt)} (conf {utt.confidence:.2f})"
                )
        lines.append("")
    return "\n".join(lines)


def _write_text(path: str, content: str) -> str:
    """Write UTF-8 text with Unix newlines and return the path."""
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(content)
    return path


def write_all(result: Result, out_dir: str) -> dict[str, str]:
    """Write ``lipread.json``, ``lipread.srt``, ``lipread.vtt`` and ``lipread.txt`` into ``out_dir``.

    Returns a mapping of kind (``json``, ``srt``, ``vtt``, ``txt``) to the written path.
    """
    os.makedirs(out_dir, exist_ok=True)
    cues = build_cues(result)

    json_path = os.path.join(out_dir, "lipread.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    return {
        "json": json_path,
        "srt": _write_text(os.path.join(out_dir, "lipread.srt"), format_srt(cues)),
        "vtt": _write_text(os.path.join(out_dir, "lipread.vtt"), format_vtt(cues)),
        "txt": _write_text(os.path.join(out_dir, "lipread.txt"), format_txt(result)),
    }
