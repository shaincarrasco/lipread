"""CTC greedy decoding into timed, confidence-scored words and utterances.

The engine emits per-frame log-probabilities (T, V) at 25 fps over SentencePiece
pieces. A piece starting with "▁" begins a new word. Special tokens (``<blank>``,
``<unk>``, ``<eos>``, ...) never emit text.

Timing and confidence come from the frames the model actually emitted, so a word's
``start`` and ``end`` cover its own frames. Confidence is the mean posterior of the
winning piece over those frames.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from .types import FPS, Confidence, Utterance, Word

CONF_HIGH = 0.90          # word-level tier thresholds (Commander recalibrates)
CONF_MEDIUM = 0.70
UTTERANCE_GAP = 0.6       # seconds of silence that split utterances
WORD_MARK = "▁"

log = logging.getLogger("lipread")

Run = tuple[int, int, int]   # (piece_id, first_frame, last_frame), inclusive


def tier_for(conf: float) -> Confidence:
    """Map a 0..1 word confidence to its tier."""
    if conf >= CONF_HIGH:
        return Confidence.HIGH
    if conf >= CONF_MEDIUM:
        return Confidence.MEDIUM
    return Confidence.LOW


def silent_mask(vocab: Sequence[str], blank: int) -> np.ndarray:
    """Boolean mask over the vocabulary: True for the blank and for special tokens."""
    mask = np.array([p.startswith("<") and p.endswith(">") for p in vocab], dtype=bool)
    mask[blank] = True
    return mask


def emission_runs(logprobs: np.ndarray, silent: np.ndarray) -> list[Run]:
    """Collapse the frame-wise argmax into runs of one emitting piece.

    A run is a maximal block of identical non-silent argmax labels. A silent frame
    (blank or special) ends the current run, so a repeated piece separated by a
    blank forms two runs, as in CTC.
    """
    lp = np.asarray(logprobs)
    if lp.shape[0] == 0:
        return []
    idx = lp.argmax(axis=1).tolist()
    runs: list[list[int]] = []
    prev = -1
    for t, k in enumerate(idx):
        if silent[k]:
            prev = -1
            continue
        if k == prev:
            runs[-1][2] = t
        else:
            runs.append([k, t, t])
            prev = k
    return [(k, a, b) for k, a, b in runs]


def greedy_words(logprobs: np.ndarray, vocab: Sequence[str], blank: int, t0: float) -> list[Word]:
    """Greedy CTC decode of one track into timed words.

    logprobs: float array (T, len(vocab)), frame 0 at source time ``t0``.
    Words whose text is empty after stripping the word mark are dropped.
    """
    lp = np.asarray(logprobs)
    if lp.ndim != 2 or lp.shape[1] != len(vocab):
        raise ValueError(f"logprobs must be (T, {len(vocab)}), got {lp.shape}")
    silent = silent_mask(vocab, blank)
    runs = emission_runs(lp, silent)

    groups: list[list[Run]] = []
    for run in runs:
        if vocab[run[0]].startswith(WORD_MARK) or not groups:
            groups.append([run])
        else:
            groups[-1].append(run)

    words: list[Word] = []
    for group in groups:
        text = "".join(vocab[k].replace(WORD_MARK, "") for k, _, _ in group).strip()
        if not text:
            continue
        probs = np.concatenate([np.exp(lp[a:b + 1, k]) for k, a, b in group])
        conf = float(probs.mean())
        start = t0 + group[0][1] / FPS
        end = t0 + (group[-1][2] + 1) / FPS
        words.append(Word(text=text, start=start, end=end, confidence=conf, tier=tier_for(conf)))
    return words


def word_frames(word: Word, t0: float) -> tuple[int, int]:
    """Frame span [a, b) of a word on the track's 25 fps timeline (frame 0 at ``t0``)."""
    return round((word.start - t0) * FPS), round((word.end - t0) * FPS)


def segment_utterances(words: Sequence[Word], gap: float = UTTERANCE_GAP) -> list[Utterance]:
    """Split words into utterances wherever the silence between them exceeds ``gap`` seconds."""
    utterances: list[Utterance] = []
    current: list[Word] = []
    for w in words:
        if current and w.start - current[-1].end > gap:
            utterances.append(Utterance(current[0].start, current[-1].end, current))
            current = []
        current.append(w)
    if current:
        utterances.append(Utterance(current[0].start, current[-1].end, current))
    return [_stretch(u) for u in utterances]


EDGE_PAD = 0.25   # seconds a word is extended before the first / after the last word of an utterance


def _stretch(u: Utterance) -> Utterance:
    """CTC emits each word as a ~1-frame spike. Give words a readable extent for subtitles:
    inside an utterance a word runs from the midpoint with its left neighbour to the midpoint
    with its right neighbour; the utterance edges get EDGE_PAD seconds of padding.
    Must run after boosting (the booster scores the original spike frames)."""
    ws = u.words
    spikes = [(w.start, w.end) for w in ws]
    for i, w in enumerate(ws):
        s, e = spikes[i]
        w.start = (spikes[i - 1][1] + s) / 2 if i > 0 else max(0.0, s - EDGE_PAD)
        w.end = (e + spikes[i + 1][0]) / 2 if i + 1 < len(ws) else e + EDGE_PAD
    return Utterance(ws[0].start, ws[-1].end, ws)


_LM_CACHE: dict = {}


def load_lm():
    """The word language model for context-aware decoding, or None if it isn't available.

    Looked up in $LIPREAD_LM, the repository's web/core/assets/lm.bin, then the models folder."""
    import os
    from pathlib import Path
    if "lm" in _LM_CACHE:
        return _LM_CACHE["lm"]
    here = Path(__file__).resolve()
    candidates = [os.environ.get("LIPREAD_LM"), here.parents[2] / "web" / "core" / "assets" / "lm.bin"]
    try:
        from .models import models_dir
        candidates.append(Path(models_dir()) / "lm.bin")
    except Exception:  # pragma: no cover
        pass
    lm = None
    for c in candidates:
        if c and Path(c).is_file():
            from .lm import NgramLM
            lm = NgramLM(str(c))
            log.info("context-aware decoding with LM %s", c)
            break
    _LM_CACHE["lm"] = lm
    return lm


def decode_track(
    logprobs: np.ndarray,
    vocab: Sequence[str],
    blank: int,
    t0: float,
    names: Sequence[str] = (),
    context: bool = True,
) -> list[Utterance]:
    """Decode one track into utterances.

    With ``context`` (and a language model available) the beam-search decoder weighs several candidate
    words using the surrounding words and attaches alternatives; otherwise greedy decoding is used.
    The roster booster then runs if names are given."""
    lm = load_lm() if context else None
    if lm is not None:
        from .beam import BeamDecoder, DecodeOptions
        words, alts = BeamDecoder(vocab, lm, DecodeOptions(names=tuple(names)), blank).decode(logprobs, t0)
        for w, a in zip(words, alts):
            w.p, w.alternatives = a["p"], a["alternatives"]
    else:
        words = greedy_words(logprobs, vocab, blank, t0)
    if names:
        from .booster import Booster  # local import: booster depends on this module

        words = Booster(names, vocab).apply(words, logprobs, t0)
    t_end = t0 + len(logprobs) / FPS
    utts = segment_utterances(words)
    for u in utts:                      # padding must not leave the analysed window
        for w in u.words:
            w.start, w.end = max(w.start, t0), min(w.end, t_end)
        u.start, u.end = u.words[0].start, u.words[-1].end
    return utts
