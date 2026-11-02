"""Roster booster: rescore name candidates against the model's own words with CTC.

For each run of 1-3 consecutive greedy words, the booster asks whether a roster name
explains the same frames better than the words the model wrote. Both texts are scored
with the CTC forward algorithm over the same padded frame span. A name replaces the
window only when ``LL(name) + weight > LL(original)``. Replacements never overlap, and
the largest gain wins. Replaced words are flagged ``boosted=True`` and keep the model's
text in ``original``. Boosting never raises confidence.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from importlib import resources
from typing import Sequence

import numpy as np
import sentencepiece as spm

from .decode import emission_runs, silent_mask, tier_for, word_frames
from .types import Word

log = logging.getLogger("lipread")

MAX_WINDOW = 3
_NEG_INF = -math.inf


def ctc_log_likelihood(logprobs_span: np.ndarray, target_ids: Sequence[int], blank: int) -> float:
    """Return log P(target | span) under CTC, via the forward algorithm in log space.

    logprobs_span: (T, V) per-frame log-probabilities. target_ids: label indices (no blanks).
    Returns -inf when the span is too short for the target (including the blanks CTC
    needs between repeated labels).
    """
    lp = np.asarray(logprobs_span, dtype=np.float64)
    T = lp.shape[0]
    target = np.asarray(target_ids, dtype=np.int64)
    L = len(target)
    if L == 0:
        return float(lp[:, blank].sum())
    repeats = int(np.sum(target[1:] == target[:-1]))
    if T < L + repeats:
        return _NEG_INF

    ext = np.full(2 * L + 1, blank, dtype=np.int64)
    ext[1::2] = target
    S = ext.shape[0]
    # The skip transition s-2 -> s is allowed onto a non-blank label that differs from s-2.
    skip_ok = np.zeros(S, dtype=bool)
    skip_ok[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])

    alpha = np.full(S, _NEG_INF)
    alpha[0] = lp[0, blank]
    alpha[1] = lp[0, ext[1]]
    emit = lp[:, ext]                       # (T, S)
    for t in range(1, T):
        prev1 = np.concatenate(([_NEG_INF], alpha[:-1]))
        prev2 = np.concatenate(([_NEG_INF, _NEG_INF], np.where(skip_ok[2:], alpha[:-2], _NEG_INF)))
        alpha = np.logaddexp(alpha, np.logaddexp(prev1, prev2)) + emit[t]
    return float(np.logaddexp(alpha[-1], alpha[-2]))


def _min_frames(target: np.ndarray) -> int:
    """Fewest frames CTC needs to emit ``target``: one per label plus a blank between repeats."""
    if len(target) == 0:
        return 0
    return len(target) + int(np.sum(target[1:] == target[:-1]))


@dataclass(frozen=True)
class _Candidate:
    text: str
    ids: np.ndarray
    min_len: int           # fewest frames CTC needs for ids
    words: frozenset[str]  # the candidate's own words


@dataclass
class _Proposal:
    gain: float
    i: int
    j: int               # window is words[i:j]
    text: str
    ll: float
    span_len: int
    original: str
    orig_conf: float


class Booster:
    """Replace greedy words by roster names where CTC says the name fits the frames better.

    names:  roster entries, e.g. ``["Travis Kelce", "Gerald Youngblood"]``. Multi-word
            names are also tried word by word.
    vocab:  the engine vocabulary (index -> piece).
    weight: log-likelihood bonus a name needs over the model's words (nats).
    pad:    frames of context added on each side of the window when scoring.
    """

    def __init__(self, names: Sequence[str], vocab: Sequence[str], weight: float = 3.0, pad: int = 3):
        self.vocab = list(vocab)
        self.weight = float(weight)
        self.pad = int(pad)
        self.blank = self.vocab.index("<blank>") if "<blank>" in self.vocab else 0
        self._silent = silent_mask(self.vocab, self.blank)
        model = resources.files("lipread.assets").joinpath("unigram5000.model").read_bytes()
        self._sp = spm.SentencePieceProcessor(model_proto=model)
        self._index: dict[str, int] = {}
        for i, piece in enumerate(self.vocab):
            self._index.setdefault(piece, i)
        self.candidates: list[_Candidate] = self._build(names)

    def _build(self, names: Sequence[str]) -> list[_Candidate]:
        seen: dict[str, np.ndarray] = {}
        for raw in names:
            full = " ".join(raw.upper().split())
            if not full:
                continue
            variants = [full]
            parts = full.split(" ")
            if len(parts) > 1:
                variants += parts
            for text in variants:
                if text in seen:
                    continue
                ids = self._encode(text)
                if ids is None:
                    log.warning("booster: cannot tokenize roster entry %r, skipped", text)
                    continue
                seen[text] = ids
        return [_Candidate(text, ids, _min_frames(ids), frozenset(text.split(" ")))
                for text, ids in seen.items()]

    def _encode(self, text: str) -> np.ndarray | None:
        ids: list[int] = []
        for piece in self._sp.encode(text, out_type=str):
            if piece.startswith("<"):          # <unk>: the name has characters the model cannot write
                return None
            k = self._index.get(piece)
            if k is None:
                return None
            ids.append(k)
        return np.array(ids, dtype=np.int64) if ids else None

    def apply(self, words: Sequence[Word], logprobs: np.ndarray, t0: float) -> list[Word]:
        """Return the words with roster names substituted where the CTC scores favour them.

        logprobs: the track's (T, V) log-probabilities, frame 0 at source time ``t0``.
        """
        words = list(words)
        if not words or not self.candidates:
            return words
        lp = np.asarray(logprobs, dtype=np.float64)
        T = lp.shape[0]
        n = len(words)
        frames = [word_frames(w, t0) for w in words]

        proposals: list[_Proposal] = []
        for i in range(n):
            for j in range(i + 1, min(n, i + MAX_WINDOW) + 1):
                a = max(0, frames[i][0])
                b = min(T, frames[j - 1][1])
                if b <= a:
                    continue
                # Padding may not reach into a neighbouring word's emitted frames,
                # otherwise a candidate can claim the next word's spike (duplicates).
                left = min(a, max(0, frames[i - 1][1])) if i > 0 else 0
                right = max(b, min(T, frames[j][0])) if j < n else T
                pa, pb = max(left, a - self.pad), min(right, b + self.pad)
                span = lp[pa:pb]
                original = " ".join(w.text for w in words[i:j])
                orig_ids = np.array([k for k, _, _ in emission_runs(lp[a:b], self._silent)], dtype=np.int64)
                orig_ll = ctc_log_likelihood(span, orig_ids, self.blank)
                best: tuple[float, _Candidate, float] | None = None
                span_frames = pb - pa
                orig_words = frozenset(original.split(" "))
                for cand in self.candidates:
                    if cand.text == original or cand.min_len > span_frames:
                        continue
                    # R1: never insert words. R2: a subset of the window's words is a
                    # silent deletion of the rest of the window, not a name.
                    if len(cand.words) > j - i or cand.words <= orig_words:
                        continue
                    ll = ctc_log_likelihood(span, cand.ids, self.blank)
                    if ll == _NEG_INF:
                        continue
                    gain = ll + self.weight - orig_ll
                    if gain > 0 and (best is None or gain > best[0]):
                        best = (gain, cand, ll)
                if best is not None:
                    gain, cand, ll = best
                    proposals.append(_Proposal(
                        gain=gain, i=i, j=j, text=cand.text, ll=ll, span_len=span_frames,
                        original=original,
                        orig_conf=float(np.mean([w.confidence for w in words[i:j]])),
                    ))

        proposals.sort(key=lambda p: p.gain, reverse=True)
        taken = np.zeros(n, dtype=bool)
        accepted: list[_Proposal] = []
        for p in proposals:
            if taken[p.i:p.j].any():
                continue
            taken[p.i:p.j] = True
            accepted.append(p)
        accepted.sort(key=lambda p: p.i)

        out: list[Word] = []
        pos = 0
        for p in accepted:
            out.extend(words[pos:p.i])
            conf = min(p.orig_conf, math.exp(p.ll / p.span_len))
            out.append(Word(
                text=p.text,
                start=words[p.i].start,
                end=words[p.j - 1].end,
                confidence=conf,
                tier=tier_for(conf),
                boosted=True,
                original=p.original,
            ))
            pos = p.j
        out.extend(words[pos:])
        return out
