"""Context-aware decoding: CTC prefix beam search with a word language model (shallow fusion).

Instead of committing to the single most likely piece at every frame (greedy), the decoder keeps the
`beam` best partial transcripts alive. Every time a word is completed, its plausibility in context
(the trigram LM, given the two previous words) is added to the hypothesis score:

    score = log P_model(pieces) + alpha * log P_LM(words) + beta * n_words

At the end it returns the best transcript plus, for every word, the alternatives that other strong
hypotheses put at the same moment, with posterior probabilities (shown as "could also be" in the UI).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .decode import tier_for
from .lm import BOS, UNK, NgramLM
from .types import FPS, Word

MARK = "▁"
NEG = -1e30


def _lse(a: float, b: float) -> float:
    if a < b:
        a, b = b, a
    return a if b <= NEG else a + math.log1p(math.exp(b - a))


@dataclass
class DecodeOptions:
    beam: int = 16
    alpha: float = 0.6          # LM weight (tuned on LRS2, see eval/beam_tune.py)
    beta: float = 1.0           # word insertion bonus
    unk_penalty: float = -4.0   # extra log-penalty for words outside the LM vocabulary
    prune: float = 10.0         # per frame, ignore pieces this far (nats) below the best
    topk: int = 24              # per frame, at most this many pieces
    n_alternatives: int = 3
    temperature: float = 1.0    # >1 softens the model's (over-confident) posteriors so context can weigh in
    names: Sequence[str] = ()   # roster names: treated as in-vocabulary with a flat probability
    name_logp: float = math.log(1e-4)


@dataclass
class _Hyp:
    pb: float = NEG             # log prob, ending in blank
    pnb: float = NEG            # log prob, ending in a piece
    lm: float = 0.0             # alpha * LM + beta * words for completed words
    hist: tuple = (None, BOS)   # (u, v) word ids of the last two completed words
    cur: str = ""               # current (incomplete) word
    cur_start: int = -1
    words: tuple = ()           # completed words: (text, start_frame, end_frame, conf_sum, conf_n)
    last_frame: int = -1
    conf_sum: float = 0.0
    conf_n: int = 0

    @property
    def total(self) -> float:
        return _lse(self.pb, self.pnb)


class BeamDecoder:
    def __init__(self, vocab: Sequence[str], lm: Optional[NgramLM], opts: DecodeOptions = DecodeOptions(), blank: int = 0):
        self.vocab, self.lm, self.o, self.blank = list(vocab), lm, opts, blank
        self.silent = np.array([i == blank or (p.startswith("<") and p.endswith(">")) for i, p in enumerate(self.vocab)])
        self.starts_word = np.array([p.startswith(MARK) for p in self.vocab])
        self.text = [p.replace(MARK, "") for p in self.vocab]
        self.names = {n.upper() for name in opts.names for n in [name, *name.split()] if n.strip()}

    # ------------------------------------------------------------------ LM
    def _word_score(self, word: str, hist: tuple) -> tuple[float, tuple]:
        o = self.o
        if self.lm is None:
            return o.beta, hist
        wid = self.lm.id(word)
        lp = self.lm.score(wid, hist[1], hist[0])
        if wid == UNK:
            lp += o.unk_penalty
            if word in self.names:
                lp = max(lp, o.name_logp)
        elif word in self.names:
            lp = max(lp, o.name_logp)
        return o.alpha * lp + o.beta, (hist[1], wid)

    def _complete(self, h: _Hyp) -> _Hyp:
        """Return a copy of h with its current word completed (LM applied)."""
        if not h.cur:
            return h
        s, hist = self._word_score(h.cur, h.hist)
        conf = (h.conf_sum, h.conf_n)
        return _Hyp(h.pb, h.pnb, h.lm + s, hist, "", -1,
                    h.words + ((h.cur, h.cur_start, h.last_frame + 1, *conf),), h.last_frame, 0.0, 0)

    # ------------------------------------------------------------------ search
    def decode(self, logprobs: np.ndarray, t0: float = 0.0) -> tuple[list[Word], list[dict]]:
        lp = np.asarray(logprobs, dtype=np.float64)
        T, V = lp.shape
        o = self.o
        conf_lp = lp                       # word confidence always uses the raw (calibrated) posteriors
        if o.temperature != 1.0:
            z = lp / o.temperature
            z = z - z.max(axis=1, keepdims=True)
            lp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        beams: dict[tuple, _Hyp] = {(): _Hyp(pb=0.0)}
        for t in range(T):
            row = lp[t]
            best = row.max()
            cand = np.argpartition(-row, min(o.topk, V - 1))[: o.topk]
            cand = [int(c) for c in cand if row[c] >= best - o.prune and not self.silent[c]]
            nxt: dict[tuple, _Hyp] = {}

            def get(key, parent: _Hyp) -> _Hyp:
                h = nxt.get(key)
                if h is None:
                    h = _Hyp(NEG, NEG, parent.lm, parent.hist, parent.cur, parent.cur_start, parent.words,
                             parent.last_frame, parent.conf_sum, parent.conf_n)
                    nxt[key] = h
                return h

            for key, h in beams.items():
                tot = h.total
                # blank: same prefix
                nb = get(key, h); nb.pb = _lse(nb.pb, tot + row[self.blank])
                # repeated last piece without a blank between: same prefix
                if key:
                    last = key[-1]
                    s = get(key, h); s.pnb = _lse(s.pnb, h.pnb + row[last])
                for c in cand:
                    p = (h.pb if key and key[-1] == c else tot) + row[c]
                    nkey = key + (c,)
                    if nkey in nxt:
                        nh = nxt[nkey]
                    else:
                        base = self._complete(h) if (self.starts_word[c] and h.cur) else h
                        nh = _Hyp(NEG, NEG, base.lm, base.hist, base.cur, base.cur_start, base.words,
                                  base.last_frame, base.conf_sum, base.conf_n)
                        if self.starts_word[c] or not nh.cur:
                            nh.cur, nh.cur_start, nh.conf_sum, nh.conf_n = self.text[c], t, 0.0, 0
                        else:
                            nh.cur = nh.cur + self.text[c]
                        nh.last_frame = t
                        nh.conf_sum += math.exp(conf_lp[t, c]); nh.conf_n += 1
                        nxt[nkey] = nh
                    nh.pnb = _lse(nh.pnb, p)
            beams = dict(sorted(nxt.items(), key=lambda kv: kv[1].total + kv[1].lm, reverse=True)[: o.beam])

        finals = []
        for key, h in beams.items():
            f = self._complete(h)
            finals.append((f.total + f.lm, f))
        finals.sort(key=lambda x: -x[0])
        return self._words(finals, t0)

    # ------------------------------------------------------------------ output
    def _words(self, finals, t0):
        if not finals:
            return [], []
        scores = np.array([s for s, _ in finals])
        post = np.exp(scores - scores.max()); post /= post.sum()
        best = finals[0][1]
        words, alts = [], []
        for (text, a, b, csum, cn) in best.words:
            if not text:
                continue
            conf = csum / max(cn, 1)
            # alternatives: words other hypotheses place at (about) the same moment
            options: dict[str, float] = {}
            for p, (_, h) in zip(post, finals):
                for (wt, wa, wb, *_r) in h.words:
                    if abs(wa - a) <= 3 or (wa < b and wb > a and abs(wa - a) <= 6):
                        options[wt] = options.get(wt, 0.0) + p
                        break
            ranked = sorted(options.items(), key=lambda kv: -kv[1])
            alt = [{"text": w, "p": round(float(pp), 4)} for w, pp in ranked if w != text][: self.o.n_alternatives]
            words.append(Word(text=text, start=t0 + a / FPS, end=t0 + b / FPS, confidence=conf, tier=tier_for(conf)))
            alts.append({"p": round(float(options.get(text, 0.0)), 4), "alternatives": alt})
        return words, alts
