"""Word trigram language model (interpolated Kneser-Ney), as built by scripts/build_lm.py.

Used by the beam-search decoder to judge which word sequence is most plausible in context.
The same binary file is read by the phone app (web/core/lm.js).
"""
from __future__ import annotations

import json
import struct
from functools import lru_cache

import numpy as np

MAGIC = b"LRLM1\n"
BOS, EOS, UNK = 0, 1, 2


class NgramLM:
    def __init__(self, path: str):
        buf = open(path, "rb").read()
        if not buf.startswith(MAGIC):
            raise ValueError(f"{path} is not a LipRead LM file")
        off = len(MAGIC)
        hlen = struct.unpack_from("<I", buf, off)[0]; off += 4
        self.header = h = json.loads(buf[off:off + hlen]); off += hlen
        wlen = struct.unpack_from("<I", buf, off)[0]; off += 4
        self.words = buf[off:off + wlen].decode().split("\n"); off += wlen
        off = (off + 3) & ~3
        V, B, T, q = h["V"], h["B"], h["T"], h["qscale"]

        def take(dtype, n):
            nonlocal off
            a = np.frombuffer(buf, dtype=dtype, count=n, offset=off)
            off += a.nbytes
            off = (off + 3) & ~3
            return a

        self.uni_lp = take(np.float32, V).astype(np.float64)
        self.uni_bow = take(np.float32, V).astype(np.float64)
        self.bi_start = take(np.uint32, V + 1)
        self.bi_w = take(np.uint16, B)
        self.bi_lp = take(np.int16, B) / q
        self.bi_bow = take(np.int16, B) / q
        self.tri_start = take(np.uint32, B + 1)
        self.tri_w = take(np.uint16, T)
        self.tri_lp = take(np.int16, T) / q
        self.wid = {w: i for i, w in enumerate(self.words)}
        self.score = lru_cache(maxsize=200_000)(self._score)

    def id(self, word: str) -> int:
        return self.wid.get(word, UNK)

    def _find(self, start: np.ndarray, keys: np.ndarray, row: int, w: int) -> int:
        a, b = int(start[row]), int(start[row + 1])
        if a == b:
            return -1
        i = a + int(np.searchsorted(keys[a:b], w))
        return i if i < b and keys[i] == w else -1

    def _bigram(self, v: int, w: int) -> float:
        i = self._find(self.bi_start, self.bi_w, v, w)
        return self.bi_lp[i] if i >= 0 else self.uni_bow[v] + self.uni_lp[w]

    def _score(self, w: int, v: int | None = None, u: int | None = None) -> float:
        """ln P(w | u v); v = previous word id, u = the word before it (None = unknown/none)."""
        if v is None:
            return float(self.uni_lp[w])
        if u is None:
            return float(self._bigram(v, w))
        bi = self._find(self.bi_start, self.bi_w, u, v)
        if bi >= 0:
            t = self._find(self.tri_start, self.tri_w, bi, w)
            if t >= 0:
                return float(self.tri_lp[t])
            return float(self.bi_bow[bi] + self._bigram(v, w))
        return float(self._bigram(v, w))

    def sentence_logprob(self, words: list[str]) -> float:
        u, v, total = None, BOS, 0.0
        for w in words + ["</s>"]:
            i = self.id(w) if w != "</s>" else EOS
            total += self.score(i, v, u)
            u, v = v, i
        return total
