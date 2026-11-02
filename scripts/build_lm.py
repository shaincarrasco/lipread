#!/usr/bin/env python3
"""Build LipRead's word trigram language model (interpolated Kneser-Ney) from plain text.

    python scripts/build_lm.py CORPUS.txt -o web/core/assets/lm.bin \
        --exclude-labels test_labels.txt --max-vocab 65000 --min-trigram 2

* One sentence/segment per line, any case (upper-cased internally; only A-Z and ' are kept).
* --exclude-labels: drop every corpus line that shares a 5-word sequence (or equals) a line of this
  file, so a benchmark's test sentences can't leak into the model.
* Output: a compact binary read by lipread.lm (Python) and web/core/lm.js (browser):

    magic "LRLM1\\n" | uint32 header_len | header JSON | words (UTF-8, '\\n'-joined) | padding to 4 |
    uni_lp f32[V] | uni_bow f32[V] |
    bi_start u32[V+1] | bi_w u16[B] (pad) | bi_lp i16[B] (pad) | bi_bow i16[B] (pad) |
    tri_start u32[B+1] | tri_w u16[T] (pad) | tri_lp i16[T] (pad)

  Log-probabilities are natural log. i16 values are quantised: value / QSCALE. Word ids are u16
  (vocabulary capped at 65,535). Ids 0,1,2 are <s>, </s>, <unk>.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
from collections import Counter, defaultdict

import numpy as np

QSCALE = 1000.0
MAGIC = b"LRLM1\n"
CLEAN = re.compile(r"[^A-Z' ]")


def norm(line: str) -> list[str]:
    return CLEAN.sub(" ", line.upper()).split()


def ngrams(ws, n):
    return {" ".join(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def load_corpus(path, exclude_path, n=5):
    lines = [norm(l) for l in open(path, encoding="utf-8", errors="ignore")]
    lines = [l for l in lines if l]
    if not exclude_path:
        return lines, 0
    bad_exact, bad_grams = set(), set()
    for l in open(exclude_path, encoding="utf-8"):
        ws = norm(l)
        if not ws:
            continue
        bad_exact.add(" ".join(ws))
        bad_grams |= ngrams(ws, n)
    kept = [l for l in lines if " ".join(l) not in bad_exact and not (ngrams(l, n) & bad_grams)]
    return kept, len(lines) - len(kept)


def build(lines, max_vocab, min_tri, min_bi):
    wc = Counter(w for l in lines for w in l)
    vocab = ["<s>", "</s>", "<unk>"] + [w for w, _ in wc.most_common(max_vocab - 3)]
    wid = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    UNK = 2

    c3, c2 = Counter(), Counter()
    for l in lines:
        ids = [0] + [wid.get(w, UNK) for w in l] + [1]
        for i in range(1, len(ids)):
            c2[(ids[i - 1], ids[i])] += 1
            if i >= 2:
                c3[(ids[i - 2], ids[i - 1], ids[i])] += 1

    def discount(counts):
        n1 = sum(1 for c in counts.values() if c == 1)
        n2 = sum(1 for c in counts.values() if c == 2)
        return n1 / (n1 + 2 * n2) if n1 + n2 else 0.5

    D3, D2 = discount(c3), None
    # continuation counts for bigrams: number of distinct left words u for (v, w)
    cc2 = Counter()
    for (u, v, w) in c3:
        cc2[(v, w)] += 1
    for (v, w), c in c2.items():       # bigrams starting a sentence have no trigram: count them directly
        if v == 0:
            cc2[(v, w)] = c
    D2 = discount(cc2)
    # continuation counts for unigrams: distinct left words v for w
    cc1 = Counter()
    for (v, w) in cc2:
        cc1[w] += 1
    total_cc1 = sum(cc1.values())

    p1 = np.full(V, 1e-12)
    for w, c in cc1.items():
        p1[w] = c / total_cc1
    p1[0] = 1e-12                       # <s> is never predicted
    p1 = 0.999 * p1 / p1.sum() + 0.001 / V   # smooth with uniform
    uni_lp = np.log(p1).astype(np.float32)

    # bigram level (from continuation counts)
    ctx2_tot = defaultdict(float)
    ctx2_types = Counter()
    for (v, w), c in cc2.items():
        ctx2_tot[v] += c
        ctx2_types[v] += 1
    bi = {}
    for (v, w), c in cc2.items():
        if c < min_bi and v != 0:
            continue
        gamma = D2 * ctx2_types[v] / ctx2_tot[v]
        bi[(v, w)] = math.log(max(c - D2, 0) / ctx2_tot[v] + gamma * p1[w])
    uni_bow = np.zeros(V, np.float32)
    for v, tot in ctx2_tot.items():
        uni_bow[v] = math.log(D2 * ctx2_types[v] / tot)

    # trigram level (raw counts)
    ctx3_tot = defaultdict(float)
    ctx3_types = Counter()
    for (u, v, w), c in c3.items():
        ctx3_tot[(u, v)] += c
        ctx3_types[(u, v)] += 1

    def p2(v, w):
        if (v, w) in bi:
            return math.exp(bi[(v, w)])
        return math.exp(uni_bow[v]) * p1[w] if ctx2_tot.get(v) else p1[w]

    tri = {}
    for (u, v, w), c in c3.items():
        if c < min_tri:
            continue
        gamma = D3 * ctx3_types[(u, v)] / ctx3_tot[(u, v)]
        tri[(u, v, w)] = math.log(max(c - D3, 0) / ctx3_tot[(u, v)] + gamma * p2(v, w))
    bi_bow = {}
    for (u, v), tot in ctx3_tot.items():
        if (u, v) in bi:
            bi_bow[(u, v)] = math.log(D3 * ctx3_types[(u, v)] / tot)
    # trigrams need their bigram context present in the table
    tri = {k: lp for k, lp in tri.items() if (k[0], k[1]) in bi}
    return vocab, uni_lp, uni_bow, bi, bi_bow, tri, {"D2": D2, "D3": D3}


def write(path, vocab, uni_lp, uni_bow, bi, bi_bow, tri, meta):
    V = len(vocab)
    assert V <= 65535
    bkeys = sorted(bi)
    bindex = {k: i for i, k in enumerate(bkeys)}
    bi_start = np.zeros(V + 1, np.uint32)
    for v, _ in bkeys:
        bi_start[v + 1] += 1
    bi_start = np.cumsum(bi_start).astype(np.uint32)
    bi_w = np.array([w for _, w in bkeys], np.uint16)
    q = lambda x: np.clip(np.round(np.asarray(x) * QSCALE), -32767, 32767).astype(np.int16)
    bi_lp = q([bi[k] for k in bkeys])
    bi_bw = q([bi_bow.get(k, 0.0) for k in bkeys])
    tkeys = sorted(tri, key=lambda k: (bindex[(k[0], k[1])], k[2]))
    tri_start = np.zeros(len(bkeys) + 1, np.uint32)
    for k in tkeys:
        tri_start[bindex[(k[0], k[1])] + 1] += 1
    tri_start = np.cumsum(tri_start).astype(np.uint32)
    tri_w = np.array([k[2] for k in tkeys], np.uint16)
    tri_lp = q([tri[k] for k in tkeys])

    header = {"order": 3, "V": V, "B": len(bkeys), "T": len(tkeys), "qscale": QSCALE, **meta}
    hj = json.dumps(header).encode()
    words = "\n".join(vocab).encode()
    with open(path, "wb") as f:
        def pad():
            while f.tell() % 4:
                f.write(b"\0")
        f.write(MAGIC); f.write(struct.pack("<I", len(hj))); f.write(hj); f.write(struct.pack("<I", len(words))); f.write(words); pad()
        for arr in (uni_lp, uni_bow, bi_start, bi_w, bi_lp, bi_bw, tri_start, tri_w, tri_lp):
            f.write(np.ascontiguousarray(arr).tobytes()); pad()
    return header


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--exclude-labels")
    ap.add_argument("--max-vocab", type=int, default=65000)
    ap.add_argument("--min-trigram", type=int, default=2)
    ap.add_argument("--min-bigram", type=int, default=1)
    a = ap.parse_args(argv)
    lines, dropped = load_corpus(a.corpus, a.exclude_labels)
    print(f"corpus lines kept {len(lines)} (dropped {dropped} for overlap with excluded labels), "
          f"words {sum(map(len, lines))}", file=sys.stderr)
    vocab, ul, ub, bi, bb, tri, meta = build(lines, a.max_vocab, a.min_trigram, a.min_bigram)
    meta.update({"source": a.corpus.split("/")[-1], "dropped_lines": dropped, "kept_lines": len(lines)})
    h = write(a.out, vocab, ul, ub, bi, bb, tri, meta)
    import os
    print(f"wrote {a.out}: V={h['V']} bigrams={h['B']} trigrams={h['T']} size={os.path.getsize(a.out)/1e6:.1f} MB", file=sys.stderr)


if __name__ == "__main__":
    main()
