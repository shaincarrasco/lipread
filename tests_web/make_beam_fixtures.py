#!/usr/bin/env python3
"""Generate the Python reference outputs for the browser beam decoder and LM (tests_web/beam.test.mjs).

    /home/claude/.venv/bin/python tests_web/make_beam_fixtures.py

Writes into tests_web/fixtures/:
  beam_<clip>.bin      per clip: the top-32 log-probs of every frame, as u16 indices (T*32) then f32 values (T*32)
  beam_refs.json       clip metadata, option sets, and the Python BeamDecoder output for each clip and option set
  beam_lm_triples.json 2000 (w, v, u) triples with NgramLM.score() values, plus word ids

The dense (T, V) matrix that both sides decode is rebuilt from the top-32 entries: every other entry is -50.
The LM read is web/core/assets/lm.bin (a copy of /home/claude/data/lm/lm_ted.bin, the file that ships).
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lipread.beam import BeamDecoder, DecodeOptions  # noqa: E402
from lipread.engine import load_vocab  # noqa: E402
from lipread.lm import NgramLM  # noqa: E402

CACHE = Path("/home/claude/data/lp_cache.pkl")
LM_PATH = ROOT / "web" / "core" / "assets" / "lm.bin"
OUT = ROOT / "tests_web" / "fixtures"
K = 32          # stored entries per frame
FILL = -50.0    # value of every entry that is not stored
T0_STEP = 0.5
OPTION_SETS = {
    "plain": {"beam": 16, "alpha": 0.3, "beta": 1.0, "posterior_scale": 0.2, "temperature": 1.0},
    "names_t2": {"beam": 16, "alpha": 0.3, "beta": 1.0, "posterior_scale": 0.2, "temperature": 2.0,
                 "names": ["Mahomes", "Kelce"]},
}
N_LRS3, N_LRS2 = 12, 8


def dense_from_topk(idx: np.ndarray, vals: np.ndarray, V: int) -> np.ndarray:
    dense = np.full((idx.shape[0], V), FILL, np.float32)
    np.put_along_axis(dense, idx.astype(np.int64), vals, axis=1)
    return dense


def topk_rows(x: np.ndarray, k: int = K) -> tuple[np.ndarray, np.ndarray]:
    """Top-k per frame, by value (descending), ties by lower index."""
    idx = np.argsort(-x, axis=1, kind="stable")[:, :k]
    return idx.astype(np.uint16), np.take_along_axis(x, idx, 1).astype(np.float32)


def word_dict(w) -> dict:
    return {
        "text": w.text, "start": w.start, "end": w.end, "confidence": w.confidence,
        "tier": w.tier.value, "p": w.p, "alternatives": w.alternatives,
    }


def make_clips(vocab: list[str], lm: NgramLM, data: dict) -> dict:
    V = len(vocab)
    OUT.mkdir(parents=True, exist_ok=True)
    picks = [("lrs3", i) for i in np.linspace(0, len(data["lrs3"]) - 1, N_LRS3).astype(int)]
    picks += [("lrs2", i) for i in np.linspace(0, len(data["lrs2"]) - 1, N_LRS2).astype(int)]
    clips = []
    for n, (src, i) in enumerate(picks):
        raw, label = data[src][int(i)]
        x = np.asarray(raw, dtype=np.float32)
        idx, vals = topk_rows(x)
        T = x.shape[0]
        clip_id = f"{src}_{n:02d}"
        fname = f"beam_{clip_id}.bin"
        (OUT / fname).write_bytes(idx.astype("<u2").tobytes() + vals.astype("<f4").tobytes())
        dense = dense_from_topk(idx, vals, V)
        t0 = round(T0_STEP * (n % 4), 3)
        results = {}
        for set_name, opts in OPTION_SETS.items():
            dec = BeamDecoder(vocab, lm, DecodeOptions(**opts), blank=0)
            words, alts = dec.decode(dense, t0)
            for w, a in zip(words, alts):
                w.p, w.alternatives = a["p"], a["alternatives"]
            results[set_name] = {
                "words": [word_dict(w) for w in words],
                "alternatives": alts,
            }
            print(f"{clip_id} {set_name}: {len(words)} words: {' '.join(w.text for w in words)}")
        clips.append({
            "id": clip_id, "source": src, "index": int(i), "file": fname, "T": T, "t0": t0,
            "label": label, "results": results,
        })
    return {"K": K, "fill": FILL, "V": V, "option_sets": OPTION_SETS, "clips": clips}


def make_lm_triples(lm: NgramLM, V: int, rng: np.random.Generator) -> dict:
    bi_start, bi_w, tri_start, tri_w = lm.bi_start, lm.bi_w, lm.tri_start, lm.tri_w
    B, T = len(bi_w), len(tri_w)
    triples: list[list] = []

    def ctx_of_bigram(b: int) -> tuple[int, int]:
        u = int(np.searchsorted(bi_start, b, side="right") - 1)   # row u holds bigram b
        return u, int(bi_w[b])

    # 800 trigram hits: (u, v, w) present in the table
    for t in rng.integers(0, T, 800):
        b = int(np.searchsorted(tri_start, t, side="right") - 1)
        u, v = ctx_of_bigram(b)
        triples.append([int(tri_w[t]), v, u])
    # 500 seen bigram contexts with a random continuation (hit or back-off)
    for b in rng.integers(0, B, 500):
        u, v = ctx_of_bigram(int(b))
        triples.append([int(rng.integers(0, V)), v, u])
    # 500 fully random ids (mostly unseen contexts)
    for _ in range(500):
        triples.append([int(rng.integers(0, V)), int(rng.integers(0, V)), int(rng.integers(0, V))])
    # 200 with missing context: 100 unigram (v = None), 100 bigram (u = None)
    for k in range(200):
        if k % 2 == 0:
            triples.append([int(rng.integers(0, V)), None, int(rng.integers(0, V))])
        else:
            triples.append([int(rng.integers(0, V)), int(rng.integers(0, V)), None])
    assert len(triples) == 2000
    scores = [lm.score(w, v, u) for w, v, u in triples]

    words = lm.words
    sample = [words[int(i)] for i in rng.integers(0, len(words), 300)]
    sample += ["XYZZY", "MAHOMES", "KELCE", "", "NOTAWORDATALL"]
    return {
        "V": V,
        "triples": triples,
        "scores": scores,
        "word_ids": [[w, lm.id(w)] for w in sample],
    }


def main() -> None:
    vocab = load_vocab()
    assert len(vocab) == 5049, len(vocab)
    lm = NgramLM(str(LM_PATH))
    rng = np.random.default_rng(20261102)
    print("loading", CACHE, file=sys.stderr)
    data = pickle.load(open(CACHE, "rb"))

    refs = make_clips(vocab, lm, data)
    (OUT / "beam_refs.json").write_text(json.dumps(refs, ensure_ascii=False, indent=1) + "\n")
    triples = make_lm_triples(lm, lm.header["V"], rng)
    (OUT / "beam_lm_triples.json").write_text(json.dumps(triples, ensure_ascii=False) + "\n")
    print(f"wrote {len(refs['clips'])} clips and {len(triples['triples'])} LM triples to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
