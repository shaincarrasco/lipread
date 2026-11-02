"""Tests for the word language model and the context-aware beam decoder."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from lipread.beam import BeamDecoder, DecodeOptions
from lipread.decode import greedy_words
from lipread.engine import StubEngine, load_vocab

ROOT = Path(__file__).resolve().parents[1]
LM_PATHS = [ROOT / "web" / "core" / "assets" / "lm.bin", Path("/home/claude/data/lm/lm_ted.bin")]
LM_PATH = next((p for p in LM_PATHS if p.is_file()), None)
needs_lm = pytest.mark.skipif(LM_PATH is None, reason="lm.bin not built (scripts/build_lm.py)")


@pytest.fixture(scope="module")
def lm():
    from lipread.lm import NgramLM
    return NgramLM(str(LM_PATH))


@needs_lm
@pytest.mark.parametrize("context", [(None, "THE"), ("OF", "THE"), (None, "<s>"), ("I", "THINK")])
def test_next_word_distribution_sums_to_one(lm, context):
    u, v = context
    uid = None if u is None else lm.id(u)
    vid = lm.id(v)
    total = sum(math.exp(lm.score(w, vid, uid)) for w in range(1, len(lm.words)))
    assert total == pytest.approx(1.0, abs=0.06), total


@needs_lm
def test_lm_prefers_fluent_english(lm):
    assert lm.sentence_logprob("I WOULD LOVE TO HEAR FROM YOU".split()) > \
        lm.sentence_logprob("I WOULD LOVE TO HERE FROM YOU".split()) + 2


def test_beam_without_lm_matches_greedy_on_stub():
    vocab = load_vocab()
    lp = StubEngine("HELLO WORLD AGAIN").logprobs(np.zeros((120, 88, 88), np.uint8))
    greedy = [w.text for w in greedy_words(lp, vocab, 0, 0.0)]
    words, alts = BeamDecoder(vocab, None, DecodeOptions(beam=8)).decode(lp)
    assert [w.text for w in words] == greedy == ["HELLO", "WORLD", "AGAIN"]
    assert len(alts) == len(words) and all(0 <= a["p"] <= 1 for a in alts)


def test_word_times_are_inside_the_window():
    vocab = load_vocab()
    lp = StubEngine("HELLO WORLD").logprobs(np.zeros((100, 88, 88), np.uint8))
    words, _ = BeamDecoder(vocab, None).decode(lp, t0=5.0)
    assert all(5.0 <= w.start < w.end <= 5.0 + 100 / 25 for w in words)


@needs_lm
def test_context_picks_the_plausible_word_between_two_candidates(lm):
    """Two pieces equally likely at one moment: the LM must choose the one that fits the sentence."""
    vocab = load_vocab()
    idx = {p: i for i, p in enumerate(vocab)}
    seq = ["▁I", "▁WOULD", "▁LOVE", "▁TO", None, "▁FROM", "▁YOU"]
    T = len(seq) * 6
    lp = np.full((T, len(vocab)), -30.0, np.float32)
    lp[:, 0] = math.log(0.98)
    for k, piece in enumerate(seq):
        f = k * 6 + 2
        lp[f, :] = -30.0
        if piece is None:              # ambiguous slot: HERE vs HEAR, model slightly prefers the wrong one
            lp[f, idx["▁HERE"]] = math.log(0.52)
            lp[f, idx["▁HEAR"]] = math.log(0.46)
        else:
            lp[f, idx[piece]] = math.log(0.98)
    greedy = [w.text for w in greedy_words(lp, vocab, 0, 0.0)]
    words, alts = BeamDecoder(vocab, lm, DecodeOptions()).decode(lp)
    assert "HERE" in greedy
    assert [w.text for w in words] == ["I", "WOULD", "LOVE", "TO", "HEAR", "FROM", "YOU"]
    hear = alts[4]
    assert any(a["text"] == "HERE" for a in hear["alternatives"])
    assert 0.5 < hear["p"] <= 1.0


@needs_lm
def test_roster_names_are_treated_as_known_words(lm):
    vocab = load_vocab()
    dec = BeamDecoder(vocab, lm, DecodeOptions(names=("Mahomes",)))
    s_name, _ = dec._word_score("MAHOMES", (None, 0))
    dec_plain = BeamDecoder(vocab, lm, DecodeOptions())
    s_plain, _ = dec_plain._word_score("MAHOMES", (None, 0))
    assert s_name > s_plain
