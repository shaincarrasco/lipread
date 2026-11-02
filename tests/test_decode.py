"""CTC greedy decoding: word timing, utterance segmentation, track clamping and tiers."""
from __future__ import annotations

import numpy as np
import pytest

from lipread.decode import (CONF_HIGH, CONF_MEDIUM, decode_track, greedy_words, segment_utterances,
                            tier_for)
from lipread.engine import StubEngine, load_vocab
from lipread.types import FPS, Confidence, Word

VOCAB = load_vocab()
BLANK = 0


def _stub_logprobs(script: str, T: int, conf: float = 0.95) -> np.ndarray:
    return StubEngine(script, confidence=conf).logprobs(np.zeros((T, 88, 88), np.uint8))


def test_greedy_words_timing_order_and_window():
    T = 150                                   # 6 s at 25 fps
    t0 = 10.0
    lp = _stub_logprobs("HELLO WORLD", T)
    words = greedy_words(lp, VOCAB, BLANK, t0)
    assert [w.text for w in words] == ["HELLO", "WORLD"]
    t_end = t0 + T / FPS
    for w in words:
        assert t0 <= w.start < w.end <= t_end
    assert words[0].end <= words[1].start           # words do not overlap
    # HELLO's spike sits in the first half, WORLD's in the second
    assert words[0].start < t0 + 3.0 < words[1].start


def test_greedy_words_rejects_wrong_vocab_width():
    with pytest.raises(ValueError):
        greedy_words(np.zeros((10, 3), np.float32), VOCAB, BLANK, 0.0)


def test_greedy_words_on_all_blank_is_empty():
    lp = np.full((50, len(VOCAB)), np.log(1e-6), np.float32)
    lp[:, BLANK] = 0.0
    assert greedy_words(lp, VOCAB, BLANK, 0.0) == []


def _w(text: str, start: float, end: float) -> Word:
    return Word(text=text, start=start, end=end, confidence=0.9, tier=Confidence.HIGH)


def test_segment_utterances_splits_on_long_gap():
    words = [_w("ONE", 1.0, 1.1), _w("TWO", 1.3, 1.4),      # 0.2 s gap: same utterance
             _w("THREE", 3.0, 3.1)]                          # 1.6 s gap: new utterance
    utts = segment_utterances(words, gap=0.6)
    assert [u.text for u in utts] == ["ONE TWO", "THREE"]


def test_segment_utterances_gap_is_strict():
    words = [_w("A", 0.0, 0.1), _w("B", 0.8, 0.9)]          # gap exactly 0.7 s
    assert len(segment_utterances(words, gap=0.6)) == 2
    assert len(segment_utterances(words, gap=0.8)) == 1


def test_segment_utterances_empty():
    assert segment_utterances([]) == []


def test_decode_track_clamps_to_window_with_nonzero_t0():
    T = 100                                   # 4 s window
    t0 = 5.0
    lp = _stub_logprobs("HELLO WORLD", T)
    utts = decode_track(lp, VOCAB, BLANK, t0)
    assert utts, "expected at least one utterance"
    lo, hi = t0, t0 + T / FPS
    for u in utts:
        assert lo <= u.start <= u.end <= hi
        for w in u.words:
            assert lo <= w.start <= w.end <= hi
    assert " ".join(w.text for u in utts for w in u.words) == "HELLO WORLD"


def test_decode_track_padding_is_clamped_at_both_edges():
    """Edge padding (0.25 s) must not leave the analysed window, even when the first and last spikes sit on the edges."""
    T = 40
    t0 = 2.0
    lp = np.full((T, len(VOCAB)), np.log(1e-6), np.float32)
    lp[:, BLANK] = np.log(0.999)
    idx = VOCAB.index("▁HELLO")
    lp[0:2, :] = np.log(1e-6)
    lp[0:2, idx] = np.log(0.999)              # spike right at the first frames
    lp[-2:, :] = np.log(1e-6)
    lp[-2:, idx] = np.log(0.999)
    utts = decode_track(lp, VOCAB, BLANK, t0)
    assert utts
    for u in utts:
        assert t0 <= u.start and u.end <= t0 + T / FPS
        assert all(t0 <= w.start <= w.end <= t0 + T / FPS for w in u.words)


@pytest.mark.parametrize("conf,tier", [
    (0.99, Confidence.HIGH),
    (CONF_HIGH, Confidence.HIGH),             # threshold is inclusive
    (0.8, Confidence.MEDIUM),
    (CONF_MEDIUM, Confidence.MEDIUM),
    (0.69, Confidence.LOW),
    (0.0, Confidence.LOW),
])
def test_tier_for_thresholds(conf, tier):
    assert tier_for(conf) == tier


@pytest.mark.parametrize("engine_conf,tier", [(0.95, Confidence.HIGH), (0.8, Confidence.MEDIUM),
                                              (0.5, Confidence.LOW)])
def test_greedy_word_confidence_maps_to_tier(engine_conf, tier):
    words = greedy_words(_stub_logprobs("HELLO", 60, conf=engine_conf), VOCAB, BLANK, 0.0)
    assert len(words) == 1
    assert words[0].confidence == pytest.approx(engine_conf, rel=1e-3)
    assert words[0].tier == tier
