"""Window stitching and the stub engine."""
from __future__ import annotations

import numpy as np
import pytest

from lipread.decode import greedy_words
from lipread.engine import StubEngine, load_vocab, stitch_windows
from lipread.types import CHUNK_FRAMES, CHUNK_OVERLAP

V = 7  # tiny vocabulary for the stitching tests


def _position_value(t: np.ndarray) -> np.ndarray:
    """A posterior-like row per frame whose values depend only on the absolute frame index."""
    base = np.stack([t * 0.01 + k for k in range(V)], axis=1)
    return base.astype(np.float32)


def _window_runner(calls: list[tuple[int, int]] | None = None):
    def run(s: int, e: int) -> np.ndarray:
        if calls is not None:
            calls.append((s, e))
        return _position_value(np.arange(s, e))
    return run


def _single_pass(T: int) -> np.ndarray:
    return _position_value(np.arange(T))


@pytest.mark.parametrize("T", [1, 7, 39])
def test_stitch_shorter_than_chunk_is_single_pass(T):
    calls: list[tuple[int, int]] = []
    out = stitch_windows(T, _window_runner(calls), chunk=40, overlap=10)
    assert calls == [(0, T)]
    np.testing.assert_allclose(out, _single_pass(T))


def test_stitch_equals_single_pass_when_exactly_one_chunk():
    T = 40
    calls: list[tuple[int, int]] = []
    out = stitch_windows(T, _window_runner(calls), chunk=40, overlap=10)
    assert calls == [(0, 40)]
    np.testing.assert_allclose(out, _single_pass(T))


@pytest.mark.parametrize("T,chunk,overlap", [
    (41, 40, 10),       # one frame over a chunk
    (120, 40, 10),
    (920, CHUNK_FRAMES, CHUNK_OVERLAP),   # 2.3 chunks with the production constants
    (1000, 40, 10),
])
def test_stitch_multi_window_equals_single_pass(T, chunk, overlap):
    calls: list[tuple[int, int]] = []
    out = stitch_windows(T, _window_runner(calls), chunk=chunk, overlap=overlap)
    assert len(calls) > 1
    assert calls[0][0] == 0 and calls[-1][1] == T
    # every frame is covered by at least one window and windows never exceed the chunk
    assert all(e - s <= chunk for s, e in calls)
    np.testing.assert_allclose(out, _single_pass(T), rtol=1e-6)


def test_stitch_2_3_chunks_uses_three_windows_with_production_constants():
    T = 920                                   # 2.3 chunks of 400 (int(2.3 * 400) would give 919)
    calls: list[tuple[int, int]] = []
    stitch_windows(T, _window_runner(calls))
    assert calls == [(0, 400), (350, 750), (700, 920)]


def test_stitch_takes_each_frame_from_its_most_central_window():
    """Each window tags its output with its own start, so the winning window is visible."""
    T = 120
    tag_calls: list[tuple[int, int]] = []

    def tagged(s, e):
        tag_calls.append((s, e))
        out = np.zeros((e - s, V), np.float32)
        out[:, 0] = s            # value = the window that produced the row
        return out

    out = stitch_windows(T, tagged, chunk=40, overlap=10)
    winner = out[:, 0].astype(int)
    for t in range(T):
        # the chosen window must contain t, and no other window may be more central for t
        s = int(winner[t])
        e = min(s + 40, T)
        assert s <= t < e
        margin = min(t - s, e - 1 - t)
        for s2, e2 in tag_calls:
            if s2 <= t < e2 and (s2, e2) != (s, e):
                other = min(t - s2, e2 - 1 - t)
                assert other <= margin + 10, f"frame {t}: window {s2} more central than {s}"


def test_stub_engine_shape_and_probability_rows_sum_to_one():
    eng = StubEngine("HELLO WORLD", confidence=0.9)
    crops = np.zeros((150, 88, 88), np.uint8)
    lp = eng.logprobs(crops)
    assert lp.shape == (150, len(load_vocab()))
    assert lp.dtype == np.float32
    np.testing.assert_allclose(np.exp(lp).sum(axis=1), 1.0, rtol=1e-4)


def test_stub_engine_words_decode_back_in_order():
    eng = StubEngine("HELLO WORLD", confidence=0.95)
    lp = eng.logprobs(np.zeros((150, 88, 88), np.uint8))
    words = greedy_words(lp, eng.vocab, eng.blank, t0=0.0)
    assert [w.text for w in words] == ["HELLO", "WORLD"]


def test_stub_engine_skips_words_outside_the_vocabulary():
    eng = StubEngine("HELLO ZZQXJ", confidence=0.95)
    words = greedy_words(eng.logprobs(np.zeros((100, 88, 88), np.uint8)), eng.vocab, eng.blank, 0.0)
    assert [w.text for w in words] == ["HELLO"]
