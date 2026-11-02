"""CTC likelihoods and the roster booster."""
from __future__ import annotations

import math

import numpy as np
import pytest

from lipread.booster import Booster, ctc_log_likelihood
from lipread.decode import greedy_words
from lipread.engine import load_vocab
from lipread.types import Confidence, Word

VOCAB = load_vocab()
BLANK = 0
V = len(VOCAB)
EPS = 1e-6


def _ids(*pieces: str) -> list[int]:
    return [VOCAB.index(p) for p in pieces]


# --------------------------------------------------------------------------- ctc_log_likelihood

def _random_logprobs(rng: np.random.Generator, T: int, V_: int) -> np.ndarray:
    logits = rng.normal(size=(T, V_))
    return logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))


@pytest.mark.parametrize("seed,T,target", [
    (0, 12, [1, 2, 3]),
    (1, 10, [2, 2, 3]),          # repeated label needs a blank in between
    (2, 16, [1, 1, 1, 2]),
    (3, 6, [4]),
    (4, 9, [5, 1, 5]),
])
def test_ctc_log_likelihood_matches_torch(seed, T, target):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(seed)
    lp = _random_logprobs(rng, T, 6)
    ours = ctc_log_likelihood(lp, target, blank=0)
    ref = -torch.nn.functional.ctc_loss(
        torch.tensor(lp, dtype=torch.float64)[:, None, :],
        torch.tensor([target], dtype=torch.long),
        torch.tensor([T]), torch.tensor([len(target)]),
        blank=0, reduction="sum", zero_infinity=False,
    ).item()
    assert ours == pytest.approx(ref, rel=1e-9, abs=1e-9)


def test_ctc_log_likelihood_too_short_is_minus_inf():
    torch = pytest.importorskip("torch")
    lp = _random_logprobs(np.random.default_rng(5), 3, 6)
    assert ctc_log_likelihood(lp, [1, 1, 1], blank=0) == -math.inf    # needs 5 frames with blanks
    loss = torch.nn.functional.ctc_loss(torch.tensor(lp, dtype=torch.float64)[:, None, :],
                                        torch.tensor([[1, 1, 1]]), torch.tensor([3]), torch.tensor([3]),
                                        blank=0, reduction="sum", zero_infinity=False).item()
    assert math.isinf(loss)


def test_ctc_log_likelihood_exact_on_tiny_case():
    """Two frames, one label: P = p0(blank)p1(a) + p0(a)p1(a) + p0(a)p1(blank)."""
    p = np.array([[0.5, 0.3, 0.2], [0.4, 0.1, 0.5]])   # blank=0, label=1 or 2
    lp = np.log(p)
    expected = math.log(0.5 * 0.1 + 0.3 * 0.1 + 0.3 * 0.4)
    assert ctc_log_likelihood(lp, [1], blank=0) == pytest.approx(expected)


# --------------------------------------------------------------------------- booster

def _crafted(p_so: float, p_ce: float) -> np.ndarray:
    """Frames: blank, KEL-spike (K then EL), then a contested final piece (SO vs CE), then blank.

    The model's greedy reading of the contested frames is ``▁SO``, but ``CE`` (the roster
    name KELCE's last piece) is nearly as likely.
    """
    T = 10
    lp = np.full((T, V), math.log(EPS), np.float64)
    spec = {2: {"▁K": 0.9}, 3: {"▁K": 0.9}, 4: {"EL": 0.9}, 5: {"EL": 0.9},
            6: {"▁SO": p_so, "CE": p_ce}, 7: {"▁SO": p_so, "CE": p_ce}}
    for t in range(T):
        probs = dict(spec.get(t, {}))
        blank = 1.0 - sum(probs.values())
        probs = {VOCAB.index(k): v for k, v in probs.items()}
        probs[BLANK] = blank
        for k, v in probs.items():
            lp[t, k] = math.log(max(v, EPS))
    return lp


def _greedy(lp: np.ndarray) -> list[Word]:
    return greedy_words(lp, VOCAB, BLANK, t0=0.0)


def test_crafted_matrix_greedy_reads_the_wrong_words():
    words = _greedy(_crafted(p_so=0.5, p_ce=0.4))
    assert [w.text for w in words] == ["KEL", "SO"]


def test_roster_name_wins_and_is_flagged_with_original_kept():
    lp = _crafted(p_so=0.5, p_ce=0.4)
    words = _greedy(lp)
    orig_conf = float(np.mean([w.confidence for w in words]))
    booster = Booster(["Kelce"], VOCAB)
    out = booster.apply(words, lp, t0=0.0)

    assert [w.text for w in out] == ["KELCE"]
    w = out[0]
    assert w.boosted is True
    assert w.original == "KEL SO"
    assert w.start == pytest.approx(words[0].start)
    assert w.end == pytest.approx(words[-1].end)
    assert w.confidence <= orig_conf + 1e-9       # boosting never raises confidence
    assert w.tier in (Confidence.HIGH, Confidence.MEDIUM, Confidence.LOW)


def test_matrix_where_model_is_confident_keeps_its_words():
    lp = _crafted(p_so=0.95, p_ce=0.01)          # e^-4.5 less likely than SO: beyond the 3-nat bonus
    words = _greedy(lp)
    out = Booster(["Kelce"], VOCAB).apply(words, lp, t0=0.0)
    assert [w.text for w in out] == ["KEL", "SO"]
    assert not any(w.boosted for w in out)
    assert all(w.original is None for w in out)


def test_booster_with_no_names_is_identity():
    lp = _crafted(p_so=0.5, p_ce=0.4)
    words = _greedy(lp)
    out = Booster([], VOCAB).apply(words, lp, t0=0.0)
    assert out == words


def test_booster_skips_names_it_cannot_tokenize():
    lp = _crafted(p_so=0.5, p_ce=0.4)
    words = _greedy(lp)
    # digits and symbols the 5000-piece vocabulary cannot write are dropped, not crashed on
    out = Booster(["Zq#9"], VOCAB).apply(words, lp, t0=0.0)
    assert [w.text for w in out] == ["KEL", "SO"]


def test_booster_candidates_are_built_per_word_for_multiword_names():
    booster = Booster(["Travis Kelce"], VOCAB)
    texts = {c.text for c in booster.candidates}
    assert {"TRAVIS KELCE", "TRAVIS", "KELCE"} <= texts
