"""Lip-reading engines: mouth crops in, per-frame CTC log-probabilities out.

Any engine implements ``Engine``. The pipeline only ever calls ``logprobs`` and
reads ``vocab`` / ``blank`` / ``info``, so newer models (USR 2.0, a fine-tuned
sports model, ...) can drop in without touching anything else.
"""
from __future__ import annotations

import json
from importlib import resources
from typing import Protocol

import numpy as np

from .align import normalize
from .types import CHUNK_FRAMES, CHUNK_OVERLAP, CROP, MAX_MODEL_FRAMES


class Engine(Protocol):
    vocab: list[str]          # index -> piece ("▁" marks word start); index `blank` is the CTC blank
    blank: int
    info: dict                # name, version, license, ...

    def logprobs(self, crops: np.ndarray) -> np.ndarray:
        """crops: uint8 (T,88,88) at 25 fps, any T >= 1. Returns float32 (T, len(vocab)) log-probs."""
        ...


def load_vocab() -> list[str]:
    return json.loads(resources.files("lipread.assets").joinpath("tokens.json").read_text())


def stitch_windows(T: int, run_window, chunk: int = CHUNK_FRAMES, overlap: int = CHUNK_OVERLAP) -> np.ndarray:
    """Run `run_window(start, end) -> (end-start, V)` over overlapping windows and stitch.

    Each output frame is taken from the window in which it lies farthest from a window
    edge, so the model always has context on both sides and words are never duplicated
    or cut at seams (stitching happens on posteriors, before decoding).
    """
    if T <= chunk:
        return run_window(0, T)
    step = chunk - overlap
    starts = list(range(0, max(T - overlap, 1), step))
    if starts[-1] + chunk < T:
        starts.append(T - chunk)
    out = None
    best = np.full(T, -1.0)
    for s in starts:
        e = min(s + chunk, T)
        lp = run_window(s, e)
        if out is None:
            out = np.empty((T, lp.shape[1]), np.float32)
        idx = np.arange(s, e)
        margin = np.minimum(idx - s, e - 1 - idx).astype(float)
        if s == 0:
            margin[: overlap] += overlap      # true video start has no left context to lose
        if e == T:
            margin[-overlap:] += overlap
        take = margin > best[s:e]
        out[idx[take]] = lp[take]
        best[s:e] = np.maximum(best[s:e], margin)
    return out


class AutoAvsrOnnx:
    """Auto-AVSR LRS3 visual-only model (Ma et al., ICASSP 2023), encoder + CTC head, int8 ONNX.

    Weights: research / non-commercial only (LRS3 terms). Downloaded at runtime, never shipped.
    """

    def __init__(self, model_path: str, threads: int | None = None):
        import onnxruntime as ort

        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        providers = [p for p in ("CoreMLExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers()]
        # CoreML can be slower for int8 graphs; keep CPU first unless the user opts in.
        self._sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        self.vocab = load_vocab()
        self.blank = 0
        self.eos = len(self.vocab) - 1
        self.info = {
            "name": "auto-avsr-lrs3-vsr-int8-onnx",
            "paper": "Ma et al., Auto-AVSR, ICASSP 2023",
            "license": "research / non-commercial (LRS3 terms)",
            "available_providers": providers,
        }

    def _window(self, x: np.ndarray, s: int, e: int) -> np.ndarray:
        lp = self._sess.run(["log_probs"], {"video": x[s:e][None, None]})[0]
        lp[:, self.eos] = -1e4        # <eos> is never a valid CTC emission here
        return lp

    def logprobs(self, crops: np.ndarray) -> np.ndarray:
        assert crops.ndim == 3 and crops.shape[1:] == (CROP, CROP), crops.shape
        assert CHUNK_FRAMES <= MAX_MODEL_FRAMES
        x = normalize(crops)
        return stitch_windows(len(x), lambda s, e: self._window(x, s, e))


class StubEngine:
    """Deterministic fake engine for tests (no model download).

    Emits the word sequence `script` spread evenly over the clip, with blanks between
    words, so timing, chunking and export logic can be tested exactly.
    """

    def __init__(self, script: str = "HELLO WORLD", confidence: float = 0.95):
        self.vocab = load_vocab()
        self.blank = 0
        self.info = {"name": "stub", "license": "n/a"}
        self.script = script.split()
        self.conf = confidence

    def logprobs(self, crops: np.ndarray) -> np.ndarray:
        T, V = len(crops), len(self.vocab)
        low = np.log((1 - self.conf) / (V - 1))
        lp = np.full((T, V), low, np.float32)
        lp[:, self.blank] = np.log(self.conf)
        index = {p: i for i, p in enumerate(self.vocab)}
        n = len(self.script)
        for k, w in enumerate(self.script):
            piece = "▁" + w
            if piece not in index:
                continue
            a = int((k + 0.25) * T / n)
            b = max(a + 1, int((k + 0.75) * T / n))
            lp[a:b, :] = low
            lp[a:b, index[piece]] = np.log(self.conf)
        return lp
