"""End-to-end check with the real licensed lip-reading model. Marked ``model``: run with ``pytest -m model``.

Needs /home/claude/models/lipread_ctc.int8.onnx and one LRS3 test clip. The clip is
read as stored (96x96 grayscale, centre-cropped to the 88x88 model input), so this
measures the model and the greedy decoder, not face detection.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from lipread.decode import greedy_words
from lipread.types import CROP

LRS3_PARQUET = Path("/home/claude/data/lrs3_test_0.parquet")

pytestmark = pytest.mark.model

MAX_WER = 0.6


def _first_lrs3_clip() -> tuple[np.ndarray, str]:
    """Return (uint8 crops (T,88,88), reference transcript) for the first clip of the LRS3 test parquet."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(LRS3_PARQUET))
    batch = next(pf.iter_batches(batch_size=1, columns=["video", "label"]))
    label = batch.column("label").to_pylist()[0]
    frames = batch.column("video")[0].as_py()                 # list of (96, 96) frames
    video = np.asarray(frames, dtype=np.uint8)
    crops = np.ascontiguousarray(video[:, 4:4 + CROP, 4:4 + CROP])
    return crops, label


def _normalise(text: str) -> str:
    return " ".join(re.sub(r"[^a-z' ]+", " ", text.lower()).split())


@pytest.fixture(scope="module")
def lrs3_clip():
    if not LRS3_PARQUET.is_file():
        pytest.skip(f"LRS3 test parquet not found: {LRS3_PARQUET}")
    return _first_lrs3_clip()


def test_real_model_reads_an_lrs3_clip_below_the_wer_bar(vsr_model_path, lrs3_clip):
    jiwer = pytest.importorskip("jiwer")
    from lipread.engine import AutoAvsrOnnx

    crops, label = lrs3_clip
    assert crops.shape[1:] == (CROP, CROP) and len(crops) > 0

    engine = AutoAvsrOnnx(vsr_model_path, threads=2)
    logprobs = engine.logprobs(crops)
    assert logprobs.shape == (len(crops), len(engine.vocab))
    np.testing.assert_allclose(np.exp(logprobs).sum(axis=1), 1.0, rtol=1e-3)

    words = greedy_words(logprobs, engine.vocab, engine.blank, t0=0.0)
    hyp = _normalise(" ".join(w.text for w in words))
    ref = _normalise(label)
    assert ref and hyp, (ref, hyp)
    wer = jiwer.wer(ref, hyp)
    assert wer < MAX_WER, f"WER {wer:.2f} >= {MAX_WER}: ref={ref!r} hyp={hyp!r}"
    assert all(0.0 <= w.confidence <= 1.0 for w in words)


def test_engine_info_names_the_research_licence(vsr_model_path):
    from lipread.engine import AutoAvsrOnnx

    engine = AutoAvsrOnnx(vsr_model_path, threads=1)
    assert "non-commercial" in engine.info["license"]
    assert engine.blank == 0
