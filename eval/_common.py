"""Shared helpers for the eval scripts: text normalisation, greedy CTC decoding with
word confidence, word-level calibration, Whisper references and engine loading.

Word confidence (used by the calibration table in lrs3_bench.py):
  for a decoded word, take the frames from its first emitted token to its last emitted
  token, and average, over those frames, the largest NON-blank posterior (exp of the
  max log-prob excluding blank and <eos>). Blank frames inside a word therefore lower
  the score, which is what we want: they are where the model was unsure of the next piece.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
REPO = EVAL_DIR.parent
if (REPO / "src").is_dir() and str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))   # allow running the scripts without `pip install -e .`

DEFAULT_MODEL = os.environ.get("LIPREAD_MODEL", "/home/claude/models/lipread_ctc.int8.onnx")
DEFAULT_PARQUET = "/home/claude/data/lrs3_test_0.parquet"
FPS = 25
SR = 16000
# word-confidence bins used for calibration: [lo, hi) except the last bin, which includes 1.0
CONF_BINS = [(0.0, 0.5, "0.0-0.5"), (0.5, 0.7, "0.5-0.7"), (0.7, 0.9, "0.7-0.9"), (0.9, 1.0001, "0.9-1.0")]
FACE_BINS = [(0, 45, "<45"), (45, 80, "45-80"), (80, 150, "80-150"), (150, 1e9, "150+")]  # face_px buckets


def norm_words(text: str) -> list[str]:
    """Lowercase, keep a-z and apostrophes, return the word list. Digits are dropped."""
    text = re.sub(r"[^a-z' ]+", " ", text.lower())
    out = []
    for tok in text.split():
        tok = tok.strip("'")
        if tok:
            out.append(tok)
    return out


def norm_text(text: str) -> str:
    return " ".join(norm_words(text))


def greedy_words(logprobs: np.ndarray, vocab: list[str], blank: int = 0):
    """Greedy CTC decode with the package decoder, so the calibration measures the same
    Word.confidence the product reports. Returns lipread.types.Word objects."""
    from lipread.decode import greedy_words as _greedy
    return _greedy(logprobs, vocab, blank, 0.0)


def hyp_from_greedy(words) -> list[tuple[str, float]]:
    """Normalise decoded Words to scoring words, keeping each word's confidence."""
    out = []
    for w in words:
        for t in norm_words(w.text):
            out.append((t, float(w.confidence)))
    return out


def hyp_correct_flags(ref_words: list[str], hyp_words: list[str]) -> list[bool]:
    """True for each hypothesis word that is aligned as 'equal' to a reference word (jiwer alignment)."""
    if not hyp_words:
        return []
    if not ref_words:
        return [False] * len(hyp_words)
    out = _process_words(" ".join(ref_words), " ".join(hyp_words))
    flags = [False] * len(hyp_words)
    for ch in out.alignments[0]:
        if ch.type == "equal":
            for k in range(ch.hyp_start_idx, ch.hyp_end_idx):
                flags[k] = True
    return flags


def _process_words(ref: str, hyp: str):
    import jiwer
    return jiwer.process_words(ref, hyp)


def corpus_wer(refs: list[str], hyps: list[str]) -> float:
    """Pooled WER over many utterances (word-count weighted). Returns a fraction."""
    import jiwer
    pairs = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
    if not pairs:
        return float("nan")
    out = jiwer.process_words([r for r, _ in pairs], [h if h.strip() else "" for _, h in pairs])
    return float(out.wer)


def calibration_rows(hyp_items: list[tuple[float, bool]]) -> list[dict]:
    """hyp_items: (confidence, correct) per hypothesis word -> one row per confidence bin."""
    total = len(hyp_items)
    rows = []
    for lo, hi, label in CONF_BINS:
        sel = [ok for c, ok in hyp_items if lo <= c < hi]
        n = len(sel)
        rows.append({
            "bin": label,
            "words": n,
            "share": n / total if total else float("nan"),
            "correct": (sum(sel) / n) if n else float("nan"),
        })
    return rows


def threshold_rows(hyp_items: list[tuple[float, bool]], thresholds=(0.5, 0.7, 0.9)) -> list[dict]:
    """Precision of words kept when the confidence threshold is applied (keep conf >= t)."""
    total = len(hyp_items)
    rows = []
    for t in thresholds:
        kept = [ok for c, ok in hyp_items if c >= t]
        rows.append({
            "threshold": t,
            "kept_share": len(kept) / total if total else float("nan"),
            "precision": (sum(kept) / len(kept)) if kept else float("nan"),
        })
    return rows


def load_engine(model_path: str = DEFAULT_MODEL, threads: int = 2):
    from lipread.engine import AutoAvsrOnnx
    return AutoAvsrOnnx(model_path, threads=threads)


def decode_audio(path: str, seconds: float | None) -> np.ndarray:
    """16 kHz mono float32 via ffmpeg (faster-whisper's own decoding breaks with the installed PyAV)."""
    cmd = ["ffmpeg", "-v", "error", "-i", path]
    if seconds:
        cmd += ["-t", str(seconds)]
    cmd += ["-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


_WHISPER = {}


def whisper_reference(path: str, seconds: float, size: str = "small.en") -> tuple[str, np.ndarray]:
    """Whisper transcript of the first `seconds` of audio. Returns (normalised text, audio)."""
    from faster_whisper import WhisperModel
    name = size if "/" in size else f"Systran/faster-whisper-{size}"
    if name not in _WHISPER:
        _WHISPER[name] = WhisperModel(name, device="cpu", compute_type="int8")
    audio = decode_audio(path, seconds)
    if audio.size == 0:
        return "", audio
    segs, _ = _WHISPER[name].transcribe(audio, language="en")
    text = " ".join(s.text for s in segs if s.start < seconds)
    return norm_text(text), audio


def parquet_clips(path: str, n: int | None, batch_size: int = 16):
    """Yield (uint8 crops (T,88,88), label) from the LRS3 parquet (video is (T,96,96), centre-cropped)."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    seen = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=["video", "label"]):
        vid = batch.column("video")
        labels = batch.column("label").to_pylist()
        clips = _nested_video(vid)
        for frames, label in zip(clips, labels):
            yield np.ascontiguousarray(frames[:, 4:92, 4:92]), label
            seen += 1
            if n is not None and seen >= n:
                return


def _nested_video(col) -> list[np.ndarray]:
    """list<list<list<uint8>>> column -> list of (T,96,96) uint8 arrays without per-pixel Python objects."""
    n_frames = col.value_lengths().to_numpy()
    lvl2 = col.flatten()
    rows_per_frame = lvl2.value_lengths().to_numpy()
    lvl3 = lvl2.flatten()
    pix = lvl3.flatten().to_numpy()
    assert (rows_per_frame == rows_per_frame[0]).all()
    H = int(rows_per_frame[0])
    W = int(lvl3.value_lengths().to_numpy()[0])
    out, pos = [], 0
    for T in n_frames:
        size = int(T) * H * W
        out.append(pix[pos: pos + size].reshape(int(T), H, W))
        pos += size
    return out
