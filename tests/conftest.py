"""Shared fixtures for the LipRead test suite.

Videos are synthesised with ffmpeg into pytest temp directories, so the default run
needs no network and no model download. Tests that need the face models are skipped
when they are absent; the licensed lip-reading model is only used by tests marked
``model``.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REAL_MODELS = Path("/home/claude/models")
FACE_MODEL_FILES = ("face_detection_yunet_2023mar.onnx", "blaze_face_short_range.tflite")
VSR_FILE = "lipread_ctc.int8.onnx"
RAW_CLIP = Path("/home/claude/data/raw/7TMJtnhiPM_0000_S1203_E1607_L345_T26_R857_B538.mp4")
LRS3_PARQUET = Path("/home/claude/data/lrs3_test_0.parquet")
NTSC = "30000/1001"        # 29.97 fps


def pytest_configure(config: pytest.Config) -> None:
    """Use the real model directory when it holds the face models and the lip reader."""
    if "LIPREAD_HOME" in os.environ:
        return
    if all((REAL_MODELS / name).is_file() for name in (*FACE_MODEL_FILES, VSR_FILE)):
        os.environ["LIPREAD_HOME"] = str(REAL_MODELS)


# --------------------------------------------------------------------------- ffmpeg

def _ffmpeg_ready() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@pytest.fixture(scope="session")
def require_ffmpeg() -> None:
    if not _ffmpeg_ready():
        pytest.skip("ffmpeg and ffprobe are needed for this test")


def make_clip(path: Path, segments: list[tuple[str, float]], *, size: tuple[int, int] = (320, 240),
              fps: str = NTSC, audio: bool = True) -> Path:
    """Write a clip made of solid-colour shots (``(colour, seconds)``) joined by hard cuts.

    Shots are joined with ``concat``, so each boundary is a real cut. The optional audio
    track is a 440 Hz sine tone spanning the whole clip.
    """
    w, h = size
    n = len(segments)
    cmd = ["ffmpeg", "-v", "error", "-y"]
    for colour, seconds in segments:
        cmd += ["-f", "lavfi", "-i", f"color=c={colour}:s={w}x{h}:r={fps}:d={seconds}"]
    graph = "".join(f"[{i}:v]" for i in range(n)) + f"concat=n={n}:v=1:a=0[v]"
    out_opts = ["-filter_complex", graph, "-map", "[v]"]
    if audio:
        total = sum(s for _, s in segments)
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=44100:duration={total}"]
        out_opts += ["-map", f"{n}:a", "-c:a", "aac", "-b:a", "64k"]
    # all -i inputs come first; -map and codec options apply to the output file
    cmd += out_opts + ["-r", fps, "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
                       "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True, capture_output=True)
    return path


@pytest.fixture(scope="session")
def two_shot_clip(tmp_path_factory, require_ffmpeg) -> Path:
    """4 s, 29.97 fps, 320x240: grey for 2 s, then red (a hard cut at frame 60), with sine audio."""
    d = tmp_path_factory.mktemp("clips")
    return make_clip(d / "two_shot.mp4", [("gray", 2.0), ("red", 2.0)])


@pytest.fixture(scope="session")
def silent_clip(tmp_path_factory, require_ffmpeg) -> Path:
    """1 s single-shot clip with no audio track."""
    d = tmp_path_factory.mktemp("clips")
    return make_clip(d / "silent.mp4", [("blue", 1.0)], audio=False)


@pytest.fixture(scope="session")
def frontal_clip(tmp_path_factory, require_ffmpeg) -> Path:
    """2 s of a talking head, cropped to 320x320 from a raw source clip.

    Skipped when the source clip is missing (it is not available in CI).
    """
    if not RAW_CLIP.is_file():
        pytest.skip(f"source clip not available: {RAW_CLIP}")
    out = tmp_path_factory.mktemp("clips") / "frontal.mp4"
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", "0", "-t", "2", "-i", str(RAW_CLIP),
           "-vf", "crop=512:512:345:26,scale=320:320", "-an",
           "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-pix_fmt", "yuv420p", str(out)]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


# --------------------------------------------------------------------------- models

@pytest.fixture
def face_models_dir() -> str:
    """The model directory when the YuNet and BlazeFace files are present, else skip."""
    from lipread import models

    d = models.models_dir()
    if not all(os.path.isfile(os.path.join(d, name)) for name in FACE_MODEL_FILES):
        pytest.skip(f"face models not found in {d} (set LIPREAD_HOME)")
    return d


@pytest.fixture(scope="session")
def vsr_model_path() -> str:
    path = REAL_MODELS / VSR_FILE
    if not path.is_file():
        pytest.skip(f"lip-reading model not found: {path}")
    return str(path)
