"""Model manifest, runtime download and sha256-verified cache.

Weights are never committed to the repo. ``ensure`` returns a local path,
downloading it on first use. Models with ``needs_acceptance`` (the LRS3-trained
lip reader) are research / non-commercial only and are only fetched after the
user accepts that licence, via ``accept_license=True``, ``--accept-license`` or
the ``LIPREAD_ACCEPT_LICENSE=1`` environment variable.
"""
from __future__ import annotations

import hashlib
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

from .errors import LipReadError

log = logging.getLogger("lipread")

ENV_HOME = "LIPREAD_HOME"
ENV_ACCEPT = "LIPREAD_ACCEPT_LICENSE"
_CHUNK = 1 << 20


@dataclass(frozen=True)
class ModelSpec:
    name: str
    url: str
    sha256: str
    size_mb: float
    license: str
    filename: str
    needs_acceptance: bool = False


MANIFEST: dict[str, ModelSpec] = {
    "vsr": ModelSpec(
        name="vsr",
        url="https://huggingface.co/eschmechel/auto-avsr-lrs3-vsr-int8-onnx/resolve/main/lipread_ctc.int8.onnx",
        sha256="da02d72ee472eb3d7d50f7dd309bfe81973acb4cac8bb770dd86aa97872f195b",
        size_mb=203.4,
        license="research / non-commercial only (LRS3 / Auto-AVSR)",
        filename="lipread_ctc.int8.onnx",
        needs_acceptance=True,
    ),
    "yunet": ModelSpec(
        name="yunet",
        url="https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
        sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        size_mb=0.23,
        license="MIT",
        filename="face_detection_yunet_2023mar.onnx",
    ),
    "blazeface": ModelSpec(
        name="blazeface",
        url="https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
        sha256="b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f",
        size_mb=0.23,
        license="Apache-2.0",
        filename="blaze_face_short_range.tflite",
    ),
}


def models_dir() -> str:
    """Directory holding model files: $LIPREAD_HOME, else ~/.cache/lipread/models."""
    env = os.environ.get(ENV_HOME)
    if env:
        return os.path.abspath(os.path.expanduser(env))
    return os.path.join(os.path.expanduser("~"), ".cache", "lipread", "models")


def _licence_accepted(accept_license: bool) -> bool:
    if accept_license:
        return True
    return os.environ.get(ENV_ACCEPT, "").strip().lower() in ("1", "true", "yes")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _get(name: str) -> ModelSpec:
    try:
        return MANIFEST[name]
    except KeyError:
        raise LipReadError(f"unknown model '{name}'; known: {', '.join(MANIFEST)}") from None


def _local_path(spec: ModelSpec) -> str:
    return os.path.join(models_dir(), spec.filename)


def cached_path(name: str) -> Optional[str]:
    """Return the local path if the file exists and its sha256 matches, else None."""
    spec = _get(name)
    path = _local_path(spec)
    if not os.path.isfile(path):
        return None
    if _sha256(path) != spec.sha256:
        log.warning("model file %s fails its sha256 check; it will be re-downloaded", path)
        return None
    return path


def installed(name: str) -> bool:
    """True if the model is present and verified."""
    return cached_path(name) is not None


def _licence_message(spec: ModelSpec) -> str:
    return (
        f"The lip-reading model '{spec.name}' is licensed for {spec.license}. "
        "It is downloaded at runtime and is not shipped with LipRead. "
        "To accept the licence and download it, run `lipread models download --accept-license` "
        f"(or set {ENV_ACCEPT}=1)."
    )


def _download(spec: ModelSpec, dest: str) -> None:
    """Stream spec.url to dest through a .part file, verifying sha256 before the rename."""
    tmp = dest + ".part"
    total_bytes = int(spec.size_mb * 1e6)
    log.info("downloading %s (%.1f MB) from %s", spec.name, spec.size_mb, spec.url)
    req = urllib.request.Request(spec.url, headers={"User-Agent": "lipread/0.1"})
    h = hashlib.sha256()
    done = 0
    next_log = 0.0
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
            length = int(resp.headers.get("Content-Length") or 0)
            if length:
                total_bytes = length
            while True:
                block = resp.read(_CHUNK)
                if not block:
                    break
                f.write(block)
                h.update(block)
                done += len(block)
                frac = done / max(total_bytes, 1)
                if frac * 100 >= next_log:
                    log.info("  %s: %3d%% (%.1f / %.1f MB)", spec.name, min(frac * 100, 100),
                             done / 1e6, total_bytes / 1e6)
                    next_log += 10.0
        if h.hexdigest() != spec.sha256:
            raise LipReadError(
                f"download of '{spec.name}' failed the sha256 check (got {h.hexdigest()[:12]}..., "
                f"expected {spec.sha256[:12]}...); the file was discarded")
        os.replace(tmp, dest)
        log.info("installed %s at %s", spec.name, dest)
    except (urllib.error.URLError, OSError) as e:
        raise LipReadError(f"could not download '{spec.name}' from {spec.url}: {e}") from e
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def ensure(name: str, accept_license: bool = False) -> str:
    """Return a verified local path for model `name`, downloading it if needed.

    Raises LipReadError if the model needs licence acceptance and it was not given,
    or if the download cannot be completed or verified.
    """
    spec = _get(name)
    path = cached_path(name)
    if path is not None:
        return path
    if spec.needs_acceptance and not _licence_accepted(accept_license):
        raise LipReadError(_licence_message(spec))
    os.makedirs(models_dir(), exist_ok=True)
    dest = _local_path(spec)
    _download(spec, dest)
    return dest


def ensure_all(accept_license: bool = False) -> dict[str, str]:
    """Ensure every model in the manifest. Returns name -> local path.

    Licence-free models are fetched first, so a refused licence does not leave a partial set.
    """
    order = sorted(MANIFEST, key=lambda n: MANIFEST[n].needs_acceptance)
    return {name: ensure(name, accept_license=accept_license) for name in order}
