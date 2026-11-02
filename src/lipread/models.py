"""Model manifest, runtime download and sha256-verified cache.

Weights are never committed to the repo. ``ensure`` returns a local path,
downloading it on first use. Models with ``needs_acceptance`` (the LRS3-trained
lip reader) are research / non-commercial only and are only fetched after the
user accepts that licence. Acceptance is persisted as a marker file
(``accept_license_persist``) by ``accept_license=True``, ``--accept-license`` or
the web UI. The ``LIPREAD_ACCEPT_LICENSE=1`` environment variable also counts,
for CI and headless use.

Verified files get a ``<file>.sha256ok`` sidecar so later calls do not re-hash
the whole file: it is trusted only while the size and mtime still match.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import http.client
import logging
import os
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

from .errors import LipReadError

log = logging.getLogger("lipread")

ENV_HOME = "LIPREAD_HOME"
ENV_ACCEPT = "LIPREAD_ACCEPT_LICENSE"
_CHUNK = 1 << 20
_ACCEPT_PREFIX = "ACCEPTED_LICENSE_"
_SIDECAR_SUFFIX = ".sha256ok"
_TRUTHY = ("1", "true", "yes")


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


def _remove_quiet(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _write_atomic(path: str, text: str) -> None:
    """Write ``text`` to ``path`` through a unique temp file and an atomic rename."""
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=os.path.basename(path) + ".",
                               suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        _remove_quiet(tmp)
        raise


# --------------------------------------------------------------------------- licence acceptance

def _marker_path(name: str) -> str:
    return os.path.join(models_dir(), f"{_ACCEPT_PREFIX}{name}")


def _env_accepts() -> bool:
    return os.environ.get(ENV_ACCEPT, "").strip().lower() in _TRUTHY


def accept_license_persist(name: str = "vsr") -> str:
    """Record that the user accepted the licence of model ``name``; return the marker path.

    Writes ``<models_dir>/ACCEPTED_LICENSE_<name>`` holding the licence and an ISO 8601
    timestamp, so later runs and the web UI do not ask again.
    """
    spec = _get(name)
    path = _marker_path(name)
    os.makedirs(models_dir(), exist_ok=True)
    stamp = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    _write_atomic(path, (f"LipRead model licence accepted: {spec.name}\n"
                         f"Licence: {spec.license}\n"
                         f"Accepted at: {stamp}\n"))
    log.info("recorded licence acceptance for %s at %s", spec.name, path)
    return path


def license_accepted(name: str) -> bool:
    """True if the licence of model ``name`` was accepted: a persisted marker exists, or
    ``LIPREAD_ACCEPT_LICENSE`` is set (CI / headless use)."""
    _get(name)
    return os.path.isfile(_marker_path(name)) or _env_accepts()


# --------------------------------------------------------------------------- verification

def _sidecar(path: str) -> str:
    return path + _SIDECAR_SUFFIX


def _read_sidecar(path: str) -> Optional[tuple[str, int, int]]:
    """Return (sha256, size, mtime_ns) recorded for ``path``, or None if absent or unreadable."""
    try:
        with open(_sidecar(path), encoding="ascii") as f:
            parts = f.read().split()
        if len(parts) != 3:
            return None
        return parts[0], int(parts[1]), int(parts[2])
    except (OSError, ValueError):
        return None


def _verified(path: str, spec: ModelSpec) -> bool:
    """True if ``path`` matches ``spec.sha256``. Re-hashes only when the sidecar does not
    vouch for the current size and mtime."""
    st = os.stat(path)
    key = (st.st_size, st.st_mtime_ns)
    recorded = _read_sidecar(path)
    if recorded is not None and recorded == (spec.sha256, *key):
        return True
    digest = _sha256(path)
    if digest != spec.sha256:
        return False
    try:
        _write_atomic(_sidecar(path), f"{digest} {key[0]} {key[1]}\n")
    except OSError as e:
        log.debug("could not write sidecar for %s: %s", path, e)
    return True


def cached_path(name: str) -> Optional[str]:
    """Return the local path if the file exists and its sha256 matches, else None."""
    spec = _get(name)
    path = _local_path(spec)
    if not os.path.isfile(path):
        return None
    if not _verified(path, spec):
        log.warning("model file %s fails its sha256 check; it will be re-downloaded", path)
        _remove_quiet(_sidecar(path))
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
    """Stream spec.url to dest through a unique temp file, verifying sha256 before the rename.

    Each call gets its own temp file, so two processes downloading the same model cannot
    corrupt each other. The temp file is removed on every failure path.
    """
    parent = os.path.dirname(dest)
    os.makedirs(parent, exist_ok=True)
    total_bytes = int(spec.size_mb * 1e6)
    log.info("downloading %s (%.1f MB) from %s", spec.name, spec.size_mb, spec.url)
    req = urllib.request.Request(spec.url, headers={"User-Agent": "lipread/0.1"})
    h = hashlib.sha256()
    done = 0
    next_log = 0.0
    tmp: Optional[str] = None
    try:
        fd, tmp = tempfile.mkstemp(dir=parent, prefix=spec.filename + ".", suffix=".part")
        with os.fdopen(fd, "wb") as f, urllib.request.urlopen(req, timeout=60) as resp:
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
        digest = h.hexdigest()
        if digest != spec.sha256:
            raise LipReadError(
                f"download of '{spec.name}' failed the sha256 check (got {digest[:12]}..., "
                f"expected {spec.sha256[:12]}...); the file was discarded")
        os.chmod(tmp, 0o644)
        os.replace(tmp, dest)
        st = os.stat(dest)
        try:
            _write_atomic(_sidecar(dest), f"{digest} {st.st_size} {st.st_mtime_ns}\n")
        except OSError as e:
            log.debug("could not write sidecar for %s: %s", dest, e)
        log.info("installed %s at %s", spec.name, dest)
    except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
        raise LipReadError(f"could not download '{spec.name}' from {spec.url}: {e}") from e
    finally:
        if tmp is not None:
            _remove_quiet(tmp)     # no-op once the rename has moved it into place


def ensure(name: str, accept_license: bool = False) -> str:
    """Return a verified local path for model `name`, downloading it if needed.

    ``accept_license=True`` persists the licence acceptance for models that need it, even
    when the file is already cached. Raises LipReadError if the model needs licence
    acceptance and none was given (argument, marker file or env var), or if the download
    cannot be completed or verified.
    """
    spec = _get(name)
    if spec.needs_acceptance and accept_license:
        accept_license_persist(name)
    path = cached_path(name)
    if path is not None:
        return path
    if spec.needs_acceptance and not license_accepted(name):
        raise LipReadError(_licence_message(spec))
    dest = _local_path(spec)
    _download(spec, dest)
    return dest


def ensure_all(accept_license: bool = False) -> dict[str, str]:
    """Ensure every model in the manifest. Returns name -> local path.

    Licence-free models are fetched first, so a refused licence does not leave a partial set.
    """
    order = sorted(MANIFEST, key=lambda n: MANIFEST[n].needs_acceptance)
    return {name: ensure(name, accept_license=accept_license) for name in order}
