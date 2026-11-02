"""Model manifest: licence gate, cache verification and download handling. No network."""
from __future__ import annotations

import dataclasses
import datetime
import glob
import hashlib
import http.client
import io
import os
import tempfile
import urllib.error

import pytest

from lipread import models
from lipread.errors import LipReadError


def _part_files(home):
    """Temp download files left behind in the model directory."""
    return glob.glob(os.path.join(str(home), "*.part"))


class _FakeResponse(io.BytesIO):
    """Stands in for the object urlopen returns: a readable stream with headers."""

    def __init__(self, payload: bytes):
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def home(tmp_path, monkeypatch):
    """An empty model directory and no licence acceptance from the environment."""
    d = tmp_path / "models"
    monkeypatch.setenv(models.ENV_HOME, str(d))
    monkeypatch.delenv(models.ENV_ACCEPT, raising=False)
    return d


@pytest.fixture
def no_network(monkeypatch):
    calls: list[str] = []

    def refuse(req, *a, **k):
        calls.append(getattr(req, "full_url", str(req)))
        raise AssertionError("the test tried to reach the network")

    monkeypatch.setattr(models.urllib.request, "urlopen", refuse)
    return calls


def test_models_dir_honours_lipread_home(home):
    assert models.models_dir() == str(home)


def test_models_dir_default_is_under_cache(monkeypatch, tmp_path):
    monkeypatch.delenv(models.ENV_HOME, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert models.models_dir() == str(tmp_path / ".cache" / "lipread" / "models")


def test_manifest_has_licence_note_and_acceptance_flag_on_vsr():
    assert models.MANIFEST["vsr"].needs_acceptance is True
    assert "non-commercial" in models.MANIFEST["vsr"].license
    for name in ("yunet", "blazeface"):
        assert models.MANIFEST[name].needs_acceptance is False
    for spec in models.MANIFEST.values():
        assert len(spec.sha256) == 64 and spec.url.startswith("https://")


def test_ensure_refuses_vsr_without_licence_and_never_downloads(home, no_network):
    with pytest.raises(LipReadError, match="licen"):
        models.ensure("vsr")
    assert no_network == []
    assert not home.exists() or not any(home.iterdir())


def test_ensure_refusal_message_explains_how_to_accept(home, no_network):
    with pytest.raises(LipReadError) as info:
        models.ensure("vsr", accept_license=False)
    assert "--accept-license" in str(info.value)
    assert models.ENV_ACCEPT in str(info.value)


def test_licence_env_var_counts_as_acceptance(home, monkeypatch):
    monkeypatch.setenv(models.ENV_ACCEPT, "1")
    monkeypatch.setattr(models.urllib.request, "urlopen",
                        lambda *a, **k: (_ for _ in ()).throw(urllib.error.URLError("offline")))
    with pytest.raises(LipReadError, match="could not download 'vsr'"):
        models.ensure("vsr")                 # got past the licence gate, then failed on the network


def test_unknown_model_name_is_a_clean_error(home):
    with pytest.raises(LipReadError, match="unknown model"):
        models.ensure("nope")


def test_sha_mismatch_discards_download_and_raises(home, monkeypatch):
    spec = dataclasses.replace(models.MANIFEST["yunet"], url="https://example.invalid/yunet.onnx")
    monkeypatch.setitem(models.MANIFEST, "yunet", spec)
    monkeypatch.setattr(models.urllib.request, "urlopen", lambda *a, **k: _FakeResponse(b"not the model"))
    with pytest.raises(LipReadError, match="sha256"):
        models.ensure("yunet")
    assert not (home / spec.filename).exists()
    assert _part_files(home) == []


def test_good_download_is_installed_once_then_cached(home, monkeypatch):
    payload = b"fake yunet weights" * 100
    spec = dataclasses.replace(models.MANIFEST["yunet"], url="https://example.invalid/y.onnx",
                               sha256=hashlib.sha256(payload).hexdigest())
    monkeypatch.setitem(models.MANIFEST, "yunet", spec)
    opened: list[str] = []

    def fake_urlopen(req, *a, **k):
        opened.append(req.full_url)
        return _FakeResponse(payload)

    monkeypatch.setattr(models.urllib.request, "urlopen", fake_urlopen)
    path = models.ensure("yunet")
    assert path == str(home / spec.filename)
    assert (home / spec.filename).read_bytes() == payload
    assert models.installed("yunet") is True
    assert models.ensure("yunet") == path
    assert opened == ["https://example.invalid/y.onnx"]         # second call used the cache


def test_corrupt_cached_file_is_not_trusted(home, monkeypatch):
    home.mkdir(parents=True, exist_ok=True)
    (home / models.MANIFEST["yunet"].filename).write_bytes(b"truncated")
    assert models.cached_path("yunet") is None
    assert models.installed("yunet") is False


def test_network_failure_becomes_lipread_error(home, monkeypatch):
    spec = dataclasses.replace(models.MANIFEST["blazeface"], url="https://example.invalid/b.tflite")
    monkeypatch.setitem(models.MANIFEST, "blazeface", spec)

    def offline(*a, **k):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(models.urllib.request, "urlopen", offline)
    with pytest.raises(LipReadError, match="could not download 'blazeface'"):
        models.ensure("blazeface")
    assert _part_files(home) == []


# --------------------------------------------------------------------------- helpers for the new tests

def _with_payload(monkeypatch, name: str, payload: bytes) -> models.ModelSpec:
    """Point manifest entry ``name`` at ``payload`` so its sha256 matches it."""
    spec = dataclasses.replace(models.MANIFEST[name], url="https://example.invalid/m.bin",
                               sha256=hashlib.sha256(payload).hexdigest())
    monkeypatch.setitem(models.MANIFEST, name, spec)
    return spec


def _count_hashes(monkeypatch) -> list[str]:
    """Wrap the real hash function and record every file it is asked to hash."""
    real = models._sha256
    calls: list[str] = []

    def counting(path):
        calls.append(path)
        return real(path)

    monkeypatch.setattr(models, "_sha256", counting)
    return calls


class _NestedResponse(_FakeResponse):
    """Runs ``on_first_read`` in the middle of a download, as a second process would."""

    def __init__(self, payload: bytes, on_first_read):
        super().__init__(payload)
        self._on_first_read = on_first_read

    def read(self, n=-1):
        if self._on_first_read is not None:
            cb, self._on_first_read = self._on_first_read, None
            cb()
        return super().read(n)


# --------------------------------------------------------------------------- HTTP errors

def test_incomplete_read_becomes_lipread_error_and_leaves_no_temp_file(home, monkeypatch):
    spec = dataclasses.replace(models.MANIFEST["yunet"], url="https://example.invalid/y.onnx")
    monkeypatch.setitem(models.MANIFEST, "yunet", spec)

    class _Truncated(_FakeResponse):
        def read(self, n=-1):
            raise http.client.IncompleteRead(b"partial", 1000)

    monkeypatch.setattr(models.urllib.request, "urlopen", lambda *a, **k: _Truncated(b""))
    with pytest.raises(LipReadError, match="could not download 'yunet'"):
        models.ensure("yunet")
    assert _part_files(home) == []
    assert not (home / spec.filename).exists()


# --------------------------------------------------------------------------- persisted licence

def test_accept_license_persist_writes_marker_with_licence_and_timestamp(home):
    path = models.accept_license_persist()
    assert path == str(home / "ACCEPTED_LICENSE_vsr")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    assert models.MANIFEST["vsr"].license in text
    stamp = next(line for line in text.splitlines() if line.startswith("Accepted at:"))
    datetime.datetime.fromisoformat(stamp.split(": ", 1)[1])     # raises if not ISO 8601


def test_persisted_marker_passes_the_gate_without_env_or_argument(home, monkeypatch, no_network):
    models.accept_license_persist()
    assert models.license_accepted("vsr") is True
    monkeypatch.setattr(models.urllib.request, "urlopen",
                        lambda *a, **k: (_ for _ in ()).throw(urllib.error.URLError("offline")))
    with pytest.raises(LipReadError, match="could not download 'vsr'"):
        models.ensure("vsr")                 # got past the gate on the marker alone


def test_no_marker_and_no_env_means_not_accepted(home):
    assert models.license_accepted("vsr") is False


def test_env_var_counts_as_acceptance_without_writing_a_marker(home, monkeypatch):
    monkeypatch.setenv(models.ENV_ACCEPT, "1")
    assert models.license_accepted("vsr") is True
    assert not (home / "ACCEPTED_LICENSE_vsr").exists()


def test_accept_license_true_persists_even_when_model_is_already_cached(home, monkeypatch, no_network):
    payload = b"already here" * 50
    spec = _with_payload(monkeypatch, "vsr", payload)
    home.mkdir(parents=True)
    (home / spec.filename).write_bytes(payload)
    assert models.ensure("vsr", accept_license=True) == str(home / spec.filename)
    assert (home / "ACCEPTED_LICENSE_vsr").is_file()
    assert no_network == []


# --------------------------------------------------------------------------- sha256 sidecar

def test_download_writes_a_sidecar_matching_the_file(home, monkeypatch):
    payload = b"sidecar payload" * 100
    spec = _with_payload(monkeypatch, "yunet", payload)
    monkeypatch.setattr(models.urllib.request, "urlopen", lambda *a, **k: _FakeResponse(payload))
    path = models.ensure("yunet")
    st = os.stat(path)
    with open(path + ".sha256ok", encoding="ascii") as f:
        assert f.read().split() == [spec.sha256, str(st.st_size), str(st.st_mtime_ns)]


def test_cached_file_is_hashed_once_then_trusted_by_its_sidecar(home, monkeypatch):
    payload = b"big weights" * 1000
    spec = _with_payload(monkeypatch, "yunet", payload)
    home.mkdir(parents=True)
    path = home / spec.filename
    path.write_bytes(payload)
    calls = _count_hashes(monkeypatch)

    assert models.cached_path("yunet") == str(path)
    assert len(calls) == 1                   # no sidecar yet: hashed, and the sidecar is written
    assert models.cached_path("yunet") == str(path)
    assert models.installed("yunet") is True
    assert len(calls) == 1                   # sidecar matched size and mtime: no second hash


def test_sidecar_is_ignored_when_mtime_changes(home, monkeypatch):
    payload = b"mtime test" * 200
    spec = _with_payload(monkeypatch, "yunet", payload)
    home.mkdir(parents=True)
    path = home / spec.filename
    path.write_bytes(payload)
    calls = _count_hashes(monkeypatch)
    models.cached_path("yunet")
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000_000))
    assert models.cached_path("yunet") == str(path)
    assert len(calls) == 2                   # mtime moved: the file was hashed again


def test_sidecar_with_wrong_size_or_digest_is_not_trusted(home, monkeypatch):
    payload = b"sidecar check" * 200
    spec = _with_payload(monkeypatch, "yunet", payload)
    home.mkdir(parents=True)
    path = home / spec.filename
    path.write_bytes(payload)
    st = os.stat(path)
    calls = _count_hashes(monkeypatch)

    (home / (spec.filename + ".sha256ok")).write_text(f"{spec.sha256} {st.st_size + 1} {st.st_mtime_ns}\n")
    assert models.cached_path("yunet") == str(path)
    assert len(calls) == 1

    (home / (spec.filename + ".sha256ok")).write_text(f"{'0' * 64} {st.st_size} {st.st_mtime_ns}\n")
    assert models.cached_path("yunet") == str(path)
    assert len(calls) == 2


def test_corrupt_file_with_a_stale_sidecar_is_rejected_and_sidecar_removed(home, monkeypatch):
    payload = b"good bytes" * 100
    spec = _with_payload(monkeypatch, "yunet", payload)
    home.mkdir(parents=True)
    path = home / spec.filename
    path.write_bytes(payload)
    models.cached_path("yunet")              # writes the sidecar
    path.write_bytes(b"bad bytes!" * 100)    # different size, so the sidecar no longer matches
    os.utime(path, ns=(1, 1))
    assert models.cached_path("yunet") is None
    assert not (home / (spec.filename + ".sha256ok")).exists()


# --------------------------------------------------------------------------- concurrent-safe temp files

def test_each_download_gets_its_own_temp_file_even_when_nested(home, monkeypatch):
    """A second download of the same model starts while the first is still reading. Both must
    finish with the right bytes, which only holds when they never share a temp file."""
    payload = b"shared target" * 300
    spec = _with_payload(monkeypatch, "yunet", payload)
    home.mkdir(parents=True)
    dest = str(home / spec.filename)

    seen: list[str] = []
    real_mkstemp = tempfile.mkstemp

    def recording(*a, **k):
        fd, path = real_mkstemp(*a, **k)
        if path.endswith(".part"):
            seen.append(path)
        return fd, path

    monkeypatch.setattr(models.tempfile, "mkstemp", recording)
    state = {"first": True}

    def fake_urlopen(req, *a, **k):
        if state["first"]:
            state["first"] = False
            return _NestedResponse(payload, lambda: models._download(spec, dest))
        return _FakeResponse(payload)

    monkeypatch.setattr(models.urllib.request, "urlopen", fake_urlopen)
    models._download(spec, dest)

    assert len(seen) == 2 and seen[0] != seen[1]
    assert all(p != dest + ".part" for p in seen)
    assert all(os.path.dirname(p) == str(home) for p in seen)
    assert open(dest, "rb").read() == payload
    assert _part_files(home) == []
