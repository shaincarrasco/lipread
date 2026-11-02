"""Model manifest: licence gate, cache verification and download handling. No network."""
from __future__ import annotations

import dataclasses
import hashlib
import io
import urllib.error

import pytest

from lipread import models
from lipread.errors import LipReadError


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
    assert not (home / (spec.filename + ".part")).exists()


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
    assert not (home / (spec.filename + ".part")).exists()
