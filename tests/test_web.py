"""FastAPI app: job submission, polling, results, file serving and labels.

``lipread.pipeline.run`` is replaced by a fast fake that writes the same files the real
pipeline would, so these tests exercise only the web layer.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

pytest.importorskip("httpx", reason="FastAPI's TestClient needs httpx (pip install httpx)")
from fastapi.testclient import TestClient  # noqa: E402

import lipread.pipeline  # noqa: E402
from lipread.errors import LipReadError  # noqa: E402
from lipread.web import server  # noqa: E402

LICENCE_MSG = ("The lip-reading model 'vsr' is licensed for research / non-commercial only. "
               "To accept the licence and download it, run `lipread models download --accept-license`.")


class _FakeResult:
    def __init__(self, job, out: Path):
        self.job = job
        self.out = out

    def to_dict(self):
        return {
            "job": {"source": self.job.source, "out_dir": str(self.out), "start": self.job.start},
            "outputs": {k: str(self.out / f"lipread.{k}") for k in ("json", "srt", "vtt", "txt")},
            "tracks": [{"track_id": 1, "thumbnail": str(self.out / "thumbs" / "track_1.jpg"),
                        "utterances": []}],
            "video": {"path": self.job.source},
        }


@pytest.fixture
def seen_jobs(monkeypatch):
    """Records each Job the fake pipeline was given. Set ``fail_with`` to make the next run fail."""
    box = {"jobs": [], "fail_with": None}

    def fake_run(job, engine=None, progress=None):
        box["jobs"].append(job)
        if box["fail_with"] is not None:
            exc, box["fail_with"] = box["fail_with"], None
            raise exc
        out = Path(job.out_dir)
        (out / "thumbs").mkdir(parents=True, exist_ok=True)
        for kind in ("json", "srt", "vtt", "txt"):
            (out / f"lipread.{kind}").write_text(f"{kind} for {Path(job.source).name}\n", encoding="utf-8")
        if progress is not None:
            progress("detect", 0.5, "half way")
            progress("decode", 1.0, "done")
        return _FakeResult(job, out)

    monkeypatch.setattr(lipread.pipeline, "run", fake_run)
    return box


@pytest.fixture
def client(tmp_path, seen_jobs):
    app = server.create_app(str(tmp_path / "jobs"))
    return TestClient(app)


def _wait_done(client: TestClient, job_id: str, timeout: float = 15.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish: {body}")


def _upload(client: TestClient, name: str = "my clip.mp4", data: dict | None = None):
    fields = {"faces": "readable"}
    fields.update(data or {})
    return client.post("/api/jobs", files={"file": (name, b"not really a video", "video/mp4")}, data=fields)


# --------------------------------------------------------------------------- happy path

def test_health_and_index(client):
    assert client.get("/api/health").json()["ok"] is True
    index = client.get("/")
    assert index.status_code == 200
    assert "text/html" in index.headers["content-type"]


def test_upload_job_runs_to_done_and_results_are_served(client, seen_jobs):
    resp = _upload(client, data={"start": "1:30", "end": "2:00", "faces": "all",
                                 "names": "Travis Kelce\nGerald Youngblood, travis kelce"})
    assert resp.status_code == 200
    job_id = resp.json()["id"]
    assert len(job_id) == 12

    body = _wait_done(client, job_id)
    assert body["status"] == "done"
    assert body["progress"] == 1.0
    assert "detect" in body["stages"] and "decode" in body["stages"]

    job = seen_jobs["jobs"][-1]
    assert job.start == pytest.approx(90.0)
    assert job.end == pytest.approx(120.0)
    assert job.faces == "all"
    assert job.names == ["Travis Kelce", "Gerald Youngblood"]     # trimmed, de-duplicated (case-insensitive)
    assert Path(job.source).name == "my_clip.mp4"                  # filename made safe
    assert Path(job.source).is_file()

    result = client.get(f"/api/jobs/{job_id}/result")
    assert result.status_code == 200
    payload = result.json()
    assert payload["outputs"]["json"] == f"/files/{job_id}/lipread.json"
    assert payload["tracks"][0]["thumbnail"] == f"/files/{job_id}/thumbs/track_1.jpg"
    assert str(client.app.state.store.root) not in json.dumps(payload)   # no local paths leak

    served = client.get(payload["outputs"]["srt"])
    assert served.status_code == 200
    assert served.text.startswith("srt for my_clip.mp4")


def test_url_job_is_accepted_and_passed_through(client, seen_jobs):
    resp = client.post("/api/jobs", data={"url": "https://example.com/watch?v=abc", "faces": "readable"})
    assert resp.status_code == 200
    body = _wait_done(client, resp.json()["id"])
    assert body["status"] == "done"
    assert seen_jobs["jobs"][-1].source == "https://example.com/watch?v=abc"


def test_job_list_and_status_for_unknown_id(client):
    resp = client.post("/api/jobs", data={"url": "https://example.com/v", "faces": "readable"})
    job_id = resp.json()["id"]
    _wait_done(client, job_id)
    listing = client.get("/api/jobs").json()
    assert any(item["id"] == job_id and item["status"] == "done" for item in listing)
    assert client.get("/api/jobs/000000000000").status_code == 404


# --------------------------------------------------------------------------- validation

@pytest.mark.parametrize("data,files", [
    ({"url": "https://example.com/v"}, True),                       # file and URL together
    ({}, False),                                                    # neither
])
def test_file_or_url_but_not_both_or_neither(client, data, files):
    if files:
        resp = _upload(client, data=data)
    else:
        resp = client.post("/api/jobs", data=data)
    assert resp.status_code == 400


def test_url_must_be_http(client):
    resp = client.post("/api/jobs", data={"url": "ftp://example.com/video.mp4"})
    assert resp.status_code == 400


@pytest.mark.parametrize("extra", [
    {"start": "abc"},
    {"start": "10", "end": "5"},
    {"faces": "bogus"},
    {"names": "x" * 61},
    {"start": "-3"},
])
def test_bad_form_values_are_rejected(client, extra):
    resp = _upload(client, data=extra)
    assert resp.status_code == 400


def test_empty_upload_is_rejected(client):
    resp = client.post("/api/jobs", files={"file": ("empty.mp4", b"", "video/mp4")}, data={"faces": "readable"})
    assert resp.status_code == 400


def test_retry_only_applies_to_failed_jobs(client):
    resp = client.post("/api/jobs", data={"url": "https://example.com/v"})
    job_id = resp.json()["id"]
    _wait_done(client, job_id)
    assert client.post(f"/api/jobs/{job_id}/retry").status_code == 409


# --------------------------------------------------------------------------- failures

def test_licence_error_is_reported_as_needs_license(client, seen_jobs):
    seen_jobs["fail_with"] = LipReadError(LICENCE_MSG)
    job_id = client.post("/api/jobs", data={"url": "https://example.com/v"}).json()["id"]
    body = _wait_done(client, job_id)
    assert body["status"] == "error"
    assert body["needs_license"] is True
    assert "accept-license" in body["error"]

    # once the licence is accepted, the same job can be run again and succeeds
    assert client.post(f"/api/jobs/{job_id}/retry").status_code == 200
    assert _wait_done(client, job_id)["status"] == "done"


def test_other_failures_are_reported_with_their_message(client, seen_jobs):
    seen_jobs["fail_with"] = LipReadError("ffmpeg failed to decode the clip")
    job_id = client.post("/api/jobs", data={"url": "https://example.com/v"}).json()["id"]
    body = _wait_done(client, job_id)
    assert body["status"] == "error"
    assert body["needs_license"] is False
    assert body["error"] == "ffmpeg failed to decode the clip"
    assert client.get(f"/api/jobs/{job_id}/result").status_code == 409


# --------------------------------------------------------------------------- file serving

@pytest.fixture
def done_job(client):
    job_id = client.post("/api/jobs", data={"url": "https://example.com/v"}).json()["id"]
    _wait_done(client, job_id)
    return job_id


def test_files_are_served_from_the_job_output(client, done_job):
    assert client.get(f"/files/{done_job}/lipread.vtt").status_code == 200
    assert client.get(f"/files/{done_job}/missing.txt").status_code == 404


@pytest.mark.parametrize("path", [
    "..%2F..%2Fstate.json",           # encoded parent traversal
    "%2Fetc%2Fpasswd",                # absolute path
    "..%2F..%2F..%2F..%2Fetc%2Fpasswd",
    "lipread.json%00.txt",            # NUL byte
    "%2e%2e%2fstate.json",
])
def test_path_traversal_is_blocked(client, done_job, path):
    resp = client.get(f"/files/{done_job}/{path}")
    assert resp.status_code == 404


def test_parent_segments_are_refused_even_unencoded(client, done_job):
    assert client.get(f"/files/{done_job}/../state.json").status_code == 404
    assert client.get(f"/files/{done_job}/sub/../../state.json").status_code == 404


def test_bad_job_ids_never_touch_the_disk(client):
    assert client.get("/files/..%2F..%2Fetc/passwd").status_code == 404
    assert client.get("/files/not-a-job/lipread.json").status_code == 404
    assert client.get("/api/jobs/..%2F..%2Fetc").status_code == 404


# --------------------------------------------------------------------------- labels

def test_labels_are_appended_as_jsonl(client, done_job, tmp_path):
    payload = {"track_id": 1, "start": 1.0, "end": 1.5, "text": "  Travis   Kelce ", "original": "TRAVIS KELSO"}
    assert client.post(f"/api/jobs/{done_job}/labels", json=payload).json() == {"ok": True}
    second = {"track_id": 2, "start": 3.0, "end": 3.2, "text": "yes"}
    assert client.post(f"/api/jobs/{done_job}/labels", json=second).json() == {"ok": True}

    store = client.app.state.store
    lines = (store.out_dir(done_job) / "labels.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first, last = json.loads(lines[0]), json.loads(lines[1])
    assert first["text"] == "Travis Kelce"                  # whitespace collapsed
    assert first["original"] == "TRAVIS KELSO"
    assert first["job"] == done_job and first["track_id"] == 1
    assert "created" in first
    assert last["text"] == "yes" and last["original"] is None


def test_label_validation(client, done_job):
    url = f"/api/jobs/{done_job}/labels"
    assert client.post(url, json={"track_id": 1, "start": 1, "end": 2, "text": "   "}).status_code == 400
    assert client.post(url, json={"track_id": 1, "start": 2, "end": 1, "text": "x"}).status_code == 400
    assert client.post(url, json={"track_id": 1, "start": 1, "end": 2, "text": ""}).status_code == 422
    assert client.post(url, json={"track_id": -1, "start": 1, "end": 2, "text": "x"}).status_code == 422
    assert client.post("/api/jobs/000000000000/labels",
                       json={"track_id": 1, "start": 1, "end": 2, "text": "x"}).status_code == 404
