"""FastAPI backend for the LipRead web app.

Jobs are queued and run one at a time on a background thread through
``lipread.pipeline.run``. Each job lives in ``<work_root>/<id>/``:

* ``in/``          the uploaded source video (URL jobs have none),
* ``out/``         pipeline outputs, served read-only under ``/files/<id>/``,
* ``state.json``   job status, persisted on every status change,
* ``result.json``  the ``Result.to_dict()`` payload once the job is done.

Run with ``uvicorn --factory lipread.web.server:create_app --host 127.0.0.1``.
"""
import json
import logging
import math
import os
import re
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from lipread import __version__

log = logging.getLogger("lipread")

STATIC_DIR = Path(__file__).resolve().parent / "static"
JOB_ID_RE = re.compile(r"^[a-f0-9]{12}$")
FACES_RE = re.compile(r"^(all|readable|largest|\d+(,\d+)*)$")
MAX_NAMES = 200
MAX_NAME_LEN = 60
MAX_TEXT_LEN = 500
MAX_TITLE_LEN = 200
CHUNK = 1 << 20
TRUTHY = {"1", "true", "on", "yes"}
ACTIVE = {"queued", "running"}
DEFAULT_MAX_UPLOAD_MB = 4096


@dataclass
class JobRecord:
    """Everything the server knows about one job. Persisted as state.json."""

    id: str
    title: str
    source: str                       # local upload path, or the http(s) URL
    source_kind: str                  # "file" | "url"
    start: float = 0.0
    end: float | None = None
    faces: str = "readable"
    names: list[str] = field(default_factory=list)
    force_unreadable: bool = False
    status: str = "queued"            # queued | running | done | error
    stage: str = ""
    stages: list[str] = field(default_factory=list)   # distinct stage names, in order
    progress: float = 0.0
    message: str = "Waiting in queue"
    error: str | None = None
    needs_license: bool = False
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None


class LabelIn(BaseModel):
    track_id: int = Field(ge=0)
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str = Field(min_length=1, max_length=MAX_TEXT_LEN)
    original: str | None = Field(default=None, max_length=MAX_TEXT_LEN)


class JobStore:
    """Job records, the FIFO queue and the single worker thread."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._cond = threading.Condition()          # guards records, _pending, state files
        self.records: dict[str, JobRecord] = {}
        self._pending: list[str] = []
        self._recover()
        self._worker = threading.Thread(target=self._work_forever, name="lipread-worker", daemon=True)
        self._worker.start()

    # ---- paths -------------------------------------------------------------
    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def out_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "out"

    # ---- persistence -------------------------------------------------------
    def _save(self, rec: JobRecord) -> None:
        """Write state.json atomically. Call with the condition held."""
        try:
            d = self.job_dir(rec.id)
            d.mkdir(parents=True, exist_ok=True)
            tmp = d / "state.json.tmp"
            tmp.write_text(json.dumps(asdict(rec)), encoding="utf-8")
            os.replace(tmp, d / "state.json")
        except OSError:
            log.exception("could not save state for job %s", rec.id)

    def _recover(self) -> None:
        """Reload jobs from disk. Running jobs died with the old process; queued ones re-run."""
        known = {f.name for f in fields(JobRecord)}
        loaded: list[JobRecord] = []
        for d in self.root.iterdir():
            if not d.is_dir() or not JOB_ID_RE.match(d.name):
                continue
            state = d / "state.json"
            if not state.is_file():
                continue
            try:
                data = json.loads(state.read_text(encoding="utf-8"))
                rec = JobRecord(**{k: v for k, v in data.items() if k in known})
            except (OSError, ValueError, TypeError):
                log.warning("skipping unreadable job state %s", state)
                continue
            if rec.status == "running":
                rec.status = "error"
                rec.message = "Interrupted"
                rec.error = "Interrupted: LipRead stopped before this job finished. Run it again."
                rec.finished = time.time()
                self._save(rec)
            loaded.append(rec)
        for rec in sorted(loaded, key=lambda r: r.created):
            self.records[rec.id] = rec
            if rec.status == "queued":
                self._pending.append(rec.id)

    # ---- queue -------------------------------------------------------------
    def submit(self, rec: JobRecord) -> None:
        with self._cond:
            self.records[rec.id] = rec
            self._save(rec)
            self._pending.append(rec.id)
            self._cond.notify()

    def requeue(self, job_id: str) -> str:
        """Re-queue a failed job. Returns "ok", "missing" or "not_failed"."""
        with self._cond:
            rec = self.records.get(job_id)
            if rec is None:
                return "missing"
            if rec.status != "error":
                return "not_failed"
            rec.status = "queued"
            rec.stage = ""
            rec.progress = 0.0
            rec.message = "Waiting in queue"
            rec.error = None
            rec.needs_license = False
            rec.started = rec.finished = None
            self._save(rec)
            self._pending.append(rec.id)
            self._cond.notify()
            return "ok"

    def queue_position(self, job_id: str) -> int | None:
        with self._cond:
            if job_id in self._pending:
                return self._pending.index(job_id) + 1
            return None

    def _next_job(self) -> str:
        with self._cond:
            while not self._pending:
                self._cond.wait()
            return self._pending.pop(0)

    def _work_forever(self) -> None:
        while True:
            job_id = self._next_job()
            try:
                self._run(job_id)
            except Exception:  # keep the worker alive whatever happens
                log.exception("worker failed on job %s", job_id)

    # ---- running one job ---------------------------------------------------
    def _progress(self, job_id: str, stage: str, fraction: float, message: str) -> None:
        try:
            frac = float(fraction)
        except (TypeError, ValueError):
            frac = 0.0
        if not math.isfinite(frac):
            frac = 0.0
        with self._cond:
            rec = self.records.get(job_id)
            if rec is None or rec.status != "running":
                return
            stage = str(stage)
            if stage and stage not in rec.stages:
                rec.stages.append(stage)
            rec.stage = stage
            rec.progress = min(max(frac, 0.0), 1.0)
            rec.message = str(message)

    def _run(self, job_id: str) -> None:
        with self._cond:
            rec = self.records.get(job_id)
            if rec is None:
                return
            rec.status = "running"
            rec.started = time.time()
            rec.stage = ""
            rec.stages = []
            rec.progress = 0.0
            rec.message = "Starting"
            rec.error = None
            rec.needs_license = False
            self._save(rec)
            snapshot = JobRecord(**asdict(rec))

        out_dir = self.out_dir(job_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Imported at call time so the web layer loads even before the pipeline exists,
            # and so tests can swap lipread.pipeline.run.
            from lipread import pipeline
            from lipread.types import Job

            job = Job(
                source=snapshot.source,
                out_dir=str(out_dir),
                start=snapshot.start,
                end=snapshot.end,
                faces=snapshot.faces,
                names=list(snapshot.names),
                force_unreadable=snapshot.force_unreadable,
            )

            def progress(stage: str, fraction: float, message: str) -> None:
                self._progress(job_id, stage, fraction, message)

            result = pipeline.run(job, progress=progress)
            payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
            (self.job_dir(job_id) / "result.json").write_text(
                json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8"
            )
        except Exception as exc:
            needs_license = _is_license_error(exc)
            if needs_license:
                log.info("job %s needs the model license accepted", job_id)
            else:
                log.exception("job %s failed", job_id)
            with self._cond:
                rec = self.records[job_id]
                rec.status = "error"
                rec.message = "Failed"
                rec.error = str(exc).strip() or exc.__class__.__name__
                rec.needs_license = needs_license
                rec.finished = time.time()
                self._save(rec)
            return

        with self._cond:
            rec = self.records[job_id]
            rec.status = "done"
            rec.stage = "done"
            rec.progress = 1.0
            rec.message = "Finished"
            rec.finished = time.time()
            self._save(rec)


def _is_license_error(exc: BaseException) -> bool:
    """True for the pipeline's LipReadError about a model license that is not accepted."""
    is_lipread_error = any(c.__name__ == "LipReadError" for c in type(exc).__mro__)
    return is_lipread_error and "license" in str(exc).lower()


# ---- request helpers ---------------------------------------------------------
def parse_time(text: str) -> float | None:
    """Parse "90", "90.5", "1:30" or "1:02:03" into seconds. Empty gives None."""
    s = text.strip()
    if not s:
        return None
    parts = s.split(":")
    if len(parts) > 3:
        raise ValueError(s)
    total = 0.0
    for part in parts:
        value = float(part)          # raises ValueError on junk
        if not math.isfinite(value) or value < 0:
            raise ValueError(s)
        total = total * 60 + value
    return total


def parse_names(raw: str) -> list[str]:
    """Split a roster on newlines or commas; trim, drop blanks and case-insensitive duplicates."""
    seen: set[str] = set()
    names: list[str] = []
    for chunk in re.split(r"[\n,]", raw):
        name = " ".join(chunk.split())
        if not name:
            continue
        if len(name) > MAX_NAME_LEN:
            raise HTTPException(400, f"Roster name is longer than {MAX_NAME_LEN} characters: {name[:20]}...")
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        names.append(name)
    if len(names) > MAX_NAMES:
        raise HTTPException(400, f"Roster has more than {MAX_NAMES} names.")
    return names


def safe_name(name: str) -> str:
    base = Path(name.replace("\\", "/")).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    return cleaned[:120] or "upload"


def _max_upload_bytes() -> int:
    try:
        mb = float(os.environ.get("LIPREAD_MAX_UPLOAD_MB", DEFAULT_MAX_UPLOAD_MB))
    except ValueError:
        mb = DEFAULT_MAX_UPLOAD_MB
    return int(mb * 1024 * 1024)


def _safe_join(base: Path, file_path: str) -> Path | None:
    """Resolve file_path under base, or None if it escapes base or is not a plain relative path."""
    if not file_path or "\x00" in file_path:
        return None
    parts = PurePosixPath(file_path).parts
    if not parts or parts[0] == "/" or any(p in ("", ".", "..") for p in parts):
        return None
    root = base.resolve()
    candidate = (root / Path(*parts)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _rewrite_paths(obj: Any, out_dir: Path, job_id: str) -> Any:
    """Turn local output paths into /files URLs; never leak other local paths."""
    root = out_dir.resolve()

    def fix(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: fix(v) for k, v in value.items()}
        if isinstance(value, list):
            return [fix(v) for v in value]
        if isinstance(value, str) and os.path.isabs(value):
            try:
                rel = Path(value).resolve().relative_to(root).as_posix()
            except (ValueError, OSError):
                return os.path.basename(value)
            return None if rel == "." else f"/files/{job_id}/{rel}"
        return value

    return fix(obj)


def _status_payload(rec: JobRecord, position: int | None) -> dict[str, Any]:
    return {
        "id": rec.id,
        "title": rec.title,
        "status": rec.status,
        "stage": rec.stage,
        "stages": list(rec.stages),
        "progress": round(rec.progress, 4),
        "message": rec.message,
        "error": rec.error,
        "needs_license": rec.needs_license,
        "queue_position": position,
    }


def _list_item(rec: JobRecord) -> dict[str, Any]:
    return {
        "id": rec.id,
        "title": rec.title,
        "source_kind": rec.source_kind,
        "status": rec.status,
        "stage": rec.stage,
        "progress": round(rec.progress, 4),
        "faces": rec.faces,
        "created": datetime.fromtimestamp(rec.created, timezone.utc).isoformat(),
        "finished": datetime.fromtimestamp(rec.finished, timezone.utc).isoformat() if rec.finished else None,
    }


def create_app(work_root: str | None = None) -> FastAPI:
    """Build the LipRead web app. Work root: argument, else $LIPREAD_JOBS, else ~/LipRead/jobs."""
    root = Path(work_root).expanduser() if work_root else Path(
        os.environ.get("LIPREAD_JOBS") or "~/LipRead/jobs"
    ).expanduser()
    store = JobStore(root)

    app = FastAPI(title="LipRead", version=__version__)
    app.state.store = store

    def require_job(job_id: str) -> JobRecord:
        if not JOB_ID_RE.match(job_id):
            raise HTTPException(404, "Unknown job")
        with store._cond:
            rec = store.records.get(job_id)
        if rec is None:
            raise HTTPException(404, "Unknown job")
        return rec

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "version": __version__}

    @app.post("/api/jobs")
    def create_job(
        file: UploadFile | None = File(None),
        url: str = Form(""),
        start: str = Form("0"),
        end: str = Form(""),
        faces: str = Form("readable"),
        names: str = Form(""),
        force_unreadable: str = Form(""),
    ) -> dict[str, str]:
        has_file = file is not None and bool(file.filename)
        url = url.strip()
        if has_file == bool(url):
            raise HTTPException(400, "Send either a video file or a URL, not both and not neither.")

        try:
            start_s = parse_time(start) or 0.0
        except ValueError:
            raise HTTPException(400, "Start must be a time such as 90 or 1:30.")
        try:
            end_s = parse_time(end)
        except ValueError:
            raise HTTPException(400, "End must be a time such as 150 or 2:30.")
        if end_s is not None and end_s <= start_s:
            raise HTTPException(400, "End must be after start.")

        faces = (faces or "readable").strip().lower()
        if not FACES_RE.match(faces):
            raise HTTPException(400, "Faces must be readable, all, largest, or a list of track ids.")
        roster = parse_names(names)
        force = force_unreadable.strip().lower() in TRUTHY

        job_id = uuid.uuid4().hex[:12]
        if has_file:
            in_dir = store.job_dir(job_id) / "in"
            in_dir.mkdir(parents=True, exist_ok=True)
            dest = in_dir / safe_name(file.filename)
            limit = _max_upload_bytes()
            written = 0
            try:
                with dest.open("wb") as fh:
                    while True:
                        chunk = file.file.read(CHUNK)
                        if not chunk:
                            break
                        written += len(chunk)
                        if written > limit:
                            raise HTTPException(413, "The video is larger than the upload limit.")
                        fh.write(chunk)
            except HTTPException:
                shutil.rmtree(store.job_dir(job_id), ignore_errors=True)
                raise
            except OSError:
                shutil.rmtree(store.job_dir(job_id), ignore_errors=True)
                raise HTTPException(500, "Could not save the upload.")
            if written == 0:
                shutil.rmtree(store.job_dir(job_id), ignore_errors=True)
                raise HTTPException(400, "The uploaded file is empty.")
            source, kind, title = str(dest), "file", file.filename[:MAX_TITLE_LEN]
        else:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                raise HTTPException(400, "The URL must start with http:// or https://.")
            source, kind, title = url, "url", url[:MAX_TITLE_LEN]

        rec = JobRecord(
            id=job_id,
            title=title,
            source=source,
            source_kind=kind,
            start=start_s,
            end=end_s,
            faces=faces,
            names=roster,
            force_unreadable=force,
        )
        store.submit(rec)
        return {"id": job_id}

    @app.get("/api/jobs")
    def list_jobs(limit: int = 50) -> list[dict[str, Any]]:
        limit = min(max(limit, 1), 200)
        with store._cond:
            recs = sorted(store.records.values(), key=lambda r: r.created, reverse=True)[:limit]
            return [_list_item(r) for r in recs]

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> dict[str, Any]:
        rec = require_job(job_id)
        with store._cond:
            return _status_payload(rec, store.queue_position(job_id) if rec.status == "queued" else None)

    @app.get("/api/jobs/{job_id}/result")
    def job_result(job_id: str) -> Any:
        rec = require_job(job_id)
        if rec.status != "done":
            raise HTTPException(409, f"Job is {rec.status}; the result is available when it is done.")
        path = store.job_dir(job_id) / "result.json"
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise HTTPException(500, "The result file is missing or unreadable.")
        return _rewrite_paths(raw, store.out_dir(job_id), job_id)

    @app.post("/api/jobs/{job_id}/labels")
    def add_label(job_id: str, body: LabelIn) -> dict[str, Any]:
        require_job(job_id)
        text = " ".join(body.text.split())
        if not text:
            raise HTTPException(400, "The corrected text is empty.")
        if body.end < body.start:
            raise HTTPException(400, "End must not be before start.")
        original = " ".join((body.original or "").split()) or None
        line = json.dumps(
            {
                "job": job_id,
                "track_id": body.track_id,
                "start": body.start,
                "end": body.end,
                "text": text,
                "original": original,
                "created": datetime.now(timezone.utc).isoformat(),
            },
            ensure_ascii=False,
        )
        out_dir = store.out_dir(job_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        with store._cond:
            with (out_dir / "labels.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        return {"ok": True}

    @app.post("/api/jobs/{job_id}/retry")
    def retry_job(job_id: str) -> dict[str, Any]:
        require_job(job_id)
        outcome = store.requeue(job_id)
        if outcome == "not_failed":
            raise HTTPException(409, "Only a failed job can be run again.")
        return {"ok": True}

    @app.post("/api/accept-license")
    def accept_license() -> dict[str, Any]:
        os.environ["LIPREAD_ACCEPT_LICENSE"] = "1"
        log.info("model license accepted for this process")
        return {"ok": True}

    @app.get("/files/{job_id}/{file_path:path}")
    def job_file(job_id: str, file_path: str) -> FileResponse:
        if not JOB_ID_RE.match(job_id):
            raise HTTPException(404, "Not found")
        target = _safe_join(store.out_dir(job_id), file_path)
        if target is None or not target.is_file():
            raise HTTPException(404, "Not found")
        return FileResponse(target)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(
            STATIC_DIR / "index.html",
            media_type="text/html; charset=utf-8",
            headers={"Cache-Control": "no-store"},
        )

    return app
