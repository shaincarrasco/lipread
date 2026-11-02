"""Command-line interface: ``lipread run | scan | models | serve``."""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import textwrap
from typing import Optional, Sequence

from . import models
from .errors import LipReadError
from .types import Job, ProgressFn, Result

log = logging.getLogger("lipread")

DEFAULT_OUT_ROOT = "lipread_out"


# --------------------------------------------------------------------------- helpers

def _setup_logging(verbosity: int, quiet: bool) -> None:
    level = logging.WARNING if quiet else (logging.DEBUG if verbosity > 0 else logging.INFO)
    log.setLevel(level)
    if not any(getattr(h, "_lipread_cli", False) for h in log.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler._lipread_cli = True  # type: ignore[attr-defined]
        log.addHandler(handler)


def _default_out(source: str) -> str:
    base = source.rstrip("/").split("/")[-1].split("?")[0]
    stem = os.path.splitext(base)[0] or "video"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)[:80]
    return os.path.join(DEFAULT_OUT_ROOT, stem)


def _split_names(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"[,;\n]", text) if p.strip()]


def _read_roster(path: str) -> list[str]:
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        raise LipReadError(f"cannot read roster {path}: {e}") from e
    if path.lower().endswith(".json"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise LipReadError(f"roster {path} is not valid JSON: {e}") from e
        if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
            raise LipReadError(f"roster {path} must be a JSON list of strings")
        return [x.strip() for x in data if x.strip()]
    names: list[str] = []
    for line in text.splitlines():
        names += _split_names(line.split("#", 1)[0])
    return names


def _collect_names(names_arg: Optional[str], roster: Optional[str]) -> list[str]:
    names: list[str] = []
    if roster:
        names += _read_roster(roster)
    if names_arg:
        names += _split_names(names_arg)
    seen: set[str] = set()
    unique: list[str] = []
    for n in names:
        if n.lower() not in seen:
            seen.add(n.lower())
            unique.append(n)
    return unique


def _table(headers: list[str], rows: list[list[str]], wrap: dict[int, int]) -> str:
    """Plain-text table. Columns in ``wrap`` (index -> width) are word-wrapped over several lines."""
    cells: list[list[list[str]]] = []
    for row in rows:
        cells.append([textwrap.wrap(c, wrap[i]) or [""] if i in wrap else [c] for i, c in enumerate(row)])
    widths = [len(h) for h in headers]
    for row in cells:
        for i, lines in enumerate(row):
            widths[i] = max(widths[i], max(len(x) for x in lines))
    fmt = "  ".join("{:<%d}" % w for w in widths)
    out = [fmt.format(*headers), "  ".join("-" * w for w in widths)]
    for row in cells:
        height = max(len(lines) for lines in row)
        for k in range(height):
            out.append(fmt.format(*[lines[k] if k < len(lines) else "" for lines in row]).rstrip())
    return "\n".join(out)


def _progress_printer() -> ProgressFn:
    state = {"stage": None, "bucket": -1}

    def cb(stage: str, frac: float, msg: str) -> None:
        bucket = int(frac * 10)
        if stage != state["stage"] or bucket != state["bucket"]:
            state["stage"], state["bucket"] = stage, bucket
            print(f"  {stage:<8} {int(frac * 100):3d}%  {msg}", file=sys.stderr, flush=True)

    return cb


# --------------------------------------------------------------------------- printing

def _print_run(result: Result) -> None:
    v = result.video
    print(f"Source: {result.job.source}")
    print(f"Video:  {v.width}x{v.height} @ {v.src_fps:.2f} fps, {v.duration:.1f} s, audio={'yes' if v.has_audio else 'no'}")
    print(f"Model:  {result.model.get('name')}, licence: {result.model.get('license')}")
    print(f"Tracks: {len(result.tracks)} kept, {len(result.shots)} shots")
    print()
    rows = []
    for t in result.tracks:
        q = t.quality
        if t.skipped:
            text = "(not decoded)"
        else:
            text = " | ".join(u.text for u in t.utterances if u.text) or "-"
        conf = "-"
        words = [w for u in t.utterances for w in u.words]
        if words and not t.skipped:
            conf = f"{sum(w.confidence for w in words) / len(words):.2f}"
        rows.append([str(t.track_id), f"{t.start:7.1f}-{t.end:7.1f}", q.tier.value, f"{q.face_px:.0f}",
                     conf, "; ".join(q.reasons) or "-", text])
    if rows:
        print(_table(["id", "time (s)", "tier", "face px", "conf", "reasons", "transcript"],
                     rows, wrap={5: 28, 6: 60}))
    else:
        print("No face tracks long enough to read.")
    print()
    print("Outputs:")
    for kind, path in sorted(result.outputs.items()):
        print(f"  {kind:<5} {path}")
    print("Timings: " + "  ".join(f"{k}={v:.1f}s" for k, v in result.timings.items()))
    print(result.disclaimer)


def _print_scan(result: Result) -> None:
    print(f"Source: {result.job.source}")
    print(f"Video:  {result.video.width}x{result.video.height} @ {result.video.src_fps:.2f} fps, "
          f"{result.video.duration:.1f} s")
    print(f"Tracks: {len(result.tracks)} kept, {len(result.shots)} shots")
    print()
    rows = []
    for t in result.tracks:
        q = t.quality
        rows.append([str(t.track_id), f"{t.start:7.1f}-{t.end:7.1f}", q.tier.value, f"{q.face_px:.0f}",
                     f"{q.mouth_px:.0f}", f"{q.yaw_deg:.0f}", f"{q.sharpness:.0f}", f"{q.coverage:.2f}",
                     f"{q.motion:.2f}", "; ".join(q.reasons) or "-"])
    if rows:
        print(_table(["id", "time (s)", "tier", "face px", "mouth px", "yaw", "sharp", "cover", "motion", "reasons"],
                     rows, wrap={9: 50}))
    else:
        print("No face tracks long enough to read.")
    print()
    print("Outputs:")
    for kind, path in sorted(result.outputs.items()):
        print(f"  {kind:<5} {path}")
    print("Timings: " + "  ".join(f"{k}={v:.1f}s" for k, v in result.timings.items()))


# --------------------------------------------------------------------------- commands

def _cmd_run(args: argparse.Namespace) -> int:
    from . import pipeline

    names = _collect_names(args.names, args.roster)
    job = Job(
        source=args.source,
        out_dir=args.out or _default_out(args.source),
        start=args.start,
        end=args.end,
        faces=args.faces,
        names=names,
        render_video=not args.no_video,
        force_unreadable=args.force_unreadable,
    )
    engine = None
    if pipeline.parse_faces(job.faces) != "none":
        engine = pipeline.default_engine(accept_license=args.accept_license)
    result = pipeline.run(job, engine=engine, progress=None if args.quiet else _progress_printer())
    _print_run(result)
    return 0


def _cmd_scan(args: argparse.Namespace) -> int:
    from . import pipeline

    job = Job(
        source=args.source,
        out_dir=args.out or _default_out(args.source),
        start=args.start,
        end=args.end,
        faces="none",
        render_video=False,
    )
    result = pipeline.run(job, engine=None, progress=None if args.quiet else _progress_printer())
    _print_scan(result)
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    if args.models_cmd == "download":
        paths = models.ensure_all(accept_license=args.accept_license)
        for name, path in paths.items():
            print(f"{name:<10} {path}")
        return 0
    rows = []
    for name, spec in models.MANIFEST.items():
        rows.append([name, "yes" if models.installed(name) else "no", f"{spec.size_mb:.1f} MB",
                     spec.license, "yes" if spec.needs_acceptance else "no"])
    print(f"Models directory: {models.models_dir()}")
    print(_table(["name", "installed", "size", "license", "needs acceptance"], rows, wrap={3: 48}))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        from .web.server import create_app
    except ImportError as e:
        raise LipReadError(f"the web UI is not available: {e}") from e
    import uvicorn

    print(f"LipRead web UI on http://{args.host}:{args.port}")
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="warning")
    return 0


# --------------------------------------------------------------------------- parser

def _add_window_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--start", type=float, default=0.0, metavar="S", help="trim start in seconds (default 0)")
    p.add_argument("--end", type=float, default=None, metavar="S", help="trim end in seconds (default: to the end)")


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("-o", "--out", metavar="OUT", help="output directory (default: lipread_out/<name>)")
    p.add_argument("-q", "--quiet", action="store_true", help="less output")
    _add_window_args(p)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lipread", description="AI lip reading for any video, tuned for sports clips.")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="debug logging")
    parser.add_argument("--version", action="version", version=f"%(prog)s {_version()}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="read the lips of every face in a video")
    run.add_argument("source", help="video file path or http(s) URL")
    _add_common(run)
    run.add_argument("--faces", default="readable", metavar="SEL",
                     help="which tracks to decode: readable (default), all, largest, or ids like 1,3")
    run.add_argument("--names", metavar="A,B", help="comma-separated names/hotwords for the roster booster")
    run.add_argument("--roster", metavar="FILE", help="file of names (one per line, or a JSON list)")
    run.add_argument("--no-video", action="store_true", help="do not render the annotated MP4")
    run.add_argument("--force-unreadable", action="store_true", help="also decode UNREADABLE tracks (labelled)")
    run.add_argument("--accept-license", action="store_true",
                     help="accept the research / non-commercial licence of the lip-reading model")
    run.set_defaults(func=_cmd_run)

    scan = sub.add_parser("scan", help="find faces and grade readability, without decoding")
    scan.add_argument("source", help="video file path or http(s) URL")
    _add_common(scan)
    scan.set_defaults(func=_cmd_scan)

    models_p = sub.add_parser("models", help="manage model weights")
    msub = models_p.add_subparsers(dest="models_cmd", required=True)
    mdl = msub.add_parser("download", help="download all models")
    mdl.add_argument("--accept-license", action="store_true",
                     help="accept the research / non-commercial licence of the lip-reading model")
    mdl.set_defaults(func=_cmd_models)
    msub.add_parser("list", help="show installed models").set_defaults(func=_cmd_models, accept_license=False)

    serve = sub.add_parser("serve", help="run the local web UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.set_defaults(func=_cmd_serve)
    return parser


def _version() -> str:
    from . import __version__
    return __version__


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the ``lipread`` command. Returns a process exit code."""
    # Silence native TFLite/mediapipe startup logs; they are imported lazily after this.
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose, getattr(args, "quiet", False))
    try:
        return int(args.func(args))
    except LipReadError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
