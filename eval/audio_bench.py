"""Pseudo-ground-truth benchmark on real video: Whisper transcript as reference, LipRead as hypothesis.

For every video in VIDEO_DIR:
  1. Whisper (faster-whisper, int8, CPU) transcribes the first --max-seconds of audio.
  2. lipread.pipeline.run(Job(faces="largest", end=--max-seconds, render_video=False)) lip-reads the same range.
  3. WER per clip, plus the face size (face_px) and readability tier of the decoded track.
Summary: pooled WER by face-size bucket (<45, 45-80, 80-150, 150+ px).

Whisper is a proxy: its errors add noise to WER, and clips where Whisper finds no speech are
skipped. Use the numbers to compare settings, not as absolute accuracy.

Usage:
  python eval/audio_bench.py /path/to/clips --whisper small.en --max-seconds 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

import _common as C

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}


def lipread_words(res) -> tuple[list[str], int]:
    """All decoded words of a Result in time order, plus the number of decoded tracks."""
    words, decoded = [], 0
    for tr in res.tracks:
        if getattr(tr, "skipped", False):
            continue
        n_before = len(words)
        for utt in tr.utterances:
            for w in utt.words:
                words.append((w.start, w.text))
        if len(words) > n_before:
            decoded += 1
    words.sort(key=lambda x: x[0])
    return [t for _, t in words], decoded


def primary_quality(res):
    """face_px and tier of the decoded track with most words (else the largest-face track)."""
    best = None
    for tr in res.tracks:
        nw = sum(len(u.words) for u in tr.utterances) if not tr.skipped else 0
        key = (nw, tr.quality.face_px)
        if best is None or key > best[0]:
            best = (key, tr)
    if best is None:
        return None, None
    tr = best[1]
    return float(tr.quality.face_px), tr.quality.tier.value if hasattr(tr.quality.tier, "value") else str(tr.quality.tier)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video_dir")
    ap.add_argument("--whisper", default="small.en", help="faster-whisper size, e.g. small.en or base.en")
    ap.add_argument("--max-seconds", type=float, default=20.0)
    ap.add_argument("--model", default=None, help="ONNX model path (default: the pipeline's default engine)")
    ap.add_argument("--work", default=str(Path.home() / ".cache" / "lipread-eval" / "audio_bench"),
                    help="job folders go here (one per clip, with result.json for labels_bench.py)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(C.EVAL_DIR / "reports" / "audio_bench.json"))
    args = ap.parse_args(argv)

    try:
        from lipread.pipeline import run
        from lipread.types import Job
    except Exception as e:  # pipeline is built in parallel; say exactly what is missing
        print(f"lipread.pipeline is not importable yet: {e}")
        return 2

    engine = None
    if args.model:
        engine = C.load_engine(args.model)

    vids = sorted(p for p in Path(args.video_dir).iterdir() if p.suffix.lower() in VIDEO_EXT)
    if args.limit:
        vids = vids[: args.limit]
    if not vids:
        print(f"no videos in {args.video_dir}")
        return 1
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in vids:
        row = {"clip": p.name, "ref": "", "hyp": "", "wer": None, "face_px": None, "tier": None,
               "words_ref": 0, "words_hyp": 0, "decoded_tracks": 0, "seconds": None, "note": ""}
        t0 = time.perf_counter()
        try:
            ref, _audio = C.whisper_reference(str(p), args.max_seconds, args.whisper)
            row["ref"] = ref
            row["words_ref"] = len(ref.split())
            if not ref:
                row["note"] = "no speech in Whisper reference (skipped)"
                rows.append(row)
                print(f"{p.name[:28]:<30} skipped: no speech")
                continue
            job_dir = work / p.stem
            job = Job(source=str(p), out_dir=str(job_dir), start=0.0, end=args.max_seconds,
                      faces="largest", render_video=False)
            res = run(job, engine=engine)
            hyp_words, decoded = lipread_words(res)
            face_px, tier = primary_quality(res)
            hyp = C.norm_text(" ".join(hyp_words))
            row.update({"hyp": hyp, "words_hyp": len(hyp.split()), "decoded_tracks": decoded,
                        "face_px": face_px, "tier": tier})
            if decoded == 0:
                # no face (voiceover over a non-face shot, etc.): not a lip-reading test, so excluded from WER
                row["note"] = "no decoded face track (excluded from WER)"
            else:
                row["wer"] = C.corpus_wer([ref], [hyp])
        except Exception as e:
            row["note"] = f"error: {type(e).__name__}: {e}"
            print(traceback.format_exc(), file=sys.stderr)
        row["seconds"] = round(time.perf_counter() - t0, 1)
        rows.append(row)
        w = "-" if row["wer"] is None else f"{row['wer'] * 100:.0f}%"
        fp = "-" if row["face_px"] is None else f"{row['face_px']:.0f}"
        print(f"{p.name[:28]:<30} face_px={fp:>5} tier={str(row['tier']):<10} "
              f"ref={row['words_ref']:>3} hyp={row['words_hyp']:>3} WER={w:>5} {row['seconds']}s {row['note']}")

    scored = [r for r in rows if r["wer"] is not None]
    print()
    print(f"{'bucket':<10}{'clips':>7}{'ref words':>11}{'WER':>9}")
    buckets = {}
    for lo, hi, label in C.FACE_BINS:
        sel = [r for r in scored if r["face_px"] is not None and lo <= r["face_px"] < hi]
        buckets[label] = _bucket_row(sel)
    for label, b in buckets.items():
        print(f"{label:<10}{b['clips']:>7}{b['ref_words']:>11}{_pct(b['wer']):>9}")
    overall = _bucket_row(scored)
    print(f"{'all':<10}{overall['clips']:>7}{overall['ref_words']:>11}{_pct(overall['wer']):>9}")
    excluded = [r["clip"] for r in rows if r["wer"] is None and r["ref"]]
    if excluded:
        print(f"excluded (no decoded face track, speech in reference): {len(excluded)} clip(s): {', '.join(excluded)}")

    report = {"bench": "audio_bench", "video_dir": args.video_dir, "whisper": args.whisper,
              "max_seconds": args.max_seconds, "model": args.model, "work": str(work),
              "summary_by_face_px": buckets, "overall": overall, "clips": rows}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"\nreport: {out}")
    return 0


def _bucket_row(sel) -> dict:
    if not sel:
        return {"clips": 0, "ref_words": 0, "wer": None}
    refs = [r["ref"] for r in sel]
    hyps = [r["hyp"] for r in sel]
    return {"clips": len(sel), "ref_words": sum(len(r.split()) for r in refs),
            "wer": float(C.corpus_wer(refs, hyps))}


def _pct(x) -> str:
    return "n/a" if x is None or x != x else f"{x * 100:.1f}%"


if __name__ == "__main__":
    raise SystemExit(main())
