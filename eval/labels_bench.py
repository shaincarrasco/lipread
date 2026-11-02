"""Score hand labels (labels.jsonl from the web UI) against a cached pipeline result.

Each label line is {"source", "track_id", "start", "end", "text"}: what a human heard in
track `track_id` between `start` and `end` seconds of the source. For each label this script
takes the cached pipeline output in the same job folder (lipread.json, or result.json), gathers
the utterances that overlap [start, end] on that track, keeps the words whose midpoint lies
inside the range, and scores them against the label with WER.

It also prints the same confidence calibration table as lrs3_bench.py, computed on your own
clips, so you can check whether the thresholds still hold for sports footage.

Usage:
  python eval/labels_bench.py /path/to/job_folder/labels.jsonl
  python eval/labels_bench.py labels.jsonl --result /path/to/lipread.json --out report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _common as C


def load_result(labels_path: Path, explicit: str | None) -> tuple[dict, Path]:
    candidates = [Path(explicit)] if explicit else [labels_path.parent / "lipread.json",
                                                    labels_path.parent / "result.json"]
    for p in candidates:
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8")), p
    raise SystemExit(f"no pipeline result next to {labels_path} (looked for: "
                     f"{', '.join(str(c) for c in candidates)}). Run the job first.")


def words_for_label(result: dict, track_id: int, start: float, end: float) -> list[dict]:
    """Words of `track_id` from utterances overlapping [start, end] whose midpoint lies in the range."""
    track = next((t for t in result.get("tracks", []) if int(t["track_id"]) == int(track_id)), None)
    if track is None:
        return []
    out = []
    for utt in track.get("utterances", []):
        if not (utt["start"] < end and utt["end"] > start):
            continue
        for w in utt.get("words", []):
            mid = 0.5 * (float(w["start"]) + float(w["end"]))
            if start <= mid <= end:
                out.append(w)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", help="labels.jsonl (or a folder containing one)")
    ap.add_argument("--result", default=None, help="pipeline result JSON (default: lipread.json or result.json beside the labels)")
    ap.add_argument("--out", default=None, help="optional JSON report path")
    args = ap.parse_args(argv)

    labels_path = Path(args.labels)
    if labels_path.is_dir():
        labels_path = labels_path / "labels.jsonl"
    result, result_path = load_result(labels_path, args.result)
    job_source = result.get("job", {}).get("source")

    labels = [json.loads(line) for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows, refs, hyps = [], [], []
    hyp_items: list[tuple[float, bool]] = []
    skipped = 0
    for i, lab in enumerate(labels):
        ref_words = C.norm_words(lab.get("text", ""))
        if not ref_words:
            skipped += 1
            continue
        note = ""
        if job_source and lab.get("source") and Path(lab["source"]).name != Path(job_source).name:
            note = "source differs from result"
        words = words_for_label(result, lab["track_id"], float(lab["start"]), float(lab["end"]))
        if not any(t["track_id"] == lab["track_id"] for t in result.get("tracks", [])):
            note = note or "track not in result"
        hyp_pairs = []
        for w in words:
            for tok in C.norm_words(w["text"]):
                hyp_pairs.append((tok, float(w.get("confidence", 0.0))))
        hyp_str = " ".join(t for t, _ in hyp_pairs)
        ref_str = " ".join(ref_words)
        wer = C.corpus_wer([ref_str], [hyp_str])
        flags = C.hyp_correct_flags(ref_words, [t for t, _ in hyp_pairs])
        hyp_items.extend((c, ok) for (_, c), ok in zip(hyp_pairs, flags))
        refs.append(ref_str)
        hyps.append(hyp_str)
        rows.append({"index": i, "track_id": lab["track_id"], "start": lab["start"], "end": lab["end"],
                     "ref": ref_str, "hyp": hyp_str, "wer": wer, "note": note})

    if not rows:
        print("no scoreable labels (empty text or no lines)")
        return 1

    print(f"result: {result_path}")
    print(f"{'#':>3} {'track':>5} {'start-end':>13} {'WER':>6}  reference -> hypothesis")
    for r in rows:
        extra = f"  [{r['note']}]" if r["note"] else ""
        print(f"{r['index']:>3} {r['track_id']:>5} {r['start']:>6.2f}-{r['end']:<6.2f} {r['wer'] * 100:>5.0f}%  "
              f"{r['ref'][:60]}\n{'':>31}-> {r['hyp'][:60]}{extra}")
    overall = C.corpus_wer(refs, hyps)
    print(f"\nlabels scored={len(rows)}  skipped (no text)={skipped}  pooled WER={overall * 100:.1f}%")
    print("\nCalibration on these clips: hypothesis words by decoder confidence")
    cal = C.calibration_rows(hyp_items)
    print(f"{'bin':<10}{'words':>8}{'share':>9}{'correct':>10}")
    for r in cal:
        print(f"{r['bin']:<10}{r['words']:>8}{_pct(r['share']):>9}{_pct(r['correct']):>10}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"labels": str(labels_path), "result": str(result_path),
                                   "pooled_wer": overall, "scored": len(rows), "skipped": skipped,
                                   "calibration": cal, "rows": rows}, indent=1))
        print(f"\nreport: {out}")
    return 0


def _pct(x) -> str:
    return "n/a" if x is None or x != x else f"{x * 100:.1f}%"


if __name__ == "__main__":
    sys.exit(main())
