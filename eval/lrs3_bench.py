"""LRS3 test-set benchmark for the visual engine: greedy WER, speed and word-confidence calibration.

Usage:
  python eval/lrs3_bench.py --n 200
  python eval/lrs3_bench.py --parquet /path/lrs3_test_0.parquet --model /path/lipread_ctc.int8.onnx --n 100

Writes a JSON report (default eval/reports/lrs3_bench.json) and prints the tables.
The calibration table is what sets the confidence thresholds: each hypothesis word is
binned by its decoder confidence and marked correct when jiwer aligns it as 'equal'.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import _common as C


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", default=C.DEFAULT_PARQUET)
    ap.add_argument("--n", type=int, default=200, help="number of test clips (0 = all)")
    ap.add_argument("--model", default=C.DEFAULT_MODEL)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default=str(C.EVAL_DIR / "reports" / "lrs3_bench.json"))
    args = ap.parse_args(argv)

    engine = C.load_engine(args.model, threads=args.threads)
    limit = None if args.n == 0 else args.n

    refs, hyps, clip_rows = [], [], []
    hyp_items: list[tuple[float, bool]] = []
    infer_s = 0.0
    audio_s = 0.0
    for i, (crops, label) in enumerate(C.parquet_clips(args.parquet, limit)):
        t0 = time.perf_counter()
        lp = engine.logprobs(crops)
        infer_s += time.perf_counter() - t0
        audio_s += len(crops) / C.FPS

        words = C.greedy_words(lp, engine.vocab, engine.blank)
        hyp_w = C.hyp_from_greedy(words)                     # [(word, conf)]
        ref_w = C.norm_words(label)
        hyp_str = " ".join(w for w, _ in hyp_w)
        ref_str = " ".join(ref_w)
        refs.append(ref_str)
        hyps.append(hyp_str)

        flags = C.hyp_correct_flags(ref_w, [w for w, _ in hyp_w])
        hyp_items.extend((c, ok) for (_, c), ok in zip(hyp_w, flags))
        clip_wer = C.corpus_wer([ref_str], [hyp_str]) if ref_str else float("nan")
        clip_rows.append({"ref": ref_str, "hyp": hyp_str, "wer": clip_wer,
                          "frames": int(len(crops)), "mean_conf": float(np.mean([c for _, c in hyp_w])) if hyp_w else None})
        if i < 4:
            print(f"REF: {ref_str}\nHYP: {hyp_str}\n")

    n = len(refs)
    if n == 0:
        print("no clips read")
        return 1
    wer = C.corpus_wer(refs, hyps)
    speed = audio_s / infer_s if infer_s else float("nan")
    per_clip = [r["wer"] for r in clip_rows if r["wer"] == r["wer"]]
    cal = C.calibration_rows(hyp_items)
    thr = C.threshold_rows(hyp_items)

    print(f"clips={n}  corpus WER={wer * 100:.1f}%  mean per-clip WER={np.mean(per_clip) * 100:.1f}%  "
          f"speed={speed:.1f}x real time ({args.threads} threads)")
    print(f"hypothesis words={len(hyp_items)}  correct={sum(ok for _, ok in hyp_items) / max(1, len(hyp_items)) * 100:.1f}%")
    print()
    print("Calibration: hypothesis words by decoder confidence")
    print(f"{'bin':<10}{'words':>8}{'share':>9}{'correct':>10}")
    for r in cal:
        print(f"{r['bin']:<10}{r['words']:>8}{_pct(r['share']):>9}{_pct(r['correct']):>10}")
    print()
    print("Threshold: keep words with conf >= t")
    print(f"{'t':<6}{'kept':>9}{'precision':>12}")
    for r in thr:
        print(f"{r['threshold']:<6}{_pct(r['kept_share']):>9}{_pct(r['precision']):>12}")

    report = {
        "bench": "lrs3_bench",
        "parquet": args.parquet, "model": args.model, "engine": engine.info, "clips": n,
        "corpus_wer": wer, "mean_clip_wer": float(np.mean(per_clip)),
        "speed_x_realtime": speed, "threads": args.threads,
        "calibration": cal, "thresholds": thr,
        "hyp_words": len(hyp_items), "clips_detail": clip_rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"\nreport: {out}")
    return 0


def _pct(x) -> str:
    return "n/a" if x is None or x != x else f"{x * 100:.1f}%"


if __name__ == "__main__":
    raise SystemExit(main())
