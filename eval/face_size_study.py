"""Face-size study: shrink talking-head clips into a 1280x720 wide shot and measure WER vs face height.

For each video and each target face height (default 200,120,80,56,40,28,20 px):
  1. the clip is scaled so its face box is `target` px tall and pasted, centred, on a 1280x720 canvas
     (optional --blur K horizontal motion blur, applied to the shrunk face in canvas pixels);
  2. lipread.faces.FaceDetector finds faces per frame, refine() gives the 4 alignment points,
     lipread.tracker.track_faces links them, the longest track is aligned (mouth_crops) and read
     by the engine (greedy CTC);
  3. WER against the Whisper transcript of the original clip's first --max-seconds.

A clip where nothing is read counts as an empty hypothesis (WER 100% for that clip), because that is
what a viewer gets. The table also reports detection rate and WER over the clips that were read.
Targets larger than the clip's own face are not applicable (we never upscale) and are skipped.

--fps 50 reads a high-fps source at 50 fps and averages each pair of frames into one 25 fps frame
(a 25 fps camera with a full-frame exposure). On a 25 fps source it changes nothing.

Usage:
  python eval/face_size_study.py /path/to/clips --max-seconds 8
  python eval/face_size_study.py /path/to/clips --targets 120,80,40 --blur 5
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import _common as C
from lipread.types import Detection  # plain dataclass, safe to import even if the rest is in flux

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
CANVAS = (720, 1280)          # H, W
GREY = 90
DEFAULT_TARGETS = [200, 120, 80, 56, 40, 28, 20]


# ---------- detector: package FaceDetector, or the reference hybrid from scratch/degrade.py ----------

def make_detector(models_dir: str):
    try:
        from lipread.faces import FaceDetector
        from lipread.tracker import track_faces
        return FaceDetector(models_dir), track_faces, "package"
    except Exception as e:  # noqa: BLE001 - fall back to the reference functions, and say so
        print(f"note: lipread.faces/tracker not importable ({e}); using the reference hybrid detector")
        return _FallbackDetector(models_dir), _simple_tracks, "reference"


class _FallbackDetector:
    """YuNet finds faces anywhere; MediaPipe refines keypoints on an upscaled face crop (as in degrade.py)."""

    def __init__(self, models_dir: str):
        import mediapipe as mp
        from mediapipe.tasks.python import vision, BaseOptions
        self._mp_img = mp
        self._yn = None
        self._dir = models_dir
        self._mpd = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=os.path.join(models_dir, "blaze_face_short_range.tflite")),
            min_detection_confidence=0.5))

    def detect(self, frame: np.ndarray, idx: int) -> list[Detection]:
        h, w = frame.shape[:2]
        if self._yn is None:
            self._yn = cv2.FaceDetectorYN.create(os.path.join(self._dir, "face_detection_yunet_2023mar.onnx"),
                                                 "", (w, h), 0.5, 0.3, 5000)
        self._yn.setInputSize((w, h))
        _, faces = self._yn.detect(frame)
        if faces is None:
            return []
        out = []
        for r in faces:
            out.append(Detection(frame=idx, box=(float(r[0]), float(r[1]), float(r[2]), float(r[3])),
                                 score=float(r[14]), points5=r[4:14].reshape(5, 2).astype(np.float64)))
        return out

    def refine(self, frame: np.ndarray, det: Detection) -> np.ndarray:
        x, y, bw, bh = det.box
        cx, cy = x + bw / 2, y + bh / 2
        s = max(bw, bh) * 1.8
        S = int(s)
        x0, y0 = int(cx - s / 2), int(cy - s / 2)
        pad = cv2.copyMakeBorder(frame, S, S, S, S, cv2.BORDER_CONSTANT)
        crop = pad[y0 + S: y0 + 2 * S, x0 + S: x0 + 2 * S]
        if crop.size == 0:
            return _yunet_kp4(det)
        sc = 256 / crop.shape[0]
        up = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_CUBIC)
        kp = self._mp_points(up)
        if kp is None:
            return _yunet_kp4(det)
        return kp / sc + np.array([x0, y0], np.float64)

    def _mp_points(self, up_bgr: np.ndarray) -> np.ndarray | None:
        mp = self._mp_img
        h, w = up_bgr.shape[:2]
        r = self._mpd.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                      data=cv2.cvtColor(up_bgr, cv2.COLOR_BGR2RGB)))
        if not r.detections:
            return None
        d = max(r.detections, key=lambda d: d.bounding_box.width * d.bounding_box.height)
        k = [(p.x * w, p.y * h) for p in d.keypoints[:4]]
        eyes = sorted(k[:2])
        return np.array([eyes[0], eyes[1], k[2], k[3]], np.float64)


def _yunet_kp4(det: Detection) -> np.ndarray:
    eyes = sorted(det.points5[:2].tolist())
    return np.array([eyes[0], eyes[1], det.points5[2], det.points5[3:5].mean(0)], np.float64)


def _simple_tracks(dets_per_frame, shots):
    """Reference tracker: one track of the largest detection per frame (single-face wide shots)."""
    from lipread.types import FaceTrack
    tr = FaceTrack(track_id=0, shot_id=0)
    for i, dets in enumerate(dets_per_frame):
        if dets:
            tr.detections.append(max(dets, key=lambda d: d.box[2] * d.box[3]))
    return [tr] if tr.detections else []


# ---------- video frames ----------

def _size(path: str) -> tuple[int, int]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height", "-of", "csv=p=0", path],
                         capture_output=True, text=True, check=True).stdout.strip()
    w, h = out.split(",")[:2]
    return int(w), int(h)


def stream_frames(path: str, seconds: float, ratio: int = 1):
    """Yield BGR frames at 25 fps. ratio>1: read at 25*ratio and average each group of `ratio` frames."""
    w, h = _size(path)
    rate = 25 * ratio
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-t", str(seconds), "-vf", f"fps={rate}",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    nbytes = w * h * 3
    group = []
    try:
        while True:
            buf = proc.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            frame = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            if ratio == 1:
                yield frame.copy()
            else:
                group.append(frame.astype(np.float32))
                if len(group) == ratio:
                    yield np.mean(group, axis=0).round().astype(np.uint8)
                    group = []
    finally:
        proc.stdout.close()
        proc.wait()


def base_face_px(det, path: str, n: int = 25) -> float | None:
    """Median face box height in the original clip (first n frames). Used to scale to the target."""
    hs = []
    for i, frame in enumerate(stream_frames(path, n / 25)):
        dets = det.detect(frame, i)
        if dets:
            hs.append(max(d.box[3] for d in dets))
    return float(np.median(hs)) if hs else None


def wide_frame(frame: np.ndarray, scale: float, blur: int) -> np.ndarray:
    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if blur > 1:
        kernel = np.ones((1, blur), np.float32) / blur       # horizontal motion blur
        small = cv2.filter2D(small, -1, kernel)
    H, W = CANVAS
    canvas = np.full((H, W, 3), GREY, np.uint8)
    h, w = small.shape[:2]
    if h > H or w > W:                                       # never larger than the canvas
        f = min(H / h, W / w)
        small = cv2.resize(small, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        h, w = small.shape[:2]
    y0, x0 = (H - h) // 2, (W - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = small
    return canvas


# ---------- one (clip, target) run ----------

def run_target(det, track_fn, engine, path: str, scale: float, seconds: float, blur: int, ratio: int) -> dict:
    from lipread.align import interpolate_keypoints, mouth_crops
    gray_frames: list[np.ndarray] = []
    dets_per_frame: list[list[Detection]] = []
    for i, frame in enumerate(stream_frames(path, seconds, ratio)):
        wide = wide_frame(frame, scale, blur)
        dets = det.detect(wide, i)
        for d in dets:
            d.keypoints4 = det.refine(wide, d)
        dets_per_frame.append(dets)
        gray_frames.append(cv2.cvtColor(wide, cv2.COLOR_BGR2GRAY))
    T = len(gray_frames)
    out = {"frames": T, "frames_with_face": sum(1 for d in dets_per_frame if d), "track_frames": 0,
           "hyp": "", "status": "ok"}
    if T == 0:
        out["status"] = "no_frames"
        return out
    tracks = track_fn(dets_per_frame, [(0, T - 1)])
    if not tracks:
        out["status"] = "no_track"
        return out
    tr = max(tracks, key=lambda t: t.num_frames)
    out["track_frames"] = tr.num_frames
    kps = [None] * tr.num_frames
    for d in tr.detections:
        kps[d.frame - tr.start_frame] = d.keypoints4
    interp = interpolate_keypoints(kps)
    if interp is None:
        out["status"] = "no_keypoints"
        return out
    crops = mouth_crops(gray_frames[tr.start_frame: tr.end_frame + 1], interp)
    lp = engine.logprobs(crops)
    words = C.greedy_words(lp, engine.vocab, engine.blank)
    out["hyp"] = C.norm_text(" ".join(w.text for w in words))
    return out


# ---------- main ----------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video_dir")
    ap.add_argument("--targets", default=",".join(map(str, DEFAULT_TARGETS)), help="target face heights in px")
    ap.add_argument("--max-seconds", type=float, default=8.0)
    ap.add_argument("--whisper", default="small.en")
    ap.add_argument("--model", default=C.DEFAULT_MODEL)
    ap.add_argument("--models-dir", default=os.environ.get("LIPREAD_HOME", os.path.expanduser("~/.cache/lipread/models")))
    ap.add_argument("--blur", type=int, default=0, help="horizontal motion-blur kernel in canvas px (0 = off)")
    ap.add_argument("--fps", type=int, default=25, choices=[25, 50, 75, 100], help="source rate to read at")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(C.EVAL_DIR / "reports" / "face_size_study.json"))
    args = ap.parse_args(argv)

    targets = [int(t) for t in args.targets.split(",") if t.strip()]
    ratio = args.fps // 25
    det, track_fn, mode = make_detector(args.models_dir)
    engine = C.load_engine(args.model)
    print(f"detector: {mode}   engine: {engine.info.get('name')}   blur={args.blur}   fps={args.fps}")

    vids = sorted(p for p in Path(args.video_dir).iterdir() if p.suffix.lower() in VIDEO_EXT)
    if args.limit:
        vids = vids[: args.limit]

    records = []
    for p in vids:
        t0 = time.perf_counter()
        ref, _ = C.whisper_reference(str(p), args.max_seconds, args.whisper)
        base = base_face_px(det, str(p))
        print(f"\n{p.name}  reference words={len(ref.split())}  own face={base and round(base)} px")
        if not ref or base is None:
            print("  skipped: " + ("no speech in reference" if not ref else "no face in the clip"))
            records.append({"clip": p.name, "status": "skipped", "ref": ref, "base_face_px": base})
            continue
        for target in targets:
            scale = target / base
            if scale > 1.0:
                records.append({"clip": p.name, "target": target, "status": "not_applicable_upscale",
                                "base_face_px": base, "ref": ref})
                continue
            r = run_target(det, track_fn, engine, str(p), scale, args.max_seconds, args.blur, ratio)
            r.update({"clip": p.name, "target": target, "scale": scale, "base_face_px": base, "ref": ref})
            r["wer"] = C.corpus_wer([ref], [r["hyp"]])          # empty hyp = 100% when nothing was read
            r["det_rate"] = r["frames_with_face"] / r["frames"] if r["frames"] else 0.0
            records.append(r)
            print(f"  target={target:>3}px  scale={scale:.2f}  det={r['det_rate'] * 100:4.0f}%  "
                  f"track={r['track_frames']:>3}f  status={r['status']:<13}  WER={r['wer'] * 100:5.1f}%  "
                  f"hyp='{r['hyp'][:60]}'", flush=True)
        print(f"  ({time.perf_counter() - t0:.0f}s)")

    print()
    print(f"{'target px':>10}{'clips':>7}{'det rate':>10}{'WER all':>10}{'read clips':>12}{'WER read':>10}")
    summary = []
    for target in targets:
        rs = [r for r in records if r.get("target") == target and "wer" in r]
        if not rs:
            continue
        refs = [r["ref"] for r in rs]
        hyps = [r["hyp"] for r in rs]
        read = [r for r in rs if r["status"] == "ok" and r["hyp"]]
        row = {"target_px": target, "clips": len(rs),
               "det_rate": float(np.mean([r["det_rate"] for r in rs])),
               "wer_all": C.corpus_wer(refs, hyps),
               "read_clips": len(read),
               "wer_read": C.corpus_wer([r["ref"] for r in read], [r["hyp"] for r in read]) if read else None}
        summary.append(row)
        wr = "n/a" if row["wer_read"] is None else f"{row['wer_read'] * 100:.1f}%"
        print(f"{target:>10}{row['clips']:>7}{row['det_rate'] * 100:>9.0f}%{row['wer_all'] * 100:>9.1f}%"
              f"{row['read_clips']:>12}{wr:>10}")

    report = {"study": "face_size", "detector": mode, "engine": engine.info, "targets": targets,
              "max_seconds": args.max_seconds, "blur": args.blur, "fps": args.fps,
              "summary": summary, "records": records}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=float))
    print(f"\nreport: {out}")
    del det, engine                      # release mediapipe before interpreter shutdown
    return 0


if __name__ == "__main__":
    sys.exit(main())
