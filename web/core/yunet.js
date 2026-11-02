// YuNet (OpenCV Zoo, face_detection_yunet_2023mar) pre/post-processing.
// The ONNX graph has a fixed 640x640 BGR float input (0..255, no normalisation) and raw per-stride
// outputs (cls/obj/bbox/kps for strides 8, 16, 32). decode() reproduces cv::FaceDetectorYN exactly
// (verified against OpenCV on test frames: max difference 0.00 px).

export const YUNET_SIZE = 640;

/** RGBA ImageData (640x640) -> Float32Array NCHW BGR 0..255, as OpenCV's blobFromImage does. */
export function toTensorData(rgba) {
  const n = YUNET_SIZE * YUNET_SIZE;
  const out = new Float32Array(3 * n);
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    out[i] = rgba[p + 2];          // B
    out[n + i] = rgba[p + 1];      // G
    out[2 * n + i] = rgba[p];      // R
  }
  return out;
}

/** outputs: {cls_8: Float32Array, ...}. Returns faces in 640-space: {x,y,w,h,score,kps:[10]}. */
export function decode(outputs, scoreTh = 0.6, nmsTh = 0.3, topK = 5000) {
  const faces = [];
  for (const s of [8, 16, 32]) {
    const n = YUNET_SIZE / s;
    const cls = outputs[`cls_${s}`], obj = outputs[`obj_${s}`];
    const bb = outputs[`bbox_${s}`], kp = outputs[`kps_${s}`];
    for (let idx = 0; idx < n * n; idx++) {
      const c = Math.min(Math.max(cls[idx], 0), 1), o = Math.min(Math.max(obj[idx], 0), 1);
      const score = Math.sqrt(c * o);
      if (score <= scoreTh) continue;
      const r = Math.floor(idx / n), col = idx % n;
      const cx = (col + bb[idx * 4]) * s, cy = (r + bb[idx * 4 + 1]) * s;
      const w = Math.exp(bb[idx * 4 + 2]) * s, h = Math.exp(bb[idx * 4 + 3]) * s;
      const k = new Array(10);
      for (let j = 0; j < 5; j++) {
        k[2 * j] = (kp[idx * 10 + 2 * j] + col) * s;
        k[2 * j + 1] = (kp[idx * 10 + 2 * j + 1] + r) * s;
      }
      faces.push({ x: cx - w / 2, y: cy - h / 2, w, h, score, kps: k });
    }
  }
  faces.sort((a, b) => b.score - a.score);
  return nms(faces.slice(0, topK), nmsTh);
}

export function iou(a, b) {
  const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w), y2 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const u = a.w * a.h + b.w * b.h - inter;
  return u > 0 ? inter / u : 0;
}

function nms(sorted, th) {
  const keep = [];
  for (const f of sorted) if (keep.every((k) => iou(k, f) <= th)) keep.push(f);
  return keep;
}

/**
 * Plan the detector passes for a frame of size W x H: one pass on the whole frame scaled to fit
 * 640, plus zoom tiles on large frames so small faces (fans, sideline) are seen at full detail.
 * Each pass: {sx, sy, sw, sh} source rect; the rect is scaled by k = 640 / max(sw, sh).
 */
export function planPasses(W, H, zoom = true) {
  const passes = [{ sx: 0, sy: 0, sw: W, sh: H }];
  const long = Math.max(W, H);
  if (zoom && long > 960) {
    if (W >= H) {                          // landscape: left and right halves, 10% overlap
      const tw = Math.round(W * 0.55);
      passes.push({ sx: 0, sy: 0, sw: tw, sh: H }, { sx: W - tw, sy: 0, sw: tw, sh: H });
    } else {                               // portrait: top and bottom halves
      const th = Math.round(H * 0.55);
      passes.push({ sx: 0, sy: 0, sw: W, sh: th }, { sx: 0, sy: H - th, sw: W, sh: th });
    }
  }
  return passes.map((p) => ({ ...p, k: YUNET_SIZE / Math.max(p.sw, p.sh) }));
}

/** Map a 640-space face back to source pixels for a pass, ordering points image-left first. */
export function toSource(face, pass) {
  const m = (v, o) => v / pass.k + o;
  const k = face.kps;
  let pts = [];
  for (let j = 0; j < 5; j++) pts.push([m(k[2 * j], pass.sx), m(k[2 * j + 1], pass.sy)]);
  // eyes and mouth corners sorted by x: [left eye, right eye, nose, left mouth, right mouth]
  const eyes = [pts[0], pts[1]].sort((a, b) => a[0] - b[0]);
  const mouth = [pts[3], pts[4]].sort((a, b) => a[0] - b[0]);
  pts = [eyes[0], eyes[1], pts[2], mouth[0], mouth[1]];
  return { x: m(face.x, pass.sx), y: m(face.y, pass.sy), w: face.w / pass.k, h: face.h / pass.k,
           score: face.score, pts };
}

/** Merge detections from several passes (same face seen by the full pass and a tile). */
export function mergePasses(faces, th = 0.4) {
  faces.sort((a, b) => b.score - a.score);
  return nms(faces, th);
}
