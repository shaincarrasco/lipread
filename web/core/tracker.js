// Online face tracker + shot-cut detector (port of lipread/tracker.py and shots.py ideas).
import { iou } from "./yunet.js";

/** Greedy IoU tracker with a centre-distance fallback for fast pans. Tracks never cross a cut. */
export class Tracker {
  constructor({ iouTh = 0.3, maxGap = 8, maxFaces = 12 } = {}) {
    this.iouTh = iouTh; this.maxGap = maxGap; this.maxFaces = maxFaces;
    this.nextId = 1; this.active = []; this.all = [];
  }

  cut() { this.active = []; }

  /** dets: faces (source px) for frame f, or null when the detector did not run on this frame. */
  update(f, dets) {
    if (dets) {
      dets = dets.slice().sort((a, b) => b.w * b.h - a.w * a.h).slice(0, this.maxFaces);
      const pairs = [];
      for (const t of this.active)
        dets.forEach((d, j) => {
          let s = iou(t.last, d);
          if (s < this.iouTh) {
            const dx = (t.last.x + t.last.w / 2) - (d.x + d.w / 2);
            const dy = (t.last.y + t.last.h / 2) - (d.y + d.h / 2);
            const ratio = d.h / t.last.h;
            if (Math.hypot(dx, dy) < 0.5 * t.last.h && ratio > 0.5 && ratio < 2) s = this.iouTh + 0.01;
          }
          if (s >= this.iouTh) pairs.push([s, t, j]);
        });
      pairs.sort((a, b) => b[0] - a[0]);
      const usedT = new Set(), usedD = new Set();
      for (const [, t, j] of pairs) {
        if (usedT.has(t) || usedD.has(j)) continue;
        usedT.add(t); usedD.add(j); this._add(t, f, dets[j]);
      }
      dets.forEach((d, j) => {
        if (usedD.has(j)) return;
        const t = { id: this.nextId++, dets: [], last: d, lastFrame: f, firstFrame: f };
        this.active.push(t); this.all.push(t); this._add(t, f, d);
      });
    }
    this.active = this.active.filter((t) => f - t.lastFrame <= this.maxGap);
    return this.active;
  }

  _add(t, f, d) { t.dets.push({ f, ...d }); t.last = d; t.lastFrame = f; }
}

/** Hue/saturation + brightness histogram cut detector on a tiny thumbnail. */
export class CutDetector {
  constructor(threshold = 0.45) { this.th = threshold; this.prev = null; }

  /** rgba: Uint8ClampedArray of a small frame (e.g. 64x36). Returns true if a cut starts here. */
  update(rgba) {
    const hs = new Float32Array(16 * 8), v = new Float32Array(16);
    for (let p = 0; p < rgba.length; p += 4) {
      const r = rgba[p] / 255, g = rgba[p + 1] / 255, b = rgba[p + 2] / 255;
      const max = Math.max(r, g, b), min = Math.min(r, g, b), d = max - min;
      let h = 0;
      if (d > 1e-6) h = max === r ? ((g - b) / d + 6) % 6 : max === g ? (b - r) / d + 2 : (r - g) / d + 4;
      const s = max > 0 ? d / max : 0;
      hs[Math.min(15, Math.floor(h / 6 * 16)) * 8 + Math.min(7, Math.floor(s * 8))]++;
      v[Math.min(15, Math.floor(max * 16))]++;
    }
    const cur = [hs, v];
    let cut = false;
    if (this.prev) cut = 1 - Math.min(corr(this.prev[0], hs), corr(this.prev[1], v)) > this.th;
    this.prev = cur;
    return cut;
  }
}

function corr(a, b) {
  const n = a.length; let ma = 0, mb = 0;
  for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
  ma /= n; mb /= n;
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < n; i++) { const x = a[i] - ma, y = b[i] - mb; num += x * y; da += x * x; db += y * y; }
  return da > 0 && db > 0 ? num / Math.sqrt(da * db) : (da === db ? 1 : 0);
}
