// Readability gate (port of lipread/quality.py, simplified for the phone).
import { GATE } from "./config.js";

const median = (a) => { const s = a.slice().sort((x, y) => x - y); const n = s.length; return n ? (n % 2 ? s[n >> 1] : (s[n / 2 - 1] + s[n / 2]) / 2) : 0; };

/** Rough absolute yaw (deg) from YuNet points: nose offset from the eye midpoint / half eye distance. */
export function yawDeg(p5) {
  const [le, re, nose] = p5;
  const half = Math.hypot(re[0] - le[0], re[1] - le[1]) / 2;
  if (half < 1e-6) return 90;
  const r = (nose[0] - (le[0] + re[0]) / 2) / half;
  return Math.abs(Math.asin(Math.max(-1, Math.min(1, r))) * 180 / Math.PI);
}

/** Mouth motion on ALIGNED 88x88 crops (alignment already cancels head motion). */
export function motionEnergy(crops, T) {
  const S = 88, n = S * S, diffs = [];
  for (let t = 1; t < T; t++) {
    let sum = 0, cnt = 0;
    for (let y = Math.floor(S / 3); y < S - 8; y++)
      for (let x = 8; x < S - 8; x++) { const i = y * S + x; sum += Math.abs(crops[t * n + i] - crops[(t - 1) * n + i]); cnt++; }
    diffs.push(sum / cnt / 255);
  }
  if (!diffs.length) return 0;
  const s = diffs.sort((a, b) => a - b);
  return s[Math.floor(0.75 * (s.length - 1))];
}

/** Median Laplacian variance of the crops (higher = sharper). Sampled every 4th frame. */
export function sharpness(crops, T) {
  const S = 88, n = S * S, vals = [];
  for (let t = 0; t < T; t += 4) {
    let sum = 0, sq = 0, cnt = 0;
    for (let y = 1; y < S - 1; y++) for (let x = 1; x < S - 1; x++) {
      const i = t * n + y * S + x;
      const l = crops[i - S] + crops[i + S] + crops[i - 1] + crops[i + 1] - 4 * crops[i];
      sum += l; sq += l * l; cnt++;
    }
    vals.push(sq / cnt - (sum / cnt) ** 2);
  }
  return median(vals);
}

/** track: {dets:[{h, pts}], nFrames}, stride: detector stride. Returns quality report. */
export function assess(track, crops, T, stride) {
  const facePx = median(track.dets.map((d) => d.h));
  const yaw = median(track.dets.map((d) => yawDeg(d.pts)));
  // the detector cadence for this face (full-frame passes every `stride`, zoom tiles less often)
  const fr = track.dets.map((d) => d.f).filter((x) => x !== undefined).sort((a, b) => a - b);
  let cadence = stride;
  if (fr.length > 1) { cadence = Infinity; for (let i = 1; i < fr.length; i++) if (fr[i] > fr[i - 1]) cadence = Math.min(cadence, fr[i] - fr[i - 1]); }
  const expected = Math.floor((track.nFrames - 1) / cadence) + 1;
  const coverage = Math.min(1, track.dets.length / expected);
  const motion = motionEnergy(crops, T), sharp = sharpness(crops, T);
  const reasons = [];
  let tier = "readable";
  const hard = (cond, msg) => { if (cond) { tier = "unreadable"; reasons.push(msg); } };
  const soft = (cond, msg) => { if (cond) { if (tier === "readable") tier = "marginal"; reasons.push(msg); } };
  hard(facePx < GATE.facePxMarginal, `Face is only ${Math.round(facePx)} px tall — too far away to read`);
  hard(yaw > GATE.yawMarginal, `Turned about ${Math.round(yaw)}° away from the camera`);
  if (tier !== "unreadable") hard(coverage < GATE.coverageMin, "Face is hidden or lost for much of the clip");
  if (tier !== "unreadable") {
    soft(facePx < GATE.facePxReadable, `Small face (${Math.round(facePx)} px) — expect more mistakes`);
    soft(yaw > GATE.yawReadable, `Partly turned away (about ${Math.round(yaw)}°)`);
    soft(sharp < GATE.sharpMarginal, "Mouth looks blurry");
    soft(motion < GATE.motionMin, "Mouth barely moves — may not be speaking");
  }
  return { tier, reasons, facePx, yaw, coverage, motion, sharpness: sharp };
}
