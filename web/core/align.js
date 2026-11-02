// Mouth-crop alignment for the phone (port of lipread/align.py, Auto-AVSR preprocessing, Apache-2.0).
// Differences from the desktop: keypoints come from YuNet + a learned correction (no MediaPipe), and
// crops are sampled from small stored mouth-region crops instead of full frames (memory).

const TARGET = 256, PATCH = 96, CROP = 88, SMOOTH_HALF = 6;

let GEO = null;
/** geometry.json: {kpmap_W: 11x8, stable_reference: 4x2, mean, std} */
export function setGeometry(g) { GEO = g; }

/** YuNet 5 points [[x,y]...] (l-eye, r-eye, nose, l-mouth, r-mouth) -> 4 alignment points. */
export function correctedKeypoints(p5) {
  const [le, re] = p5;
  const mx = (le[0] + re[0]) / 2, my = (le[1] + re[1]) / 2;
  let vx = re[0] - le[0], vy = re[1] - le[1];
  const d = Math.hypot(vx, vy) || 1;
  const ex = [vx / d, vy / d], ey = [-ex[1], ex[0]];
  const x = [];
  for (const p of p5) {
    const px = p[0] - mx, py = p[1] - my;
    x.push((px * ex[0] + py * ex[1]) / d, (px * ey[0] + py * ey[1]) / d);
  }
  x.push(1);
  const W = GEO.kpmap_W, out = [];
  for (let j = 0; j < 4; j++) {
    let u = 0, v = 0;
    for (let i = 0; i < 11; i++) { u += x[i] * W[i][2 * j]; v += x[i] * W[i][2 * j + 1]; }
    // back to image coordinates: p = mid + d * (u*ex + v*ey)
    out.push([mx + d * (u * ex[0] + v * ey[0]), my + d * (u * ex[1] + v * ey[1])]);
  }
  return out;
}

export const interEye = (kp) => Math.hypot(kp[1][0] - kp[0][0], kp[1][1] - kp[0][1]);

/** Fill missing frames (null) by linear interpolation; edges held. kps: array of 4x2 or null. */
export function interpolate(kps) {
  const idx = kps.map((k, i) => (k ? i : -1)).filter((i) => i >= 0);
  if (!idx.length) return null;
  const out = new Array(kps.length);
  for (let i = 0; i < kps.length; i++) {
    if (kps[i]) { out[i] = kps[i]; continue; }
    let a = -1, b = -1;
    for (const j of idx) { if (j < i) a = j; else { b = j; break; } }
    if (a < 0) out[i] = kps[b]; else if (b < 0) out[i] = kps[a];
    else {
      const w = (i - a) / (b - a);
      out[i] = kps[a].map((p, n) => [p[0] + w * (kps[b][n][0] - p[0]), p[1] + w * (kps[b][n][1] - p[1])]);
    }
  }
  return out;
}

/** Least-squares similarity transform src(4x2) -> dst(4x2). Returns [a, b, tx, ty] with
 *  x' = a*x - b*y + tx, y' = b*x + a*y + ty. */
export function similarity(src, dst) {
  const n = src.length;
  let sx = 0, sy = 0, dx = 0, dy = 0;
  for (let i = 0; i < n; i++) { sx += src[i][0]; sy += src[i][1]; dx += dst[i][0]; dy += dst[i][1]; }
  sx /= n; sy /= n; dx /= n; dy /= n;
  let num1 = 0, num2 = 0, den = 0;
  for (let i = 0; i < n; i++) {
    const x = src[i][0] - sx, y = src[i][1] - sy, u = dst[i][0] - dx, v = dst[i][1] - dy;
    num1 += x * u + y * v; num2 += x * v - y * u; den += x * x + y * y;
  }
  const a = den > 0 ? num1 / den : 1, b = den > 0 ? num2 / den : 0;
  return [a, b, dx - (a * sx - b * sy), dy - (b * sx + a * sy)];
}

/**
 * Build the model's 88x88 crops for one track.
 * kps:   per-frame 4x2 keypoints (already interpolated), length T
 * crops: per-frame stored crop {data: Uint8Array(S*S), cx, cy, k, size:S}; crop coords
 *        q = (p - c) * k + S/2 for a frame point p.
 * Returns Uint8Array(T*88*88).
 */
export function mouthCrops(kps, crops) {
  const T = kps.length, ref = GEO.stable_reference;
  const out = new Uint8Array(T * CROP * CROP);
  for (let i = 0; i < T; i++) {
    const m = Math.min(SMOOTH_HALF, i, T - 1 - i);
    // window mean, re-centred on the current frame (as in Auto-AVSR)
    const sm = [[0, 0], [0, 0], [0, 0], [0, 0]];
    for (let j = i - m; j <= i + m; j++) for (let q = 0; q < 4; q++) { sm[q][0] += kps[j][q][0]; sm[q][1] += kps[j][q][1]; }
    const cnt = 2 * m + 1;
    let cw = [0, 0], cc = [0, 0];
    for (let q = 0; q < 4; q++) {
      sm[q][0] /= cnt; sm[q][1] /= cnt;
      cw[0] += sm[q][0] / 4; cw[1] += sm[q][1] / 4; cc[0] += kps[i][q][0] / 4; cc[1] += kps[i][q][1] / 4;
    }
    for (let q = 0; q < 4; q++) { sm[q][0] += cc[0] - cw[0]; sm[q][1] += cc[1] - cw[1]; }

    const [a, b, tx, ty] = similarity(sm, ref);
    const mcx = a * sm[3][0] - b * sm[3][1] + tx, mcy = b * sm[3][0] + a * sm[3][1] + ty;
    const x0 = Math.round(Math.min(Math.max(mcx - PATCH / 2, 0), TARGET - PATCH)) + (PATCH - CROP) / 2;
    const y0 = Math.round(Math.min(Math.max(mcy - PATCH / 2, 0), TARGET - PATCH)) + (PATCH - CROP) / 2;
    // inverse similarity: p = R^-1 (r - t) / s
    const s2 = a * a + b * b;
    const c = crops[i], S = c.size, half = S / 2;
    const base = i * CROP * CROP;
    for (let v = 0; v < CROP; v++) {
      for (let u = 0; u < CROP; u++) {
        const rx = x0 + u - tx, ry = y0 + v - ty;
        const px = (a * rx + b * ry) / s2, py = (-b * rx + a * ry) / s2;
        out[base + v * CROP + u] = bilinear(c.data, S, (px - c.cx) * c.k + half, (py - c.cy) * c.k + half);
      }
    }
  }
  return out;
}

function bilinear(img, S, x, y) {
  const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0;
  const at = (xx, yy) => (xx < 0 || yy < 0 || xx >= S || yy >= S ? 0 : img[yy * S + xx]);
  const v = (1 - fy) * ((1 - fx) * at(x0, y0) + fx * at(x0 + 1, y0)) + fy * ((1 - fx) * at(x0, y0 + 1) + fx * at(x0 + 1, y0 + 1));
  return v + 0.5 | 0;
}
