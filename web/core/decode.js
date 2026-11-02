// CTC greedy decoding into timed, confidence-scored words (port of lipread/decode.py).
import { TIERS, UTTERANCE_GAP, EDGE_PAD, CONFIG } from "./config.js";

const MARK = "▁";   // "▁"

export const tierFor = (c) => (c >= TIERS.high ? "high" : c >= TIERS.medium ? "medium" : "low");

/** logprobs: Float32Array (T x V) row-major. Returns words [{text,start,end,confidence,tier}]. */
export function greedyWords(lp, T, V, vocab, t0, blank = 0) {
  const silent = vocab.map((p, i) => i === blank || (p.startsWith("<") && p.endsWith(">")));
  const runs = [];
  let prev = -1;
  for (let t = 0; t < T; t++) {
    let k = 0, best = -Infinity;
    const off = t * V;
    for (let j = 0; j < V; j++) if (lp[off + j] > best) { best = lp[off + j]; k = j; }
    if (silent[k]) { prev = -1; continue; }
    if (k === prev) runs[runs.length - 1][2] = t;
    else { runs.push([k, t, t]); prev = k; }
  }
  const groups = [];
  for (const r of runs) {
    if (vocab[r[0]].startsWith(MARK) || !groups.length) groups.push([r]);
    else groups[groups.length - 1].push(r);
  }
  const fps = CONFIG.fps, words = [];
  for (const g of groups) {
    const text = g.map(([k]) => vocab[k].replaceAll(MARK, "")).join("").trim();
    if (!text) continue;
    let sum = 0, n = 0;
    for (const [k, a, b] of g) for (let t = a; t <= b; t++) { sum += Math.exp(lp[t * V + k]); n++; }
    const confidence = sum / n;
    words.push({ text, start: t0 + g[0][1] / fps, end: t0 + (g[g.length - 1][2] + 1) / fps,
                 confidence, tier: tierFor(confidence), boosted: false, original: null });
  }
  return words;
}

/** Split into utterances at silences > gap, stretch words to readable extents, clamp to window. */
export function segment(words, t0, tEnd, gap = UTTERANCE_GAP) {
  const utts = [];
  let cur = [];
  for (const w of words) {
    if (cur.length && w.start - cur[cur.length - 1].end > gap) { utts.push(cur); cur = []; }
    cur.push(w);
  }
  if (cur.length) utts.push(cur);
  return utts.map((ws) => {
    const spikes = ws.map((w) => [w.start, w.end]);
    ws.forEach((w, i) => {
      const [s, e] = spikes[i];
      w.start = i > 0 ? (spikes[i - 1][1] + s) / 2 : Math.max(t0, s - EDGE_PAD);
      w.end = i + 1 < ws.length ? (e + spikes[i + 1][0]) / 2 : Math.min(tEnd, e + EDGE_PAD);
    });
    const confidence = ws.reduce((a, w) => a + w.confidence, 0) / ws.length;
    return { start: ws[0].start, end: ws[ws.length - 1].end, text: ws.map((w) => w.text).join(" "),
             confidence, words: ws };
  });
}
