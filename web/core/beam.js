// Context-aware decoding: CTC prefix beam search with a word trigram LM (shallow fusion).
// Port of lipread/beam.py: same algorithm, same defaults, stable sorts for the same tie-breaking.
//
//   score = log P_model(pieces) + alpha * log P_LM(words) + beta * n_words
//
// Word confidence and tiers come from the raw (unsoftened) posteriors, as in Python.
// Prefixes are trie nodes, so a prefix key is one integer and comparing keys is comparing ids.

import { CONFIG } from "./config.js";
import { tierFor } from "./decode.js";
import { BOS, UNK } from "./lm.js";

const MARK = "▁";   // "▁": a piece starting with it begins a new word
const NEG = -1e30;
const FPS = CONFIG.fps;

export const DEFAULT_OPTIONS = Object.freeze({
  beam: 16,
  alpha: 0.3,              // LM weight (tuned on LRS2 + half of LRS3)
  beta: 1.0,               // word insertion bonus
  unk_penalty: -4.0,       // extra penalty for words outside the LM vocabulary
  prune: 10.0,             // per frame, ignore pieces this far (nats) below the best
  topk: 24,                // per frame, at most this many pieces
  n_alternatives: 3,
  posterior_scale: 0.2,    // hypothesis scores are scaled by this before the word-posterior softmax
  temperature: 1.0,        // > 1 softens the over-confident posteriors so context can weigh in
  names: Object.freeze([]), // roster names: in-vocabulary with a flat probability
  name_logp: Math.log(1e-4),
  blank: 0,
});

// log(exp(a) + exp(b)) with NEG as "log 0" (same as _lse in beam.py).
function lse(a, b) {
  if (a < b) { const t = a; a = b; b = t; }
  return b <= NEG ? a : a + Math.log1p(Math.exp(b - a));
}

// Python round(x, 4): exact decimal rounding, exact ties to even.
function round4(x) {
  const s = x.toFixed(60);
  const dot = s.indexOf(".");
  let n = Number(s.slice(0, dot) + s.slice(dot + 1, dot + 5));
  const d = s.charCodeAt(dot + 5) - 48;
  const restNonZero = /[1-9]/.test(s.slice(dot + 6));
  if (d > 5 || (d === 5 && (restNonZero || n % 2 === 1))) n += 1;
  return n / 10000;
}

function toArray(list) {
  const out = [];
  for (let n = list; n; n = n.prev) out.push(n);
  return out.reverse();
}

// A hypothesis. `words` is an immutable linked list of completed words (newest first).
class Hyp {
  constructor(pb, pnb, lm, hu, hv, cur, curStart, words, lastFrame, confSum, confN, node) {
    this.pb = pb;                // log prob ending in blank
    this.pnb = pnb;              // log prob ending in a piece
    this.lm = lm;                // alpha * LM + beta * words, for completed words
    this.hu = hu;                // word id two back (null = none)
    this.hv = hv;                // word id of the last completed word
    this.cur = cur;              // current (incomplete) word text
    this.curStart = curStart;
    this.words = words;          // { prev, text, start, end, cs, cn }
    this.lastFrame = lastFrame;
    this.confSum = confSum;
    this.confN = confN;
    this.node = node;            // trie node of the prefix
  }
}

export class BeamDecoder {
  /** vocab: piece strings; lm: NgramLM or null; opts: any subset of DEFAULT_OPTIONS (Python names). */
  constructor(vocab, lm, opts = {}) {
    this.vocab = vocab;
    this.V = vocab.length;
    this.lm = lm ?? null;
    this.o = { ...DEFAULT_OPTIONS, ...opts };
    const blank = this.o.blank;
    this.blank = blank;
    this.silent = new Uint8Array(this.V);
    this.startsWord = new Uint8Array(this.V);
    this.text = new Array(this.V);
    vocab.forEach((p, i) => {
      this.silent[i] = i === blank || (p.startsWith("<") && p.endsWith(">")) ? 1 : 0;
      this.startsWord[i] = p.startsWith(MARK) ? 1 : 0;
      this.text[i] = p.replaceAll(MARK, "");
    });
    this.names = new Set();
    for (const name of this.o.names) {
      for (const n of [name, ...name.split(/\s+/)]) if (n.trim()) this.names.add(n.toUpperCase());
    }
    this._row = new Float64Array(this.V);
    const k = Math.min(this.o.topk, this.V);
    this._topI = new Int32Array(k);
    this._topV = new Float64Array(k);
  }

  /**
   * logprobs: Float32Array (T x V) row-major, raw model log-probs. t0: source time of frame 0.
   * Returns { words, alternatives }: words as in decode.js plus p and alternatives [{text, p}];
   * alternatives[i] = { p, alternatives } for words[i], as the Python decoder returns them.
   */
  decode(logprobs, T, V, t0 = 0) {
    if (V !== this.V) throw new Error(`logprobs must have ${this.V} columns, got ${V}`);
    if (logprobs.length < T * V) throw new Error(`logprobs too short for T=${T}, V=${V}`);
    const o = this.o, Vn = this.V, row = this._row, blank = this.blank;
    const soft = o.temperature !== 1.0;
    const topI = this._topI, topV = this._topV, k = topI.length;
    this._trie = new Map();      // (parent node * V + piece) -> child node
    this._piece = [-1];          // node -> piece (node 0 is the empty prefix)

    let beams = [new Hyp(0, NEG, 0, null, BOS, "", -1, null, -1, 0, 0, 0)];
    for (let t = 0; t < T; t++) {
      const off = t * Vn;
      // Frame row: softened (temperature) or raw log-probs, as in Python.
      if (soft) {
        const temp = o.temperature;
        let zmax = -Infinity;
        for (let j = 0; j < Vn; j++) {
          const z = logprobs[off + j] / temp;
          row[j] = z;
          if (z > zmax) zmax = z;
        }
        let S = 0;
        for (let j = 0; j < Vn; j++) { row[j] -= zmax; S += Math.exp(row[j]); }
        const ls = Math.log(S);
        for (let j = 0; j < Vn; j++) row[j] -= ls;
      } else {
        for (let j = 0; j < Vn; j++) row[j] = logprobs[off + j];
      }

      // Top-k pieces of the frame (by value, ties to the lower index), then the prune window.
      let n = 0;
      for (let j = 0; j < Vn; j++) {
        const v = row[j];
        let p;
        if (n < k) {
          p = n++;
        } else if (v > topV[k - 1]) {
          p = k - 1;
        } else {
          continue;
        }
        while (p > 0 && topV[p - 1] < v) { topV[p] = topV[p - 1]; topI[p] = topI[p - 1]; p--; }
        topV[p] = v; topI[p] = j;
      }
      const thr = topV[0] - o.prune;
      const cand = [];
      for (let i = 0; i < n; i++) {
        const c = topI[i];
        if (topV[i] >= thr && !this.silent[c]) cand.push(c);
      }

      const nxt = new Map();     // node -> Hyp, in insertion order (as the Python dict)
      const get = (node, parent) => {
        let h = nxt.get(node);
        if (h === undefined) {
          h = new Hyp(NEG, NEG, parent.lm, parent.hu, parent.hv, parent.cur, parent.curStart, parent.words,
            parent.lastFrame, parent.confSum, parent.confN, node);
          nxt.set(node, h);
        }
        return h;
      };

      for (const h of beams) {
        const tot = lse(h.pb, h.pnb);
        // blank: same prefix
        const nb = get(h.node, h);
        nb.pb = lse(nb.pb, tot + row[blank]);
        // repeated last piece without a blank in between: same prefix
        if (h.node !== 0) {
          const s = get(h.node, h);
          const rep = h.pnb + row[this._piece[h.node]];
          s.pnb = lse(s.pnb, rep);
          if (rep > tot + row[blank]) s.lastFrame = t;   // piece still being "spoken": extend the word
        }
        let done = null;         // h with its current word completed (made at most once)
        for (const c of cand) {
          const p = (h.node !== 0 && this._piece[h.node] === c ? h.pb : tot) + row[c];
          const nkey = this._child(h.node, c);
          let nh = nxt.get(nkey);
          if (nh === undefined) {
            let base = h;
            if (this.startsWord[c] && h.cur) {
              if (done === null) done = this._complete(h);
              base = done;
            }
            nh = new Hyp(NEG, NEG, base.lm, base.hu, base.hv, base.cur, base.curStart, base.words,
              base.lastFrame, base.confSum, base.confN, nkey);
            if (this.startsWord[c] || !nh.cur) {
              nh.cur = this.text[c]; nh.curStart = t; nh.confSum = 0; nh.confN = 0;
            } else {
              nh.cur += this.text[c];
            }
            nh.lastFrame = t;
            nh.confSum += Math.exp(logprobs[off + c]);   // raw posterior
            nh.confN += 1;
            nxt.set(nkey, nh);
          }
          nh.pnb = lse(nh.pnb, p);
        }
      }

      // Keep the `beam` best by total score; ties keep insertion order (stable, as Python's sorted).
      const arr = [...nxt.values()];
      const sc = arr.map((h) => lse(h.pb, h.pnb) + h.lm);
      const order = arr.map((_, i) => i).sort((i, j) => (sc[j] > sc[i] ? 1 : sc[j] < sc[i] ? -1 : i - j));
      beams = order.slice(0, o.beam).map((i) => arr[i]);
    }

    const fin = beams.map((h) => {
      const f = this._complete(h);
      return { score: lse(f.pb, f.pnb) + f.lm, h: f };
    });
    fin.sort((a, b) => (b.score > a.score ? 1 : b.score < a.score ? -1 : 0));
    return this._words(fin, t0);
  }

  _child(node, c) {
    const key = node * this.V + c;
    let ch = this._trie.get(key);
    if (ch === undefined) {
      ch = this._piece.length;
      this._piece.push(c);
      this._trie.set(key, ch);
    }
    return ch;
  }

  // Score the current word with the LM and append it to the word list (returns h itself if none).
  _complete(h) {
    if (!h.cur) return h;
    const o = this.o, word = h.cur;
    let s, hu = h.hu, hv = h.hv;
    if (this.lm === null) {
      s = o.beta;
    } else {
      const wid = this.lm.id(word);
      let lp = this.lm.score(wid, h.hv, h.hu);
      if (wid === UNK) {
        lp += o.unk_penalty;
        if (this.names.has(word)) lp = Math.max(lp, o.name_logp);
      } else if (this.names.has(word)) {
        lp = Math.max(lp, o.name_logp);
      }
      s = o.alpha * lp + o.beta;
      hu = h.hv;
      hv = wid;
    }
    return new Hyp(h.pb, h.pnb, h.lm + s, hu, hv, "", -1,
      { prev: h.words, text: word, start: h.curStart, end: h.lastFrame + 1, cs: h.confSum, cn: h.confN },
      h.lastFrame, 0, 0, h.node);
  }

  _words(fin, t0) {
    if (!fin.length) return { words: [], alternatives: [] };
    const ps = this.o.posterior_scale;
    const scores = fin.map((f) => f.score * ps);
    let mx = -Infinity;
    for (const sc of scores) if (sc > mx) mx = sc;
    const post = scores.map((sc) => Math.exp(sc - mx));
    let Z = 0;
    for (const p of post) Z += p;
    for (let i = 0; i < post.length; i++) post[i] /= Z;

    const hypWords = fin.map((f) => toArray(f.h.words));
    const words = [], alternatives = [];
    for (const bw of hypWords[0]) {
      if (!bw.text) continue;
      const a = bw.start, b = bw.end;
      const conf = bw.cs / Math.max(bw.cn, 1);
      // For every hypothesis, the word it puts at this moment: same text first, then the largest time
      // overlap, then the nearest start. Words are time-ordered within a hypothesis.
      const options = new Map();
      for (let i = 0; i < fin.length; i++) {
        let match = null;   // [text, key0, key1, key2]
        for (const x of hypWords[i]) {
          if (x.end <= a - 3) continue;
          if (x.start >= b + 3) break;
          const k0 = x.text === bw.text ? 1 : 0;
          const k1 = Math.min(b, x.end) - Math.max(a, x.start);
          const k2 = -Math.abs(x.start - a);
          if (match === null || k0 > match[1] || (k0 === match[1] && (k1 > match[2] || (k1 === match[2] && k2 > match[3])))) {
            match = [x.text, k0, k1, k2];
          }
        }
        if (match !== null) options.set(match[0], (options.get(match[0]) ?? 0) + post[i]);
      }
      // Ranked by posterior (desc), then text (asc); alternatives need p >= 0.01.
      const ranked = [...options.entries()].sort((x, y) => (y[1] !== x[1] ? (y[1] - x[1]) : (x[0] < y[0] ? -1 : x[0] > y[0] ? 1 : 0)));
      const alt = ranked
        .filter(([w, pp]) => w !== bw.text && pp >= 0.01)
        .slice(0, this.o.n_alternatives)
        .map(([w, pp]) => ({ text: w, p: round4(pp) }));
      const p = round4(options.get(bw.text) ?? 0);
      words.push({
        text: bw.text, start: t0 + a / FPS, end: t0 + b / FPS, confidence: conf, tier: tierFor(conf),
        boosted: false, original: null, p, alternatives: alt,
      });
      alternatives.push({ p, alternatives: alt });
    }
    return { words, alternatives };
  }
}
