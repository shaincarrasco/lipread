// Roster booster, browser port of src/lipread/booster.py (plus the SentencePiece
// segmentation it gets from the sentencepiece library).
//
// For each run of 1-3 consecutive greedy words, the booster asks whether a roster name
// explains the same frames better than the words the model wrote. Both texts are scored
// with the CTC forward algorithm over the same padded frame span. A name replaces the
// window only when LL(name) + weight > LL(original). Replacements never overlap, and the
// largest gain wins. Replaced words are flagged boosted=true and keep the model's text in
// `original`. Boosting never raises confidence.
//
// No dependencies, no build step. Words are plain objects:
//   { text, start, end, confidence, tier, boosted, original }
// tier is "high" | "medium" | "low".

export const FPS = 25;
export const MAX_WINDOW = 3;
export const CONF_HIGH = 0.90;
export const CONF_MEDIUM = 0.70;

const WORD_MARK = "▁";   // SentencePiece "▁": marks the start of a word
const FIRST_REAL_PIECE = 3;    // pieces 0-2 are <unk>, <s>, </s>: never emitted

// ----------------------------------------------------------------------------- helpers

/** Python's round(): halves go to the nearest even integer. */
export function roundHalfEven(x) {
  const f = Math.floor(x);
  const d = x - f;
  if (d < 0.5) return f;
  if (d > 0.5) return f + 1;
  return f % 2 === 0 ? f : f + 1;
}

/** Word confidence tier, same thresholds as decode.py tier_for. */
export function tierFor(conf) {
  if (conf >= CONF_HIGH) return "high";
  if (conf >= CONF_MEDIUM) return "medium";
  return "low";
}

/** Frame span [a, b) of a word on the 25 fps track timeline (frame 0 at t0), as word_frames. */
export function wordFrames(word, t0) {
  return [roundHalfEven((word.start - t0) * FPS), roundHalfEven((word.end - t0) * FPS)];
}

/** log(exp(a) + exp(b)) with -Infinity handled exactly. */
export function logAddExp(a, b) {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  const m = a > b ? a : b;
  return m + Math.log1p(Math.exp(-Math.abs(a - b)));
}

/** Uint8Array over the vocabulary: 1 for the blank and for special tokens like <eos>. */
export function silentMask(vocab, blank) {
  const mask = new Uint8Array(vocab.length);
  for (let i = 0; i < vocab.length; i++) {
    const p = vocab[i];
    mask[i] = p.startsWith("<") && p.endsWith(">") ? 1 : 0;
  }
  mask[blank] = 1;
  return mask;
}

// ----------------------------------------------------------------------------- tokenize

const tableCache = new WeakMap();

// Map piece -> score for the real pieces (index >= 3), plus the longest piece in code points.
function pieceTable(pieces) {
  let table = tableCache.get(pieces);
  if (!table) {
    const scores = new Map();
    let maxLen = 1;
    for (let i = FIRST_REAL_PIECE; i < pieces.length; i++) {
      const [piece, score] = pieces[i];
      if (!scores.has(piece)) scores.set(piece, score);
      const len = Array.from(piece).length;
      if (len > maxLen) maxLen = len;
    }
    table = { scores, maxLen };
    tableCache.set(pieces, table);
  }
  return table;
}

// Best-scoring segmentation of one "▁"-prefixed word. Returns null if some character is
// covered by no piece.
function segmentWord(word, table) {
  const cps = Array.from(word);
  const n = cps.length;
  const best = new Float64Array(n + 1).fill(-Infinity);
  const back = new Int32Array(n + 1);
  best[0] = 0;
  for (let end = 1; end <= n; end++) {
    const maxStart = Math.max(0, end - table.maxLen);
    for (let start = end - 1; start >= maxStart; start--) {
      if (best[start] === -Infinity) continue;
      const score = table.scores.get(cps.slice(start, end).join(""));
      if (score === undefined) continue;
      const total = best[start] + score;
      if (total > best[end]) {
        best[end] = total;
        back[end] = start;
      }
    }
  }
  if (best[n] === -Infinity) return null;
  const pieces = [];
  for (let end = n; end > 0; end = back[end]) {
    pieces.push(cps.slice(back[end], end).join(""));
  }
  return pieces.reverse();
}

/**
 * SentencePiece unigram segmentation of a roster name, as sentencepiece's encode() does it.
 * The name is uppercased, NFKC-normalised and split on whitespace; each word gets a "▁"
 * prefix and is segmented by Viterbi over the sum of piece scores.
 *
 * pieces: [[piece, score], ...] from spm_pieces.json.
 * Returns the piece strings, or null when a character has no piece.
 */
export function tokenize(name, pieces) {
  const table = pieceTable(pieces);
  const words = String(name).toUpperCase().normalize("NFKC").split(/\s+/).filter(Boolean);
  const out = [];
  for (const word of words) {
    const seg = segmentWord(WORD_MARK + word, table);
    if (seg === null) return null;
    for (const piece of seg) out.push(piece);
  }
  return out;
}

// ----------------------------------------------------------------------------- CTC

/**
 * CTC log-likelihood log P(target | frames [a, b)), forward algorithm in log space.
 *
 * logprobs: Float32Array or Float64Array, row-major (T x V); row t starts at t * V.
 * T: total number of rows. [a, b): the span. targetIds: label indices, no blanks.
 * Returns -Infinity when the span is too short for the target (including the blanks
 * CTC needs between repeated labels).
 */
export function ctcLogLikelihood(logprobs, T, V, a, b, targetIds, blank = 0) {
  const n = b - a;
  const L = targetIds.length;
  if (L === 0) {
    let sum = 0;
    for (let t = a; t < b; t++) sum += logprobs[t * V + blank];
    return sum;
  }
  let repeats = 0;
  for (let i = 1; i < L; i++) if (targetIds[i] === targetIds[i - 1]) repeats++;
  if (n < L + repeats) return -Infinity;

  const S = 2 * L + 1;
  const ext = new Int32Array(S).fill(blank);
  for (let i = 0; i < L; i++) ext[2 * i + 1] = targetIds[i];
  // The skip transition s-2 -> s is allowed onto a non-blank label that differs from s-2.
  const skipOk = new Uint8Array(S);
  for (let s = 2; s < S; s++) skipOk[s] = ext[s] !== blank && ext[s] !== ext[s - 2] ? 1 : 0;

  let alpha = new Float64Array(S).fill(-Infinity);
  let next = new Float64Array(S);
  alpha[0] = logprobs[a * V + blank];
  alpha[1] = logprobs[a * V + ext[1]];
  for (let t = a + 1; t < b; t++) {
    const row = t * V;
    for (let s = 0; s < S; s++) {
      let v = alpha[s];
      if (s >= 1) v = logAddExp(v, alpha[s - 1]);
      if (s >= 2 && skipOk[s]) v = logAddExp(v, alpha[s - 2]);
      next[s] = v + logprobs[row + ext[s]];
    }
    const tmp = alpha;
    alpha = next;
    next = tmp;
  }
  return logAddExp(alpha[S - 1], alpha[S - 2]);
}

/** Minimum frames CTC needs to emit targetIds: one per label plus a blank between repeats. */
function minFrames(ids) {
  let n = ids.length;
  for (let i = 1; i < ids.length; i++) if (ids[i] === ids[i - 1]) n++;
  return n;
}

/**
 * Greedy argmax runs over frames [a, b): [[piece_id, first_frame, last_frame], ...], inclusive,
 * as decode.py emission_runs. A silent frame ends the current run.
 */
export function emissionRuns(logprobs, V, a, b, silent) {
  const runs = [];
  let prev = -1;
  for (let t = a; t < b; t++) {
    const off = t * V;
    let k = 0;
    let best = logprobs[off];
    for (let v = 1; v < V; v++) {
      const x = logprobs[off + v];
      if (x > best) {
        best = x;
        k = v;
      }
    }
    if (silent[k]) {
      prev = -1;
      continue;
    }
    if (k === prev) {
      runs[runs.length - 1][2] = t;
    } else {
      runs.push([k, t, t]);
      prev = k;
    }
  }
  return runs;
}

// ----------------------------------------------------------------------------- Booster

/**
 * Replace greedy words by roster names where CTC says the name fits the frames better.
 *
 * names:   roster entries, e.g. ["Travis Kelce", "Gerald Youngblood"]. Multi-word names are
 *          also tried word by word.
 * vocab:   the engine vocabulary (index -> piece), from tokens.json. Index 0 is "<blank>".
 * pieces:  the SentencePiece table [[piece, score], ...], from spm_pieces.json.
 * weight:  log-likelihood bonus a name needs over the model's words (nats).
 * pad:     frames of context added on each side of the window when scoring.
 */
export class Booster {
  constructor(names, vocab, pieces, { weight = 3.0, pad = 3 } = {}) {
    this.vocab = Array.from(vocab);
    this.pieces = pieces;
    this.weight = Number(weight);
    this.pad = Math.trunc(pad);
    const blankIndex = this.vocab.indexOf("<blank>");
    this.blank = blankIndex >= 0 ? blankIndex : 0;
    this._silent = silentMask(this.vocab, this.blank);
    this._index = new Map();
    for (let i = 0; i < this.vocab.length; i++) {
      if (!this._index.has(this.vocab[i])) this._index.set(this.vocab[i], i);
    }
    this.candidates = this._build(names);
  }

  _build(names) {
    const seen = new Map();   // text -> ids, in insertion order (as in Python)
    for (const raw of names) {
      const full = String(raw).toUpperCase().split(/\s+/).filter(Boolean).join(" ");
      if (!full) continue;
      const variants = [full];
      const parts = full.split(" ");
      if (parts.length > 1) variants.push(...parts);
      for (const text of variants) {
        if (seen.has(text)) continue;
        const ids = this._encode(text);
        if (ids === null) {
          console.warn(`booster: cannot tokenize roster entry "${text}", skipped`);
          continue;
        }
        seen.set(text, ids);
      }
    }
    const out = [];
    for (const [text, ids] of seen) {
      out.push({ text, ids, minLen: minFrames(ids), words: new Set(text.split(" ")) });
    }
    return out;
  }

  // Piece ids for a name, or null when the name has a character the model cannot write.
  _encode(text) {
    const tokens = tokenize(text, this.pieces);
    if (tokens === null) return null;
    const ids = [];
    for (const piece of tokens) {
      if (piece.startsWith("<")) return null;
      const k = this._index.get(piece);
      if (k === undefined) return null;
      ids.push(k);
    }
    return ids.length ? Int32Array.from(ids) : null;
  }

  /**
   * Return the words with roster names substituted where the CTC scores favour them.
   *
   * words:     greedy words (plain objects), frame 0 of logprobs at source time t0.
   * logprobs:  Float32Array or Float64Array, row-major (T x V).
   */
  apply(words, logprobs, T, V, t0) {
    words = Array.from(words);
    if (!words.length || !this.candidates.length) return words;
    if (V !== this.vocab.length) {
      throw new RangeError(`booster: V=${V} does not match vocab length ${this.vocab.length}`);
    }
    if (logprobs.length < T * V) {
      throw new RangeError(`booster: logprobs has ${logprobs.length} values, need ${T * V}`);
    }
    const n = words.length;
    const frames = words.map((w) => wordFrames(w, t0));
    const blank = this.blank;

    const proposals = [];
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j <= Math.min(n, i + MAX_WINDOW); j++) {
        const a = Math.max(0, frames[i][0]);
        const b = Math.min(T, frames[j - 1][1]);
        if (b <= a) continue;
        // Padding may not reach into a neighbouring word's emitted frames, otherwise a
        // candidate can claim the next word's spike (duplicates).
        const left = i > 0 ? Math.min(a, Math.max(0, frames[i - 1][1])) : 0;
        const right = j < n ? Math.max(b, Math.min(T, frames[j][0])) : T;
        const pa = Math.max(left, a - this.pad);
        const pb = Math.min(right, b + this.pad);
        const spanFrames = pb - pa;

        const original = words.slice(i, j).map((w) => w.text).join(" ");
        const origIds = emissionRuns(logprobs, V, a, b, this._silent).map((r) => r[0]);
        const origLL = ctcLogLikelihood(logprobs, T, V, pa, pb, origIds, blank);
        const origWords = new Set(original.split(" "));

        let best = null;
        for (const cand of this.candidates) {
          if (cand.text === original || cand.minLen > spanFrames) continue;
          // R1: never insert words. R2: a subset of the window's words is a silent
          // deletion of the rest of the window, not a name.
          if (cand.words.size > j - i) continue;
          if (isSubset(cand.words, origWords)) continue;
          const ll = ctcLogLikelihood(logprobs, T, V, pa, pb, cand.ids, blank);
          if (ll === -Infinity) continue;
          const gain = ll + this.weight - origLL;
          if (gain > 0 && (best === null || gain > best.gain)) best = { gain, cand, ll };
        }
        if (best !== null) {
          let sum = 0;
          for (let k = i; k < j; k++) sum += words[k].confidence;
          proposals.push({
            gain: best.gain, i, j, text: best.cand.text, ll: best.ll,
            spanLen: spanFrames, original, origConf: sum / (j - i),
          });
        }
      }
    }

    // Largest gain first; equal gains keep their order (stable, as Python's sort).
    proposals.sort((x, y) => (x.gain === y.gain ? 0 : x.gain > y.gain ? -1 : 1));
    const taken = new Uint8Array(n);
    const accepted = [];
    for (const p of proposals) {
      let clash = false;
      for (let k = p.i; k < p.j; k++) if (taken[k]) { clash = true; break; }
      if (clash) continue;
      for (let k = p.i; k < p.j; k++) taken[k] = 1;
      accepted.push(p);
    }
    accepted.sort((x, y) => x.i - y.i);

    const out = [];
    let pos = 0;
    for (const p of accepted) {
      for (let k = pos; k < p.i; k++) out.push(words[k]);
      const conf = Math.min(p.origConf, Math.exp(p.ll / p.spanLen));
      out.push({
        text: p.text,
        start: words[p.i].start,
        end: words[p.j - 1].end,
        confidence: conf,
        tier: tierFor(conf),
        boosted: true,
        original: p.original,
      });
      pos = p.j;
    }
    for (let k = pos; k < n; k++) out.push(words[k]);
    return out;
  }
}

function isSubset(small, big) {
  for (const x of small) if (!big.has(x)) return false;
  return true;
}
