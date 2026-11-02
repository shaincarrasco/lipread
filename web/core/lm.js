// Word trigram language model reader (port of lipread/lm.py). Reads the binary written by
// scripts/build_lm.py: interpolated Kneser-Ney, natural-log scores, i16 values divided by qscale.
//
// Layout (little-endian). Every array starts on a 4-byte boundary, so typed-array views need no copy:
//   "LRLM1\n" | u32 header_len | header JSON | u32 words_len | words (UTF-8, "\n"-joined) | pad to 4 |
//   f32 uni_lp[V] | f32 uni_bow[V] | u32 bi_start[V+1] | u16 bi_w[B] | i16 bi_lp[B] | i16 bi_bow[B] |
//   u32 tri_start[B+1] | u16 tri_w[T] | i16 tri_lp[T]        (each array padded to 4 bytes)

export const BOS = 0;   // <s>
export const EOS = 1;   // </s>
export const UNK = 2;   // <unk>: any word outside the vocabulary

const MAGIC = "LRLM1\n";
const KEY = 65537;             // cache key radix: ids are < 65536 and 0 stands for "no word"
const CACHE_MAX = 200000;      // same cap as the Python lru_cache
const dec = new TextDecoder();

const align4 = (n) => (n + 3) & ~3;

export class NgramLM {
  /** src: a URL string (fetched), an ArrayBuffer, or a typed-array view of the file bytes. */
  static async load(src) {
    let buf;
    if (src instanceof ArrayBuffer) buf = src;
    else if (ArrayBuffer.isView(src)) buf = src.buffer.slice(src.byteOffset, src.byteOffset + src.byteLength);
    else {
      const res = await fetch(src);
      if (!res.ok) throw new Error(`language model fetch failed: HTTP ${res.status}`);
      buf = await res.arrayBuffer();
    }
    return new NgramLM(buf);
  }

  constructor(buf) {
    const bytes = new Uint8Array(buf);
    if (bytes.length < MAGIC.length || dec.decode(bytes.subarray(0, MAGIC.length)) !== MAGIC) {
      throw new Error("not a LipRead LM file");
    }
    const dv = new DataView(buf);
    let off = MAGIC.length;
    const hlen = dv.getUint32(off, true); off += 4;
    this.header = JSON.parse(dec.decode(bytes.subarray(off, off + hlen))); off += hlen;
    const wlen = dv.getUint32(off, true); off += 4;
    this.words = dec.decode(bytes.subarray(off, off + wlen)).split("\n"); off += wlen;
    off = align4(off);

    const { V, B, T, qscale } = this.header;
    this.q = qscale;
    const take = (Type, n) => {
      const a = new Type(buf, off, n);
      off = align4(off + n * Type.BYTES_PER_ELEMENT);
      return a;
    };
    this.uniLp = take(Float32Array, V);
    this.uniBow = take(Float32Array, V);
    this.biStart = take(Uint32Array, V + 1);
    this.biW = take(Uint16Array, B);
    this.biLp = take(Int16Array, B);
    this.biBow = take(Int16Array, B);
    this.triStart = take(Uint32Array, B + 1);
    this.triW = take(Uint16Array, T);
    this.triLp = take(Int16Array, T);
    if (off > buf.byteLength) throw new Error("truncated LM file");

    this.wid = new Map();
    this.words.forEach((w, i) => this.wid.set(w, i));
    this._cache = new Map();
  }

  /** Vocabulary id of a word, or UNK. */
  id(word) {
    const i = this.wid.get(word);
    return i === undefined ? UNK : i;
  }

  /** ln P(w | u v); v = previous word id (null = none), u = the word before it (null = unknown). */
  score(w, v = null, u = null) {
    const key = ((u == null ? 0 : u + 1) * KEY + (v == null ? 0 : v + 1)) * KEY + w;
    let s = this._cache.get(key);
    if (s === undefined) {
      s = this._score(w, v, u);
      if (this._cache.size >= CACHE_MAX) this._cache.delete(this._cache.keys().next().value);
      this._cache.set(key, s);
    }
    return s;
  }

  // Binary search for `w` in the sorted keys[start[row] .. start[row+1]); index or -1.
  _find(start, keys, row, w) {
    const a = start[row], b = start[row + 1];
    if (a === b) return -1;
    let lo = a, hi = b;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (keys[mid] < w) lo = mid + 1;
      else hi = mid;
    }
    return lo < b && keys[lo] === w ? lo : -1;
  }

  _bigram(v, w) {
    const i = this._find(this.biStart, this.biW, v, w);
    return i >= 0 ? this.biLp[i] / this.q : this.uniBow[v] + this.uniLp[w];
  }

  _score(w, v, u) {
    if (v == null) return this.uniLp[w];
    if (u == null) return this._bigram(v, w);
    const bi = this._find(this.biStart, this.biW, u, v);
    if (bi >= 0) {
      const t = this._find(this.triStart, this.triW, bi, w);
      if (t >= 0) return this.triLp[t] / this.q;
      return this.biBow[bi] / this.q + this._bigram(v, w);
    }
    return this._bigram(v, w);
  }
}
