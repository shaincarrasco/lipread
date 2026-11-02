//! Word trigram language model reader (port of `src/lipread/lm.py`).
//!
//! Reads the binary written by `scripts/build_lm.py` in place: the arrays are never copied out of the
//! file bytes, every value is decoded little-endian at its offset when it is looked up.
//!
//! Layout (little-endian, every array starts on a 4-byte boundary):
//!   "LRLM1\n" | u32 header_len | header JSON | u32 words_len | words (UTF-8, "\n"-joined) | pad to 4 |
//!   f32 uni_lp[V] | f32 uni_bow[V] | u32 bi_start[V+1] | u16 bi_w[B] | i16 bi_lp[B] | i16 bi_bow[B] |
//!   u32 tri_start[B+1] | u16 tri_w[T] | i16 tri_lp[T]        (each array padded to 4 bytes)
//!
//! Scores are natural-log probabilities computed in f64 exactly as Python does: f32 values are widened
//! to f64, i16 values are widened and divided by `qscale`.

use std::collections::HashMap;

pub const BOS: u32 = 0; // <s>
pub const EOS: u32 = 1; // </s>
pub const UNK: u32 = 2; // <unk>

const MAGIC: &[u8] = b"LRLM1\n";

#[derive(serde::Deserialize, Debug, Clone)]
pub struct Header {
    #[serde(rename = "V")]
    pub v: usize,
    #[serde(rename = "B")]
    pub b: usize,
    #[serde(rename = "T")]
    pub t: usize,
    pub qscale: f64,
    #[serde(default)]
    pub order: Option<u32>,
}

/// Byte offsets of the arrays inside the file.
#[derive(Debug, Clone, Copy)]
struct Offsets {
    uni_lp: usize,
    uni_bow: usize,
    bi_start: usize,
    bi_w: usize,
    bi_lp: usize,
    bi_bow: usize,
    tri_start: usize,
    tri_w: usize,
    tri_lp: usize,
}

pub struct NgramLM {
    buf: Vec<u8>,
    off: Offsets,
    pub header: Header,
    q: f64,
    pub words: Vec<String>,
    wid: HashMap<String, u32>,
}

#[inline]
fn align4(n: usize) -> usize {
    (n + 3) & !3
}

#[inline]
fn rd_u32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

impl NgramLM {
    /// Parse the file bytes (takes ownership; the arrays are read in place).
    pub fn from_bytes(buf: Vec<u8>) -> Result<NgramLM, String> {
        if buf.len() < MAGIC.len() + 4 || &buf[..MAGIC.len()] != MAGIC {
            return Err("not a LipRead LM file".into());
        }
        let mut off = MAGIC.len();
        let hlen = rd_u32(&buf, off) as usize;
        off += 4;
        if off + hlen + 4 > buf.len() {
            return Err("truncated LM file".into());
        }
        let header: Header = serde_json::from_slice(&buf[off..off + hlen]).map_err(|e| format!("bad LM header: {e}"))?;
        off += hlen;
        let wlen = rd_u32(&buf, off) as usize;
        off += 4;
        if off + wlen > buf.len() {
            return Err("truncated LM file".into());
        }
        let words: Vec<String> = std::str::from_utf8(&buf[off..off + wlen])
            .map_err(|_| "LM word list is not UTF-8".to_string())?
            .split('\n')
            .map(str::to_owned)
            .collect();
        off = align4(off + wlen);

        let (v, b, t) = (header.v, header.b, header.t);
        let mut take = |elem: usize, n: usize| {
            let at = off;
            off = align4(off + elem * n);
            at
        };
        let o = Offsets {
            uni_lp: take(4, v),
            uni_bow: take(4, v),
            bi_start: take(4, v + 1),
            bi_w: take(2, b),
            bi_lp: take(2, b),
            bi_bow: take(2, b),
            tri_start: take(4, b + 1),
            tri_w: take(2, t),
            tri_lp: take(2, t),
        };
        // The last array need not be padded at the end of the file.
        if o.tri_lp + 2 * t > buf.len() {
            return Err("truncated LM file".into());
        }
        if words.len() != v {
            return Err(format!("LM header says V={v} but the file lists {} words", words.len()));
        }
        let mut wid = HashMap::with_capacity(words.len());
        for (i, w) in words.iter().enumerate() {
            wid.insert(w.clone(), i as u32);
        }
        let q = header.qscale;
        Ok(NgramLM { buf, off: o, header, q, words, wid })
    }

    /// Vocabulary id of a word, or UNK.
    #[inline]
    pub fn id(&self, word: &str) -> u32 {
        self.wid.get(word).copied().unwrap_or(UNK)
    }

    #[inline]
    fn f32_at(&self, base: usize, i: usize) -> f64 {
        let o = base + 4 * i;
        f32::from_le_bytes([self.buf[o], self.buf[o + 1], self.buf[o + 2], self.buf[o + 3]]) as f64
    }
    #[inline]
    fn u32_at(&self, base: usize, i: usize) -> u32 {
        rd_u32(&self.buf, base + 4 * i)
    }
    #[inline]
    fn u16_at(&self, base: usize, i: usize) -> u16 {
        let o = base + 2 * i;
        u16::from_le_bytes([self.buf[o], self.buf[o + 1]])
    }
    #[inline]
    fn q_at(&self, base: usize, i: usize) -> f64 {
        let o = base + 2 * i;
        i16::from_le_bytes([self.buf[o], self.buf[o + 1]]) as f64 / self.q
    }

    /// Binary search for `w` in the sorted keys[start[row] .. start[row+1]); index or None.
    #[inline]
    fn find(&self, start: usize, keys: usize, row: u32, w: u32) -> Option<usize> {
        let a = self.u32_at(start, row as usize) as usize;
        let b = self.u32_at(start, row as usize + 1) as usize;
        let (mut lo, mut hi) = (a, b);
        while lo < hi {
            let mid = (lo + hi) >> 1;
            if (self.u16_at(keys, mid) as u32) < w {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if lo < b && self.u16_at(keys, lo) as u32 == w {
            Some(lo)
        } else {
            None
        }
    }

    #[inline]
    fn bigram(&self, v: u32, w: u32) -> f64 {
        match self.find(self.off.bi_start, self.off.bi_w, v, w) {
            Some(i) => self.q_at(self.off.bi_lp, i),
            None => self.f32_at(self.off.uni_bow, v as usize) + self.f32_at(self.off.uni_lp, w as usize),
        }
    }

    /// ln P(w | u v); `v` = previous word id (None = no context), `u` = the word before it (None = unknown).
    pub fn score(&self, w: u32, v: Option<u32>, u: Option<u32>) -> f64 {
        let v = match v {
            None => return self.f32_at(self.off.uni_lp, w as usize),
            Some(v) => v,
        };
        let u = match u {
            None => return self.bigram(v, w),
            Some(u) => u,
        };
        if let Some(bi) = self.find(self.off.bi_start, self.off.bi_w, u, v) {
            if let Some(t) = self.find(self.off.tri_start, self.off.tri_w, bi as u32, w) {
                return self.q_at(self.off.tri_lp, t);
            }
            return self.q_at(self.off.bi_bow, bi) + self.bigram(v, w);
        }
        self.bigram(v, w)
    }

    /// Sum of ln P over a sentence, with <s> context and a closing </s> (as `sentence_logprob`).
    pub fn sentence_logprob(&self, words: &[&str]) -> f64 {
        let (mut u, mut v, mut total) = (None, Some(BOS), 0.0);
        for i in words.iter().map(|w| self.id(w)).chain(std::iter::once(EOS)) {
            total += self.score(i, v, u);
            u = v;
            v = Some(i);
        }
        total
    }
}
