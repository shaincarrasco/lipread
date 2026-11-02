//! Context-aware decoding: CTC prefix beam search with a word trigram LM (shallow fusion).
//! Exact port of `src/lipread/beam.py` (`BeamDecoder.decode`, `_complete`, `_word_score`, `_words`):
//! same defaults, same candidate selection, same tie rules (stable sorts in insertion order).
//!
//!   score = log P_model(pieces) + alpha * log P_LM(words) + beta * n_words
//!
//! Prefixes are nodes of a trie (a `Vec` arena), so a prefix key is one integer. The current word is a
//! node of a second trie over pieces, and completed words live in an arena as an immutable linked list,
//! so extending a hypothesis copies a small `Copy` struct and never clones strings or vectors.
//! Scratch buffers are kept on the decoder and reused across frames and calls.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use serde::{Deserialize, Serialize};

use crate::lm::{NgramLM, BOS, UNK};

pub const MARK: &str = "\u{2581}"; // "▁": a piece starting with it begins a new word
pub const FPS: f64 = 25.0;
pub const CONF_HIGH: f64 = 0.90;
pub const CONF_MEDIUM: f64 = 0.70;
const NEG: f64 = -1e30;
const NIL: u32 = u32::MAX;

/// Decoder options (Python `DecodeOptions`; same names and defaults). Unknown JSON fields are ignored.
#[derive(Deserialize, Debug, Clone)]
#[serde(default)]
pub struct DecodeOptions {
    pub beam: usize,
    pub alpha: f64,
    pub beta: f64,
    pub unk_penalty: f64,
    pub prune: f64,
    pub topk: usize,
    pub n_alternatives: usize,
    pub posterior_scale: f64,
    pub temperature: f64,
    pub names: Vec<String>,
    pub name_logp: f64,
    pub blank: usize,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        DecodeOptions {
            beam: 16,
            alpha: 0.3,
            beta: 1.0,
            unk_penalty: -4.0,
            prune: 10.0,
            topk: 24,
            n_alternatives: 3,
            posterior_scale: 0.2,
            temperature: 1.0,
            names: Vec::new(),
            name_logp: (1e-4f64).ln(),
            blank: 0,
        }
    }
}

#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct Alternative {
    pub text: String,
    pub p: f64,
}

/// One decoded word, with the same fields as the JS decoders' words.
#[derive(Serialize, Debug, Clone)]
pub struct Word {
    pub text: String,
    pub start: f64,
    pub end: f64,
    pub confidence: f64,
    pub tier: &'static str,
    pub boosted: bool,
    pub original: Option<String>,
    pub p: f64,
    pub alternatives: Vec<Alternative>,
}

/// Map a 0..1 word confidence to its tier (decode.tier_for).
pub fn tier_for(conf: f64) -> &'static str {
    if conf >= CONF_HIGH {
        "high"
    } else if conf >= CONF_MEDIUM {
        "medium"
    } else {
        "low"
    }
}

#[inline]
fn lse(a: f64, b: f64) -> f64 {
    let (a, b) = if a < b { (b, a) } else { (a, b) };
    if b <= NEG {
        a
    } else {
        a + (b - a).exp().ln_1p()
    }
}

/// numpy's pairwise summation (the order `np.sum` adds a contiguous float64 array in).
pub fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0;
        for &x in a {
            res += x;
        }
        res
    } else if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i < n - (n % 8) {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
    }
}

/// Python `round(x, 4)`: decimal rounding of the exact binary value, exact ties to even.
pub fn round4(x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    let neg = x < 0.0;
    let s = format!("{:.90}", x.abs()); // exact decimal expansion (Rust formats floats exactly)
    let dot = s.find('.').unwrap();
    let digits = s.as_bytes();
    let mut n: f64 = 0.0;
    for &ch in digits[..dot].iter().chain(digits[dot + 1..dot + 5].iter()) {
        n = n * 10.0 + (ch - b'0') as f64;
    }
    let d = digits[dot + 5] - b'0';
    let rest_nonzero = digits[dot + 6..].iter().any(|&c| c != b'0');
    if d > 5 || (d == 5 && (rest_nonzero || n % 2.0 == 1.0)) {
        n += 1.0;
    }
    let r = n / 10000.0;
    if neg {
        -r
    } else {
        r
    }
}

// ---------------------------------------------------------------------------------------------- hashing

/// Multiplicative hasher for the u64 trie keys (no deps, no DoS concerns: keys are our own ids).
#[derive(Default, Clone, Copy)]
pub struct IdHasher(u64);

impl Hasher for IdHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(8) ^ b as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        }
    }
    #[inline]
    fn write_u64(&mut self, x: u64) {
        self.0 = (x ^ (x >> 31)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.0 ^ (self.0 >> 29)
    }
}

type IdMap = HashMap<u64, u32, BuildHasherDefault<IdHasher>>;

// ---------------------------------------------------------------------------------------------- state

/// A hypothesis (Python `_Hyp`). `Copy`: all variable-size data lives in the arenas.
#[derive(Clone, Copy, Debug)]
struct Hyp {
    pb: f64,        // log prob, ending in blank
    pnb: f64,       // log prob, ending in a piece
    lm: f64,        // alpha * LM + beta * words for completed words
    hu: u32,        // word id two back (NIL = none)
    hv: u32,        // word id of the last completed word
    cur: u32,       // word-trie node of the current (incomplete) word; 0 = ""
    cur_start: i32, // first frame of the current word
    words: u32,     // newest completed word in the word arena (NIL = none)
    last_frame: i32,
    conf_sum: f64,
    conf_n: u32,
    node: u32, // prefix-trie node (0 = empty prefix)
}

/// A completed word: (text, start_frame, end_frame, conf_sum, conf_n), linked to the previous one.
#[derive(Clone, Copy, Debug)]
struct WordEntry {
    prev: u32,
    wnode: u32,
    start: i32,
    end: i32,
    cs: f64,
    cn: u32,
}

/// Per-decode arenas and scratch buffers (cleared, never freed, between calls).
#[derive(Default)]
struct Scratch {
    // prefix trie
    ptrie: IdMap,
    piece: Vec<u32>,      // node -> last piece
    slot_stamp: Vec<u32>, // node -> frame stamp of its entry in `nxt`
    slot_idx: Vec<u32>,   // node -> index in `nxt`
    // word trie (current word as a piece sequence) and its LM lookups
    wtrie: IdMap,
    wparent: Vec<u32>,
    wpiece: Vec<u32>,
    wlen: Vec<u32>,  // byte length of the node's text (0 = empty word)
    wid: Vec<u32>,   // LM id of the node's text (NIL = not looked up yet)
    wname: Vec<bool>, // node's text is a roster name
    // completed words
    went: Vec<WordEntry>,
    // per frame
    row: Vec<f64>,
    ex: Vec<f64>,
    top_i: Vec<u32>,
    top_v: Vec<f64>,
    cand: Vec<u32>,
    beams: Vec<Hyp>,
    nxt: Vec<Hyp>,
    order: Vec<u32>,
    sc: Vec<f64>,
    pieces_tmp: Vec<u32>,
}

/// The beam decoder for one piece vocabulary. Reusable: buffers persist across `decode` calls.
pub struct BeamDecoder {
    vocab: Vec<String>,
    v: usize,
    blank: usize,
    silent: Vec<bool>,
    starts_word: Vec<bool>,
    text: Vec<String>,
    s: Scratch,
}

struct Ctx<'a> {
    lm: Option<&'a NgramLM>,
    o: &'a DecodeOptions,
    names: &'a std::collections::HashSet<String>,
}

impl BeamDecoder {
    pub fn new(vocab: &[String], blank: usize) -> BeamDecoder {
        let v = vocab.len();
        BeamDecoder {
            vocab: vocab.to_vec(),
            v,
            blank,
            silent: vocab
                .iter()
                .enumerate()
                .map(|(i, p)| i == blank || (p.starts_with('<') && p.ends_with('>')))
                .collect(),
            starts_word: vocab.iter().map(|p| p.starts_with(MARK)).collect(),
            text: vocab.iter().map(|p| p.replace(MARK, "")).collect(),
            s: Scratch::default(),
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.v
    }

    /// Roster names as Python builds them: each full name and each of its words, upper-cased.
    pub fn name_set(names: &[String]) -> std::collections::HashSet<String> {
        let mut set = std::collections::HashSet::new();
        for name in names {
            for n in std::iter::once(name.as_str()).chain(name.split_whitespace()) {
                if !n.trim().is_empty() {
                    set.insert(n.to_uppercase());
                }
            }
        }
        set
    }

    // ------------------------------------------------------------------ tries

    fn reset(&mut self) {
        let s = &mut self.s;
        s.ptrie.clear();
        s.piece.clear();
        s.slot_stamp.clear();
        s.slot_idx.clear();
        s.piece.push(NIL);
        s.slot_stamp.push(0);
        s.slot_idx.push(0);
        s.wtrie.clear();
        s.wparent.clear();
        s.wpiece.clear();
        s.wlen.clear();
        s.wid.clear();
        s.wname.clear();
        s.wparent.push(NIL);
        s.wpiece.push(NIL);
        s.wlen.push(0);
        s.wid.push(NIL);
        s.wname.push(false);
        s.went.clear();
    }

    #[inline]
    fn child(&mut self, node: u32, c: u32) -> u32 {
        let key = node as u64 * self.v as u64 + c as u64;
        let s = &mut self.s;
        let next = s.piece.len() as u32;
        let ch = *s.ptrie.entry(key).or_insert(next);
        if ch == next {
            s.piece.push(c);
            s.slot_stamp.push(0);
            s.slot_idx.push(0);
        }
        ch
    }

    #[inline]
    fn wchild(&mut self, wnode: u32, c: u32) -> u32 {
        let key = wnode as u64 * self.v as u64 + c as u64;
        let s = &mut self.s;
        let next = s.wparent.len() as u32;
        let ch = *s.wtrie.entry(key).or_insert(next);
        if ch == next {
            let len = s.wlen[wnode as usize] + self.text[c as usize].len() as u32;
            s.wparent.push(wnode);
            s.wpiece.push(c);
            s.wlen.push(len);
            s.wid.push(NIL);
            s.wname.push(false);
        }
        ch
    }

    /// Text of a word-trie node.
    fn wtext(&mut self, wnode: u32) -> String {
        let s = &mut self.s;
        s.pieces_tmp.clear();
        let mut n = wnode;
        while n != 0 && n != NIL {
            s.pieces_tmp.push(s.wpiece[n as usize]);
            n = s.wparent[n as usize];
        }
        let mut out = String::with_capacity(s.wlen[wnode as usize] as usize);
        for &c in s.pieces_tmp.iter().rev() {
            out.push_str(&self.text[c as usize]);
        }
        out
    }

    // ------------------------------------------------------------------ LM

    /// `_word_score` + `_complete`: h with its current word completed (h itself if there is none).
    fn complete(&mut self, h: &Hyp, cx: &Ctx) -> Hyp {
        if self.s.wlen[h.cur as usize] == 0 {
            return *h;
        }
        let o = cx.o;
        let (s, hu, hv) = match cx.lm {
            None => (o.beta, h.hu, h.hv),
            Some(lm) => {
                let wi = h.cur as usize;
                if self.s.wid[wi] == NIL {
                    let text = self.wtext(h.cur);
                    self.s.wname[wi] = cx.names.contains(&text);
                    self.s.wid[wi] = lm.id(&text);
                }
                let (wid, is_name) = (self.s.wid[wi], self.s.wname[wi]);
                let u = if h.hu == NIL { None } else { Some(h.hu) };
                let mut lp = lm.score(wid, Some(h.hv), u);
                if wid == UNK {
                    lp += o.unk_penalty;
                    if is_name {
                        lp = lp.max(o.name_logp);
                    }
                } else if is_name {
                    lp = lp.max(o.name_logp);
                }
                (o.alpha * lp + o.beta, h.hv, wid)
            }
        };
        let idx = self.s.went.len() as u32;
        self.s.went.push(WordEntry {
            prev: h.words,
            wnode: h.cur,
            start: h.cur_start,
            end: h.last_frame + 1,
            cs: h.conf_sum,
            cn: h.conf_n,
        });
        Hyp {
            pb: h.pb,
            pnb: h.pnb,
            lm: h.lm + s,
            hu,
            hv,
            cur: 0,
            cur_start: -1,
            words: idx,
            last_frame: h.last_frame,
            conf_sum: 0.0,
            conf_n: 0,
            node: h.node,
        }
    }

    // ------------------------------------------------------------------ search

    /// Decode a (T, V) row-major matrix of raw model log-probs; frame 0 is at source time `t0`.
    pub fn decode(
        &mut self,
        logprobs: &[f32],
        t_len: usize,
        v: usize,
        t0: f64,
        lm: Option<&NgramLM>,
        o: &DecodeOptions,
    ) -> Result<Vec<Word>, String> {
        if v != self.v {
            return Err(format!("logprobs must have {} columns, got {v}", self.v));
        }
        if logprobs.len() < t_len * v {
            return Err(format!("logprobs too short for T={t_len}, V={v}"));
        }
        if o.blank != self.blank {
            if o.blank >= self.v {
                return Err(format!("blank id {} is outside the vocabulary", o.blank));
            }
            self.set_blank(o.blank);
        }
        let names = Self::name_set(&o.names);
        let cx = Ctx { lm, o, names: &names };
        self.reset();
        let vn = self.v;
        let blank = self.blank;
        let soft = o.temperature != 1.0;
        let k = o.topk.min(vn);

        let mut row = std::mem::take(&mut self.s.row);
        let mut ex = std::mem::take(&mut self.s.ex);
        let mut top_i = std::mem::take(&mut self.s.top_i);
        let mut top_v = std::mem::take(&mut self.s.top_v);
        let mut cand = std::mem::take(&mut self.s.cand);
        let mut beams = std::mem::take(&mut self.s.beams);
        let mut nxt = std::mem::take(&mut self.s.nxt);
        let mut order = std::mem::take(&mut self.s.order);
        let mut sc = std::mem::take(&mut self.s.sc);
        row.resize(vn, 0.0);
        ex.resize(vn, 0.0);
        top_i.resize(k, 0);
        top_v.resize(k, 0.0);

        beams.clear();
        beams.push(Hyp {
            pb: 0.0,
            pnb: NEG,
            lm: 0.0,
            hu: NIL,
            hv: BOS,
            cur: 0,
            cur_start: -1,
            words: NIL,
            last_frame: -1,
            conf_sum: 0.0,
            conf_n: 0,
            node: 0,
        });

        for t in 0..t_len {
            let raw = &logprobs[t * vn..(t + 1) * vn];
            // Frame row: softened (temperature) or raw log-probs, as in Python.
            if soft {
                let temp = o.temperature;
                let mut zmax = f64::NEG_INFINITY;
                for j in 0..vn {
                    let z = raw[j] as f64 / temp;
                    row[j] = z;
                    if z > zmax {
                        zmax = z;
                    }
                }
                for j in 0..vn {
                    row[j] -= zmax;
                    ex[j] = row[j].exp();
                }
                let ls = pairwise_sum(&ex).ln();
                for x in row.iter_mut() {
                    *x -= ls;
                }
            } else {
                for j in 0..vn {
                    row[j] = raw[j] as f64;
                }
            }

            // Top-k pieces of the frame (by value desc, ties to the lower index), then the prune window.
            let mut n = 0usize;
            for (j, &val) in row.iter().enumerate() {
                let mut p;
                if n < k {
                    p = n;
                    n += 1;
                } else if k > 0 && val > top_v[k - 1] {
                    p = k - 1;
                } else {
                    continue;
                }
                while p > 0 && top_v[p - 1] < val {
                    top_v[p] = top_v[p - 1];
                    top_i[p] = top_i[p - 1];
                    p -= 1;
                }
                top_v[p] = val;
                top_i[p] = j as u32;
            }
            cand.clear();
            if n > 0 {
                let thr = top_v[0] - o.prune;
                for i in 0..n {
                    let c = top_i[i];
                    if top_v[i] >= thr && !self.silent[c as usize] {
                        cand.push(c);
                    }
                }
            }

            nxt.clear();
            let stamp = t as u32 + 1;
            let rb = row[blank];

            for bi in 0..beams.len() {
                let h = beams[bi];
                let tot = lse(h.pb, h.pnb);
                // blank: same prefix
                let ni = self.get(&mut nxt, stamp, h.node, &h);
                nxt[ni].pb = lse(nxt[ni].pb, tot + rb);
                // repeated last piece without a blank in between: same prefix
                let last = self.s.piece[h.node as usize];
                if h.node != 0 {
                    let rep = h.pnb + row[last as usize];
                    nxt[ni].pnb = lse(nxt[ni].pnb, rep);
                    if rep > tot + rb {
                        nxt[ni].last_frame = t as i32; // the piece is still being "spoken"
                    }
                }
                let mut done: Option<Hyp> = None;
                for ci in 0..cand.len() {
                    let c = cand[ci];
                    let p = (if h.node != 0 && last == c { h.pb } else { tot }) + row[c as usize];
                    let nkey = self.child(h.node, c);
                    let idx = if self.s.slot_stamp[nkey as usize] == stamp {
                        self.s.slot_idx[nkey as usize] as usize
                    } else {
                        let sw = self.starts_word[c as usize];
                        let base = if sw && self.s.wlen[h.cur as usize] != 0 {
                            match done {
                                Some(d) => d,
                                None => {
                                    let d = self.complete(&h, &cx);
                                    done = Some(d);
                                    d
                                }
                            }
                        } else {
                            h
                        };
                        let mut nh = base;
                        nh.pb = NEG;
                        nh.pnb = NEG;
                        nh.node = nkey;
                        if sw || self.s.wlen[nh.cur as usize] == 0 {
                            nh.cur = self.wchild(0, c);
                            nh.cur_start = t as i32;
                            nh.conf_sum = 0.0;
                            nh.conf_n = 0;
                        } else {
                            nh.cur = self.wchild(nh.cur, c);
                        }
                        nh.last_frame = t as i32;
                        nh.conf_sum += (raw[c as usize] as f64).exp(); // raw posterior
                        nh.conf_n += 1;
                        let idx = nxt.len();
                        nxt.push(nh);
                        self.s.slot_stamp[nkey as usize] = stamp;
                        self.s.slot_idx[nkey as usize] = idx as u32;
                        idx
                    };
                    nxt[idx].pnb = lse(nxt[idx].pnb, p);
                }
            }

            // Keep the `beam` best by total + lm; ties keep insertion order (stable, as Python's sorted).
            sc.clear();
            sc.extend(nxt.iter().map(|h| lse(h.pb, h.pnb) + h.lm));
            order.clear();
            order.extend(0..nxt.len() as u32);
            order.sort_unstable_by(|&i, &j| {
                sc[j as usize].partial_cmp(&sc[i as usize]).unwrap_or(std::cmp::Ordering::Equal).then(i.cmp(&j))
            });
            order.truncate(o.beam);
            beams.clear();
            beams.extend(order.iter().map(|&i| nxt[i as usize]));
        }

        // Final scores: complete the current word of every beam; stable sort, best first.
        let mut finals: Vec<(f64, Hyp)> = Vec::with_capacity(beams.len());
        for bi in 0..beams.len() {
            let f = self.complete(&beams[bi], &cx);
            finals.push((lse(f.pb, f.pnb) + f.lm, f));
        }
        finals.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        self.s.row = row;
        self.s.ex = ex;
        self.s.top_i = top_i;
        self.s.top_v = top_v;
        self.s.cand = cand;
        self.s.beams = beams;
        self.s.nxt = nxt;
        self.s.order = order;
        self.s.sc = sc;

        Ok(self.words(&finals, t0, o))
    }

    /// Recompute the silent mask for another blank id.
    fn set_blank(&mut self, blank: usize) {
        self.blank = blank;
        for (i, p) in self.vocab.iter().enumerate() {
            self.silent[i] = i == blank || (p.starts_with('<') && p.ends_with('>'));
        }
    }

    /// `get(key, parent)`: the entry of `node` in `nxt`, created from `parent` (with NEG scores) if new.
    #[inline]
    fn get(&mut self, nxt: &mut Vec<Hyp>, stamp: u32, node: u32, parent: &Hyp) -> usize {
        let ni = node as usize;
        if self.s.slot_stamp[ni] == stamp {
            return self.s.slot_idx[ni] as usize;
        }
        let mut h = *parent;
        h.pb = NEG;
        h.pnb = NEG;
        h.node = node;
        let idx = nxt.len();
        nxt.push(h);
        self.s.slot_stamp[ni] = stamp;
        self.s.slot_idx[ni] = idx as u32;
        idx
    }

    // ------------------------------------------------------------------ output

    fn words(&mut self, finals: &[(f64, Hyp)], t0: f64, o: &DecodeOptions) -> Vec<Word> {
        if finals.is_empty() {
            return Vec::new();
        }
        let scores: Vec<f64> = finals.iter().map(|(sc, _)| sc * o.posterior_scale).collect();
        let mx = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut post: Vec<f64> = scores.iter().map(|s| (s - mx).exp()).collect();
        let z = pairwise_sum(&post);
        for p in post.iter_mut() {
            *p /= z;
        }

        // Word lists (time-ordered) of every final hypothesis, with interned texts.
        let mut text_of: HashMap<u32, u32> = HashMap::new(); // word node -> text id
        let mut texts: Vec<String> = Vec::new();
        let mut text_ids: HashMap<String, u32> = HashMap::new();
        let mut hyp_words: Vec<Vec<(u32, i32, i32, f64, u32)>> = Vec::with_capacity(finals.len());
        for (_, h) in finals {
            let mut ws = Vec::new();
            let mut e = h.words;
            while e != NIL {
                let we = self.s.went[e as usize];
                let tid = match text_of.get(&we.wnode) {
                    Some(&id) => id,
                    None => {
                        let s = self.wtext(we.wnode);
                        let id = match text_ids.get(&s) {
                            Some(&id) => id,
                            None => {
                                let id = texts.len() as u32;
                                text_ids.insert(s.clone(), id);
                                texts.push(s);
                                id
                            }
                        };
                        text_of.insert(we.wnode, id);
                        id
                    }
                };
                ws.push((tid, we.start, we.end, we.cs, we.cn));
                e = we.prev;
            }
            ws.reverse();
            hyp_words.push(ws);
        }

        let mut out = Vec::new();
        let mut options: Vec<(u32, f64)> = Vec::new(); // text id -> posterior, in insertion order
        for &(tid, a, b, csum, cn) in &hyp_words[0] {
            if texts[tid as usize].is_empty() {
                continue;
            }
            let conf = csum / (cn.max(1) as f64);
            // For every hypothesis, the word it puts at this moment: same text first, then the largest
            // time overlap, then the nearest start (words are time-ordered within a hypothesis).
            options.clear();
            for (i, ws) in hyp_words.iter().enumerate() {
                let mut best: Option<((bool, i32, i32), u32)> = None;
                for &(wt, wa, wb, _, _) in ws {
                    if wb <= a - 3 {
                        continue;
                    }
                    if wa >= b + 3 {
                        break;
                    }
                    let key = (wt == tid, b.min(wb) - a.max(wa), -(wa - a).abs());
                    if best.map_or(true, |(k, _)| key > k) {
                        best = Some((key, wt));
                    }
                }
                if let Some((_, wt)) = best {
                    match options.iter_mut().find(|(t, _)| *t == wt) {
                        Some(e) => e.1 += post[i],
                        None => options.push((wt, 0.0 + post[i])),
                    }
                }
            }
            let mut ranked = options.clone();
            ranked.sort_by(|x, y| {
                y.1.partial_cmp(&x.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| texts[x.0 as usize].cmp(&texts[y.0 as usize]))
            });
            let alternatives: Vec<Alternative> = ranked
                .iter()
                .filter(|(w, pp)| *w != tid && *pp >= 0.01)
                .take(o.n_alternatives)
                .map(|(w, pp)| Alternative { text: texts[*w as usize].clone(), p: round4(*pp) })
                .collect();
            let p = round4(options.iter().find(|(t, _)| *t == tid).map_or(0.0, |e| e.1));
            out.push(Word {
                text: texts[tid as usize].clone(),
                start: t0 + a as f64 / FPS,
                end: t0 + b as f64 / FPS,
                confidence: conf,
                tier: tier_for(conf),
                boosted: false,
                original: None,
                p,
                alternatives,
            });
        }
        out
    }
}
