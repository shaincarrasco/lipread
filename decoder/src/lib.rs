//! LipRead's context-aware decoder in Rust: word trigram LM reader + CTC prefix beam search with
//! shallow fusion + greedy decoding. Exact port of `src/lipread/{lm,beam,decode}.py` (and of the JS
//! port in `web/core/`), compiled to WebAssembly for the phone app and usable natively as an rlib.
//!
//! JS API (wasm-bindgen, `web/core/wasm/lipread_decoder.js`):
//!
//! ```js
//! import init, { Decoder } from "./wasm/lipread_decoder.js";
//! await init();                                   // or initSync({ module: wasmBytes })
//! const dec = new Decoder(lmBytes, JSON.stringify(vocab));      // lmBytes: Uint8Array (empty = no LM)
//! const { words } = JSON.parse(dec.decode(logprobs, T, V, t0, JSON.stringify({ names: ["Mahomes"] })));
//! const greedy = JSON.parse(dec.greedy(logprobs, T, V, t0));    // { words }
//! ```

pub mod beam;
pub mod greedy;
pub mod lm;

use serde::Serialize;
use wasm_bindgen::prelude::*;

pub use beam::{BeamDecoder, DecodeOptions, Word};
pub use greedy::{greedy_words, GreedyWord};
pub use lm::NgramLM;

#[derive(Serialize)]
struct Out<'a, W: Serialize> {
    words: &'a [W],
}

/// A decoder for one piece vocabulary, with an optional language model. Buffers are reused across calls.
#[wasm_bindgen]
pub struct Decoder {
    lm: Option<NgramLM>,
    vocab: Vec<String>,
    beam: BeamDecoder,
}

impl Decoder {
    /// `lm_bytes`: the LM file (empty = decode without an LM); `vocab_json`: JSON array of piece strings.
    pub fn try_new(lm_bytes: &[u8], vocab_json: &str) -> Result<Decoder, String> {
        Decoder::try_from_vec(lm_bytes.to_vec(), vocab_json)
    }

    /// Same as `try_new`, taking ownership of the bytes (no extra copy of the 11 MB LM).
    pub fn try_from_vec(lm_bytes: Vec<u8>, vocab_json: &str) -> Result<Decoder, String> {
        let vocab: Vec<String> = serde_json::from_str(vocab_json).map_err(|e| format!("bad vocab JSON: {e}"))?;
        if vocab.is_empty() {
            return Err("empty vocabulary".into());
        }
        let lm = if lm_bytes.is_empty() { None } else { Some(NgramLM::from_bytes(lm_bytes)?) };
        let beam = BeamDecoder::new(&vocab, 0);
        Ok(Decoder { lm, vocab, beam })
    }

    pub fn parse_options(opts_json: &str) -> Result<DecodeOptions, String> {
        if opts_json.trim().is_empty() {
            return Ok(DecodeOptions::default());
        }
        serde_json::from_str(opts_json).map_err(|e| format!("bad options JSON: {e}"))
    }

    /// Beam decode into words (native API).
    pub fn decode_words(&mut self, logprobs: &[f32], t: usize, v: usize, t0: f64, o: &DecodeOptions) -> Result<Vec<Word>, String> {
        self.beam.decode(logprobs, t, v, t0, self.lm.as_ref(), o)
    }

    /// Beam decode; returns `{"words": [...]}` JSON.
    pub fn decode_json(&mut self, logprobs: &[f32], t: usize, v: usize, t0: f64, opts_json: &str) -> Result<String, String> {
        let o = Self::parse_options(opts_json)?;
        let words = self.decode_words(logprobs, t, v, t0, &o)?;
        serde_json::to_string(&Out { words: &words }).map_err(|e| e.to_string())
    }

    /// Greedy decode; returns `{"words": [...]}` JSON.
    pub fn greedy_json(&self, logprobs: &[f32], t: usize, v: usize, t0: f64, blank: usize) -> Result<String, String> {
        let words = greedy_words(logprobs, t, v, &self.vocab, t0, blank)?;
        serde_json::to_string(&Out { words: &words }).map_err(|e| e.to_string())
    }

    pub fn lm(&self) -> Option<&NgramLM> {
        self.lm.as_ref()
    }

    pub fn vocab(&self) -> &[String] {
        &self.vocab
    }
}

#[wasm_bindgen]
impl Decoder {
    #[wasm_bindgen(constructor)]
    pub fn new(lm_bytes: Vec<u8>, vocab_json: &str) -> Result<Decoder, JsError> {
        Decoder::try_from_vec(lm_bytes, vocab_json).map_err(|e| JsError::new(&e))
    }

    /// logprobs: Float32Array (T x V) row-major raw model log-probs; t0: source time of frame 0;
    /// opts_json: any subset of the Python DecodeOptions ("" = defaults).
    /// Returns JSON {words:[{text,start,end,confidence,tier,boosted,original,p,alternatives:[{text,p}]}]}.
    pub fn decode(&mut self, logprobs: &[f32], t: usize, v: usize, t0: f64, opts_json: &str) -> Result<String, JsError> {
        self.decode_json(logprobs, t, v, t0, opts_json).map_err(|e| JsError::new(&e))
    }

    /// Greedy CTC decode. Returns JSON {words:[{text,start,end,confidence,tier,boosted,original}]}.
    pub fn greedy(&self, logprobs: &[f32], t: usize, v: usize, t0: f64, blank: Option<u32>) -> Result<String, JsError> {
        self.greedy_json(logprobs, t, v, t0, blank.unwrap_or(0) as usize).map_err(|e| JsError::new(&e))
    }

    /// ln P(w | u v) from the LM (negative v / u = no such context word); NaN without an LM.
    #[wasm_bindgen(js_name = lmScore)]
    pub fn lm_score(&self, w: u32, v: i32, u: i32) -> f64 {
        match &self.lm {
            None => f64::NAN,
            Some(lm) => lm.score(w, (v >= 0).then_some(v as u32), (u >= 0).then_some(u as u32)),
        }
    }

    /// LM word id (2 = <unk>); -1 without an LM.
    #[wasm_bindgen(js_name = lmId)]
    pub fn lm_id(&self, word: &str) -> i32 {
        self.lm.as_ref().map_or(-1, |lm| lm.id(word) as i32)
    }

    #[wasm_bindgen(getter, js_name = hasLm)]
    pub fn has_lm(&self) -> bool {
        self.lm.is_some()
    }

    #[wasm_bindgen(getter, js_name = vocabSize)]
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}
