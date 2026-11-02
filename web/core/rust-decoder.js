// Context-aware decoder compiled from Rust (decoder/ -> web/core/wasm, built by scripts/build_wasm.sh).
// Same results as the JS port in beam.js (checked against the Python reference in tests_web/wasm.test.mjs),
// about 2.5-3x faster. Callers fall back to the JS decoder when this can't load.
import init, { Decoder } from "./wasm/lipread_decoder.js";

let ready = null;

/** A Rust decoder built from the LM bytes and the token list, or throws if WebAssembly can't load. */
export async function createRustDecoder(lmBuf, tokens) {
  ready ??= init({ module_or_path: new URL("./wasm/lipread_decoder_bg.wasm", import.meta.url) });
  try {
    await ready;
  } catch (err) {
    ready = null;
    throw err;
  }
  const dec = new Decoder(new Uint8Array(lmBuf), JSON.stringify(tokens));
  return {
    kind: "rust",
    /** Same contract as BeamDecoder.decode: {words}. */
    decode(logprobs, T, V, t0, opts = {}) {
      return JSON.parse(dec.decode(logprobs, T, V, t0, JSON.stringify(opts)));
    },
    free() { dec.free(); },
  };
}
