// Parity + speed: the Rust decoder compiled to WebAssembly (web/core/wasm/, built by scripts/build_wasm.sh)
// vs the Python reference fixtures (tests_web/make_beam_fixtures.py) and the JS port (web/core/beam.js).
// Run with: node --test tests_web/*.test.mjs

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { initSync, Decoder } from "../web/core/wasm/lipread_decoder.js";
import { NgramLM } from "../web/core/lm.js";
import { BeamDecoder } from "../web/core/beam.js";

const FIX = new URL("./fixtures/", import.meta.url);
const readFix = (name) => readFileSync(new URL(name, FIX));
const toArrayBuffer = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
const near = (a, b, tol = 1e-6) => Math.abs(a - b) < tol;

initSync({ module: readFileSync(new URL("../web/core/wasm/lipread_decoder_bg.wasm", import.meta.url)) });

const vocabText = readFileSync(new URL("../web/core/assets/tokens.json", import.meta.url), "utf8");
const vocab = JSON.parse(vocabText);
const lmBytes = readFileSync(new URL("../web/core/assets/lm.bin", import.meta.url));
const dec = new Decoder(new Uint8Array(lmBytes), vocabText);
const jsLm = await NgramLM.load(toArrayBuffer(lmBytes));
const refs = JSON.parse(readFix("beam_refs.json"));

// Rebuild the dense (T, V) matrix exactly as make_beam_fixtures.py does: top-K stored, the rest -50.
function denseFromFixture(clip) {
  const { T, file } = clip, K = refs.K, V = refs.V;
  const raw = readFix(file);
  const idxBytes = T * K * 2;
  const idx = new Uint16Array(toArrayBuffer(raw.subarray(0, idxBytes)));
  const vals = new Float32Array(toArrayBuffer(raw.subarray(idxBytes, idxBytes + T * K * 4)));
  const dense = new Float32Array(T * V).fill(refs.fill);
  for (let t = 0; t < T; t++) {
    for (let k = 0; k < K; k++) dense[t * V + idx[t * K + k]] = vals[t * K + k];
  }
  return dense;
}
const dense = new Map(refs.clips.map((c) => [c.id, denseFromFixture(c)]));
const wasmDecode = (lp, clip, opts) => JSON.parse(dec.decode(lp, clip.T, refs.V, clip.t0, JSON.stringify(opts))).words;

function assertSameWords(out, ref, label, { checkConf = true } = {}) {
  assert.deepEqual(out.map((w) => w.text), ref.map((w) => w.text), `${label}: word texts`);
  assert.deepEqual(out.map((w) => w.alternatives.map((a) => a.text)), ref.map((w) => w.alternatives.map((a) => a.text)),
    `${label}: alternative texts`);
  out.forEach((w, i) => {
    const r = ref[i];
    assert.equal(w.tier, r.tier, `${label}: word ${i} tier`);
    for (const k of ["start", "end", "p", ...(checkConf ? ["confidence"] : [])]) {
      assert.ok(near(w[k], r[k]), `${label}: word ${i} ${k} ${w[k]} vs ${r[k]}`);
    }
    w.alternatives.forEach((a, j) => assert.ok(near(a.p, r.alternatives[j].p), `${label}: word ${i} alt ${j} p`));
  });
}

test("WASM: module loads and describes the vocabulary and LM", () => {
  assert.equal(dec.vocabSize, vocab.length);
  assert.equal(dec.hasLm, true);
  assert.throws(() => new Decoder(new Uint8Array(64), vocabText), /not a LipRead LM file/);
  assert.throws(() => dec.decode(new Float32Array(10), 1, vocab.length, 0, ""), /too short/);
});

test("WASM: LM scores equal Python on 2000 (w, v, u) triples; word ids equal", (t) => {
  const f = JSON.parse(readFix("beam_lm_triples.json"));
  let worst = 0;
  f.triples.forEach(([w, v, u], i) => {
    const s = dec.lmScore(w, v ?? -1, u ?? -1);
    worst = Math.max(worst, Math.abs(s - f.scores[i]));
    assert.ok(near(s, f.scores[i], 1e-9), `triple ${i}: ${s} vs ${f.scores[i]}`);
  });
  for (const [w, id] of f.word_ids) assert.equal(dec.lmId(w), id, `id(${JSON.stringify(w)})`);
  t.diagnostic(`max |WASM - Python| over ${f.triples.length} triples: ${worst.toExponential(2)}`);
});

for (const name of ["booster_case0", "booster_case3"]) {
  test(`WASM: greedy words equal Python (${name})`, () => {
    const meta = JSON.parse(readFix(`${name}.json`));
    const buf = readFix(meta.logprobs_file);
    const lp = new Float32Array(toArrayBuffer(buf));
    const words = JSON.parse(dec.greedy(lp, meta.T, meta.V, meta.t0)).words;
    assert.equal(words.length, meta.greedy.length);
    words.forEach((w, i) => {
      const p = meta.greedy[i];
      assert.equal(w.text, p.text);
      assert.equal(w.tier, p.tier);
      assert.ok(near(w.start, p.start, 1e-9) && near(w.end, p.end, 1e-9));
      assert.ok(near(w.confidence, p.confidence, 1e-5), `${w.confidence} vs ${p.confidence}`);
    });
  });
}

for (const clip of refs.clips) {
  for (const [setName, opts] of Object.entries(refs.option_sets)) {
    test(`WASM beam decode equals Python: ${clip.id} / ${setName}`, () => {
      const ref = clip.results[setName];
      const out = wasmDecode(dense.get(clip.id), clip, opts);
      assertSameWords(out, ref.words, `${clip.id}/${setName}`);
      assert.deepEqual(out.map((w) => w.alternatives.map((a) => a.text)),
        ref.alternatives.map((a) => a.alternatives.map((x) => x.text)), "per-word alternative texts");
      out.forEach((w, i) => {
        assert.equal(w.boosted, false);
        assert.equal(w.original, null);
        assert.ok(near(w.p, ref.alternatives[i].p));
      });
    });
  }
}

test("WASM equals the JS port at beam 64 (no Python reference for this setting)", () => {
  const opts = { beam: 64, n_alternatives: 4 };
  for (const clip of refs.clips) {
    const lp = dense.get(clip.id);
    const js = new BeamDecoder(vocab, jsLm, opts).decode(lp, clip.T, refs.V, clip.t0).words;
    assertSameWords(wasmDecode(lp, clip, opts), js, `${clip.id}/beam64`);
  }
});

test("report: Rust-WASM vs JS decode time (each clip decoded 3x)", (t) => {
  const settings = [["beam 16", {}], ["beam 64", { beam: 64, n_alternatives: 4 }]];
  const frames = refs.clips.reduce((a, c) => a + c.T, 0);
  const rows = [];
  for (const [label, opts] of settings) {
    const js = new BeamDecoder(vocab, jsLm, opts);
    const optsJson = JSON.stringify(opts);
    const time = (fn) => {
      for (const clip of refs.clips) fn(clip, dense.get(clip.id));   // warm-up (JIT, caches, buffers)
      const t1 = performance.now();
      for (const clip of refs.clips) {
        const lp = dense.get(clip.id);
        for (let rep = 0; rep < 3; rep++) fn(clip, lp);
      }
      return (performance.now() - t1) / (3 * frames);
    };
    const wasmMs = time((clip, lp) => JSON.parse(dec.decode(lp, clip.T, refs.V, clip.t0, optsJson)));
    const jsMs = time((clip, lp) => js.decode(lp, clip.T, refs.V, clip.t0));
    rows.push({ label, wasmMs, jsMs });
  }
  t.diagnostic(`${refs.clips.length} clips, ${frames} frames, 3 decodes each (ms per frame):`);
  t.diagnostic("setting   | Rust-WASM | JS     | speed-up");
  for (const r of rows) {
    t.diagnostic(`${r.label.padEnd(9)} | ${r.wasmMs.toFixed(4).padStart(9)} | ${r.jsMs.toFixed(4).padStart(6)} | ${(r.jsMs / r.wasmMs).toFixed(2)}x`);
  }
  for (const r of rows) assert.ok(r.wasmMs > 0 && r.jsMs > 0);
});
