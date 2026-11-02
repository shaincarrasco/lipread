// Parity: browser LM (web/core/lm.js) and context-aware beam decoder (web/core/beam.js) vs the Python
// reference (src/lipread/lm.py, src/lipread/beam.py). Fixtures come from tests_web/make_beam_fixtures.py.
// Run with: node --test tests_web/*.test.mjs

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { NgramLM } from "../web/core/lm.js";
import { BeamDecoder } from "../web/core/beam.js";

const FIX = new URL("./fixtures/", import.meta.url);
const readFix = (name) => readFileSync(new URL(name, FIX));
const toArrayBuffer = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
const vocab = JSON.parse(readFileSync(new URL("../web/core/assets/tokens.json", import.meta.url), "utf8"));
const lm = await NgramLM.load(toArrayBuffer(readFileSync(new URL("../web/core/assets/lm.bin", import.meta.url))));
const refs = JSON.parse(readFix("beam_refs.json"));
const near = (a, b, tol = 1e-6) => Math.abs(a - b) < tol;

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

test("LM: header describes a trigram model over the word vocabulary", () => {
  assert.equal(lm.header.order, 3);
  assert.equal(lm.words.length, lm.header.V);
  assert.equal(lm.words[0], "<s>");
  assert.equal(lm.words[2], "<unk>");
});

test("LM: rejects a file that is not an LM", async () => {
  await assert.rejects(NgramLM.load(new ArrayBuffer(64)), /not a LipRead LM file/);
});

test("LM: scores equal Python on 2000 (w, v, u) triples", (t) => {
  const f = JSON.parse(readFix("beam_lm_triples.json"));
  assert.equal(f.V, lm.header.V);
  let worst = 0;
  f.triples.forEach(([w, v, u], i) => {
    const s = lm.score(w, v, u);
    worst = Math.max(worst, Math.abs(s - f.scores[i]));
    assert.ok(near(s, f.scores[i]), `triple ${i} (w=${w}, v=${v}, u=${u}): ${s} vs ${f.scores[i]}`);
  });
  t.diagnostic(`max |JS - Python| over ${f.triples.length} triples: ${worst.toExponential(2)}`);
});

test("LM: word ids equal Python (unknown words map to <unk>)", () => {
  const f = JSON.parse(readFix("beam_lm_triples.json"));
  for (const [w, id] of f.word_ids) assert.equal(lm.id(w), id, `id(${JSON.stringify(w)})`);
});

const timings = [];
for (const clip of refs.clips) {
  for (const [setName, opts] of Object.entries(refs.option_sets)) {
    test(`beam decode equals Python: ${clip.id} / ${setName}`, (t) => {
      const ref = clip.results[setName];
      const lp = denseFromFixture(clip);
      const dec = new BeamDecoder(vocab, lm, opts);
      const out = dec.decode(lp, clip.T, refs.V, clip.t0);

      assert.deepEqual(out.words.map((w) => w.text), ref.words.map((w) => w.text), "word texts");
      assert.deepEqual(out.words.map((w) => w.alternatives.map((a) => a.text)),
        ref.words.map((w) => w.alternatives.map((a) => a.text)), "alternative texts");
      assert.deepEqual(out.alternatives.map((a) => a.alternatives.map((x) => x.text)),
        ref.alternatives.map((a) => a.alternatives.map((x) => x.text)), "per-word alternative texts");
      out.words.forEach((w, i) => {
        const r = ref.words[i];
        assert.ok(near(w.start, r.start) && near(w.end, r.end), `word ${i} timing`);
        assert.ok(near(w.confidence, r.confidence), `word ${i} confidence ${w.confidence} vs ${r.confidence}`);
        assert.equal(w.tier, r.tier, `word ${i} tier`);
        assert.ok(near(w.p, r.p), `word ${i} p ${w.p} vs ${r.p}`);
        assert.equal(w.boosted, false);
        assert.equal(w.original, null);
        assert.equal(w.alternatives.length, r.alternatives.length);
        w.alternatives.forEach((a, j) => assert.ok(near(a.p, r.alternatives[j].p), `word ${i} alt ${j} p`));
      });

      // Timing: best of three runs on a warm decoder (the first run is the parity run above).
      let best = Infinity;
      for (let rep = 0; rep < 3; rep++) {
        const t1 = performance.now();
        dec.decode(lp, clip.T, refs.V, clip.t0);
        best = Math.min(best, performance.now() - t1);
      }
      timings.push({ id: clip.id, setName, T: clip.T, ms: best });
      t.diagnostic(`${clip.id} ${setName}: T=${clip.T}, ${out.words.length} words, decode ${best.toFixed(1)} ms`);
    });
  }
}

test("report: JS decode time summary", (t) => {
  const total = timings.reduce((a, x) => a + x.ms, 0);
  const frames = timings.reduce((a, x) => a + x.T, 0);
  const worst = timings.reduce((a, x) => (x.ms > a.ms ? x : a), { ms: 0, id: "" });
  t.diagnostic(`${timings.length} decodes, total ${total.toFixed(0)} ms, ${(total / frames).toFixed(3)} ms/frame; `
    + `slowest ${worst.id} ${worst.setName} ${worst.ms.toFixed(1)} ms`);
  assert.equal(timings.length, refs.clips.length * Object.keys(refs.option_sets).length);
});
