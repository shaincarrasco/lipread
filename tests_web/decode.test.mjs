// Parity: JS greedy CTC decoding (web/core/decode.js) vs the Python reference, on real model output.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { greedyWords, segment, tierFor } from "../web/core/decode.js";

const here = new URL("./fixtures/", import.meta.url);
const vocab = JSON.parse(readFileSync(new URL("../web/core/assets/tokens.json", import.meta.url)));

for (const name of ["booster_case0", "booster_case3"]) {
  test(`greedy words match Python (${name})`, () => {
    const meta = JSON.parse(readFileSync(new URL(`${name}.json`, here)));
    const buf = readFileSync(new URL(meta.logprobs_file, here));
    const lp = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
    const words = greedyWords(lp, meta.T, meta.V, vocab, meta.t0);
    assert.equal(words.length, meta.greedy.length);
    words.forEach((w, i) => {
      const p = meta.greedy[i];
      assert.equal(w.text, p.text);
      assert.ok(Math.abs(w.start - p.start) < 1e-9 && Math.abs(w.end - p.end) < 1e-9);
      assert.ok(Math.abs(w.confidence - p.confidence) < 1e-5, `${w.confidence} vs ${p.confidence}`);
      assert.equal(w.tier, p.tier);
    });
  });
}

test("segment stretches words, splits on silence and clamps to the window", () => {
  const w = (text, s, e) => ({ text, start: s, end: e, confidence: 0.95, tier: "high" });
  const utts = segment([w("A", 1.0, 1.04), w("B", 1.3, 1.34), w("C", 3.0, 3.04)], 0.9, 3.1);
  assert.equal(utts.length, 2);
  assert.equal(utts[0].text, "A B");
  assert.ok(utts[0].words[0].start >= 0.9 && Math.abs(utts[0].words[0].end - 1.17) < 1e-9);
  assert.ok(utts[1].words[0].end <= 3.1);
});

test("tiers follow the LRS3 calibration thresholds", () => {
  assert.equal(tierFor(0.95), "high"); assert.equal(tierFor(0.8), "medium"); assert.equal(tierFor(0.5), "low");
});
