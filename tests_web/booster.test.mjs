// Browser booster tests. Run with: node tests_web/booster.test.mjs (Node 20+, no packages).
//
// (a) tokenize() reproduces sentencepiece on the golden names and on 400 random names.
// (b) ctcLogLikelihood() reproduces lipread.booster.ctc_log_likelihood (Python).
// (c) Booster.apply() reproduces Booster.apply() on real LRS3 clips (Python engine output).

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import {
  Booster,
  ctcLogLikelihood,
  roundHalfEven,
  tierFor,
  tokenize,
} from "../web/core/booster.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const ASSETS = join(HERE, "..", "web", "core", "assets");
const FIXTURES = join(HERE, "fixtures");

const readJson = (path) => JSON.parse(readFileSync(path, "utf8"));
const pieces = readJson(join(ASSETS, "spm_pieces.json"));
const vocab = readJson(join(ASSETS, "tokens.json"));

// JSON cannot hold -Infinity; the fixtures write it as a string.
const fromJsonNum = (x) => (x === "-Infinity" ? -Infinity : x === "Infinity" ? Infinity : x);

function f32FromBase64(b64) {
  const bytes = Buffer.from(b64, "base64");
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return new Float32Array(copy.buffer);
}

function f32FromFile(path) {
  const bytes = readFileSync(path);
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return new Float32Array(copy.buffer);
}

// ----------------------------------------------------------------------------- (a) tokenize

test("tokenize matches spm_golden.json (sentencepiece output) for every entry", () => {
  const golden = readJson(join(ASSETS, "spm_golden.json"));
  const names = Object.keys(golden);
  assert.ok(names.length >= 8, "golden file should carry the spec names");
  for (const name of names) {
    assert.deepEqual(tokenize(name, pieces), golden[name], `tokenize(${JSON.stringify(name)})`);
  }
});

test("tokenize matches sentencepiece on 400 random roster-like names", () => {
  const random = readJson(join(FIXTURES, "spm_random_golden.json")).entries;
  for (const [name, expected] of Object.entries(random)) {
    assert.deepEqual(tokenize(name, pieces), expected, `tokenize(${JSON.stringify(name)})`);
  }
});

test("tokenize never emits the special pieces <unk>, <s>, </s>", () => {
  const specials = new Set(pieces.slice(0, 3).map(([p]) => p));
  for (const name of ["HUMILITY", "Travis Kelce", "O'Neal"]) {
    for (const piece of tokenize(name, pieces)) assert.ok(!specials.has(piece));
  }
});

test("tokenize returns null when a character has no piece", () => {
  assert.equal(tokenize("Zq#9", pieces), null);
});

test("tokenize of a blank or whitespace-only name is empty", () => {
  assert.deepEqual(tokenize("   ", pieces), []);
});

// ----------------------------------------------------------------------------- (b) CTC

test("ctcLogLikelihood matches the Python ctc_log_likelihood golden values", () => {
  const g = readJson(join(FIXTURES, "ctc_golden.json"));
  const lp = f32FromBase64(g.logprobs_f32_base64);
  assert.equal(lp.length, g.T * g.V);
  for (const c of g.cases) {
    const expected = fromJsonNum(c.ll);
    const got = ctcLogLikelihood(lp, g.T, g.V, c.a, c.b, c.target, g.blank);
    const label = `span [${c.a},${c.b}) target ${JSON.stringify(c.target)}`;
    if (expected === -Infinity) {
      assert.equal(got, -Infinity, label);
    } else {
      assert.ok(Math.abs(got - expected) < 1e-9, `${label}: got ${got}, want ${expected}`);
    }
  }
});

test("ctcLogLikelihood is exact on a tiny case", () => {
  // Two frames, one label: P = p0(blank)p1(a) + p0(a)p1(a) + p0(a)p1(blank), blank=0, label=1.
  const p = [[0.5, 0.3, 0.2], [0.4, 0.1, 0.5]];
  const lp = Float64Array.from(p.flat().map(Math.log));
  const expected = Math.log(0.5 * 0.1 + 0.3 * 0.1 + 0.3 * 0.4);
  assert.ok(Math.abs(ctcLogLikelihood(lp, 2, 3, 0, 2, [1], 0) - expected) < 1e-12);
});

// ----------------------------------------------------------------------------- (c) Booster parity

function wordsFromJson(list) {
  return list.map((w) => ({
    text: w.text,
    start: fromJsonNum(w.start),
    end: fromJsonNum(w.end),
    confidence: fromJsonNum(w.confidence),
    tier: w.tier,
    boosted: w.boosted,
    original: w.original,
  }));
}

for (const name of ["booster_case0", "booster_case3"]) {
  test(`Booster.apply matches Python on ${name}`, () => {
    const meta = readJson(join(FIXTURES, `${name}.json`));
    const lp = f32FromFile(join(FIXTURES, meta.logprobs_file));
    assert.equal(lp.length, meta.T * meta.V);
    assert.equal(vocab.length, meta.V);

    const booster = new Booster(meta.names, vocab, pieces, { weight: meta.weight, pad: meta.pad });
    const greedy = wordsFromJson(meta.greedy);
    const got = booster.apply(greedy, lp, meta.T, meta.V, meta.t0);
    const want = wordsFromJson(meta.expected);

    assert.deepEqual(got.map((w) => w.text), want.map((w) => w.text), "texts");
    assert.deepEqual(got.map((w) => w.boosted), want.map((w) => w.boosted), "boosted flags");
    assert.deepEqual(got.map((w) => w.original ?? null), want.map((w) => w.original ?? null), "original");
    assert.deepEqual(got.map((w) => w.tier), want.map((w) => w.tier), "tiers");
    got.forEach((w, k) => {
      assert.ok(Math.abs(w.confidence - want[k].confidence) < 1e-4,
        `confidence of ${w.text}: got ${w.confidence}, want ${want[k].confidence}`);
      assert.ok(Math.abs(w.start - want[k].start) < 1e-9, `start of ${w.text}`);
      assert.ok(Math.abs(w.end - want[k].end) < 1e-9, `end of ${w.text}`);
    });
  });
}

test("booster_case0 really is a boost (sanity for the parity fixture)", () => {
  const meta = readJson(join(FIXTURES, "booster_case0.json"));
  assert.ok(meta.expected.some((w) => w.boosted), "the fixture should contain a replacement");
});

// ----------------------------------------------------------------------------- crafted matrix

// Mirrors tests/test_booster.py: frames blank, KEL spike, a contested final piece (SO vs CE).
const VOCAB_INDEX = new Map(vocab.map((p, i) => [p, i]));
const EPS = 1e-6;

function craftedMatrix(pSo, pCe) {
  const T = 10;
  const V = vocab.length;
  const lp = new Float64Array(T * V).fill(Math.log(EPS));
  const spec = {
    2: { "▁K": 0.9 }, 3: { "▁K": 0.9 }, 4: { EL: 0.9 }, 5: { EL: 0.9 },
    6: { "▁SO": pSo, CE: pCe }, 7: { "▁SO": pSo, CE: pCe },
  };
  for (let t = 0; t < T; t++) {
    const probs = { ...(spec[t] ?? {}) };
    const total = Object.values(probs).reduce((s, v) => s + v, 0);
    const byId = new Map();
    for (const [piece, v] of Object.entries(probs)) byId.set(VOCAB_INDEX.get(piece), v);
    byId.set(0, 1.0 - total);
    for (const [k, v] of byId) lp[t * V + k] = Math.log(Math.max(v, EPS));
  }
  return { lp, T, V };
}

// Greedy reading of craftedMatrix: "KEL" (frames 2-5) then "SO" (frames 6-7).
function craftedGreedy() {
  return [
    { text: "KEL", start: 2 / 25, end: 6 / 25, confidence: 0.9, tier: "high", boosted: false, original: null },
    { text: "SO", start: 6 / 25, end: 8 / 25, confidence: 0.5, tier: "low", boosted: false, original: null },
  ];
}

test("roster name wins a contested window and is flagged, original kept", () => {
  const { lp, T, V } = craftedMatrix(0.5, 0.4);
  const words = craftedGreedy();
  const origConf = (words[0].confidence + words[1].confidence) / 2;
  const out = new Booster(["Kelce"], vocab, pieces).apply(words, lp, T, V, 0);

  assert.deepEqual(out.map((w) => w.text), ["KELCE"]);
  const w = out[0];
  assert.equal(w.boosted, true);
  assert.equal(w.original, "KEL SO");
  assert.equal(w.start, words[0].start);
  assert.equal(w.end, words[1].end);
  assert.ok(w.confidence <= origConf + 1e-9, "boosting never raises confidence");
});

test("a confident model keeps its words", () => {
  const { lp, T, V } = craftedMatrix(0.95, 0.01);
  const words = craftedGreedy();
  const out = new Booster(["Kelce"], vocab, pieces).apply(words, lp, T, V, 0);
  assert.deepEqual(out.map((w) => w.text), ["KEL", "SO"]);
  assert.ok(out.every((w) => !w.boosted && w.original === null));
});

test("no names is the identity", () => {
  const { lp, T, V } = craftedMatrix(0.5, 0.4);
  const words = craftedGreedy();
  assert.deepEqual(new Booster([], vocab, pieces).apply(words, lp, T, V, 0), words);
});

test("names the vocabulary cannot write are skipped, not fatal", () => {
  const { lp, T, V } = craftedMatrix(0.5, 0.4);
  const words = craftedGreedy();
  const out = new Booster(["Zq#9"], vocab, pieces).apply(words, lp, T, V, 0);
  assert.deepEqual(out.map((w) => w.text), ["KEL", "SO"]);
});

test("multi-word names are also tried word by word", () => {
  const booster = new Booster(["Travis Kelce"], vocab, pieces);
  const texts = new Set(booster.candidates.map((c) => c.text));
  for (const t of ["TRAVIS KELCE", "TRAVIS", "KELCE"]) assert.ok(texts.has(t), t);
});

test("apply rejects a logprob matrix that does not match T and V", () => {
  const booster = new Booster(["Kelce"], vocab, pieces);
  assert.throws(() => booster.apply(craftedGreedy(), new Float64Array(10), 10, vocab.length, 0), RangeError);
});

// ----------------------------------------------------------------------------- helpers

test("roundHalfEven matches Python round()", () => {
  assert.equal(roundHalfEven(2.5), 2);
  assert.equal(roundHalfEven(3.5), 4);
  assert.equal(roundHalfEven(-2.5), -2);
  assert.equal(roundHalfEven(2.4), 2);
  assert.equal(roundHalfEven(2.6), 3);
  assert.equal(roundHalfEven(-3.5), -4);
});

test("tierFor uses HIGH >= 0.90 and MEDIUM >= 0.70", () => {
  assert.equal(tierFor(0.9), "high");
  assert.equal(tierFor(0.8999), "medium");
  assert.equal(tierFor(0.7), "medium");
  assert.equal(tierFor(0.6999), "low");
});
