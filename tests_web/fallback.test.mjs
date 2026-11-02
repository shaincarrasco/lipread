// A decoder that fails in the middle of a run must not cost the user their result:
// Rust -> JS -> greedy, retrying the same face each time.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

globalThis.self ??= globalThis;
const { LipRead } = await import("../web/core/lipread.js");
const tokens = JSON.parse(readFileSync(new URL("../web/core/assets/tokens.json", import.meta.url), "utf8"));

// "HELLO" emitted once on frame 2, blank everywhere else.
function clip() {
  const T = 10, V = tokens.length, lp = new Float32Array(T * V).fill(-30);
  for (let t = 0; t < T; t++) lp[t * V] = -0.01;
  const k = tokens.indexOf("▁HELLO");
  lp[2 * V] = -30; lp[2 * V + k] = -0.01;
  return { lp, T, V };
}
const throwing = (kind) => ({ kind, decode() { throw new Error(`${kind} trapped`); } });
const quiet = (fn) => async (...a) => { const w = console.warn, i = console.info; console.warn = console.info = () => {};
  try { return await fn(...a); } finally { console.warn = w; console.info = i; } };

test("a failing Rust decoder falls back to JS for the same face", quiet(async () => {
  const lr = new LipRead();
  const calls = [];
  lr._makeDecoder = async (tryRust) => { calls.push(tryRust); return { kind: "js", decode: () => ({ words: [{ text: "FROM-JS" }] }) }; };
  const assets = { tokens, decoder: throwing("rust") };
  const { lp, T, V } = clip();
  const words = await lr._decodeWords(assets, lp, T, V, 0, []);
  assert.deepEqual(words.map((w) => w.text), ["FROM-JS"]);
  assert.deepEqual(calls, [false]);
  assert.equal(assets.decoder.kind, "js");   // later faces use JS directly
}));

test("if every decoder fails, greedy decoding still gives the words", quiet(async () => {
  const lr = new LipRead();
  lr._makeDecoder = async () => throwing("js");
  const assets = { tokens, decoder: throwing("rust") };
  const { lp, T, V } = clip();
  const words = await lr._decodeWords(assets, lp, T, V, 0, []);
  assert.deepEqual(words.map((w) => w.text), ["HELLO"]);
  assert.equal(assets.decoder, null);
}));
