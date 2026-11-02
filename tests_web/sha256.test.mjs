import { test } from "node:test";
import assert from "node:assert/strict";
import { createHash, randomBytes } from "node:crypto";
import { readFileSync, existsSync } from "node:fs";
import { Sha256 } from "../web/core/sha256.js";

test("matches node:crypto for empty, short, block-boundary and random-chunked inputs", () => {
  for (const n of [0, 1, 55, 56, 63, 64, 65, 119, 120, 1000, 100003]) {
    const data = randomBytes(n);
    const h = new Sha256();
    let i = 0;
    while (i < n) { const step = 1 + Math.floor(Math.random() * 300); h.update(data.subarray(i, i + step)); i += step; }
    assert.equal(h.hex(), createHash("sha256").update(data).digest("hex"), `length ${n}`);
  }
});

const model = "/home/claude/models/lipread_ctc.int8.onnx";
test("verifies the real 203 MB model in 1 MB chunks", { skip: !existsSync(model) }, () => {
  const data = readFileSync(model), h = new Sha256(), t = Date.now();
  for (let i = 0; i < data.length; i += 1 << 20) h.update(data.subarray(i, i + (1 << 20)));
  assert.equal(h.hex(), "da02d72ee472eb3d7d50f7dd309bfe81973acb4cac8bb770dd86aa97872f195b");
  console.log(`# hashed ${(data.length / 1e6).toFixed(0)} MB in ${((Date.now() - t) / 1000).toFixed(1)} s`);
});
