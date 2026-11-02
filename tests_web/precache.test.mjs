// The installed app must start offline: every module and asset the app loads has to be in the
// service worker's precache list, or the offline launch fails with a module import error.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";

const web = new URL("../web/", import.meta.url);
const sw = readFileSync(new URL("sw.js", web), "utf8");
const block = sw.match(/const REQUIRED_PATHS = \[([\s\S]*?)\];/)[1];
const required = new Set([...block.matchAll(/"([^"]+)"/g)].map((m) => m[1]));

test("every core module is precached", () => {
  const files = [
    ...readdirSync(new URL("core/", web)).filter((f) => f.endsWith(".js")).map((f) => `core/${f}`),
    ...readdirSync(new URL("core/wasm/", web)).map((f) => `core/wasm/${f}`),
  ];
  const missing = files.filter((f) => !required.has(f));
  assert.deepEqual(missing, [], `add to REQUIRED_PATHS in web/sw.js: ${missing.join(", ")}`);
});

test("every relative import in core/ resolves to a precached file", () => {
  for (const f of readdirSync(new URL("core/", web)).filter((x) => x.endsWith(".js"))) {
    const src = readFileSync(new URL(`core/${f}`, web), "utf8");
    for (const [, spec] of src.matchAll(/(?:from\s+|import\()\s*"(\.{1,2}\/[^"]+)"/g)) {
      const path = new URL(spec, new URL(`core/${f}`, web)).href.slice(web.href.length);
      assert.ok(required.has(path), `${f} imports ${spec} (${path}), which is not precached`);
    }
  }
});

test("every precached path exists", () => {
  // vendor/ is filled at deploy time by scripts/vendor_ort.sh, so it is absent in a fresh checkout.
  for (const p of required) {
    if (p.startsWith("vendor/")) continue;
    assert.doesNotThrow(() => readFileSync(new URL(p, web)), p);
  }
});
