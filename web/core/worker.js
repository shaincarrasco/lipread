// Inference worker: YuNet face detection and the lip-reading model, both in onnxruntime-web (WASM).
// Runs off the main thread so the UI stays smooth. Messages: {id, type, ...} -> {id, ok, result|error}.
import { CONFIG } from "./config.js";
import { decode as decodeYunet, toTensorData, YUNET_SIZE } from "./yunet.js";

let ort = null, yunet = null, lips = null, lipsOut = null, eos = -1;

async function getOrt() {
  if (ort) return ort;
  ort = await import(CONFIG.ortModule);
  ort.env.wasm.wasmPaths = CONFIG.ortWasmPaths;
  // GitHub Pages can't send COOP/COEP headers, so threads are only used if the page is isolated.
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
  return ort;
}

const handlers = {
  config({ overrides }) { Object.assign(CONFIG, overrides || {}); return { threads: self.crossOriginIsolated }; },

  async initYunet() {
    if (yunet) return true;
    const o = await getOrt();
    const buf = await (await fetch(CONFIG.yunetUrl)).arrayBuffer();
    yunet = await o.InferenceSession.create(buf, { executionProviders: ["wasm"] });
    return true;
  },

  async detect({ rgba, scoreTh, nmsTh }) {
    const o = await getOrt();
    const input = new o.Tensor("float32", toTensorData(rgba), [1, 3, YUNET_SIZE, YUNET_SIZE]);
    const res = await yunet.run({ input });
    const outputs = {};
    for (const [k, v] of Object.entries(res)) outputs[k] = v.data;
    return decodeYunet(outputs, scoreTh, nmsTh);
  },

  async initLips() {
    if (lips) return true;
    const o = await getOrt();
    const cache = await caches.open(CONFIG.modelCache);
    const resp = await cache.match(CONFIG.modelUrl);
    if (!resp) throw new Error("MODEL_MISSING");
    let buf = await resp.arrayBuffer();
    lips = await o.InferenceSession.create(new Uint8Array(buf), { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    buf = null;
    lipsOut = lips.outputNames[0];
    return true;
  },

  async lips({ crops, T }) {
    const o = await getOrt();
    const S = 88 * 88;
    const x = new Float32Array(T * S);
    for (let i = 0; i < x.length; i++) x[i] = (crops[i] / 255 - 0.421) / 0.165;
    let V = 0, out = null;
    const run = async (s, e) => {
      const n = e - s;
      const t = new o.Tensor("float32", x.subarray(s * S, e * S), [1, 1, n, 88, 88]);
      const r = await lips.run({ video: t });
      const lp = r[lipsOut].data;
      V = lp.length / n;
      if (eos < 0) eos = V - 1;
      for (let f = 0; f < n; f++) lp[f * V + eos] = -1e4;   // <eos> is never a valid emission
      return lp;
    };
    out = await stitch(T, run, CONFIG.chunkFrames, CONFIG.chunkOverlap, () => V);
    return { logprobs: out, T, V };
  },

  dispose() { yunet = null; lips = null; return true; },
};

/** Port of engine.stitch_windows: each frame comes from the window where it is farthest from an edge. */
async function stitch(T, run, chunk, overlap, getV) {
  if (T <= chunk) return run(0, T);
  const step = chunk - overlap, starts = [];
  for (let s = 0; s < Math.max(T - overlap, 1); s += step) starts.push(s);
  if (starts[starts.length - 1] + chunk < T) starts.push(T - chunk);
  let out = null;
  const best = new Float32Array(T).fill(-1);
  for (const s of starts) {
    const e = Math.min(s + chunk, T);
    const lp = await run(s, e);
    const V = getV();
    if (!out) out = new Float32Array(T * V);
    for (let i = s; i < e; i++) {
      let margin = Math.min(i - s, e - 1 - i);
      if (s === 0 && i - s < overlap) margin += overlap;
      if (e === T && e - 1 - i < overlap) margin += overlap;
      if (margin > best[i]) { out.set(lp.subarray((i - s) * V, (i - s + 1) * V), i * V); best[i] = margin; }
    }
  }
  return out;
}

self.onmessage = async (e) => {
  const { id, type, ...args } = e.data;
  try {
    const result = await handlers[type](args);
    const transfer = result && result.logprobs ? [result.logprobs.buffer] : [];
    self.postMessage({ id, ok: true, result }, transfer);
  } catch (err) {
    self.postMessage({ id, ok: false, error: String((err && err.message) || err) });
  }
};
