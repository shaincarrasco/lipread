// One-time download of the lip-reading model into Cache Storage. The file is hashed as it arrives
// (no extra copies), a stalled connection is retried with an HTTP Range request, partial progress
// survives "Try Again" for as long as the page is open, and the bytes are streamed into the cache.
import { CONFIG } from "./config.js";
import { Sha256 } from "./sha256.js";

const IDLE_MS = 20000;       // no bytes for this long = stalled connection
const MAX_ATTEMPTS = 6;

// Partial download kept across calls (same page session), so "Try Again" continues.
let partial = null;          // {chunks: Uint8Array[], loaded, hash: Sha256}

export async function modelStatus() {
  if (!("caches" in self)) return { cached: false, bytes: 0, supported: false };
  const cache = await caches.open(CONFIG.modelCache);
  const r = await cache.match(CONFIG.modelUrl);
  return { cached: !!r, bytes: r ? CONFIG.modelBytes : 0, supported: true, total: CONFIG.modelBytes,
           partialBytes: partial ? partial.loaded : 0 };
}

/** onBytes(loaded, total). Honours `signal` (cancel keeps the partial download for later). */
export async function downloadModel(onBytes = () => {}, signal) {
  const total = CONFIG.modelBytes;
  if (!partial) partial = { chunks: [], loaded: 0, hash: new Sha256() };
  const p = partial;
  onBytes(p.loaded, total);
  let attempts = 0;
  while (p.loaded < total) {
    if (signal?.aborted) throw new DOMException("Download cancelled", "AbortError");
    const ctrl = new AbortController();
    const onAbort = () => ctrl.abort();
    signal?.addEventListener("abort", onAbort, { once: true });
    let idle = null;
    const arm = () => { clearTimeout(idle); idle = setTimeout(() => ctrl.abort(), IDLE_MS); };
    try {
      arm();
      const headers = p.loaded ? { Range: `bytes=${p.loaded}-` } : {};
      const resp = await fetch(CONFIG.modelUrl, { headers, signal: ctrl.signal, cache: "no-store" });
      if (!resp.ok && resp.status !== 206) throw new Error(`HTTP ${resp.status}`);
      if (p.loaded && resp.status === 200) {          // server ignored the Range header: start over
        p.chunks = []; p.loaded = 0; p.hash = new Sha256();
      }
      const reader = resp.body.getReader();
      for (;;) {
        arm();
        const { done, value } = await reader.read();
        if (done) break;
        p.chunks.push(value); p.hash.update(value); p.loaded += value.byteLength;
        onBytes(Math.min(p.loaded, total), total);
        attempts = 0;                                   // progress resets the retry budget
      }
      if (p.loaded < total) throw new Error("connection closed early");
    } catch (err) {
      if (signal?.aborted) throw new DOMException("Download cancelled", "AbortError");
      if (++attempts > MAX_ATTEMPTS) throw new Error(`DOWNLOAD_FAILED: ${err.message || err}`);
      await new Promise((r) => setTimeout(r, Math.min(8000, 1000 * 2 ** attempts)));
    } finally {
      clearTimeout(idle);
      signal?.removeEventListener("abort", onAbort);
    }
  }
  if (p.loaded !== total || p.hash.hex() !== CONFIG.modelSha256) {
    partial = null;
    throw new Error("CHECKSUM_MISMATCH");
  }
  // Stream the chunks into the cache, releasing each one as it is written.
  const chunks = p.chunks;
  partial = null;
  const body = new ReadableStream({
    pull(controller) { chunks.length ? controller.enqueue(chunks.shift()) : controller.close(); },
  });
  const cache = await caches.open(CONFIG.modelCache);
  try {
    await cache.put(CONFIG.modelUrl, new Response(body, {
      headers: { "Content-Type": "application/octet-stream", "Content-Length": String(total) } }));
  } catch (err) {
    throw new Error(`STORAGE_FULL: ${err.message || err}`);
  }
  try { await navigator.storage?.persist?.(); } catch { /* best effort */ }
  return true;
}

export async function deleteModel() {
  partial = null;
  return caches.delete(CONFIG.modelCache);
}
