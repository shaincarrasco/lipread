// One-time download of the lip-reading model into Cache Storage, with progress, resume and a
// SHA-256 check. Afterwards the app works offline.
import { CONFIG } from "./config.js";

export async function modelStatus() {
  if (!("caches" in self)) return { cached: false, bytes: 0, supported: false };
  const cache = await caches.open(CONFIG.modelCache);
  const r = await cache.match(CONFIG.modelUrl);
  return { cached: !!r, bytes: r ? CONFIG.modelBytes : 0, supported: true, total: CONFIG.modelBytes };
}

/** onBytes(loaded, total). Resumes with HTTP Range after a dropped connection (same session). */
export async function downloadModel(onBytes = () => {}, signal) {
  const total = CONFIG.modelBytes;
  const chunks = [];
  let loaded = 0, attempts = 0;
  while (loaded < total) {
    try {
      const headers = loaded ? { Range: `bytes=${loaded}-` } : {};
      const resp = await fetch(CONFIG.modelUrl, { headers, signal, cache: "no-store" });
      if (!resp.ok && resp.status !== 206) throw new Error(`HTTP ${resp.status}`);
      if (loaded && resp.status === 200) { chunks.length = 0; loaded = 0; }   // server ignored Range
      const reader = resp.body.getReader();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value); loaded += value.byteLength; onBytes(loaded, total);
      }
      if (loaded < total) throw new Error("connection closed early");
    } catch (err) {
      if (signal && signal.aborted) throw new DOMException("Download cancelled", "AbortError");
      if (++attempts > 4) throw new Error(`DOWNLOAD_FAILED: ${err.message || err}`);
      await new Promise((r) => setTimeout(r, 1500 * attempts));
    }
  }
  const blob = new Blob(chunks, { type: "application/octet-stream" });
  chunks.length = 0;
  const digest = await crypto.subtle.digest("SHA-256", await blob.arrayBuffer());
  const hex = [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
  if (hex !== CONFIG.modelSha256) throw new Error("CHECKSUM_MISMATCH");
  const cache = await caches.open(CONFIG.modelCache);
  try {
    await cache.put(CONFIG.modelUrl, new Response(blob, { headers: { "Content-Type": "application/octet-stream", "Content-Length": String(blob.size) } }));
  } catch (err) {
    throw new Error(`STORAGE_FULL: ${err.message || err}`);
  }
  try { await navigator.storage?.persist?.(); } catch { /* best effort */ }
  return true;
}

export async function deleteModel() {
  return caches.delete(CONFIG.modelCache);
}
