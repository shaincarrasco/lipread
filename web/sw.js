/*
 * LipRead service worker.
 *
 * - App shell: versioned precache, cache-first. Each deploy gets a new cache name
 *   (the Pages workflow replaces the build placeholder below with the commit SHA),
 *   and activate() deletes only older "lipread-shell-*" caches.
 * - Navigations: network-first, falling back to the cached index.html when offline.
 * - ONNX Runtime from cdn.jsdelivr.net: runtime cache, cache-first (URLs are versioned).
 * - The 203 MB model lives in "lipread-model-v1" (web/core/model-store.js). This worker
 *   never precaches it, never intercepts Hugging Face, and never deletes that cache.
 *
 * Everything is resolved against the registration scope, so the app works at a subpath
 * such as https://<user>.github.io/lipread/.
 */

const BUILD_ID = "__BUILD_ID__";
const SHELL_PREFIX = "lipread-shell-";
const SHELL_CACHE = SHELL_PREFIX + BUILD_ID;
const CDN_CACHE = "lipread-cdn-v1";
const CDN_HOST = "cdn.jsdelivr.net";

// Relative to the scope. Missing files are tolerated (cached individually).
const SHELL_PATHS = [
  "index.html",
  "styles.css",
  "app.js",
  "sw-register.js",
  "manifest.webmanifest",
  "404.html",
  "icons/icon.svg",
  "icons/icon-180.png",
  "icons/icon-192.png",
  "icons/icon-512.png",
  "icons/maskable-512.png",
  "icons/favicon-32.png",
  "core/config.js",
  "core/video.js",
  "core/yunet.js",
  "core/tracker.js",
  "core/align.js",
  "core/engine.js",
  "core/decode.js",
  "core/quality.js",
  "core/booster.js",
  "core/model-store.js",
  "core/lipread.js",
  "core/worker.js",
  "core/assets/face_detection_yunet_2023mar.onnx",
  "core/assets/tokens.json",
  "core/assets/geometry.json",
  "core/assets/spm_pieces.json",
  "core/assets/spm_golden.json",
];

const scopeURL = () => self.registration.scope;
const inScope = (url) => url.href.startsWith(scopeURL());
const keyFor = (path) => new URL(path, scopeURL()).href;
const SHELL_KEYS = new Set(SHELL_PATHS.map(keyFor));
const INDEX_KEY = keyFor("index.html");

self.addEventListener("install", (event) => {
  event.waitUntil(precacheShell().then(() => self.skipWaiting()));
});

async function precacheShell() {
  const cache = await caches.open(SHELL_CACHE);
  await Promise.allSettled(
    SHELL_PATHS.map(async (path) => {
      const key = keyFor(path);
      // "reload" bypasses the HTTP cache so the precache is always the freshly deployed file.
      const res = await fetch(new Request(key, { cache: "reload" }));
      if (!res.ok) throw new Error(`HTTP ${res.status} for ${path}`);
      await cache.put(key, res);
    })
  );
}

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      const names = await caches.keys();
      // Only shell caches from earlier builds. Never the model cache, never the CDN cache.
      await Promise.all(
        names
          .filter((name) => name.startsWith(SHELL_PREFIX) && name !== SHELL_CACHE)
          .map((name) => caches.delete(name))
      );
      await self.clients.claim();
    })()
  );
});

self.addEventListener("message", (event) => {
  if (event.data === "SKIP_WAITING") self.skipWaiting();
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;

  const url = new URL(req.url);

  if (url.hostname === CDN_HOST) {
    event.respondWith(cdnCacheFirst(req));
    return;
  }

  // Everything else that is not ours (Hugging Face model download, other origins) goes straight to the network.
  if (url.origin !== self.location.origin || !inScope(url)) return;

  if (req.mode === "navigate") {
    event.respondWith(navigationNetworkFirst(req).then(isolate));
    return;
  }

  event.respondWith(shellCacheFirst(req).then(isolate));
});

// Cross-origin isolation (COOP/COEP) lets onnxruntime-web use several CPU threads. GitHub Pages can't
// send these headers, so the service worker adds them to our own responses (the coi-serviceworker
// technique). Measured on a 2-core desktop: lip model 14.1 s -> 8.6 s for 12 s of video. Browsers that
// don't honour it simply stay single-threaded.
function isolate(res) {
  if (!res || res.status === 0 || res.type === "opaque" || res.type === "error") return res;
  const headers = new Headers(res.headers);
  headers.set("Cross-Origin-Opener-Policy", "same-origin");
  headers.set("Cross-Origin-Embedder-Policy", "require-corp");
  headers.set("Cross-Origin-Resource-Policy", "cross-origin");
  return new Response(res.body, { status: res.status, statusText: res.statusText, headers });
}

async function shellCacheFirst(req) {
  const key = new URL(req.url);
  key.search = "";
  key.hash = "";
  const cache = await caches.open(SHELL_CACHE);
  const hit = await cache.match(key.href);
  if (hit) return hit;
  try {
    const res = await fetch(req);
    if (res.ok && SHELL_KEYS.has(key.href)) cache.put(key.href, res.clone()).catch(() => {});
    return res;
  } catch (err) {
    return Response.error();
  }
}

async function navigationNetworkFirst(req) {
  const cache = await caches.open(SHELL_CACHE);
  try {
    const res = await fetch(req);
    if (res.ok) cache.put(INDEX_KEY, res.clone()).catch(() => {});
    return res;
  } catch (err) {
    const hit = await cache.match(INDEX_KEY);
    return hit || offlineResponse();
  }
}

async function cdnCacheFirst(req) {
  const cache = await caches.open(CDN_CACHE);
  const hit = await cache.match(req.url);
  if (hit) return hit;
  const res = await fetch(req);
  // Opaque responses (no-cors) cannot be inspected, but are cacheable and used by some loaders.
  if (res.ok || res.type === "opaque") cache.put(req.url, res.clone()).catch(() => {});
  return res;
}

function offlineResponse() {
  const html =
    '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">' +
    "<title>LipRead</title><body style=\"font:17px -apple-system,system-ui,sans-serif;padding:24px\">" +
    "<h1 style=\"font-size:28px\">You are offline</h1>" +
    "<p>Open LipRead once while online to set it up. After that it works without a connection.</p>";
  return new Response(html, {
    status: 503,
    headers: { "Content-Type": "text/html; charset=utf-8" },
  });
}
