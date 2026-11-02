/*
 * LipRead service worker.
 *
 * - App shell: versioned precache. The Pages workflow stamps the build id below (the commit SHA), so each
 *   deploy gets its own "lipread-shell-<build>" cache. Install is all-or-nothing over the required files:
 *   if one fails, the previous worker keeps serving.
 * - Navigations: the app shell (scope URL or index.html) is served cache-first from the current build and
 *   refreshed in the background. Any other navigation goes to the network and is never cached.
 * - Cross-origin isolation: same-origin responses get COOP/COEP/CORP (threads for onnxruntime-web) and a
 *   Content-Security-Policy.
 * - No mid-session swap: a new build waits. The page activates it only when the app is idle (sw-register.js).
 * - The 203 MB model lives in "lipread-model-v1" (web/core/model-store.js). This worker never touches it and
 *   never intercepts Hugging Face. The old jsDelivr cache is deleted on activate.
 *
 * Everything is resolved against the registration scope, so the app works at a subpath.
 */

const BUILD_ID = "__BUILD_ID__";
const SHELL_PREFIX = "lipread-shell-";
const SHELL_CACHE = SHELL_PREFIX + BUILD_ID;
const LEGACY_CDN_CACHE = "lipread-cdn-v1";
const NAV_TIMEOUT_MS = 3500;

// Relative to the scope. Every one must cache, or the install fails and the old worker stays.
const REQUIRED_PATHS = [
  "index.html",
  "styles.css",
  "app.js",
  "sw-register.js",
  "manifest.webmanifest",
  "core/align.js",
  "core/booster.js",
  "core/config.js",
  "core/decode.js",
  "core/lipread.js",
  "core/model-store.js",
  "core/quality.js",
  "core/tracker.js",
  "core/video.js",
  "core/worker.js",
  "core/yunet.js",
  "core/assets/face_detection_yunet_2023mar.onnx",
  "core/assets/tokens.json",
  "core/assets/geometry.json",
  "core/assets/spm_pieces.json",
  "core/assets/spm_golden.json",
  "vendor/ort/ort.wasm.min.mjs",
  "vendor/ort/ort-wasm-simd-threaded.mjs",
  "vendor/ort/ort-wasm-simd-threaded.wasm",
];

// Best effort: a missing icon or the 404 page does not block the install.
const OPTIONAL_PATHS = [
  "404.html",
  "icons/icon.svg",
  "icons/icon-180.png",
  "icons/icon-192.png",
  "icons/icon-512.png",
  "icons/maskable-512.png",
  "icons/favicon-32.png",
];

const keyFor = (path) => new URL(path, self.registration.scope).href;
const SHELL_KEYS = new Set([...REQUIRED_PATHS, ...OPTIONAL_PATHS].map(keyFor));
const INDEX_KEY = keyFor("index.html");
const SCOPE_KEY = self.registration.scope;

const withoutQuery = (href) => {
  const url = new URL(href);
  url.search = "";
  url.hash = "";
  return url.href;
};

self.addEventListener("install", (event) => {
  // No skipWaiting() here. A new build waits until the open app is idle (see the message handler).
  event.waitUntil(precacheShell());
});

async function precacheShell() {
  const cache = await caches.open(SHELL_CACHE);
  const fetchInto = async (path) => {
    const key = keyFor(path);
    // "reload" bypasses the HTTP cache so the precache holds the freshly deployed file.
    const res = await fetch(new Request(key, { cache: "reload" }));
    if (!res.ok) throw new Error(`HTTP ${res.status} for ${path}`);
    await cache.put(key, res);
  };
  try {
    await Promise.all(REQUIRED_PATHS.map(fetchInto));
  } catch (err) {
    // A half-filled cache is useless. Drop it so the previous build keeps serving.
    await caches.delete(SHELL_CACHE);
    throw err;
  }
  await Promise.allSettled(OPTIONAL_PATHS.map(fetchInto));
}

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      const names = await caches.keys();
      // Shell caches from earlier builds, and the retired CDN cache. Never the model cache.
      const stale = names.filter(
        (name) => (name.startsWith(SHELL_PREFIX) && name !== SHELL_CACHE) || name === LEGACY_CDN_CACHE
      );
      await Promise.all(stale.map((name) => caches.delete(name)));
      await self.clients.claim();
    })()
  );
});

self.addEventListener("message", (event) => {
  if (event.data && event.data.type === "SKIP_WAITING") self.skipWaiting();
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;

  const url = new URL(req.url);
  // Hugging Face (the model), other origins and anything outside the scope go straight to the network.
  if (url.origin !== self.location.origin || !url.href.startsWith(SCOPE_KEY)) return;

  if (req.mode === "navigate") {
    event.respondWith(navigate(event, req).then(isolate));
    return;
  }
  event.respondWith(shellCacheFirst(req).then(isolate));
});

// Cross-origin isolation (COOP/COEP) lets onnxruntime-web use several CPU threads. GitHub Pages can't send
// these headers, so the service worker adds them to our own responses (the coi-serviceworker technique).
// Browsers that don't honour it simply stay single-threaded.
const CSP = [
  "default-src 'self'",
  "script-src 'self' 'wasm-unsafe-eval'",
  "worker-src 'self' blob:",
  "connect-src 'self' https://huggingface.co https://*.huggingface.co https://*.hf.co",
  "img-src 'self' data: blob:",
  "media-src 'self' blob:",
  "style-src 'self' 'unsafe-inline'",
  "object-src 'none'",
  "base-uri 'self'",
  "form-action 'none'",
].join("; ");

function isolate(res) {
  if (!res || res.status === 0 || res.type === "opaque" || res.type === "error") return res;
  const headers = new Headers(res.headers);
  headers.set("Cross-Origin-Opener-Policy", "same-origin");
  headers.set("Cross-Origin-Embedder-Policy", "require-corp");
  headers.set("Cross-Origin-Resource-Policy", "cross-origin");
  headers.set("Content-Security-Policy", CSP);
  return new Response(res.body, { status: res.status, statusText: res.statusText, headers });
}

async function shellCacheFirst(req) {
  const key = withoutQuery(req.url);
  const cache = await caches.open(SHELL_CACHE);
  const hit = await cache.match(key);
  if (hit) return hit;
  try {
    const res = await fetch(req);
    if (res.ok && SHELL_KEYS.has(key)) cache.put(key, res.clone()).catch(() => {});
    return res;
  } catch (err) {
    return Response.error();
  }
}

async function navigate(event, req) {
  const key = withoutQuery(req.url);
  if (key !== SCOPE_KEY && key !== INDEX_KEY) return otherNavigation(req);

  const cache = await caches.open(SHELL_CACHE);
  const cached = await cache.match(INDEX_KEY);
  if (cached) {
    // Serve the current build at once. Refresh index.html in the background for the next visit.
    event.waitUntil(refreshIndex(cache));
    return cached;
  }

  // Nothing cached (not expected after a successful install): network, bounded by a timeout.
  const net = fetch(req)
    .then(async (res) => {
      if (res.ok) await cache.put(INDEX_KEY, res.clone());
      return res;
    })
    .catch(() => null);
  const winner = await Promise.race([net, after(NAV_TIMEOUT_MS)]);
  if (winner) return winner;
  event.waitUntil(net);
  return offlineResponse();
}

// Never cached: a 404 page or a JSON file opened in a tab must not replace the app shell.
async function otherNavigation(req) {
  try {
    return await fetch(req);
  } catch (err) {
    return offlineResponse();
  }
}

async function refreshIndex(cache) {
  try {
    // Plain URL, not the navigation Request: a navigate-mode Request cannot be re-created with a RequestInit.
    const res = await fetch(INDEX_KEY, { cache: "no-cache" });
    if (res.ok) await cache.put(INDEX_KEY, res.clone());
  } catch (err) {
    // Offline: keep the cached copy.
  }
}

const after = (ms) => new Promise((resolve) => setTimeout(() => resolve(null), ms));

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
