// Registers the LipRead service worker at the app scope and manages its updates. Renders nothing.
//
// - Cross-origin isolation: the worker adds COOP/COEP, which only applies to pages it controls. On the first
//   visit we reload once after it takes control, so the page is isolated (multi-threaded lip reading).
// - Updates: a new build installs and waits. We activate it, and reload once, only when the app is idle.
// Nothing here interrupts work: every reload waits for window.__lipreadBusy to drop to zero.
if ("serviceWorker" in navigator) {
  const KEY = "lipread-coi-reload";
  const sw = navigator.serviceWorker;
  // Read before the worker can take control of this page.
  const hadController = !!sw.controller;

  const store = {
    get: () => { try { return sessionStorage.getItem(KEY); } catch { return null; } },
    set: () => { try { sessionStorage.setItem(KEY, "1"); } catch { /* private mode */ } },
    clear: () => { try { sessionStorage.removeItem(KEY); } catch { /* private mode */ } },
  };
  const isIdle = () => !(window.__lipreadBusy > 0);

  // Reload as soon as the app is idle. Gives up after maxMs (Infinity keeps trying), retrying every 2 s.
  const reloadWhenIdle = (maxMs, beforeReload) => {
    const giveUp = Date.now() + maxMs;
    const attempt = () => {
      if (isIdle()) {
        if (beforeReload) beforeReload();
        location.reload();
        return;
      }
      if (Date.now() < giveUp) setTimeout(attempt, 2000);
    };
    attempt();
  };

  // First visit (or a page that is controlled but not isolated): reload once, while idle, within 20 s.
  const isolateOnce = () => {
    if (self.crossOriginIsolated) { store.clear(); return; }
    if (store.get()) return;   // already tried this session; never loop
    reloadWhenIdle(20000, store.set);
  };

  // Activate a waiting build, but only while the app is idle. Checked every 5 s.
  const activateWhenIdle = (reg) => {
    const tryActivate = () => {
      if (!reg.waiting) return;   // already activated, or replaced by a newer build
      if (!isIdle()) { setTimeout(tryActivate, 5000); return; }
      reg.waiting.postMessage({ type: "SKIP_WAITING" });
    };
    tryActivate();
  };

  // Attached once, before registration, so no takeover is missed. Activation of an update reloads the page;
  // the first claim of a new install isolates it.
  sw.addEventListener("controllerchange", () => {
    if (hadController) reloadWhenIdle(Infinity);
    else isolateOnce();
  });

  const start = async () => {
    let reg;
    try {
      reg = await sw.register("./sw.js", { scope: "./" });
    } catch (err) {
      console.warn("LipRead: offline support unavailable", err);
      return;
    }
    if (reg.waiting && sw.controller) activateWhenIdle(reg);
    reg.addEventListener("updatefound", () => {
      const worker = reg.installing;
      if (!worker) return;
      worker.addEventListener("statechange", () => {
        if (worker.state === "installed" && sw.controller) activateWhenIdle(reg);
      });
    });
    if (self.crossOriginIsolated) store.clear();
    else if (sw.controller) isolateOnce();
  };

  if (document.readyState === "complete") start();
  else window.addEventListener("load", start, { once: true });
}
