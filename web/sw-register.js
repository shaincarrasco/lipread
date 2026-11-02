// Registers the LipRead service worker at the app scope. Renders nothing.
// The worker also makes the page cross-origin isolated (faster multi-threaded lip reading). That only
// takes effect for pages it controls, so on the very first visit we reload once after it activates —
// before the user has started anything.
if ("serviceWorker" in navigator) {
  const KEY = "lipread-coi-reload";
  const register = async () => {
    try {
      const reg = await navigator.serviceWorker.register("./sw.js", { scope: "./" });
      if (self.crossOriginIsolated) { sessionStorage.removeItem(KEY); return; }
      const reloadOnce = () => {
        if (sessionStorage.getItem(KEY) || window.__lipreadBusy) return;   // never interrupt work
        sessionStorage.setItem(KEY, "1");
        location.reload();
      };
      if (navigator.serviceWorker.controller) reloadOnce();
      else navigator.serviceWorker.addEventListener("controllerchange", reloadOnce, { once: true });
      void reg;
    } catch (err) {
      console.warn("LipRead: offline support unavailable", err);
    }
  };
  if (document.readyState === "complete") register();
  else window.addEventListener("load", register, { once: true });
}
