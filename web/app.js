// LipRead phone UI. ES module, no framework.
// Talks to the engine only through the contract in web/core/lipread.js (or the dev fake with ?fake=1).

const params = new URLSearchParams(location.search);
const FAKE = params.get("fake") === "1";

const MODEL_MB = 203;
const REPO_URL = "https://github.com/shaincarrasco/lipread";
const KEYS = {
  names: "lipread.names",
  readAll: "lipread.readAll",
  firstRunDismissed: "lipread.firstRunDismissed",
  hintDismissed: "lipread.hintDismissed",
};

const IS_IOS = /iPhone|iPad|iPod/.test(navigator.userAgent) ||
  (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
const IS_IOS_SAFARI = IS_IOS && !/CriOS|FxiOS|EdgiOS|OPiOS/.test(navigator.userAgent);
const IS_STANDALONE = navigator.standalone === true ||
  matchMedia("(display-mode: standalone)").matches;
const REDUCE_MOTION = matchMedia("(prefers-reduced-motion: reduce)").matches;
const DEVICE = IS_IOS ? "iPhone" : "device";
const NS = "http://www.w3.org/2000/svg";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const store = {
  get(key, fallback = null) {
    try { const v = localStorage.getItem(key); return v === null ? fallback : v; } catch { return fallback; }
  },
  set(key, value) {
    try { value == null ? localStorage.removeItem(key) : localStorage.setItem(key, value); } catch { /* private mode */ }
  },
};

/* Busy holds. sw-register.js reloads the page once (first visit) only when window.__lipreadBusy is falsy, so it
   must never reload during a picker, the first-run sheet, a download or a run. Each hold is counted once and
   released once, so a repeated or late release can never take the count below zero. */
const busyHolds = new Set();
function hold(key) {
  if (busyHolds.has(key)) return;
  busyHolds.add(key);
  window.__lipreadBusy = (window.__lipreadBusy | 0) + 1;
}
function release(key) {
  if (!busyHolds.delete(key)) return;
  window.__lipreadBusy = Math.max(0, (window.__lipreadBusy | 0) - 1);
}

const state = {
  engine: null,
  modelCached: false,
  screen: "home",
  openSheet: null,
  sheetOpener: null,
  pendingPick: null,     // "file" | "record" when the user chose a video before the model was ready
  run: null,             // the processing run in progress
  lastRun: null,         // { file, faces } for retry
  recent: [],            // results from this session only (kept in memory)
  current: null,         // the results view
  onProgress: null,
  wakeLock: null,
  toastTimer: 0,
};

/* ---------- Small helpers ---------- */

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value == null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key === "dataset") Object.assign(node.dataset, value);
    else if (key.startsWith("on") && typeof value === "function") node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : String(value));
  }
  for (const child of children.flat()) {
    if (child == null || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function icon(name, cls = "") {
  const s = document.createElementNS(NS, "svg");
  s.setAttribute("class", `icon ${cls}`.trim());
  s.setAttribute("aria-hidden", "true");
  s.setAttribute("focusable", "false");
  const u = document.createElementNS(NS, "use");
  u.setAttribute("href", `#i-${name}`);
  s.append(u);
  return s;
}

const clamp = (n, lo, hi) => Math.min(hi, Math.max(lo, n));

function fmtTime(sec) {
  if (!isFinite(sec) || sec < 0) sec = 0;
  const s = Math.floor(sec);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  const mm = h ? String(m).padStart(2, "0") : String(m);
  const ss = String(r).padStart(2, "0");
  return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

const fmtMB = (bytes) => `${Math.round(bytes / 1e6)} MB`;

function baseName(name) {
  return String(name || "LipRead").replace(/\.[^.]+$/, "").replace(/[^\w\- ]+/g, "").trim() || "LipRead";
}

function nowLabel(date) {
  const diff = (Date.now() - date.getTime()) / 1000;
  if (diff < 60) return "Just now";
  return date.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
}

function hasWasmSimd() {
  try {
    if (typeof WebAssembly !== "object" || typeof WebAssembly.validate !== "function") return false;
    // Smallest module that uses a SIMD opcode (v128.const). Same probe as wasm-feature-detect.
    const bytes = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11]);
    return WebAssembly.validate(bytes);
  } catch {
    return false;
  }
}

function toast(message) {
  const node = $("#toast");
  node.textContent = message;
  node.hidden = false;
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => { node.hidden = true; }, 2400);
}

/* ---------- Engine ---------- */

let EngineClass = null;

async function getEngine() {
  if (!state.engine) {
    if (!EngineClass) {
      const mod = FAKE
        ? await import("./dev/fake-lipread.js")
        : await import("./core/lipread.js");
      EngineClass = mod.LipRead;
    }
    state.engine = new EngineClass({ onProgress: (p) => state.onProgress?.(p) });
  }
  return state.engine;
}

async function refreshModelStatus() {
  try {
    const engine = await getEngine();
    const status = await engine.modelStatus();
    state.modelCached = !!status.cached;
  } catch (err) {
    console.warn("LipRead engine did not load", err);
    state.modelCached = false;
    throw err;
  }
  return state.modelCached;
}

/* Map engine errors to the copy the user sees. The core should set err.code. */
function classify(err) {
  const code = err?.code || "";
  if (code === "aborted" || err?.name === "AbortError") return "aborted";
  if (code === "decode" || err?.name === "NotSupportedError") return "decode";
  if (code === "nofaces") return "nofaces";
  if (code === "storage" || err?.name === "QuotaExceededError") return "storage";
  if (code === "download" || err?.name === "NetworkError") return "download";
  if (code === "unsupported") return "unsupported";
  if (code === "model") return "model";
  if (code === "memory") return "memory";
  if (code === "short") return "short";
  if (code === "autoplay") return "autoplay";
  if (code === "stalled") return "stalled";
  return "generic";
}

/* ---------- Screens and navigation ---------- */

const SCREENS = {
  home: $("#screen-home"),
  processing: $("#screen-processing"),
  results: $("#screen-results"),
};

const HEADINGS = { home: "#largeTitle", processing: "#procTitle", results: "#resTitle" };

function showScreen(name, { animate = true, focus = true } = {}) {
  for (const [key, node] of Object.entries(SCREENS)) {
    const on = key === name;
    node.hidden = !on;
    if (on && animate && !REDUCE_MOTION) {
      node.classList.remove("enter");
      void node.offsetWidth; // restart the entrance animation
      node.classList.add("enter");
    }
  }
  state.screen = name;

  const nav = $("#navbar");
  // Back is offered on results, and on processing only once a run has ended (error state).
  $("#backBtn").hidden = !(name === "results" || (name === "processing" && !state.run));
  $("#shareBtn").hidden = name !== "results";
  $("#settingsBtn").hidden = name !== "home";
  $("#backLabel").textContent = "Home";

  if (name === "home") {
    $("#navTitle").textContent = "LipRead";
    updateNavTitle();
  } else {
    nav.classList.add("show-title");
    nav.classList.toggle("is-scrolled", window.scrollY > 4);
  }

  // Move focus to the new screen's heading so screen readers announce it.
  if (focus) $(HEADINGS[name])?.focus({ preventScroll: true });
}

function updateNavTitle() {
  const nav = $("#navbar");
  if (state.screen !== "home") { nav.classList.add("show-title"); return; }
  const title = $("#largeTitle");
  const passed = title.getBoundingClientRect().bottom < 44 + 4;
  nav.classList.toggle("show-title", passed);
  nav.classList.toggle("is-scrolled", window.scrollY > 4);
}

/* Back to Home. `back` removes the Results or processing history entry (the in-app Back button does that);
   `keep` leaves it for the caller, which replaces it (see openLayer). */
function goHome({ back = true, keep = false } = {}) {
  if (back && !keep) {
    const lr = history.state?.lr;
    if (lr === "results" || lr === "processing") dropLayer(lr);
  }
  pauseVideo();
  state.current = null;
  showScreen("home");
  renderRecent();
  window.scrollTo(0, 0);
  maybeShowHint();
}

window.addEventListener("scroll", updateNavTitle, { passive: true });
window.addEventListener("resize", () => { updateNavTitle(); renderOverlay(); }, { passive: true });

/* ---------- Home ---------- */

function renderRecent() {
  const list = $("#recentList");
  list.replaceChildren();
  $("#recentSection").hidden = state.recent.length === 0;
  for (const item of state.recent) {
    const sub = [
      item.faceCount === 0 ? "No faces" : `${item.faceCount} ${item.faceCount === 1 ? "face" : "faces"}`,
      nowLabel(item.date),
    ].join(" · ");
    const thumb = item.poster
      ? el("img", { class: "thumb", src: item.poster, alt: "" })
      : el("span", { class: "thumb", "aria-hidden": "true" });
    const row = el("button", { type: "button", class: "row row--recent row--pressable", onclick: () => openResults(item) },
      thumb,
      el("span", { class: "row__main" },
        el("span", { class: "row__name", text: item.name }),
        el("span", { class: "row__sub", text: sub }),
      ),
      icon("chev-right", "chev"),
    );
    list.append(el("li", {}, row));
  }
}

function renderHomeSupport() {
  const supported = hasWasmSimd();
  $("#unsupported").hidden = supported;
  $("#heroCard").hidden = !supported;
  $("#homeFooter").textContent = IS_IOS
    ? "Runs on your iPhone. Videos never leave your device."
    : "Runs on your device. Videos never leave your device.";
  return supported;
}

function requestPick(kind) {
  if (!state.modelCached) {
    state.pendingPick = kind;
    openFirstRun();
    return;
  }
  openPicker(kind);
}

function inputFor(kind) {
  return kind === "record" ? $("#recordInput") : $("#fileInput");
}

/* The file picker. Called synchronously inside a tap, so iOS allows it to open. */
let picker = null; // { input, key } while the system picker may be open

function openPicker(kind) {
  endPicker();
  const input = inputFor(kind);
  picker = { input, key: `picker:${kind}` };
  hold(picker.key);
  input.click();
}

function endPicker() {
  if (!picker) return;
  const { key } = picker;
  picker = null;
  release(key);
}

$("#chooseBtn").addEventListener("click", () => requestPick("file"));
$("#recordBtn").addEventListener("click", () => requestPick("record"));

for (const input of [$("#fileInput"), $("#recordInput")]) {
  input.addEventListener("change", () => {
    const file = input.files?.[0];
    input.value = "";
    endPicker();
    if (file) startRun(file, { faces: store.get(KEYS.readAll) === "1" ? "all" : "readable" });
  });
  // Supported on recent iOS and Chrome: the picker was dismissed without a file.
  input.addEventListener("cancel", endPicker);
}

// Fallback for browsers without the 'cancel' event: when focus returns and no file arrived, the picker is closed.
window.addEventListener("focus", () => {
  if (!picker) return;
  setTimeout(() => {
    if (picker && !picker.input.files?.length) endPicker();
  }, 600);
});

$("#backBtn").addEventListener("click", goHome);
$("#settingsBtn").addEventListener("click", () => openSettings($("#settingsBtn")));

/* ---------- Sheets and alerts ---------- */

const SHEETS = {
  firstRun: $("#firstRunSheet"),
  settings: $("#settingsSheet"),
};

/* While a sheet or the alert is up, the page behind it is inert: no focus, no taps, hidden from screen readers. */
function syncInert() {
  const alertOpen = !$("#alert").hidden;
  const modal = !!state.openSheet || alertOpen;
  $("#navbar").inert = modal;
  $("#main").inert = modal;
  for (const [name, sheet] of Object.entries(SHEETS)) sheet.inert = alertOpen && state.openSheet === name;
}

function openSheet(name, opener = null) {
  openLayer("sheet");
  const sheet = SHEETS[name];
  sheet.classList.remove("is-closing");
  sheet.hidden = false;
  $("#backdrop").hidden = false;
  state.openSheet = name;
  state.sheetOpener = opener;
  hold(`sheet:${name}`);
  $("#iosHint").hidden = true;
  syncInert();
  const focusTarget = sheet.querySelector("[data-autofocus]") || sheet.querySelector(".sheet__title, .sheet__nav-title") || sheet;
  focusTarget.focus({ preventScroll: true });
}

function closeSheet(name, { instant = false } = {}) {
  const sheet = SHEETS[name];
  if (state.openSheet !== name) return;
  state.openSheet = null;
  release(`sheet:${name}`);
  syncInert();
  const finish = () => {
    if (state.openSheet) return; // another sheet opened meanwhile
    sheet.hidden = true;
    sheet.classList.remove("is-closing", "is-dragging");
    sheet.style.transform = "";
    sheet.style.transition = "";
    $("#backdrop").hidden = true;
    maybeShowHint();
  };
  if (REDUCE_MOTION || instant) {
    finish();
  } else {
    sheet.classList.add("is-closing");
    setTimeout(() => { if (sheet.classList.contains("is-closing")) { sheet.hidden = true; sheet.classList.remove("is-closing"); finish(); } }, 280);
  }
  const opener = state.sheetOpener;
  state.sheetOpener = null;
  if (opener && opener.isConnected && state.screen === "home") opener.focus({ preventScroll: true });
}

document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  if (!$("#alert").hidden) { closeAlert(false); return; }
  if (state.openSheet === "firstRun") closeFirstRun();
  else if (state.openSheet === "settings") closeSettings();
});

/* Sheets can be dragged down to dismiss, as on iOS. The drag starts on a handle (the grabber, the header, or the
   top of the first-run sheet). A sheet closes when it is dragged past a third of its height or flicked down. */
function bindSheetDrag(name) {
  const sheet = SHEETS[name];
  for (const handle of sheet.querySelectorAll("[data-drag-handle]")) {
    let pid = null, startY = 0, lastY = 0, lastT = 0, speed = 0, dragging = false;

    // Moves and the release are tracked on the window: the finger crosses other handles on its way down.
    const onMove = (e) => {
      if (e.pointerId !== pid) return;
      const dy = e.clientY - startY;
      if (!dragging) {
        if (dy < 8) return; // a small wobble is not a drag
        dragging = true;
        try { handle.setPointerCapture(pid); } catch { /* ignore */ }
        sheet.classList.add("is-dragging");
      }
      const dt = e.timeStamp - lastT;
      if (dt > 0) speed = (e.clientY - lastY) / dt;
      lastY = e.clientY;
      lastT = e.timeStamp;
      sheet.style.transform = `translateY(${Math.max(0, dy)}px)`;
    };

    const onEnd = (e) => {
      if (e.pointerId !== pid) return;
      pid = null;
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onEnd);
      window.removeEventListener("pointercancel", onEnd);
      if (!dragging) return;
      dragging = false;
      sheet.classList.remove("is-dragging");
      const dy = Math.max(0, e.clientY - startY);
      if (dy > Math.min(160, sheet.offsetHeight / 3) || speed > 0.7) {
        dismissDragged(name);
      } else {
        sheet.style.transition = "transform 0.3s var(--ease-sheet)";
        sheet.style.transform = "";
        setTimeout(() => { sheet.style.transition = ""; }, 320);
      }
    };

    handle.addEventListener("pointerdown", (e) => {
      if (!e.isPrimary || state.openSheet !== name) return;
      if (name === "firstRun" && dl.phase === "downloading") return; // closing would stop the download
      if (e.target.closest("button, a, input, textarea, select, label")) return;
      pid = e.pointerId; startY = lastY = e.clientY; lastT = e.timeStamp; speed = 0; dragging = false;
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onEnd);
      window.addEventListener("pointercancel", onEnd);
    });
  }
}

function dismissDragged(name) {
  const sheet = SHEETS[name];
  sheet.style.transition = REDUCE_MOTION ? "none" : "transform 0.25s ease-in";
  sheet.style.transform = "translateY(100%)";
  setTimeout(() => {
    if (name === "firstRun") closeFirstRun({ instant: true });
    else closeSettings({ instant: true });
  }, REDUCE_MOTION ? 0 : 250);
}

$("#backdrop").addEventListener("click", () => {
  if (state.openSheet === "firstRun") closeFirstRun();
  else if (state.openSheet === "settings") closeSettings();
});

/* First-run sheet */

// Phases: idle | downloading | paused | failed | ready.
// `interrupted` is set once an attempt has stopped part way in this page session. The engine picks up where it
// stopped when downloadModel is called again in the same page session, so the button then reads "Resume download".
const dl = {
  phase: "idle",
  interrupted: false,
  abort: null,     // AbortController for the attempt in progress
  loaded: 0,       // bytes so far, from the last progress callback
  total: 0,
};

function openFirstRun() {
  renderFirstRun();
  openSheet("firstRun", document.activeElement);
}

function closeFirstRun({ instant = false, history: useHistory = true } = {}) {
  // Closing the sheet during a download is the same as Not Now: the download stops.
  if (dl.phase === "downloading") dl.abort?.abort();
  store.set(KEYS.firstRunDismissed, "1");
  closeSheet("firstRun", { instant });
  if (useHistory) dropLayer("sheet");
}

// "Not Now" becomes "Cancel" while a download runs, and stops it without closing the sheet.
function onNotNow() {
  if (dl.phase === "downloading") dl.abort?.abort();
  else closeFirstRun();
}

function renderFirstRun(message = "") {
  const ready = dl.phase === "ready";
  const downloading = dl.phase === "downloading";
  const showProgress = downloading || dl.phase === "paused" || dl.phase === "failed";
  $("#readyRow").hidden = !ready;
  $("#dlProgressRow").hidden = !showProgress;
  $("#dlBtn").hidden = ready;
  $("#frDone").hidden = !ready;
  $("#notNowBtn").hidden = ready;
  $("#notNowBtn").textContent = downloading ? "Cancel" : "Not Now";
  $("#dlGroup").querySelector(".row").hidden = ready; // "One-time download 203 MB" row
  const btn = $("#dlBtn");
  btn.disabled = downloading;
  btn.textContent = downloading ? "Downloading…"
    : dl.interrupted ? "Resume Download"
    : dl.phase === "failed" ? "Try Again"
    : "Download";
  const status = $("#dlStatus");
  status.classList.toggle("dl-error", dl.phase === "failed");
  if (message) status.textContent = message;
}

async function downloadModel() {
  if (dl.phase === "downloading") return;
  const controller = new AbortController();
  dl.abort = controller;
  dl.phase = "downloading";
  progressTextAt = 0;
  progressTextPct = -100;
  hold("download");
  renderFirstRun();
  setDownloadProgress(dl.loaded, dl.total); // a resumed download starts where it stopped
  try {
    const engine = await getEngine();
    await engine.downloadModel((loaded, total) => {
      dl.loaded = loaded;
      dl.total = total;
      setDownloadProgress(loaded, total);
    }, controller.signal);
    state.modelCached = true;
    dl.phase = "ready";
    dl.interrupted = false;
    renderFirstRun();
    $("#frDone").focus({ preventScroll: true });
  } catch (err) {
    const kind = classify(err);
    if (controller.signal.aborted || kind === "aborted") {
      dl.phase = "paused";
      dl.interrupted = true;
      renderFirstRun("Download paused. Tap Resume Download to pick up where it stopped.");
    } else if (kind === "storage") {
      dl.phase = "failed";
      dl.interrupted = false;
      renderFirstRun("Not enough space. The model needs about 210 MB free on this device.");
    } else {
      dl.phase = "failed";
      dl.interrupted = true;
      renderFirstRun("Download paused. Check your connection, then resume. Keep this page open to pick up where it stopped.");
    }
  } finally {
    dl.abort = null;
    release("download");
  }
}

// The bar moves on every chunk. The text (and the aria-live status) updates at most once a second, and only
// after a 5-point change, plus once at 100%.
let progressTextAt = 0;
let progressTextPct = -100;

function setDownloadProgress(loaded, total) {
  if (dl.phase !== "downloading") return;
  const frac = total ? clamp(loaded / total, 0, 1) : 0;
  const pct = Math.round(frac * 100);
  $("#dlFill").style.width = `${pct}%`;
  $("#dlBar").setAttribute("aria-valuenow", String(pct));
  const now = Date.now();
  const due = now - progressTextAt >= 1000 && Math.abs(pct - progressTextPct) >= 5;
  const finishing = pct === 100 && progressTextPct !== 100;
  if (!due && !finishing) return;
  progressTextAt = now;
  progressTextPct = pct;
  $("#dlStatus").textContent = `Downloading ${fmtMB(loaded)} of ${fmtMB(total || MODEL_MB * 1e6)}`;
}

$("#dlBtn").addEventListener("click", downloadModel);
$("#notNowBtn").addEventListener("click", onNotNow);
$("#frDone").addEventListener("click", () => {
  const pick = state.pendingPick;
  state.pendingPick = null;
  closeFirstRun();
  // Called synchronously inside the click, so iOS allows the file picker to open.
  if (pick) openPicker(pick);
});

/* Settings sheet */

function openSettings(opener) {
  $("#names").value = store.get(KEYS.names, "");
  renderSettings();
  openSheet("settings", opener);
}

function closeSettings({ instant = false, history: useHistory = true } = {}) {
  closeSheet("settings", { instant });
  if (useHistory) dropLayer("sheet");
}

function renderSettings() {
  const on = store.get(KEYS.readAll) === "1";
  const sw = $("#faceSwitch");
  sw.setAttribute("aria-checked", String(on));
  const del = $("#deleteModel");
  del.disabled = !state.modelCached;
  del.textContent = state.modelCached ? `Delete Downloaded Model (${MODEL_MB} MB)` : "No Model Downloaded";
}

function parseNames() {
  return store.get(KEYS.names, "").split(/\r?\n/).map((s) => s.trim()).filter(Boolean);
}

$("#setDone").addEventListener("click", () => closeSettings());
$("#names").addEventListener("input", (e) => store.set(KEYS.names, e.target.value));
$("#faceSwitch").addEventListener("click", () => {
  const next = store.get(KEYS.readAll) !== "1";
  store.set(KEYS.readAll, next ? "1" : null);
  renderSettings();
});
$("#repoLink").href = REPO_URL;
// The whole row toggles the switch, as on iOS, not only the small switch.
$("#faceSwitch").closest(".row").addEventListener("click", (e) => {
  if (!e.target.closest("#faceSwitch")) $("#faceSwitch").click();
});

/* Confirm delete alert */

function openAlert() {
  openLayer("alert");
  $("#alertBackdrop").hidden = false;
  $("#alert").hidden = false;
  syncInert();
  $("#alertCancel").focus({ preventScroll: true });
}

function closeAlert(confirmed, { history: useHistory = true } = {}) {
  $("#alert").hidden = true;
  $("#alertBackdrop").hidden = true;
  if (useHistory) dropLayer("alert");
  syncInert();
  if (confirmed) deleteModel();
  else $("#deleteModel").focus({ preventScroll: true });
}

async function deleteModel() {
  try {
    const engine = await getEngine();
    await engine.deleteModel();
    state.modelCached = false;
    dl.phase = "idle";
    dl.interrupted = false;
    dl.loaded = 0;
    dl.total = 0;
    toast("Model deleted");
  } catch {
    toast("Couldn’t delete the model. Try again.");
  }
  renderSettings();
}

$("#deleteModel").addEventListener("click", openAlert);
$("#alertCancel").addEventListener("click", () => closeAlert(false));
$("#alertOk").addEventListener("click", () => closeAlert(true));
$("#alertBackdrop").addEventListener("click", () => closeAlert(false));

/* Add to Home Screen hint, shown once on iOS Safari. */

function maybeShowHint() {
  if (!IS_IOS_SAFARI || IS_STANDALONE) return;
  if (store.get(KEYS.hintDismissed) === "1") return;
  if (state.openSheet || state.screen !== "home") return;
  if (!state.recent.length) return; // only after the user has got a result
  $("#iosHint").hidden = false;
}

$("#hintClose").addEventListener("click", () => {
  store.set(KEYS.hintDismissed, "1");
  $("#iosHint").hidden = true;
});

/* ---------- Processing ---------- */

const STAGE_COPY = {
  0: { name: "Preparing model", short: "Preparing model" },
  1: { name: "Finding faces", short: "Finding faces" },
  2: { name: "Reading lips", short: "Reading lips" },
};

const ERRORS = {
  nofaces: {
    tone: "orange",
    title: "No faces found",
    text: "We couldn’t find a face in this clip. Try a clip where people face the camera and are reasonably close.",
    primary: "Choose Another Video", action: "pick",
  },
  small: {
    tone: "gray",
    title: "Faces are too far away",
    text: "Every face here is too small or turned away to read lips reliably. Try a closer clip, or read them anyway and expect more mistakes.",
    primary: "Read Anyway", action: "readAll",
    secondary: "Choose Another Video", secondaryAction: "pick",
  },
  decode: {
    tone: "orange",
    title: "Can’t open this video",
    text: "This format isn’t supported here. Some Android phones record HEVC, which browsers often can’t read. Share or export the clip in a more compatible format (H.264), then try again. On iPhone, Settings > Camera > Formats > Most Compatible makes future recordings compatible.",
    primary: "Choose Another Video", action: "pick",
  },
  download: {
    tone: "orange",
    title: "Download paused",
    text: "The model didn’t finish downloading. Check your connection (Wi‑Fi is best), then try again.",
    primary: "Try Again", action: "redownload",
  },
  storage: {
    tone: "red",
    title: "Not enough storage",
    text: "The model needs about 210 MB free on this device. Free up some space, then try again.",
    primary: "Try Again", action: "redownload",
  },
  unsupported: {
    tone: "gray",
    title: "This browser can’t run LipRead",
    text: "LipRead needs WebAssembly with SIMD. Update to a recent version of iOS or Chrome, then try again.",
    primary: "Back to Home", action: "home",
  },
  memory: {
    tone: "orange",
    title: "Your phone ran out of memory",
    text: "This clip was too much to read in one go. Close other tabs and apps, or trim the clip to a shorter moment, then try again.",
    primary: "Choose Another Video", action: "pick",
  },
  short: {
    tone: "gray",
    title: "Clip is too short",
    text: "LipRead needs at least half a second of video. Choose a longer clip.",
    primary: "Choose Another Video", action: "pick",
  },
  model: {
    tone: "orange",
    title: "Model not downloaded",
    text: "LipRead needs its one-time model download before it can read lips.",
    primary: "Download", action: "redownload",
  },
  autoplay: {
    tone: "orange",
    title: "Tap to start",
    text: "iPhone blocked automatic video playback. This can happen in Low Power Mode. Tap Start to read the clip.",
    primary: "Start", action: "retry",
  },
  stalled: {
    tone: "orange",
    title: "Video stopped loading",
    text: "This clip stopped decoding partway through. Try again, or re-save the clip and choose it once more.",
    primary: "Try Again", action: "retry",
    secondary: "Choose Another Video", secondaryAction: "pick",
  },
  generic: {
    tone: "orange",
    title: "Something went wrong",
    text: "LipRead couldn’t finish reading this video. Try again, or choose a different clip.",
    primary: "Try Again", action: "retry",
  },
};

function resetProcessingView(name) {
  $("#procEta").textContent = "Step 1 of 3 · Estimating time left…";
  $("#procTitle").textContent = `Analyzing ${name}`;
  $("#posterName").textContent = name;
  $("#posterImg").hidden = true;
  $("#posterImg").removeAttribute("src");
  $("#procProgress").hidden = false;
  $("#stageList").hidden = false;
  $("#procError").hidden = true;
  $("#procFootnote").hidden = false;
  $("#procActions").hidden = false;
  $("#procLive").textContent = "";
  for (const row of $$(".stage-row")) {
    row.dataset.state = "pending";
    row.querySelector(".stage-sub").textContent = "";
  }
  setOverall(0, STAGE_COPY[0].short);
}

function setOverall(frac, label) {
  const pct = Math.round(clamp(frac, 0, 1) * 100);
  $("#procFill").style.width = `${pct}%`;
  $("#procBar").setAttribute("aria-valuenow", String(pct));
  $("#procPct").textContent = `${pct}%`;
  if (label) $("#procStatus").textContent = label;
}

function stageIndex(stage) {
  if (stage === "model") return 0;
  if (stage === "frames" || stage === "faces") return 1;
  if (stage === "lips") return 2;
  if (stage === "done") return 3;
  return 0;
}

let lastLiveStage = -1;

function onEngineProgress(p, run) {
  if (state.run !== run) return;
  const idx = stageIndex(p?.stage);
  const frac = clamp(Number(p?.fraction) || 0, 0, 1);
  const overall = idx >= 3 ? 1 : (idx + frac) / 3;
  run.progress = Math.max(run.progress || 0, overall);

  // One progress measure only: the bar and its percentage. The rows show which step is done, active or waiting.
  const rows = $$(".stage-row");
  rows.forEach((row, i) => {
    const next = i < idx ? "done" : i === idx ? "active" : "pending";
    if (row.dataset.state !== next) row.dataset.state = next;
  });

  const active = STAGE_COPY[Math.min(idx, 2)];
  setOverall(run.progress, idx >= 3 ? "Finishing" : active.short);
  run.idx = idx;
  updateEta(run, idx);
  if (idx !== lastLiveStage && idx < 3) {
    lastLiveStage = idx;
    $("#procLive").textContent = active.name;
  }
}

/* Time left, from the recent rate of progress (the last 15 s), so a stall makes the estimate grow instead of
   freezing. Smoothed, and shown in steps of 5 s so it does not jitter. Called on every progress event and once a second. */
function updateEta(run, idx) {
  const now = performance.now();
  const p = run.progress;
  const samples = run.samples;
  if (!samples.length || now - samples[samples.length - 1].t >= 500) samples.push({ t: now, p });
  while (samples.length > 2 && now - samples[0].t > 15000) samples.shift();
  if (p > run.lastP + 0.002) { run.lastP = p; run.lastMove = now; }
  const first = samples[0];
  const span = (now - first.t) / 1000;
  const stalled = now - run.lastMove > 4000;
  if (stalled) {
    run.eta = null; // no progress for a few seconds: say so rather than keep an old figure
  } else if (span >= 3 && p > first.p + 0.01) {
    const rate = (p - first.p) / span;
    const remaining = (1 - p) / rate;
    run.eta = run.eta == null ? remaining : run.eta * 0.6 + remaining * 0.4;
  }
  let left;
  if (idx >= 3 || (run.eta != null && run.eta < 8)) left = "Almost done";
  else if (run.eta == null) left = "Estimating time left…";
  else if (run.eta < 90) left = `About ${Math.round(run.eta / 5) * 5} s left`;
  else left = `About ${Math.round(run.eta / 60)} min left`;
  $("#procEta").textContent = `Step ${Math.min(idx + 1, 3)} of 3 · ${left}`;
}

let wakeWanted = false;   // a run is using the screen-wake lock
let wakePending = false;  // a request is in flight

async function acquireWakeLock() {
  wakeWanted = true;
  if (state.wakeLock || wakePending) return;
  if (!("wakeLock" in navigator && navigator.wakeLock?.request)) return;
  wakePending = true;
  try {
    const sentinel = await navigator.wakeLock.request("screen");
    // Released while the request was pending: let go of the sentinel as soon as it arrives.
    if (!wakeWanted) {
      try { sentinel.release()?.catch?.(() => {}); } catch { /* ignore */ }
      return;
    }
    state.wakeLock = sentinel;
    sentinel.addEventListener?.("release", () => { if (state.wakeLock === sentinel) state.wakeLock = null; });
  } catch { /* not allowed or not supported */ }
  finally { wakePending = false; }
}

function releaseWakeLock() {
  wakeWanted = false;
  try { state.wakeLock?.release?.()?.catch?.(() => {}); } catch { /* ignore */ }
  state.wakeLock = null;
}

document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible" && state.run && !state.wakeLock) acquireWakeLock();
});

/* Draw one frame of a video to a small JPEG data URL, or null if the browser can't. */
function makeFrame(url, at = 0.5, maxWidth = 640) {
  return new Promise((resolve) => {
    const v = document.createElement("video");
    v.muted = true;
    v.playsInline = true;
    v.preload = "auto";
    let settled = false;
    const finish = (value) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      v.removeAttribute("src");
      v.load();
      resolve(value);
    };
    const timer = setTimeout(() => finish(null), 5000);
    const draw = () => {
      try {
        const scale = Math.min(1, maxWidth / v.videoWidth);
        const c = document.createElement("canvas");
        c.width = Math.max(1, Math.round(v.videoWidth * scale));
        c.height = Math.max(1, Math.round(v.videoHeight * scale));
        c.getContext("2d").drawImage(v, 0, 0, c.width, c.height);
        finish(c.toDataURL("image/jpeg", 0.85));
      } catch {
        finish(null);
      }
    };
    v.onloadeddata = () => {
      const t = Math.min(at, Math.max(0, (v.duration || 1) - 0.1));
      if (t > 0.01) v.currentTime = t; else draw();
    };
    v.onseeked = draw;
    v.onerror = () => finish(null);
    v.src = url;
  });
}

let runSeq = 0;

/* Frees a run's object URL for a run that produced no result. The poster frame may still be loading from the URL,
   so the revoke waits until that has settled (makeFrame always settles, within its 5 s timeout). */
function revokeRunUrl(run) {
  const url = run?.url;
  if (!url) return;
  run.url = null;
  Promise.resolve(run.framing).then(() => {
    try { URL.revokeObjectURL(url); } catch { /* ignore */ }
  });
}

async function startRun(file, { faces = "readable" } = {}) {
  if (state.run) return;
  state.lastRun = { file, faces };
  const name = file.name || "Video";
  const url = URL.createObjectURL(file);
  const abort = new AbortController();
  const run = { id: ++runSeq, file, url, name, faces, poster: null, progress: 0, abort, eta: null, samples: [], idx: 0, timer: 0, lastP: 0, lastMove: performance.now() };
  state.run = run;
  run.timer = setInterval(() => { if (state.run === run) updateEta(run, run.idx); }, 1000);
  hold(`run:${run.id}`);
  state.onProgress = (p) => onEngineProgress(p, run);
  lastLiveStage = -1;

  openLayer("processing");
  resetProcessingView(baseName(name));
  $("#navTitle").textContent = "Analyzing";
  showScreen("processing");
  window.scrollTo(0, 0);
  acquireWakeLock();

  run.framing = makeFrame(url, 0.5).then((poster) => {
    if (state.run !== run || !poster) return;
    run.poster = poster;
    const img = $("#posterImg");
    img.src = poster;
    img.hidden = false;
  });

  try {
    const engine = await getEngine();
    const result = await engine.process(file, { names: parseNames(), faces, signal: abort.signal });
    // Canceled or replaced while it ran: its result is dropped and the UI is left alone.
    if (state.run !== run) { revokeRunUrl(run); return; }
    releaseWakeLock();
    finishRun(run, result);
  } catch (err) {
    // A late rejection from a canceled run must not touch the UI, so this check comes first.
    if (state.run !== run) { revokeRunUrl(run); return; }
    releaseWakeLock();
    const kind = classify(err);
    state.run = null;
    revokeRunUrl(run);
    if (kind === "aborted" || abort.signal.aborted) {
      goHome();
      return;
    }
    if (kind === "model") {
      goHome({ keep: true });
      state.pendingPick = "file";
      openFirstRun();
      return;
    }
    showProcessingError(kind);
  } finally {
    clearInterval(run.timer);
    release(`run:${run.id}`);
  }
}

function finishRun(run, result) {
  state.run = null;
  const tracks = Array.isArray(result?.tracks) ? result.tracks : [];
  const item = {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    name: run.name,
    date: new Date(),
    file: run.file,
    url: run.url,
    poster: run.poster,
    faces: run.faces,
    result,
    faceCount: tracks.length,
  };
  state.recent.unshift(item);
  if (state.recent.length > 8) {
    const old = state.recent.pop();
    try { URL.revokeObjectURL(old.url); } catch { /* ignore */ }
  }

  const allUnreadable = tracks.length > 0 && tracks.every((t) => t.tier === "unreadable" || t.skipped);
  if (tracks.length === 0) {
    run.failed = true;
    showProcessingError("nofaces", item);
    return;
  }
  if (allUnreadable && run.faces !== "all") {
    run.failed = true;
    showProcessingError("small", item);
    return;
  }
  openResults(item);
}

function showProcessingError(kind, item = null) {
  const copy = ERRORS[kind] || ERRORS.generic;
  $("#procProgress").hidden = true;
  $("#stageList").hidden = true;
  $("#procFootnote").hidden = true;
  $("#procActions").hidden = true;
  if (item) $("#posterName").textContent = item.name;
  const card = $("#procError");
  const iconBox = $("#errIcon");
  iconBox.dataset.tone = copy.tone;
  iconBox.replaceChildren(icon(kind === "nofaces" ? "faces" : kind === "small" ? "distance" : "warning"));
  $("#errTitle").textContent = copy.title;
  $("#errText").textContent = copy.text;

  const primary = $("#errPrimary");
  primary.textContent = copy.primary;
  primary.onclick = () => runErrorAction(copy.action, kind, item);
  const secondary = $("#errSecondary");
  if (copy.secondary) {
    secondary.hidden = false;
    secondary.textContent = copy.secondary;
    secondary.onclick = () => runErrorAction(copy.secondaryAction, kind, item);
  } else {
    secondary.hidden = true;
  }
  card.hidden = false;
  $("#procTitle").textContent = copy.title;
  $("#procLive").textContent = copy.title;
  $("#navTitle").textContent = "Analyzing";
  showScreen("processing", { animate: false });
}

function runErrorAction(action, kind, item) {
  if (action === "pick") {
    // Still inside the tap, so the picker can open.
    goHome();
    openPicker("file");
  } else if (action === "readAll") {
    const file = item?.file || state.lastRun?.file;
    if (file) startRun(file, { faces: "all" });
  } else if (action === "redownload") {
    goHome({ keep: true });
    state.pendingPick = null;
    dl.phase = "failed";
    openFirstRun();
    renderFirstRun("Download didn’t finish. Check your connection, then try again.");
  } else if (action === "home") {
    goHome();
  } else if (action === "retry") {
    const last = state.lastRun;
    if (last) startRun(last.file, { faces: last.faces });
    else goHome();
  }
}

$("#cancelBtn").addEventListener("click", cancelRun);

function cancelRun() {
  const run = state.run;
  if (run) {
    state.run = null;          // from here on, this run's late rejection is ignored
    clearInterval(run.timer);
    run.abort.abort();
    releaseWakeLock();
    revokeRunUrl(run);
    release(`run:${run.id}`);
  }
  goHome();
}

/* ---------- Results ---------- */

const video = $("#video");
const stage = $("#stage");
const overlay = $("#overlay");

function openResults(item) {
  openLayer("results");
  state.current = {
    item,
    result: item.result,
    mode: "likely",
    views: [],
    chips: [],
    expanded: new Set(),
  };

  const result = item.result;
  stage.style.aspectRatio = `${result.width || 16} / ${result.height || 9}`;
  if (video.dataset.src !== item.url) {
    video.dataset.src = item.url;
    video.src = item.url;
    video.load();
  }
  $("#navTitle").textContent = "Results";
  $("#resTitle").textContent = `Results for ${item.name}`;
  $("#scrub").value = "0";
  $("#scrub").style.setProperty("--p", "0%");
  renderTimeLabel();

  renderFaces();
  setMode("likely");
  const tm = result.timings || {};
  const totalS = tm.total_ms ? tm.total_ms / 1000 : Object.values(tm).reduce((a, v) => a + (typeof v === "number" ? v : 0), 0);
  const secs = totalS > 0 ? Math.max(1, Math.round(totalS)) : null;
  const partial = result.end != null && result.duration != null && result.end < result.duration - 0.5;
  $("#resultFooter").textContent = (secs
    ? `Analyzed on this ${DEVICE} in ${secs} s. Results are estimates; check them against the video.`
    : `Analyzed on this ${DEVICE}. Results are estimates; check them against the video.`)
    + (partial ? ` Only the first ${Math.round(result.end - (result.start || 0))} seconds were read — trim the clip to read a different moment.` : "");

  showScreen("results");
  window.scrollTo(0, 0);
  requestAnimationFrame(renderOverlay);
}

function tierLabel(tier) {
  return tier === "readable" ? "Readable" : tier === "marginal" ? "Marginal" : "Unreadable";
}

function wordTier(w) {
  return w.tier || (w.confidence >= 0.9 ? "high" : w.confidence >= 0.7 ? "medium" : "low");
}

function renderFaces() {
  const cur = state.current;
  const list = $("#facesList");
  list.replaceChildren();
  overlay.replaceChildren();
  cur.views = [];
  cur.chips = [];

  cur.result.tracks.forEach((track, i) => {
    const label = `Face ${i + 1}`;
    const decoded = !track.skipped && Array.isArray(track.utterances) && track.utterances.length > 0;
    const reasons = (track.reasons || []).join(" · ");
    const bodyId = `face-body-${i}`;

    const thumb = track.thumb
      ? el("img", { class: "face__thumb", src: track.thumb, alt: "" })
      : el("span", { class: "face__thumb", "aria-hidden": "true" });

    const subtitle = track.skipped
      ? "Not read. Enable it in Settings."
      : reasons || "Lip movements tracked";

    const main = el("span", { class: "face__main" },
      el("span", { class: "face__name", text: label }),
      el("span", { class: "face__reasons", text: subtitle }),
    );
    const pill = el("span", { class: "tier", "data-tier": track.tier, text: tierLabel(track.tier) });

    let row;
    let body = null;
    if (decoded) {
      row = el("button", {
        type: "button",
        class: "face__row",
        "aria-expanded": "false",
        "aria-controls": bodyId,
        onclick: () => toggleFace(i),
      }, thumb, main, pill, icon("chev-right", "chev face__chev"));
      body = el("div", { class: "face__body", id: bodyId, hidden: true });
    } else {
      row = el("div", { class: "face__row", "aria-disabled": "true", disabled: true }, thumb, main, pill, icon("chev-right", "chev face__chev"));
    }

    list.append(el("li", { class: "face", "data-index": i }, row, body));

    // Overlay pieces for this face (only decoded faces get a box and captions).
    if (decoded) {
      const box = el("div", { class: "face-box", hidden: true });
      const pillEl = el("div", { class: "caption", hidden: true });
      overlay.append(box, pillEl);
      cur.views.push({ index: i, track, box, pill: pillEl, key: "" });
    }
  });

  if (cur.result.tracks.length === 0) {
    list.append(el("li", { class: "face" }, el("p", { class: "face__hint", text: "No faces were found in this clip." })));
  }
}

function toggleFace(i) {
  const cur = state.current;
  const li = $(`#facesList li[data-index="${i}"]`);
  if (!li) return;
  const row = li.querySelector(".face__row");
  const body = li.querySelector(".face__body");
  const open = !cur.expanded.has(i);
  if (open) {
    cur.expanded.add(i);
    if (!body.childElementCount) buildTranscript(i, body);
  } else {
    cur.expanded.delete(i);
  }
  body.hidden = !open;
  row.setAttribute("aria-expanded", String(open));
}

function buildTranscript(i, body) {
  const cur = state.current;
  const track = cur.result.tracks[i];
  if (!track.utterances.length) {
    body.append(el("p", { class: "face__hint", text: "No words were read for this face." }));
    return;
  }
  for (const u of track.utterances) {
    const words = el("div", { class: "words" });
    for (const w of u.words) {
      const tier = wordTier(w);
      const textNode = document.createTextNode(displayWord(w));
      const classes = ["word"];
      if (tier === "low") classes.push("is-low");
      const chipChildren = [el("span", { class: "w-text" }, textNode)];
      if (w.boosted) chipChildren.push(el("span", { class: "tag", text: "name match" }));
      const label = tier === "low" ? `${w.text}, uncertain` : w.boosted ? `${w.text}, name match` : w.text;
      const chip = el("button", {
        type: "button",
        class: classes.join(" "),
        "aria-label": label,
        onclick: () => seekTo(w.start),
      }, ...chipChildren);
      chip._word = w;
      chip._text = textNode;
      words.append(chip);
      cur.chips.push(chip);
    }
    body.append(el("div", { class: "utt" },
      el("span", { class: "utt__time", text: fmtTime(u.start) }),
      words,
    ));
  }
}

function displayWord(w) {
  return state.current?.mode === "likely" && wordTier(w) === "low" ? "…" : w.text;
}

function setMode(mode) {
  const cur = state.current;
  if (!cur) return;
  cur.mode = mode;
  for (const btn of $$(".segmented button")) {
    btn.setAttribute("aria-pressed", String(btn.dataset.mode === mode));
  }
  $("#modeNote").textContent = mode === "likely"
    ? "Uncertain words show as “…”. Choose All to see every guess."
    : "Every word the model read, including its guesses.";
  for (const chip of cur.chips) {
    chip._text.textContent = displayWord(chip._word);
  }
  cur.views.forEach((v) => { v.key = ""; });
  renderOverlay();
}

$$(".segmented button").forEach((btn) => {
  btn.addEventListener("click", () => setMode(btn.dataset.mode));
});

/* Player */

function seekTo(t) {
  if (!state.current) return;
  video.currentTime = Math.max(0, t - 0.1);
  renderOverlay();
  const rect = stage.getBoundingClientRect();
  if (rect.bottom < 80) {
    stage.scrollIntoView({ behavior: REDUCE_MOTION ? "auto" : "smooth", block: "start" });
  }
}

function pauseVideo() {
  try { video.pause(); } catch { /* ignore */ }
}

function updatePlayButton() {
  const playing = !video.paused && !video.ended;
  const btn = $("#playBtn");
  btn.replaceChildren(icon(playing ? "pause" : "play", "icon--solid"));
  btn.setAttribute("aria-label", playing ? "Pause" : "Play");
}

function renderTimeLabel() {
  const dur = isFinite(video.duration) ? video.duration : 0;
  $("#timeLabel").textContent = `${fmtTime(video.currentTime)} / ${fmtTime(dur)}`;
  const scrub = $("#scrub");
  const frac = dur ? video.currentTime / dur : 0;
  scrub.value = String(Math.round(frac * 1000));
  scrub.style.setProperty("--p", `${(frac * 100).toFixed(2)}%`);
  scrub.setAttribute("aria-valuetext", fmtTime(video.currentTime));
}

let rafId = 0;
function frameLoop() {
  renderOverlay();
  renderTimeLabel();
  highlightNow();
  rafId = (!video.paused && !video.ended) ? requestAnimationFrame(frameLoop) : 0;
}

video.addEventListener("play", () => {
  updatePlayButton();
  if (!rafId) rafId = requestAnimationFrame(frameLoop);
});
video.addEventListener("pause", () => { updatePlayButton(); renderOverlay(); renderTimeLabel(); highlightNow(); });
video.addEventListener("ended", () => { updatePlayButton(); renderOverlay(); });
video.addEventListener("seeked", () => { renderOverlay(); renderTimeLabel(); highlightNow(); });
video.addEventListener("timeupdate", () => { renderTimeLabel(); if (!rafId) { renderOverlay(); highlightNow(); } });
video.addEventListener("loadedmetadata", () => { renderTimeLabel(); renderOverlay(); });
video.addEventListener("error", () => {
  const cur = state.current;
  if (!cur) return;
  toast("This clip can’t play in this browser. Results are still shown below.");
});

// Tapping the clip or the play button toggles playback, as in Photos and the system video player.
function togglePlayback() {
  if (video.paused || video.ended) {
    const p = video.play();
    if (p && p.catch) p.catch(() => {});
  } else {
    video.pause();
  }
}
$("#playBtn").addEventListener("click", togglePlayback);
video.addEventListener("click", togglePlayback);

$("#scrub").addEventListener("input", (e) => {
  const dur = isFinite(video.duration) ? video.duration : 0;
  video.currentTime = (Number(e.target.value) / 1000) * dur;
  renderTimeLabel();
  renderOverlay();
  highlightNow();
});

/* Captions and boxes, positioned over the displayed video rect (letterboxing included). */

function interpolateBox(boxes, t) {
  if (!boxes || !boxes.length) return null;
  const first = boxes[0];
  const last = boxes[boxes.length - 1];
  if (t < first.t - 0.25 || t > last.t + 0.25) return null;
  if (t <= first.t) return first;
  if (t >= last.t) return last;
  let lo = 0;
  let hi = boxes.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (boxes[mid].t <= t) lo = mid; else hi = mid;
  }
  const a = boxes[lo];
  const b = boxes[hi];
  const f = (t - a.t) / ((b.t - a.t) || 1);
  return {
    x: a.x + (b.x - a.x) * f,
    y: a.y + (b.y - a.y) * f,
    w: a.w + (b.w - a.w) * f,
    h: a.h + (b.h - a.h) * f,
  };
}

function recentWords(track, t, mode) {
  const words = [];
  for (const u of track.utterances) {
    if (u.end < t - 4 || u.start > t + 0.05) continue;
    for (const w of u.words) {
      if (w.start <= t && w.start >= t - 4) words.push(w);
    }
  }
  const tail = words.slice(-6);
  // Keep the caption short enough to read in one line.
  let total = 0;
  const kept = [];
  for (let i = tail.length - 1; i >= 0; i--) {
    const text = mode === "likely" && wordTier(tail[i]) === "low" ? "…" : tail[i].text;
    total += text.length + 1;
    if (total > 34 && kept.length) break;
    kept.unshift(tail[i]);
  }
  return kept;
}

function renderOverlay() {
  const cur = state.current;
  if (!cur || state.screen !== "results" || !cur.views.length) return;
  const t = video.currentTime || 0;
  const cw = stage.clientWidth;
  const ch = stage.clientHeight;
  const vw = cur.result.width || video.videoWidth || 16;
  const vh = cur.result.height || video.videoHeight || 9;
  if (!cw || !ch) return;
  const s = Math.min(cw / vw, ch / vh);
  const ox = (cw - vw * s) / 2;
  const oy = (ch - vh * s) / 2;

  for (const view of cur.views) {
    const box = interpolateBox(view.track.boxes, t);
    if (!box) {
      view.box.hidden = true;
      view.pill.hidden = true;
      continue;
    }
    const left = ox + box.x * s;
    const top = oy + box.y * s;
    const w = box.w * s;
    const h = box.h * s;
    view.box.hidden = false;
    view.box.style.transform = `translate(${left.toFixed(1)}px, ${top.toFixed(1)}px)`;
    view.box.style.width = `${w.toFixed(1)}px`;
    view.box.style.height = `${h.toFixed(1)}px`;

    const words = recentWords(view.track, t, cur.mode);
    if (!words.length) {
      view.pill.hidden = true;
      view.key = "";
      continue;
    }
    const currentIdx = words.length - 1;
    const isCurrent = t <= words[currentIdx].end + 0.25;
    const key = words.map((w) => `${w.start}`).join(",") + `|${cur.mode}|${isCurrent ? currentIdx : -1}`;
    if (view.key !== key) {
      view.key = key;
      view.pill.replaceChildren();
      words.forEach((w, idx) => {
        const hidden = cur.mode === "likely" && wordTier(w) === "low";
        // a run of hidden low-confidence words collapses into a single "…"
        if (hidden && idx > 0 && cur.mode === "likely" && wordTier(words[idx - 1]) === "low") return;
        const span = el("span", {
          class: `w${wordTier(w) === "low" ? " is-low" : ""}${isCurrent && idx === currentIdx ? " is-current" : ""}`,
          text: hidden ? "…" : w.text,
        });
        if (view.pill.childNodes.length) view.pill.append(document.createTextNode(" "));
        view.pill.append(span);
      });
    }
    view.pill.hidden = false;
    // Place above the box, or below it when there is no room at the top.
    const pw = view.pill.offsetWidth;
    const ph = view.pill.offsetHeight;
    const cx = clamp(left + w / 2, pw / 2 + 6, cw - pw / 2 - 6);
    const above = top - ph - 8 >= 0;
    const y = above ? top - 8 : top + h + 8;
    view.pill.style.transform = `translate(${(cx - pw / 2).toFixed(1)}px, ${(above ? y - ph : y).toFixed(1)}px)`;
  }
}

function highlightNow() {
  const cur = state.current;
  if (!cur) return;
  const t = video.currentTime || 0;
  for (const chip of cur.chips) {
    const w = chip._word;
    const now = t >= w.start && t < w.end + 0.05;
    if (chip.classList.contains("is-now") !== now) chip.classList.toggle("is-now", now);
  }
}

/* Share */

async function shareCurrent() {
  // Keep this synchronous until navigator.share is called, so the user gesture is still valid.
  const cur = state.current;
  const engine = state.engine;
  if (!cur || !engine) return;
  const body = engine.toText(cur.result).trim();
  if (!body) {
    toast("Nothing to share yet");
    return;
  }
  const text = `AI estimate — don’t treat as a quote.\n\n${body}`;
  const data = { title: `LipRead: ${cur.item.name}`, text };
  try {
    const srt = engine.toSRT(cur.result);
    const file = new File([srt], `${baseName(cur.item.name)}-captions.srt`, { type: "application/x-subrip" });
    if (navigator.canShare?.({ files: [file] })) data.files = [file];
  } catch { /* the SRT file is optional */ }
  if (navigator.share) {
    try {
      await navigator.share(data);
      return;
    } catch (err) {
      if (err?.name === "AbortError") return;
    }
  }
  try {
    await navigator.clipboard.writeText(text);
    toast("Transcript copied");
  } catch {
    toast("Sharing isn’t available here");
  }
}

$("#shareBtn").addEventListener("click", shareCurrent);

/* ---------- Back, as on the system ---------- */

/* Android Back, the iOS edge swipe and the browser's back button close the top layer (a sheet, the alert,
   Results or a run that is being read) instead of leaving the page. Each layer has a history entry. Closing a layer
   from the UI removes its entry; histSkip counts the popstate events our own history.back() calls cause, so they
   are not handled twice. */
let histSkip = 0;

function openLayer(lr) {
  const cur = history.state?.lr || "home";
  if (cur === lr) return;
  try {
    // The processing entry becomes whatever the run leads to: Results, or a sheet after an error.
    if (cur === "processing") history.replaceState({ lr }, "");
    else history.pushState({ lr }, "");
  } catch { /* no history API */ }
}

function dropLayer(lr) {
  if ((history.state?.lr || "home") !== lr) return;
  histSkip++;
  try { history.back(); } catch { histSkip--; }
}

window.addEventListener("popstate", (e) => {
  if (histSkip > 0) { histSkip--; return; }
  const lr = e.state?.lr || "home";
  // A run or a model download cannot be left with Back. The screen stays and says how to stop it.
  const busy = state.run ? { lr: "processing", msg: "Tap Cancel to stop reading." }
    : (state.openSheet === "firstRun" && dl.phase === "downloading") ? { lr: "sheet", msg: "Tap Cancel to stop the download." }
    : null;
  if (busy && lr !== busy.lr) {
    history.pushState({ lr: busy.lr }, "");
    toast(busy.msg);
    return;
  }
  if (!$("#alert").hidden && lr !== "alert") closeAlert(false, { history: false });
  if (state.openSheet && lr !== "sheet" && lr !== "alert") {
    if (state.openSheet === "firstRun") closeFirstRun({ history: false });
    else closeSettings({ history: false });
  }
  if (lr === "home" && state.screen !== "home") goHome({ back: false });
});

/* ---------- Boot ---------- */

async function init() {
  const supported = renderHomeSupport();
  renderRecent();
  window.scrollTo(0, 0);
  showScreen("home", { animate: false, focus: false });
  updateNavTitle();
  if (!supported) return;

  try {
    await refreshModelStatus();
  } catch {
    toast("LipRead couldn’t load. Check your connection and reload.");
    return;
  }

  if (!state.modelCached && store.get(KEYS.firstRunDismissed) !== "1") {
    setTimeout(() => {
      if (state.screen === "home" && !state.openSheet) {
        dl.phase = "idle";
        openFirstRun();
      }
    }, 450);
  } else {
    maybeShowHint();
  }
}

bindSheetDrag("firstRun");
bindSheetDrag("settings");
try { history.replaceState({ lr: "home" }, ""); } catch { /* no history API */ }
init();
