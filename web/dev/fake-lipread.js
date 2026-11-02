// DEV ONLY. Fake of web/core/lipread.js with the same API, for ?fake=1.
// Simulates a model download and processing. Honours ?fail=download|storage|decode|autoplay|stalled|nofaces|small
// (read from location.search) so the UI error states can be exercised.
// ?slow=1 paces the download and processing about 3x slower, and makes a cancelled process() reject late
// (about 900 ms after the abort), like a real engine that is still winding down.

const MODEL_BYTES = 203 * 1000 * 1000;
const MODEL = { name: "LipRead phone model (fake)", license: "Non-commercial research licence (LRS3 terms)" };
const LS_CACHED = "fake.cached";
const LS_FAILED_ONCE = "fake.downloadFailedOnce";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const QS = (() => {
  try { return new URLSearchParams(location.search); } catch { return new URLSearchParams(); }
})();
const FAIL = QS.get("fail") || "";
const SLOW = QS.get("slow") === "1";
const pace = (ms) => (SLOW ? ms * 3 : ms);
// Bytes downloaded in this page session. Kept across calls, so a second downloadModel resumes from here.
let dlLoaded = 0;

function coded(code, message) {
  const e = new Error(message);
  e.code = code;
  return e;
}

function abortError() {
  const e = new Error("Aborted");
  e.name = "AbortError";
  e.code = "aborted";
  return e;
}

function lsGet(k) { try { return localStorage.getItem(k); } catch { return null; } }
function lsSet(k, v) { try { v == null ? localStorage.removeItem(k) : localStorage.setItem(k, v); } catch { /* ignore */ } }

function fmt(t) {
  const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = Math.floor(t % 60), ms = Math.round((t % 1) * 1000);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")},${String(ms).padStart(3, "0")}`;
}

function tierOf(conf) {
  return conf >= 0.9 ? "high" : conf >= 0.7 ? "medium" : "low";
}

// Read duration and size of a video file. Rejects with code "decode" if the browser cannot open it.
function readMeta(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const v = document.createElement("video");
    v.muted = true;
    v.preload = "metadata";
    const done = (fn, val) => { v.removeAttribute("src"); v.load(); URL.revokeObjectURL(url); fn(val); };
    v.onloadedmetadata = () => {
      if (!v.videoWidth || !v.duration || !isFinite(v.duration)) return done(reject, coded("decode", "No video track"));
      done(resolve, { duration: v.duration, width: v.videoWidth, height: v.videoHeight });
    };
    v.onerror = () => done(reject, coded("decode", "Cannot decode video"));
    v.src = url;
  });
}

// Crop a square thumbnail from a frame at time t. Returns a data URL or a placeholder.
async function thumbAt(file, t, box, meta) {
  const placeholder = () => {
    const c = document.createElement("canvas");
    c.width = c.height = 144;
    const g = c.getContext("2d");
    g.fillStyle = "#8e8e93"; g.fillRect(0, 0, 144, 144);
    return c.toDataURL("image/jpeg", 0.8);
  };
  try {
    const url = URL.createObjectURL(file);
    const v = document.createElement("video");
    v.muted = true; v.preload = "auto"; v.src = url;
    await new Promise((res, rej) => { v.onloadeddata = res; v.onerror = rej; });
    await new Promise((res) => { v.onseeked = res; v.currentTime = Math.min(Math.max(t, 0), meta.duration - 0.05); });
    const c = document.createElement("canvas");
    c.width = c.height = 144;
    const g = c.getContext("2d");
    const side = Math.min(box.w, box.h) * 1.6;
    const sx = Math.max(0, Math.min(meta.width - side, box.x + box.w / 2 - side / 2));
    const sy = Math.max(0, Math.min(meta.height - side, box.y + box.h / 2 - side / 2));
    g.drawImage(v, sx, sy, side, side, 0, 0, 144, 144);
    v.removeAttribute("src"); v.load();
    URL.revokeObjectURL(url);
    return c.toDataURL("image/jpeg", 0.85);
  } catch {
    return placeholder();
  }
}

// Scripted speech. Each sentence is a list of words; confidence is assigned per word.
const SCRIPT = [
  "Come on we need this one now",
  "Keep it tight keep it tight",
  "Rodriguez you have got space",
  "Switch it switch it",
  "Good work that was big",
  "Hold the line hold the line",
  "Watch the left side now",
  "Yes yes yes that is the one",
];
const CONF_PATTERN = [0.96, 0.93, 0.84, 0.97, 0.66, 0.91, 0.78, 0.94, 0.59, 0.95, 0.88, 0.72, 0.97, 0.52, 0.9, 0.81];

function buildTrack({ id, tier, cx, cy, size, a, b, meta, names, sentenceOffset, confScale, reasons, facePx, yaw }) {
  const W = meta.width, H = meta.height, D = meta.duration;
  const start = a * D, end = b * D;
  // Boxes sampled every 0.4 s, drifting slightly.
  const boxes = [];
  const phase = id * 1.7;
  for (let t = start; t <= end + 1e-6; t += 0.4) {
    const x = (cx + 0.025 * Math.sin(t * 0.8 + phase)) * W;
    const y = (cy + 0.02 * Math.cos(t * 0.6 + phase)) * H;
    const w = size * W;
    const h = w * 1.25;
    boxes.push({ t: +t.toFixed(2), x: x - w / 2, y: y - h / 2, w, h });
  }
  // Words.
  const utterances = [];
  let t = start + 0.6;
  let wi = sentenceOffset;
  let k = 0;
  let boostedDone = false;
  while (t < end - 0.5 && utterances.length < 40) {
    const sentence = SCRIPT[(sentenceOffset + utterances.length) % SCRIPT.length].split(" ");
    const uStart = t;
    const words = [];
    for (const raw of sentence) {
      let conf = CONF_PATTERN[(wi + k) % CONF_PATTERN.length] * confScale;
      conf = Math.min(0.99, Math.max(0.05, conf));
      let text = raw;
      let original;
      let boosted = false;
      if (!boostedDone && raw === "Rodriguez" && id === 1) {
        text = names[0] || "Rodriguez";
        original = "Rodrigues";
        boosted = true;
        conf = 0.8;
        boostedDone = true;
      }
      const dur = 0.22 + 0.06 * (raw.length / 6);
      words.push({ text, start: +t.toFixed(2), end: +(t + dur).toFixed(2), confidence: +conf.toFixed(2), tier: tierOf(conf), boosted, ...(boosted ? { original } : {}) });
      t += dur + 0.1;
      k++;
      if (t > end - 0.4) break;
    }
    if (words.length === 0) break;
    utterances.push({
      start: +uStart.toFixed(2),
      end: +words[words.length - 1].end.toFixed(2),
      text: words.map((w) => w.text).join(" ") + ".",
      confidence: +(words.reduce((s, w) => s + w.confidence, 0) / words.length).toFixed(2),
      words,
    });
    t += 0.9;
    wi++;
  }
  return {
    id,
    tier,
    reasons,
    facePx,
    yaw,
    start: +start.toFixed(2),
    end: +end.toFixed(2),
    boxes,
    thumb: null,
    skipped: false,
    utterances,
  };
}

export class LipRead {
  constructor({ modelUrl, onProgress } = {}) {
    this.modelUrl = modelUrl;
    this.onProgress = onProgress || null;
  }

  async modelStatus() {
    const cached = lsGet(LS_CACHED) === "1";
    return { cached, bytes: cached ? MODEL_BYTES : 0 };
  }

  async downloadModel(onBytes, signal) {
    if (lsGet(LS_CACHED) === "1") { onBytes?.(MODEL_BYTES, MODEL_BYTES); return; }
    if (FAIL === "storage") { await sleep(400); throw coded("storage", "Not enough storage"); }
    const steps = 50;
    const failAt = FAIL === "download" && lsGet(LS_FAILED_ONCE) !== "1" ? MODEL_BYTES * 0.4 : Infinity;
    while (dlLoaded < MODEL_BYTES) {
      if (signal?.aborted) throw abortError();
      await sleep(pace(55));
      if (signal?.aborted) throw abortError();
      dlLoaded = Math.min(MODEL_BYTES, dlLoaded + MODEL_BYTES / steps);
      onBytes?.(dlLoaded, MODEL_BYTES);
      if (dlLoaded >= failAt) {
        lsSet(LS_FAILED_ONCE, "1");
        throw coded("download", "Network connection lost");
      }
    }
    lsSet(LS_CACHED, "1");
  }

  async process(file, opts = {}) {
    try {
      return await this._run(file, opts);
    } catch (err) {
      if (SLOW && err?.name === "AbortError") { await sleep(900); console.info("fake: late AbortError after cancel"); } // late rejection after cancel
      throw err;
    }
  }

  async _run(file, opts = {}) {
    const t0 = performance.now();
    const names = (opts.names || []).map((s) => s.trim()).filter(Boolean);
    const faces = opts.faces || "readable";
    const signal = opts.signal;
    const emit = (stage, fraction, message) => this.onProgress?.({ stage, fraction, message });
    const check = () => { if (signal?.aborted) throw abortError(); };

    if (lsGet(LS_CACHED) !== "1") throw coded("model", "Model not downloaded");
    check();
    emit("frames", 0, "Opening video");
    const meta = await readMeta(file);
    if (FAIL === "decode") throw coded("decode", "Cannot decode video");
    if (FAIL === "autoplay") throw coded("autoplay", "VIDEO_AUTOPLAY_BLOCKED");
    if (FAIL === "stalled") throw coded("stalled", "Video stalled");
    check();

    for (let i = 1; i <= 16; i++) {
      check();
      emit("frames", i / 16, "Reading frames");
      await sleep(pace(70));
    }
    for (let i = 1; i <= 20; i++) {
      check();
      emit("faces", i / 20, "Finding faces");
      await sleep(pace(80));
    }
    for (let i = 1; i <= 30; i++) {
      check();
      emit("lips", i / 30, "Reading lips");
      await sleep(pace(70));
    }
    emit("done", 1, "Done");

    let tracks;
    if (FAIL === "nofaces") {
      tracks = [];
    } else if (FAIL === "small") {
      tracks = [
        buildTrack({ id: 1, tier: "unreadable", cx: 0.3, cy: 0.35, size: 0.04, a: 0.02, b: 0.9, meta, names, sentenceOffset: 2, confScale: 0.5, reasons: ["too small (18 px)", "far from camera"], facePx: 18, yaw: 4 }),
        buildTrack({ id: 2, tier: "unreadable", cx: 0.7, cy: 0.3, size: 0.035, a: 0.1, b: 0.8, meta, names, sentenceOffset: 5, confScale: 0.5, reasons: ["too small (16 px)", "turned away"], facePx: 16, yaw: -52 }),
      ].map((tr) => ({ ...tr, skipped: faces !== "all" }));
      if (faces !== "all") tracks.forEach((tr) => { tr.utterances = []; });
    } else {
      const t1 = buildTrack({ id: 1, tier: "readable", cx: 0.24, cy: 0.32, size: 0.17, a: 0.03, b: 0.95, meta, names, sentenceOffset: 0, confScale: 1.0, reasons: ["mouth clearly visible", "facing camera"], facePx: 96, yaw: 3 });
      const t2 = buildTrack({ id: 2, tier: "marginal", cx: 0.68, cy: 0.3, size: 0.09, a: 0.2, b: 0.8, meta, names, sentenceOffset: 3, confScale: 0.85, reasons: ["far from camera", "turned 25 degrees"], facePx: 41, yaw: -24 });
      const t3 = buildTrack({ id: 3, tier: "unreadable", cx: 0.46, cy: 0.62, size: 0.05, a: 0.1, b: 0.6, meta, names, sentenceOffset: 6, confScale: 0.6, reasons: ["too small (22 px)", "turned away"], facePx: 22, yaw: -58 });
      tracks = [t1, t2, t3];
      if (faces !== "all") {
        t3.skipped = true;
        t3.utterances = [];
      }
    }

    // Thumbnails from real frames (cropped around the first box).
    for (const tr of tracks) {
      const b0 = tr.boxes[0] || { x: 0, y: 0, w: meta.width / 4, h: meta.height / 4 };
      tr.thumb = await thumbAt(file, tr.start + 0.3, b0, meta);
    }

    return {
      duration: meta.duration,
      width: meta.width,
      height: meta.height,
      tracks,
      timings: { total_ms: Math.round(performance.now() - t0), decode_ms: 120, detect_ms: 1400, lips_ms: 2100 },
      model: MODEL,
    };
  }

  toText(result) {
    const lines = [];
    result.tracks.forEach((tr, i) => {
      if (tr.skipped || !tr.utterances.length) return;
      lines.push(`Face ${i + 1}`);
      for (const u of tr.utterances) lines.push(u.text);
      lines.push("");
    });
    return lines.join("\n").trim();
  }

  toSRT(result) {
    const items = [];
    result.tracks.forEach((tr, i) => {
      if (tr.skipped) return;
      for (const u of tr.utterances) items.push({ start: u.start, end: u.end, text: `Face ${i + 1}: ${u.text}` });
    });
    items.sort((a, b) => a.start - b.start);
    return items.map((it, n) => `${n + 1}\n${fmt(it.start)} --> ${fmt(it.end)}\n${it.text}\n`).join("\n");
  }

  async deleteModel() {
    lsSet(LS_CACHED, null);
    lsSet(LS_FAILED_ONCE, null);
    dlLoaded = 0;
  }
}
