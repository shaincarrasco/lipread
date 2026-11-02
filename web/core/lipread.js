// LipRead on-device engine: the only module the UI talks to.
//   const lr = new LipRead({ onProgress });
//   await lr.modelStatus(); await lr.downloadModel((loaded, total) => ...);
//   const result = await lr.process(file, { names, faces: "readable" | "all", start, end, signal });
import { CONFIG, assetUrl } from "./config.js";
import { openVideo, stepFrames } from "./video.js";
import { planPasses, toSource, mergePasses, YUNET_SIZE } from "./yunet.js";
import { Tracker, CutDetector } from "./tracker.js";
import { setGeometry, correctedKeypoints, interEye, interpolate, mouthCrops } from "./align.js";
import { assess } from "./quality.js";
import { greedyWords, segment } from "./decode.js";
import * as store from "./model-store.js";
import { NgramLM } from "./lm.js";
import { BeamDecoder } from "./beam.js";

const json = async (name) => (await fetch(assetUrl(name))).json();

export class LipRead {
  constructor({ onProgress, config } = {}) {
    this.onProgress = onProgress || (() => {});
    this.overrides = { ...(self.LIPREAD_CONFIG || {}), ...(config || {}) };
    Object.assign(CONFIG, this.overrides);
    this._worker = null; this._seq = 0; this._pending = new Map();
    this._assets = null;
  }

  // ---------------------------------------------------------------- model
  modelStatus() { return store.modelStatus(); }
  async downloadModel(onBytes, signal) {
    busy(+1);
    try { return await store.downloadModel(onBytes, signal); } catch (err) { throw coded(err); } finally { busy(-1); }
  }
  async deleteModel() { await this._call("dispose").catch(() => {}); return store.deleteModel(); }
  get model() { return { name: CONFIG.modelName, license: CONFIG.modelLicense, bytes: CONFIG.modelBytes }; }

  /** Feature check for the UI's "device not supported" screen. */
  static support() {
    const wasm = typeof WebAssembly === "object";
    let simd = false;
    try {   // minimal SIMD module (v128.const) validation
      simd = WebAssembly.validate(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 22, 1, 20, 0, 253, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11]));
    } catch { simd = false; }
    return { ok: wasm && "caches" in self && "Worker" in self && !!self.crypto?.subtle, wasm, simd,
             caches: "caches" in self, worker: "Worker" in self, secure: !!self.crypto?.subtle };
  }

  // ---------------------------------------------------------------- worker plumbing
  _ensureWorker() {
    if (this._worker) return;
    this._worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    this._worker.onmessage = (e) => {
      const { id, ok, result, error } = e.data;
      const p = this._pending.get(id);
      if (!p) return;
      this._pending.delete(id);
      ok ? p.resolve(result) : p.reject(new Error(error));
    };
    this._worker.onerror = (e) => this._killWorker(new Error(`WORKER_CRASHED: ${(e && e.message) || "out of memory?"}`));
    this._worker.onmessageerror = () => this._killWorker(new Error("WORKER_CRASHED: bad message"));
    this._call("config", { overrides: JSON.parse(JSON.stringify(this.overrides)) });
  }

  /** Stop the worker immediately (cancel or crash): frees the model's memory, rejects pending calls. */
  _killWorker(err) {
    if (this._worker) { try { this._worker.terminate(); } catch { /* ignore */ } }
    this._worker = null;
    for (const p of this._pending.values()) p.reject(err);
    this._pending.clear();
  }

  _call(type, args = {}, transfer = []) {
    this._ensureWorker();
    const id = ++this._seq;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this._worker.postMessage({ id, type, ...args }, transfer);
    });
  }

  async _loadAssets(withPieces) {
    if (!this._assets) {
      const [tokens, geometry] = await Promise.all([json("tokens.json"), json("geometry.json")]);
      this._assets = { tokens, geometry, pieces: null };
      setGeometry(geometry);
    }
    if (withPieces && !this._assets.pieces) this._assets.pieces = await json("spm_pieces.json");
    if (CONFIG.context && this._assets.decoder === undefined) {
      this._assets.decoder = null;
      let buf = null;
      try { buf = await store.ensureLm(); } catch (err) {
        console.warn("LipRead: language model unavailable, using greedy decoding", err);
      }
      if (buf) {
        // Rust (WebAssembly) decoder first; the identical JS decoder if it can't load.
        if (CONFIG.rustDecoder) {
          try {
            const { createRustDecoder } = await import("./rust-decoder.js");
            this._assets.decoder = await createRustDecoder(buf, this._assets.tokens);
          } catch (err) {
            console.warn("LipRead: Rust decoder unavailable, using the JS decoder", err);
          }
        }
        if (!this._assets.decoder) {
          try {
            const lm = new NgramLM(buf), tokens = this._assets.tokens;
            this._assets.decoder = { kind: "js", decode: (lp, T, V, t0, opts) => new BeamDecoder(tokens, lm, opts).decode(lp, T, V, t0) };
          } catch (err) {
            console.warn("LipRead: language model unreadable, using greedy decoding", err);
          }
        }
      }
      console.info(`LipRead decoder: ${this._assets.decoder?.kind || "greedy"}`);
    }
    return this._assets;
  }

  _progress(stage, fraction, message = "") { this.onProgress({ stage, fraction: Math.max(0, Math.min(1, fraction)), message }); }

  // ---------------------------------------------------------------- main entry
  async process(file, opts = {}) {
    busy(+1);
    // Cancelling really stops the work: the worker (and its 200+ MB model session) is terminated.
    const onAbort = () => this._killWorker(new DOMException("Cancelled", "AbortError"));
    opts.signal?.addEventListener("abort", onAbort, { once: true });
    try { return await this._process(file, opts); } catch (err) { throw coded(err); }
    finally { busy(-1); opts.signal?.removeEventListener("abort", onAbort); }
  }

  async _process(file, { names = [], faces = "readable", start = 0, end = null, signal } = {}) {
    const token = (this._runToken = Symbol("run"));
    const report = (stage, fraction, message) => {
      if (token === this._runToken && !signal?.aborted) this._progress(stage, fraction, message);
    };
    const timings = {}; let tick = performance.now();
    const lap = (k) => { const n = performance.now(); timings[k] = (timings[k] || 0) + (n - tick) / 1000; tick = n; };
    const check = () => { if (signal && signal.aborted) throw new DOMException("Cancelled", "AbortError"); };
    names = names.map((n) => n.trim()).filter(Boolean);

    report("model", 0, "Preparing");
    // Open the video first, as close to the user's tap as possible (iOS playback rules).
    const vid = await openVideo(file);
    let result;
    try {
      const status = await store.modelStatus();
      if (!status.cached) throw new Error("MODEL_MISSING");
      const assets = await this._loadAssets(names.length > 0);
      await this._call("initYunet");
      check();
      lap("setup");
      result = await this._run(vid, assets, { names, faces, start, end, check, lap, signal, report });
    } finally {
      vid.close();
    }
    result.timings = timings;
    result.profile = Object.fromEntries(Object.entries(this.prof || {}).map(([k, v]) => [k, Math.round(v)]));
    report("done", 1, "Done");
    return result;
  }

  async _run(vid, assets, { names, faces, start, end, check, lap, signal, report }) {
    const fps = CONFIG.fps, stride = CONFIG.detectStride;
    start = Math.max(0, start || 0);
    const stop = Math.min(vid.duration, end ?? vid.duration, start + CONFIG.maxSeconds);
    const N = Math.max(0, Math.floor((stop - start) * fps));
    if (N < CONFIG.minTrackFrames) throw new Error("CLIP_TOO_SHORT");
    const s = Math.min(1, CONFIG.maxSide / Math.max(vid.width, vid.height));
    const W = Math.round(vid.width * s), H = Math.round(vid.height * s);

    const frame = new OffscreenOrCanvas(W, H), fctx = frame.getContext("2d", { willReadFrequently: false });
    const det = new OffscreenOrCanvas(YUNET_SIZE, YUNET_SIZE), dctx = det.getContext("2d", { willReadFrequently: true });
    const tiny = new OffscreenOrCanvas(64, 36), tctx = tiny.getContext("2d", { willReadFrequently: true });
    const S = CONFIG.cropSize;
    const crop = new OffscreenOrCanvas(S, S), cctx = crop.getContext("2d", { willReadFrequently: true });
    const thumbC = new OffscreenOrCanvas(112, 112), thctx = thumbC.getContext("2d");
    const passes = planPasses(W, H);

    const tracker = new Tracker({ maxFaces: CONFIG.maxFaces });
    const cuts = new CutDetector();
    const prof = this.prof = { seek: 0, draw: 0, prep: 0, detect: 0, crop: 0, passes: 0 };
    const data = new Map();   // track id -> {kps: Map, crops: Map, dets: [], boxes: [], lastKp, thumb}

    let qWait = performance.now();
    const onFrame = async (f) => {
      prof.seek += performance.now() - qWait;     // time spent waiting for the decoder
      const t = start + f / fps;
      let q0 = performance.now();
      fctx.drawImage(vid.video, 0, 0, W, H);
      prof.draw += performance.now() - q0;

      tctx.drawImage(frame, 0, 0, 64, 36);
      if (cuts.update(tctx.getImageData(0, 0, 64, 36).data) && f > 0) tracker.cut();

      let found = null;
      if (f % stride === 0) {
        found = [];
        // zoom tiles (small far-away faces) on every 3rd detector run; full frame every run
        const runPasses = (f / stride) % 3 === 0 ? passes : passes.slice(0, 1);
        for (const p of runPasses) {
          dctx.fillStyle = "#000"; dctx.fillRect(0, 0, YUNET_SIZE, YUNET_SIZE);
          dctx.drawImage(frame, p.sx, p.sy, p.sw, p.sh, 0, 0, p.sw * p.k, p.sh * p.k);
          let q1 = performance.now();
          const rgba = dctx.getImageData(0, 0, YUNET_SIZE, YUNET_SIZE).data;
          prof.prep += performance.now() - q1; q1 = performance.now();
          const faces640 = await this._call("detect", { rgba, scoreTh: CONFIG.detectScore, nmsTh: CONFIG.nmsIou }, [rgba.buffer]);
          prof.detect += performance.now() - q1; prof.passes++;
          for (const face of faces640) found.push(toSource(face, p));
        }
        found = mergePasses(found).filter((d) => d.h >= 8);
      }
      const active = tracker.update(f, found);

      for (const tr of active) {
        let td = data.get(tr.id);
        if (!td) { td = { kps: new Map(), crops: new Map(), dets: [], boxes: [], lastKp: null, thumb: null, thumbArea: 0 }; data.set(tr.id, td); }
        const d = tr.lastFrame === f ? tr.last : null;
        if (d) {
          const kp = correctedKeypoints(d.pts);
          td.kps.set(f, kp); td.lastKp = kp;
          td.dets.push({ f, h: d.h / s, pts: d.pts.map(([x, y]) => [x / s, y / s]) });
          td.boxes.push({ t, x: d.x / s, y: d.y / s, w: d.w / s, h: d.h / s });
          if (d.w * d.h > td.thumbArea) {
            td.thumbArea = d.w * d.h;
            const side = Math.max(d.w, d.h) * 1.4, cx = d.x + d.w / 2, cy = d.y + d.h / 2;
            thctx.drawImage(frame, cx - side / 2, cy - side / 2, side, side, 0, 0, 112, 112);
            td.thumb = thumbC.toDataURL ? thumbC.toDataURL("image/jpeg", 0.8) : null;
          }
        }
        if (!td.lastKp) continue;
        // store a small grey mouth-region crop for this frame (never the full frame)
        const kp = td.lastKp, iod = Math.max(interEye(kp), 2), k = S / (CONFIG.cropSpan * iod);
        const qc = performance.now();
        const [mx, my] = kp[3], half = S / 2 / k;
        cctx.fillStyle = "#000"; cctx.fillRect(0, 0, S, S);
        cctx.drawImage(frame, mx - half, my - half, 2 * half, 2 * half, 0, 0, S, S);
        const px = cctx.getImageData(0, 0, S, S).data, g = new Uint8Array(S * S);
        for (let i = 0, p = 0; i < g.length; i++, p += 4) g[i] = (0.299 * px[p] + 0.587 * px[p + 1] + 0.114 * px[p + 2] + 0.5) | 0;
        td.crops.set(f, { data: g, cx: mx, cy: my, k, size: S });
        prof.crop += performance.now() - qc;
      }
      if (f % 5 === 0) report("faces", f / N, `Frame ${f + 1} of ${N}`);
      qWait = performance.now();
    };
    await stepFrames(vid, start, N, fps, onFrame, check, signal);
    lap("faces");

    // ---- per track: alignment, readability, selection
    const tracks = [];
    for (const tr of tracker.all) {
      const td = data.get(tr.id);
      if (!td || tr.lastFrame - tr.firstFrame + 1 < CONFIG.minTrackFrames) continue;
      const f0 = tr.firstFrame, f1 = tr.lastFrame, T = f1 - f0 + 1;
      const kpList = [], cropList = [];
      for (let f = f0; f <= f1; f++) { kpList.push(td.kps.get(f) || null); cropList.push(td.crops.get(f)); }
      if (cropList.some((c) => !c)) continue;
      const kps = interpolate(kpList);
      const crops = mouthCrops(kps.map((k) => k.map((p) => [p[0], p[1]])), cropList);
      const q = assess({ dets: td.dets, nFrames: T }, crops, T, stride);
      tracks.push({ id: tr.id, f0, T, crops, q, td });
    }
    data.clear();
    lap("align");

    const wanted = tracks.filter((t) => faces === "all" || t.q.tier !== "unreadable");
    if (wanted.length) {
      report("lips", 0, "Loading lip-reading model");
      await this._call("initLips");
      lap("modelLoad");
    }
    let Booster = null;
    if (names.length) Booster = (await import("./booster.js")).Booster;

    const out = [];
    let done = 0;
    for (const t of tracks) {
      check();
      const t0 = start + t.f0 / fps, tEnd = t0 + t.T / fps;
      const item = { id: t.id, tier: t.q.tier, reasons: t.q.reasons, facePx: Math.round(t.q.facePx), yaw: Math.round(t.q.yaw),
                     coverage: t.q.coverage, start: t0, end: tEnd, boxes: t.td.boxes, thumb: t.td.thumb,
                     skipped: !wanted.includes(t), utterances: [] };
      if (!item.skipped) {
        report("lips", done / wanted.length, `Reading face ${done + 1} of ${wanted.length}`);
        const crops = t.crops;
        const { logprobs, T, V } = await this._call("lips", { crops, T: t.T }, [crops.buffer]);
        // Context-aware decoding: keep several candidate transcripts and let the word language model
        // pick the most plausible one; every word carries alternatives. Greedy if the LM is missing.
        let words = assets.decoder
          ? assets.decoder.decode(logprobs, T, V, t0, { names }).words
          : greedyWords(logprobs, T, V, assets.tokens, t0);
        if (Booster) words = new Booster(names, assets.tokens, assets.pieces).apply(words, logprobs, T, V, t0);
        item.utterances = segment(words, t0, tEnd);
        done++;
      }
      out.push(item);
    }
    lap("lips");
    out.sort((a, b) => (a.skipped - b.skipped) || b.facePx - a.facePx);
    return { duration: vid.duration, width: vid.width, height: vid.height, start, end: stop,
             tracks: out, model: this.model, decoder: assets.decoder?.kind || "greedy", disclaimer: "AI lip-reading estimate. Words can be wrong. Don't treat as a quote." };
  }

  // ---------------------------------------------------------------- exports
  toText(result) {
    const lines = ["LipRead — AI lip-reading estimate (words can be wrong; not a quote)", ""];
    result.tracks.filter((t) => !t.skipped).forEach((t, i) => {
      lines.push(`Face ${i + 1} (${t.tier})`);
      for (const u of t.utterances) lines.push(`  [${fmt(u.start)}] ${u.words.map((w) => (w.tier === "low" ? w.text.toLowerCase() + "?" : w.text)).join(" ")}`);
      lines.push("");
    });
    return lines.join("\n");
  }

  toSRT(result) {
    const cues = [{ start: result.start, end: result.start + 3, text: "AI lip-read estimate — may be wrong" }];
    result.tracks.filter((t) => !t.skipped).forEach((t, i) => {
      for (const u of t.utterances) cues.push({ start: u.start, end: Math.max(u.end, u.start + 0.8),
        text: `[Face ${i + 1}] ` + u.words.map((w) => (w.tier === "low" ? w.text + "?" : w.text)).join(" ") });
    });
    cues.sort((a, b) => a.start - b.start);
    const ts = (x) => { const ms = Math.round(x * 1000); const h = Math.floor(ms / 3600000), m = Math.floor(ms / 60000) % 60, s2 = Math.floor(ms / 1000) % 60;
      return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s2).padStart(2, "0")},${String(ms % 1000).padStart(3, "0")}`; };
    return cues.map((c, i) => `${i + 1}\n${ts(c.start)} --> ${ts(c.end)}\n${c.text}\n`).join("\n");
  }
}

/** Busy counter read by sw-register.js: never reload the page while work is running. */
function busy(d) { self.__lipreadBusy = Math.max(0, (self.__lipreadBusy | 0) + d); }

/** Attach the short error code the UI keys its messages on. */
function coded(err) {
  const msg = String((err && err.message) || err);
  const code = err?.name === "AbortError" ? "aborted"
    : /^VIDEO_AUTOPLAY_BLOCKED/.test(msg) ? "autoplay"
    : /^VIDEO_STALLED/.test(msg) ? "stalled"
    : /^VIDEO_/.test(msg) ? "decode"
    : /^MODEL_MISSING/.test(msg) ? "model"
    : /^(DOWNLOAD_FAILED|CHECKSUM_MISMATCH)/.test(msg) ? "download"
    : /^STORAGE_FULL/.test(msg) || err?.name === "QuotaExceededError" ? "storage"
    : /^WORKER_CRASHED|out of memory|memory access out of bounds|Aborted\(|RangeError|alloc/i.test(msg) ? "memory"
    : /^CLIP_TOO_SHORT/.test(msg) ? "short" : "";
  const e = err instanceof Error ? err : new Error(msg);
  if (code && !e.code) e.code = code;
  return e;
}

function fmt(t) { const m = Math.floor(t / 60), s = (t % 60).toFixed(1).padStart(4, "0"); return `${m}:${s}`; }

/** Canvas that supports toDataURL (thumbnails) — a plain <canvas> on the main thread. */
function OffscreenOrCanvas(w, h) {
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  return c;
}
