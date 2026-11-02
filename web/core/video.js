// Frame access for on-device processing: a hidden, muted, inline <video>.
// Frames are visited on a 25 fps grid by playing the video and pausing on each grid frame
// (requestVideoFrameCallback): the decoder runs forward instead of re-decoding from a keyframe on
// every seek (seeking measured 183 ms/frame in Chromium). Seeking is the fallback.

const STALL_NUDGE_MS = 5000, STALL_FAIL_MS = 15000;

/** Open a picked file. Call this as close to the user's tap as possible (iOS playback rules). */
export async function openVideo(file) {
  const url = URL.createObjectURL(file);
  const v = document.createElement("video");
  v.muted = true; v.defaultMuted = true; v.playsInline = true; v.preload = "auto";
  v.setAttribute("playsinline", ""); v.setAttribute("muted", "");
  Object.assign(v.style, { position: "fixed", left: "-10px", top: "-10px", width: "2px", height: "2px",
                           opacity: "0", pointerEvents: "none" });
  document.body.appendChild(v);   // some browsers only present frames for attached elements
  v.src = url;
  const close = () => { try { v.pause(); } catch { /* ignore */ } v.removeAttribute("src"); v.load(); v.remove(); URL.revokeObjectURL(url); };
  try {
    const metadata = new Promise((resolve, reject) => {
      const t = setTimeout(() => reject(new Error("VIDEO_TIMEOUT")), 20000);
      v.addEventListener("loadedmetadata", () => { clearTimeout(t); resolve(); }, { once: true });
      v.addEventListener("error", () => { clearTimeout(t); reject(new Error("VIDEO_UNSUPPORTED")); }, { once: true });
    });
    // iOS only starts decoding after play(); a muted inline video may play without a gesture,
    // except when the system refuses (e.g. Low Power Mode).
    let canPlay = true;
    try { await v.play(); v.pause(); }
    catch (err) {
      if (err && err.name === "NotAllowedError") canPlay = false;
      else if (err && err.name === "NotSupportedError") throw new Error("VIDEO_UNSUPPORTED");
    }
    await metadata;
    if (!v.videoWidth || !v.videoHeight || !isFinite(v.duration)) throw new Error("VIDEO_UNSUPPORTED");
    if (!canPlay) {
      // Without playback, seeking still decodes on most browsers; prove we can get a frame.
      const ok = await seek(v, Math.min(0.05, v.duration / 2), 6000);
      if (!ok || v.readyState < 2) throw new Error("VIDEO_AUTOPLAY_BLOCKED");
    }
    return { video: v, url, canPlay, width: v.videoWidth, height: v.videoHeight, duration: v.duration, close };
  } catch (err) {
    close();
    throw err;
  }
}

/**
 * Visit N frames on a 25 fps grid starting at `start`, calling `await onFrame(f)` while the video
 * shows frame f. `check()` throws to cancel. Rejects with VIDEO_STALLED if decoding stops.
 */
export async function stepFrames(vid, start, N, fps, onFrame, check = () => {}, signal) {
  const v = vid.video;
  if (!vid.canPlay || !("requestVideoFrameCallback" in HTMLVideoElement.prototype)) {
    for (let f = 0; f < N; f++) {
      check();
      if (!(await seek(v, start + f / fps, 8000))) throw new Error("VIDEO_STALLED");
      await onFrame(f);
    }
    return;
  }
  await seek(v, start, 8000);
  let f = 0, dt = 1 / 30, lastMT = null;
  await new Promise((resolve, reject) => {
    let settled = false, busy = false, lastFrameAt = performance.now();
    const watchdog = setInterval(() => {
      if (busy || settled) return;
      const idle = performance.now() - lastFrameAt;
      if (idle > STALL_FAIL_MS) finish(new Error("VIDEO_STALLED"));
      else if (idle > STALL_NUDGE_MS && v.paused) v.play().catch(() => {});
    }, 1000);
    const onVisible = () => { if (!document.hidden && !busy && !settled && v.paused) v.play().catch(() => {}); };
    const onAbort = () => finish(new DOMException("Cancelled", "AbortError"));
    document.addEventListener("visibilitychange", onVisible);
    signal?.addEventListener("abort", onAbort, { once: true });
    function finish(err) {
      if (settled) return;
      settled = true;
      clearInterval(watchdog);
      document.removeEventListener("visibilitychange", onVisible);
      signal?.removeEventListener("abort", onAbort);
      v.onended = null;
      try { v.pause(); } catch { /* ignore */ }
      err ? reject(err) : resolve();
    }
    const cb = async (_now, meta) => {
      if (settled) return;
      lastFrameAt = performance.now();
      try {
        check();
        const mt = meta.mediaTime;
        if (lastMT !== null && mt - lastMT > 1e-4) dt = Math.min(dt, mt - lastMT);
        lastMT = mt;
        if (mt + dt / 2 >= start + f / fps) {
          v.pause();
          busy = true;
          // one presented frame can serve several grid points when the source is below 25 fps
          while (f < N && mt + dt / 2 >= start + f / fps) { await onFrame(f); f++; }
          busy = false; lastFrameAt = performance.now();
          if (f >= N) return finish();
        }
        if (settled) return;
        v.requestVideoFrameCallback(cb);
        if (v.paused) await v.play();
      } catch (err) { finish(err); }
    };
    v.onended = async () => {   // the clip ended before the grid did: reuse the last frame
      if (settled || busy) return;
      try { busy = true; while (f < N) { check(); await onFrame(f); f++; } finish(); } catch (err) { finish(err); }
    };
    v.requestVideoFrameCallback(cb);
    v.play().catch((err) => finish(new Error(err && err.name === "NotAllowedError" ? "VIDEO_AUTOPLAY_BLOCKED" : `VIDEO_STALLED: ${err.message || err}`)));
  });
}

/** Seek to time t (s); resolves true once the frame is ready to draw, false on timeout. */
export function seek(v, t, timeoutMs = 4000) {
  return new Promise((resolve) => {
    let finished = false, timer = null;
    const done = (ok) => {
      if (finished) return; finished = true;
      clearTimeout(timer);
      v.removeEventListener("seeked", onSeeked);
      resolve(ok);
    };
    const onSeeked = () => done(true);
    v.addEventListener("seeked", onSeeked);
    timer = setTimeout(() => done(false), timeoutMs);
    v.currentTime = Math.min(t, Math.max(0, v.duration - 0.001));
  });
}
