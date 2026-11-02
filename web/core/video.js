// Frame access for on-device processing: a hidden, muted, inline <video> that we seek frame by frame
// at 25 fps. Seeking is deterministic (every model frame is exactly where it should be) and works the
// same on iPhone Safari and Android Chrome.

export async function openVideo(file) {
  const url = URL.createObjectURL(file);
  const v = document.createElement("video");
  v.muted = true; v.playsInline = true; v.preload = "auto";
  v.setAttribute("playsinline", ""); v.setAttribute("muted", "");
  v.src = url;
  try {
    await new Promise((resolve, reject) => {
      const t = setTimeout(() => reject(new Error("VIDEO_TIMEOUT")), 20000);
      v.addEventListener("loadeddata", () => { clearTimeout(t); resolve(); }, { once: true });
      v.addEventListener("error", () => { clearTimeout(t); reject(new Error("VIDEO_UNSUPPORTED")); }, { once: true });
      // iOS only decodes after play(); a muted inline video may play without a gesture.
      v.play().then(() => v.pause()).catch(() => {});
    });
  } catch (err) {
    URL.revokeObjectURL(url);
    throw err;
  }
  if (!v.videoWidth || !v.videoHeight || !isFinite(v.duration)) {
    URL.revokeObjectURL(url);
    throw new Error("VIDEO_UNSUPPORTED");
  }
  v.pause();
  return { video: v, url, width: v.videoWidth, height: v.videoHeight, duration: v.duration,
           close() { v.removeAttribute("src"); v.load(); URL.revokeObjectURL(url); } };
}

/**
 * Visit N frames on a 25 fps grid starting at `start`, calling `await onFrame(f)` with the video
 * showing frame f. Uses playback + requestVideoFrameCallback, pausing while each frame is processed:
 * the decoder runs forward continuously instead of re-decoding from a keyframe on every seek
 * (seeking measured 183 ms/frame in Chromium; long-GOP sports broadcasts are worse).
 * Falls back to seeking when requestVideoFrameCallback is missing.
 */
export async function stepFrames(v, start, N, fps, onFrame, check = () => {}) {
  if (!("requestVideoFrameCallback" in HTMLVideoElement.prototype)) {
    for (let f = 0; f < N; f++) { check(); await seek(v, start + f / fps); await onFrame(f); }
    return;
  }
  if (!v.isConnected) {   // some browsers only present frames for attached elements
    Object.assign(v.style, { position: "fixed", left: "-10px", top: "-10px", width: "2px", height: "2px", opacity: "0", pointerEvents: "none" });
    document.body.appendChild(v);
  }
  await seek(v, start);
  let f = 0, dt = 1 / 30, lastMT = null;
  await new Promise((resolve, reject) => {
    let settled = false;
    const finish = (err) => { if (settled) return; settled = true; v.pause(); v.onended = null; err ? reject(err) : resolve(); };
    const cb = async (_now, meta) => {
      try {
        check();
        const mt = meta.mediaTime;
        if (lastMT !== null && mt - lastMT > 1e-4) dt = Math.min(dt, mt - lastMT);
        lastMT = mt;
        if (mt + dt / 2 >= start + f / fps) {
          v.pause();
          // one presented frame can serve several grid points when the source is below 25 fps
          while (f < N && mt + dt / 2 >= start + f / fps) { await onFrame(f); f++; }
          if (f >= N) return finish();
        }
        v.requestVideoFrameCallback(cb);
        if (v.paused) await v.play();
      } catch (err) { finish(err); }
    };
    v.onended = async () => {   // clip ended before the grid did: reuse the last frame
      try { while (f < N) { await onFrame(f); f++; } finish(); } catch (err) { finish(err); }
    };
    v.requestVideoFrameCallback(cb);
    v.play().catch((err) => finish(new Error(`VIDEO_PLAY_FAILED: ${err.message || err}`)));
  });
}

/** Seek to time t (s) and resolve once the frame is ready to draw. */
export function seek(v, t) {
  return new Promise((resolve) => {
    let finished = false;
    const done = () => {
      if (finished) return; finished = true;
      v.removeEventListener("seeked", onSeeked);
      resolve();
    };
    const onSeeked = () => {
      done();   // after "seeked" the new frame can be drawn (Safari and Chrome)
    };
    v.addEventListener("seeked", onSeeked);
    v.currentTime = Math.min(t, Math.max(0, v.duration - 0.001));
    setTimeout(done, 4000);
  });
}
