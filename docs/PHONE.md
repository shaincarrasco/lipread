# Phone app: architecture, privacy model, and test plan

The phone app is a static web app in `web/`. It runs entirely in the browser on the phone. The
video is read on the device and is never uploaded. This page describes how it works, what has been
tested, and what still has to be tested on real phones.

**Status: pre-release.** Checks so far used desktop Chromium with iPhone-size emulation. Nothing here
has been run on a real iPhone or Android phone, and no phone speed or memory figure exists yet.

## 1. Pipeline

```
video file
  -> frame stepper (play + pause at each 25 fps frame) main thread
  -> YuNet face detection (onnxruntime-web, WASM)     Web Worker
  -> IoU tracker (one track per person; cuts reset it)
  -> keypoint correction (learned, 5 -> 4 points)
  -> readability gate (face size, yaw)
  -> mouth crops (144 x 144 grayscale, stored per track per frame)
  -> lip-reading model (onnxruntime-web, WASM)        Web Worker
  -> CTC decode (words, times, confidence)
  -> name booster (roster rescoring, flags boosted words)
  -> UI (captions, Likely / All, share sheet)         main thread
```

Each stage:

1. **Video in.** A muted, `playsinline` `<video>` element loads the file the user picked.
2. **Frames.** The video plays muted and pauses on each 25 fps grid frame (`requestVideoFrameCallback`),
   so the decoder runs forward instead of re-decoding from a keyframe on every seek. In desktop Chromium
   this cut frame waiting from 55 s to 9 s for a 12 s clip (seeking: 183 ms per frame). Browsers without
   `requestVideoFrameCallback` fall back to seeking.
3. **Detection.** YuNet runs on every 2nd frame, on a 640 x 640 letterbox of the whole frame. When the
   long side is over 960 px, two zoomed tiles (each 55% of the frame, overlapping) are also searched on every 3rd detector run
   (every 6th frame), so small faces in wide shots are found (a 29 px fan face in a 720p test clip was
   only found this way). The YuNet decode matches OpenCV's output exactly on the test frames.
4. **Tracking.** An IoU tracker links detections into one track per person. A cut resets the tracks,
   so a track never crosses a change of shot.
5. **Keypoint correction.** A learned correction maps YuNet's five face points to the four keypoints
   used for alignment. It replaces the MediaPipe refinement used on the desktop. It was fitted on 13 clips (1,234 face detections).
6. **Readability gate.** Face size, yaw (from the YuNet points), how often the face was found,
   mouth sharpness and mouth motion set each track to `readable`,
   `marginal` or `unreadable`, with reasons. Thresholds: 56 px and above readable, 32 to 56 px marginal,
   under 32 px unreadable. By default readable and marginal tracks are decoded; unreadable ones only if the user asks.
7. **Mouth crops.** For each track, in each frame, a 144 x 144 grayscale crop is cut around the mouth,
   sized at 2.6 times the inter-eye distance. After the pass, keypoints are interpolated and smoothed
   (+-6 frames) and the 88 x 88 model crops are sampled from these stored crops with the Auto-AVSR
   alignment (least-squares similarity transform).
8. **Lip model.** The int8 ONNX lip-reading model runs in the worker on the aligned crops. It is the
   same kind of model as the desktop app. See `MODEL_LICENSES.md` for its terms.
9. **CTC decode.** The model's output becomes words with start and end times. Each word has a
   confidence, which sets its tier: high (0.9 and above), medium (0.7 to 0.9), low (below 0.7).
10. **Name booster.** Words close to a name in the roster are rescored and flagged as boosted. The
    original word is kept. The booster is a port of the desktop booster, not a library.
11. **UI.** Captions appear as pills anchored to each face, timed to the track's boxes. A segmented
    control switches between *Likely* and *All*. The share sheet offers the transcript text and SRT.

### Worker API (what the UI calls)

```js
import { LipRead } from "./core/lipread.js";
const lr = new LipRead({ onProgress?, config? });   // config overrides web/core/config.js
await lr.modelStatus();              // { cached, bytes }
await lr.downloadModel(onBytes);     // resolves once cached and SHA-256 verified
const result = await lr.process(file, { names, faces: "readable" | "all", start?, end?, signal? });
lr.toSRT(result);  lr.toText(result);
await lr.deleteModel();
```

Progress stages are `model`, `faces`, `lips` and `done`.

## 2. Why there is no MediaPipe on the phone

- **One detector, one model file, one license to check.** YuNet is MIT-licensed and is the only face
  detector in the phone app. The desktop app also uses MediaPipe BlazeFace, whose model card has its
  own terms to check (see `MODEL_LICENSES.md`).
- **The correction closes the gap.** On the 8-clip check, the corrected YuNet scored 31.8% word error
  rate. The desktop MediaPipe hybrid scored 31.4% on the same clips, so the difference is within what
  8 clips can show (the correction was fitted on 13 clips that include them).
- **MediaPipe alone was weak on small faces.** In the desktop tests it lost faces at 120 px and below.

## 3. Memory strategy

- **Full-resolution frames are never kept.** Each frame is decoded, searched, cropped and released.
- **Only mouth crops are stored.** One 144 x 144 grayscale crop is about 20 KB. Storage grows with
  the number of tracks times the number of sampled frames, so a long clip with several faces uses
  more memory. Limits on a real phone are untested.
- **The model runs in the worker,** not on the main thread, so the model's memory and work stay out
  of the UI's path.
- **Detection runs on every 2nd frame.** Between detections a track keeps its last box and keypoints
  (the crop has margin for movement); keypoints are interpolated afterwards.

## 4. Model download and caching

- The lip-reading model is about **203 MB**, fetched from Hugging Face. Hugging Face serves it with
  CORS headers, which were verified.
- The download is checked against its **SHA-256** before it is used. A file that fails the check is not
  kept.
- The verified model is stored in the browser's **Cache Storage**. The app asks for persistent storage
  with `navigator.storage.persist()`, so the browser is less likely to clear it under storage pressure.
  The browser may still refuse, so the app must handle a missing model.
- A **service worker** caches the app's own files, so the app is designed to open with no network.
  Airplane-mode reopen is in the device checklist and has not been tested.
- A first run on a slow connection will take a while. Wi-Fi is recommended.

## 5. Threading

- Multi-thread WASM needs `SharedArrayBuffer`, which needs cross-origin isolation (`Cross-Origin-Opener-Policy`
  and `Cross-Origin-Embedder-Policy` headers). **GitHub Pages cannot send them**, so the app's service worker
  adds them to its own responses (the coi-serviceworker technique) and the page reloads once on the first
  visit, before anything starts.
- Measured in desktop Chromium (2 cores), 12 s clip: lip model 14.1 s single-thread vs 8.6 s with threads;
  total 38 s vs 31 s. Phone numbers are not measured.
- If a browser does not honour the service-worker headers (possible on iPhone Safari, untested), the app
  runs single-threaded automatically.
- The two neural networks (face detector and lip model) run in a Web Worker. Frame drawing, crops,
  tracking and alignment run on the main thread between frames.

## 6. Privacy model

- **The video is read on the device.** No frame, crop or transcript is sent anywhere.
- **Network use is downloads only:** the app's files from GitHub Pages, and the model from Hugging Face.
  Both see the request. Their logs are not controlled by this project.
- **No accounts, no analytics, no tracking code.**
- **Stored on the phone:** the model (Cache Storage), the app's files (service worker cache), and the
  roster names the user types, kept in the browser's `localStorage` on the device.
- **Shared only by the user.** Transcript text and SRT leave the phone only through the share sheet,
  and only when the user chooses to send them.

The full plain-language version is in `PRIVACY.md`.

## 7. Testing

### Done

- **YuNet decoding in the browser** matches OpenCV's YuNet decoding exactly on test frames.
- **8-clip accuracy check (word error rate, WER):**
  - corrected YuNet (the shipped path): **31.8%**
  - MediaPipe hybrid (desktop path): **31.4%**
  - raw YuNet with no correction: **32.2%**
  - The correction was fitted on 13 clips that include these 8, so the 31.8% figure is slightly optimistic.
- **Desktop Chromium with iPhone-size emulation.** The app was exercised at phone size in this browser.

### Not done

- **No real iPhone.** Not on any iOS version.
- **No WebKit.** WebKit (the engine behind Safari) was not available, so Safari-specific behaviour is
  untested.
- **No real Android phone.**
- **No phone speed, memory, battery or heat figures.**
- **No HEVC (H.265) decoding on any real device.**
- **No tests of** airplane-mode reopen, backgrounding mid-run, long clips, or the first download on a
  real connection.
- **Automated browser tests:** `tests_web/e2e_phone.py` walks the real UI with the real engine in headless
  Chromium at iPhone size; `node --test tests_web/*.test.mjs` runs the JS unit tests (booster and decoder
  parity with Python, SHA-256).
- **Booster parity:** the JS booster matches the Python booster on real model output, and its name
  tokenizer matches SentencePiece on 400 random names.

## 8. Real-device test checklist

Record the device model, OS version, browser version, date, clip length, and the time each step took.

### iPhone (Safari, iOS 16.4 or later)

- [ ] Open the app link in Safari. Note the iOS version and the phone model.
- [ ] Add to Home Screen. Launch from the icon and check it opens full screen.
- [ ] Download the model on Wi-Fi. Record the time, that progress shows, and that the SHA-256 check passes.
- [ ] Process a 10-second clip with one face. Record total time and whether the captions match the speaker.
- [ ] Turn on airplane mode. Close the app, reopen it, and process a clip using the cached model.
- [ ] Process a portrait HEVC clip (a common iPhone recording format). Record whether it decodes, or whether
      the message is clear if it does not.
- [ ] Process a long clip (for example 5 minutes). Record whether it finishes, how long it takes, and
      whether Safari reloads or drops the page.
- [ ] Switch to another app for 30 seconds during a run, then return. Record whether the job resumes,
      fails with a clear message, or is lost.
- [ ] Rotate to landscape and back during a run. Check for no crash and no broken layout.
- [ ] Turn on Low Power Mode and process a 10-second clip. Record the time.
- [ ] Check dark mode and Reduce Motion. Check nothing is broken.

### Android (Chrome, recent version)

- [ ] Open the app link in Chrome. Note the Android version, phone model and Chrome version.
- [ ] Install or add to the Home screen. Check it opens full screen.
- [ ] Download the model on Wi-Fi. Record the time, progress, and that the SHA-256 check passes.
- [ ] Process a 10-second clip with one face. Record total time and caption quality.
- [ ] Turn on airplane mode. Close the app, reopen it, and process a clip using the cached model.
- [ ] Process an HEVC clip. Confirm the HEVC failure message appears clearly and nothing crashes.
- [ ] Process a long clip (for example 5 minutes). Confirm only the first 30 s are read and the screen says so.
- [ ] Switch to another app for 30 seconds during a run, then return. Record the outcome.
- [ ] Rotate to landscape and back during a run. Check for no crash.
- [ ] Turn on battery saver and process a 10-second clip. Record the time.
- [ ] Check dark mode and reduced motion. Check nothing is broken.

## 9. Open questions and risks

- **Memory limits on phones.** The main unknown. Long clips with many faces may fail, and the failure
  mode on iPhone Safari (a reloaded or dropped page) is not known.
- **HEVC.** Whether each browser can decode common phone recordings is not known. The app must say so
  when it cannot.
- **Background runs.** Mobile browsers may pause or kill a tab that is in the background. Behaviour is
  untested.
- **Speed on phones.** Desktop Chromium needs about 3x the clip length (12 s single-face clip: 31-39 s; 12 s three-face 720p clip: 46 s). Phones are
  likely slower; only a real-device test will tell.
- **Clip length.** Only the first 30 seconds of a clip are read; the results screen says so.
- **Accuracy claims.** The 31.8% figure is from 8 clips (correction fitted on 13 clips that include them) and is optimistic. Phone accuracy needs a larger,
  labelled set.
- **Roster storage.** Roster names are saved in the browser's `localStorage` on the device only.
