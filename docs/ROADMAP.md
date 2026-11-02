# Roadmap

Status key: **Shipped** = in the code now and measured. **Planned** = decided, not built.
**Goal** = what we are aiming for. A goal is not a result. No accuracy number below is a
claim about LipRead unless it is in the README's measured table.

## v0.1: now (shipped)

- Any local file or http(s) URL in, 25 fps frame stream out.
- Shot cuts, small-face detection (YuNet), keypoint refinement (MediaPipe), track linking.
- Readability gate: readable, marginal, unreadable, with reasons.
- Visual-only Auto-AVSR (LRS3) int8 ONNX engine, greedy CTC decoding, word timing and confidence.
- Roster booster: names passed in are rescored and flagged as boosted.
- Outputs: JSON, SRT, WebVTT, TXT, annotated MP4, face thumbnails.
- CLI (`scan`, `run`, `models`, `serve`) and a local neon web UI on 127.0.0.1:8765.
- macOS install script and a Desktop launcher that opens iTerm2.
- Measured: 31.6% WER greedy on 150 LRS3 test clips; confidence tiers separate good and bad output
  (see README). Beam search without a language model did not help.

## v0.2: calibrate and upgrade the engine

- **Calibrate on labelled sports clips.** The owner's own clips, with human-written transcripts,
  set the confidence thresholds and the readability gate. Goal: thresholds that match sports footage,
  not TED talks.
- **USR 2.0 engine.** The model card reports 17.6% WER on LRS3 VSR. That number is the model
  authors' claim and is **unverified** here. It will only count after we measure it on the same
  150-clip set. License to be checked first.
- **Full Auto-AVSR decoding.** Beam search with a language model. The published figure for this
  setup is 19.1% WER on LRS3 VSR (Ma et al., ICASSP 2023). That is published, not measured here.
  Risk: KenLM would not install in our build environment, so this may need a different LM toolkit.

## v0.3: sports, audio and names

- **Sports fine-tuning.** Fine-tune on sports footage. **Blocked for now:** the LRS3 data is listed
  as CC BY-NC-ND 4.0 (no derivatives) on the KAIST mirror. Check the terms with the rights holders
  first, or use training data that allows derivatives. Fine-tuned weights must not be distributed
  until this is cleared. See MODEL_LICENSES.md.
- **Audio-visual mode.** Use the audio track where it is clean (a close-mic'd coach, a stadium
  with low noise). Lip reading covers the rest.
- **Jersey OCR to name players.** Read jersey numbers and match them to a roster, so the booster
  can name people without you typing names. Flagged as a guess. Not in v0.1 on purpose.
- **Browser version.** Now in progress as the phone web app in `web/`, running the engine with
  onnxruntime-web. Phone speed, memory and HEVC are unmeasured on real devices. See `docs/PHONE.md`.

## v1.0: goals

- A published accuracy table on a sports test set we label ourselves, with confidence intervals.
- A signed, notarized Mac app (no Terminal, no iTerm).
- Pluggable engines documented and tested with at least two real models.
- Human-review workflow in the web UI: accept, edit, reject each word before export.

## Phone app milestones

The phone app (`web/`, see `docs/PHONE.md`) is free and non-commercial, like the rest of the project.
These milestones are planned. None is done yet.

| Milestone | What it means | Done when |
|---|---|---|
| P1: real-device test | Run the checklist in `docs/PHONE.md` on at least one real iPhone (Safari) and one real Android phone (Chrome). | Results for every checklist item are recorded, with device and OS versions. |
| P2: speed and memory tuning | Use the P1 numbers to tune frame reading, detection stride and crop storage. Record the memory limit found on each device. | Each device has a measured speed and memory figure in `docs/PHONE.md`. |
| P3: `requestVideoFrameCallback` frame capture | Read frames with `requestVideoFrameCallback` where the browser supports it, instead of seeking. | Output on the test clips matches the seek path, and the change is faster on the P1 phones. |
| P4: threads, only if measured faster | Multi-thread WASM needs cross-origin isolation, which GitHub Pages cannot send. Any workaround needs its own review. | Threads are measured faster on the same phones, and the hosting change is decided. |
| P5: captioned video export | Burn captions into a copy of the clip and save it to the phone (was planned as v1.1 in the phone plan). The AI-estimate label stays on the video. | Export works on the P1 phones, and the label is visible on the exported file. |
| P6: sample clip with cleared rights | A sample clip that the owner or the rights holder has cleared for public use, so anyone can try the app. | Written permission is on file, and the clip's license is in the repo. |

## Path to monetizing

Nothing here is planned. It is the list of what must change before the project can charge money or
show ads. The owner may want to do this later.

**Today the answer is no.** The lip-reading model's weights are licensed for research and
non-commercial use only, and its training data (LRS3) carries no-derivatives terms on the mirror we
checked. See `MODEL_LICENSES.md`.

### Blockers, in order

1. **Replace the lip-reading model** with one whose weights and training data both allow commercial
   use. Options:
   - a permissively licensed model whose training data also allows commercial use, with its license
     checked at both levels; or
   - train or fine-tune on footage the project has the right to use: licensed footage, or footage
     recorded for this purpose with written consent from the people filmed.
   The engine interface (`src/lipread/engine.py` on desktop, the worker API in `docs/PHONE.md` on the
   phone) is built so the swap is local. The swap still needs a new accuracy measurement on the same
   test set.
2. **Verify the full license chain.** Check every file the app ships or downloads, and record each
   result in `MODEL_LICENSES.md`:
   - the model weights and the ONNX export (the export's own card says research and non-commercial);
   - `tokens.json`, the mean-face file and the SentencePiece model (from Auto-AVSR, Apache-2.0 per
     `MODEL_LICENSES.md`, to re-check for the exported copies);
   - YuNet (MIT) and onnxruntime-web (license to check);
   - the learned keypoint correction. It was fitted on clips, so the terms of those clips may carry over
     to the weights. Check this before the correction is distributed;
   - icons, fonts and any sample clip.
   The Hugging Face hosting terms for the model file also apply.
3. **Privacy policy and terms.** `PRIVACY.md` is a plain-language draft. Before charging or showing
   ads, have it reviewed for each country you sell in, and add terms of use (acceptable use, no
   targeting of private people, the AI-estimate label). If you add ads or analytics, the promise
   "no analytics" in `PRIVACY.md` stops being true and must be rewritten, with the consent flows that
   the countries you sell in require.
4. **Accuracy claims you can sell.** Current numbers are from LRS3 and an 8-clip check. Any sales claim
   needs the sports benchmark from ROADMAP v1.0: a labelled sports test set, with confidence intervals.
5. **Safety review for paid use.** Paid use means more people using the tool, and more chances for
   harm. Review `ETHICS.md` against the paid use case before launch.

### Ways to ship it as an app

These are options, not decisions. Every fee and rate below is **[unverified]** and must be checked on
the store's current pages before any budget is set.

- **Keep it a web app** (what it is now). No store fees. Any payment would use a web checkout, whose
  processor fees are **[unverified]**.
- **Wrap it in a native shell** (for example Capacitor or a similar tool) and submit it to the Apple App
  Store and Google Play. Apple's developer program annual fee is **[unverified]**. Google Play's one-time
  registration fee is **[unverified]**. Store commission on digital purchases is **[unverified]** and
  varies by store and by program. Store review rules for AI and video apps are **[unverified]**.
- **Ads** in either form are governed by the ad network's policies, which may restrict apps that process
  video of people. Check those policies first **[unverified]**.

Wrapping the app does not change the model license. Blockers 1 to 3 still apply.

## Considered and parked

These are not on the path above. Each one was considered and set aside for a reason.

| Idea | Why parked |
|---|---|
| Replay fusion (combine several angles) | Adds much complexity before single-angle output is calibrated. |
| Scoreboard OCR prior | Useful later; it would guess who is talking from the game state. Risk of wrong attribution. |
| Viseme output | Interesting for dubbing research; not useful for a readable transcript. |
| Super-resolution of faces | Can invent mouth shapes that were never in the video. Fabricated words are worse than no words. |
| Caption-first mode | Captions transcribe commentators, not the people on screen. Would mislead. |
| Gradio UI | Custom neon UI matches the look and keeps the app small. |
