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
- **Browser version.** Run the engine with onnxruntime-web, so a clip can be checked without a Mac
  install. Goal, not promised: the model is large and browser speed is unmeasured.

## v1.0: goals

- A published accuracy table on a sports test set we label ourselves, with confidence intervals.
- A signed, notarized Mac app (no Terminal, no iTerm).
- Pluggable engines documented and tested with at least two real models.
- Human-review workflow in the web UI: accept, edit, reject each word before export.

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
