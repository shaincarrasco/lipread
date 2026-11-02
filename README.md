<p align="center">
  <img src="web/icons/icon-512.png" width="152" alt="LipRead app icon">
</p>

<h1 align="center">LipRead</h1>

<p align="center"><b>Read the lips in a video, right on your phone.</b><br>
Built for sports clips: see what a coach, a player or a referee appears to say.</p>

> **Status: pre-release.** The phone app has been checked in desktop Chromium at phone size only.
> It has **not yet been tested on a real iPhone or Android phone.** The desktop app (below) is
> v0.1, early, and has measured results.

## Open on your phone

**[Open LipRead → shaincarrasco.github.io/lipread](https://shaincarrasco.github.io/lipread/)**

> The link goes live after the repository is published with GitHub Pages ([docs/DEPLOY.md](docs/DEPLOY.md)).
> If you publish under a different account, change `shaincarrasco` here and in `web/app.js` (`REPO_URL`).

### How to use it

1. **Open the link** in Safari on an iPhone, or Chrome on an Android phone.
2. **Add it to your Home Screen.** In Safari: Share, then *Add to Home Screen*. In Chrome: the menu,
   then *Add to Home screen* or *Install app*. It then opens full screen, like an app.
3. **Tap Choose Video** (or Record) and pick a clip. Reading starts straight away and shows its progress.
   Optional: in **Settings**, type the names you expect (players, coaches) so the app can check for them.

The first run downloads the lip-reading model: **about 203 MB**. Use Wi-Fi. After that the model
stays on the phone.

## How accurate is it?

The phone app runs the same kind of lip-reading model as the desktop app. It is usable on clear,
face-on speakers and unreliable elsewhere. The confidence on each word tells you which words to doubt.

- **Confidence tiers.** Every word is marked **high**, **medium** or **low**. In the desktop
  measurements, high-confidence words were right 96% of the time, medium 74%, and low 32% to 50%.
  Treat low words as guesses.
- **Phone pipeline, 8-clip check.** **31.8% word error rate.** On the same clips the desktop pipeline
  scored 31.4%, and raw face detection without the correction scored 32.2%. The phone correction was
  fitted on 13 clips (1,234 face detections) that include these 8, so expect the phone figure to be
  slightly optimistic. Eight clips cannot tell these three apart.
- **Context-aware decoding (new).** Instead of taking the single best guess for each word, LipRead keeps
  16 candidate transcripts and lets a word language model (built from 4 million words of TED transcripts,
  with every test sentence removed) pick the most plausible one. Measured on data not used for tuning:
  LRS3 test **31.9% → 30.6%** word error, LRS2 **22.9% → 20.7%**. Tap any word to see what else it could
  be; those percentages are calibrated (words shown at about 57% were right 57% of the time).
- **Phone app in a desktop browser.** The phone app's engine running in desktop Chromium (iPhone-size
  emulation, 2-core machine) gave the same transcripts as the desktop app on the test clips. A 12 s
  single-face clip took 31–39 s; a 12 s three-face 720p clip took 46 s.
- **Phone speed and accuracy on real phones have not been measured.** The test checklist is in
  [docs/PHONE.md](docs/PHONE.md).

The full desktop measurements, and what each one means, are in
[the accuracy table](#how-accurate-is-it) under *Desktop and command line*.

## Privacy

- **Your video stays on your phone.** The app reads the frames on your device. Nothing you process
  is uploaded.
- **No accounts and no analytics.** The app does not track you.
- **Network use is limited to downloads.** The app (including its on-device AI runtime) comes from
  GitHub Pages, and the model comes from Hugging Face. Both see the request. Details are in
  [PRIVACY.md](PRIVACY.md).

## Supported browsers

These are **targets**. They have not been tested on real devices yet.

- **iPhone:** Safari on iOS 16.4 or later
- **Android:** recent Chrome

Other browsers are not supported yet.

## Known limits on the phone

- **Not tested on real phones.** Checks so far used desktop Chromium at phone size.
- **Speed and memory are unmeasured on phones.** In a desktop browser, reading takes about 3x the
  clip length. Only the **first 30 seconds** of a clip are read; the results screen says so.
- **Keep the app open while it works.** The app asks the phone to keep the screen on where the
  browser allows it. What happens when you switch apps mid-run is untested.
- **Some videos may not open.** Many phone recordings use HEVC (H.265), and not every browser can
  read them. The app should say so rather than fail silently. That message is untested on devices.
- **English speech only.** Names are boosted only if you type them in the roster.
- **Not in the phone app yet:** captioned video export, live captions, a sample clip.
- **The limits below still apply:** side views, small faces, covered mouths, blur and low frame rates
  all hurt accuracy.

## Responsible use

LipRead gives **estimates, not quotes**. Check the output against the video before you share it. Do
not use it to read the lips of private people in crowds, and do not put words in a named person's
mouth. The full policy is in [ETHICS.md](ETHICS.md).

---

## Desktop and command line

LipRead takes a video, finds the faces, decides which faces are readable, and writes out what
each person appears to say. You get word-level timestamps, a confidence score on every word,
and an annotated video you can check. It uses the picture only (no audio) and is built to run locally on your Mac (so far it has been tested in a Linux sandbox, not on a Mac).

> **Status: v0.1, early.** It has measured results, and it is not perfect. It is **not yet
> tuned for sports**. That is on the roadmap. The default model is **research and non-commercial
> only** (see [MODEL_LICENSES.md](MODEL_LICENSES.md)).

### What it does

- Finds shot cuts and the faces in each shot (up to 20 per frame), including small ones.
- Tracks each person across the shot and rates each track: `readable`, `marginal` or `unreadable`, with reasons.
- Runs a lip-reading model (Auto-AVSR, visual only) on the readable and marginal tracks (unreadable ones only if you force it).
- Gives each word a start time, end time, confidence, and tier.
- Boosts names you give it (a roster of players and coaches). Boosted words are marked, and the model's original word is kept.
- Writes JSON, SRT, WebVTT, TXT, an annotated MP4 with a neon overlay, and face thumbnails.
- Ships a local web UI and a CLI. Both run the same pipeline.

### How accurate is it?

Short answer: **usable on clear, frontal faces, unreliable elsewhere, and the confidence score
tells you which words to doubt.** The numbers below are the ones we measured. Nothing else here is
a result.

| What we measured | Result | What it means |
|---|---|---|
| LRS3 test set, 150 clips, greedy decoding | **31.6% word error rate (WER)** (28.6% on a 100-clip rerun with the packaged decoder) | LRS3 is clean TED and TEDx talking heads. This is the standard benchmark. |
| Word confidence vs. correctness (LRS3, 761 words) | conf ≥ 0.9: **96% correct** (59% of words) · 0.7–0.9: **74%** · 0.5–0.7: **50%** · < 0.5: **32%** | This is why words show as high / medium / low. Low words get a "?". |
| Raw talking-head clips, scored against Whisper transcripts | **6% to 39% WER** on 5 of 7 frontal clip segments (69% and 76% on the two hardest, one full of names) | The reference is an automatic transcript, not a human. Treat it as a rough guide. |
| Synthetic "sideline" wide shot (3 people + a camera cut) | 182 px face: near-perfect sentence · 85 px face: partly right, low confidence · 29 px face: flagged **unreadable** (confidence 0.10) | The readability gate stops it inventing words for tiny faces. |
| Face-size study (faces shrunk into a 1280x720 frame, 2 clips) | Accuracy roughly flat from 200 px down to ~40 px, worse at 28 px, collapses at 20 px | Thresholds: **56 px+ readable, 32–56 px marginal, < 32 px unreadable**. Set conservatively because real broadcast footage is blurrier. |
| Name booster, 50-name sports roster on 100 LRS3 clips | **1 false boost in 761 words (0.13%)**; fixes near-misses (HUMAN → HUMILITY, RADIOIL → RADIO) | It cannot rescue a name the model never considered (GENERAL YOUNG BLOOD stayed, not GERALD YOUNGBLOOD). Every boost is flagged. |
| Speed | LRS3 model only: **7.9x real time**. Full pipeline: 9 s single-face clip in **16 s**; 12 s three-face 720p clip in **31 s** (incl. rendering) | Measured on a 2-core cloud sandbox, not a Mac. Speed on Apple Silicon has not been measured yet. |

Published numbers, not measured here: the Auto-AVSR paper reports **19.1% WER** on LRS3 visual
speech with beam search and a language model (Ma et al., ICASSP 2023). LipRead v0.1 uses greedy
decoding and a different test subset, so the two are not directly comparable.

#### What makes it fail

- **Side views.** A face turned away from the camera loses the lips. The readability gate checks yaw and marks these down.
- **Tiny faces.** MediaPipe alone lost the face at 120 px and below in our test. LipRead pairs it with YuNet, which finds smaller faces. In a face-size study accuracy held steady down to about 40 px and collapsed at 20 px; faces under 32 px are marked unreadable and not decoded unless you force it.
- **Covered mouths.** Hands, microphones, masks, and cups hide the lips.
- **Low frame rate.** The model runs at 25 fps. Footage well below that smears mouth movement.
- **Blur and heavy compression.** Motion blur on fast sports footage hurts accuracy.
- **Crowds and overlap.** Several people talking at once, or a crowd chanting, are hard to separate.
- **Other languages.** The model was trained on English (LRS3). Output in other languages is not reliable.
- **Confident mistakes.** The model can be sure and still wrong. Confidence is a signal, not a guarantee.

**Treat every word as an estimate until a person has checked it.**

### Quickstart (macOS, Apple Silicon)

#### One-click setup

```bash
bash scripts/install_mac.sh          # installs ffmpeg and Python 3.12 if needed, sets up .venv, asks before downloading the model
bash scripts/make_mac_launcher.sh    # puts LipRead.app on your Desktop with a custom icon
```

Double-click **LipRead** on your Desktop from then on. It opens iTerm2, starts the server, and opens
http://127.0.0.1:8765 in your browser. The first time, macOS asks whether LipRead may control iTerm2.
Allow it. Install iTerm2 with `brew install --cask iterm2`. If iTerm2 is missing, the launcher falls
back to Terminal.app so it still starts.

#### By hand

```bash
brew install ffmpeg                                   # video decoding
# Python 3.11 or newer must be the python3 on your PATH (brew install python@3.12 works)
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[url]"                               # ".[url]" adds yt-dlp for http(s) links
lipread models download --accept-license              # prints the license, records your acceptance; weights are never bundled
lipread serve                                         # then open http://127.0.0.1:8765
```

### Command line

```bash
lipread scan game.mp4                                  # list shots and faces, no lip reading
lipread run game.mp4 -o out/game                       # read every readable face
lipread run game.mp4 --start 95 --end 130              # trim to a window (seconds)
lipread run sideline.mp4 --roster chiefs.txt           # boost names from a roster file
lipread run sideline.mp4 --names "Smith,Jones"         # or pass names on the command line
lipread run clip.mp4 --faces 1,3 --no-video            # only track IDs 1 and 3, no annotated MP4
lipread run clip.mp4 --faces largest                   # just the biggest face
lipread run clip.mp4 --force-unreadable                # decode faces marked unreadable (labelled as such)
lipread models list                                    # show the models and their licenses
```

`--faces` takes `readable` (default), `all`, `largest`, `none`, or a comma list of track IDs. A roster file has one name per line, or is a JSON list; `#` starts a comment.
`-o` defaults to `lipread_out/<name>`.
`SOURCE` can be a file or an http(s) URL. URLs need the `url` extra. Only download video you have
the right to use (see [ETHICS.md](ETHICS.md)).

### Output files

Everything goes to the output folder (`-o`):

- **`lipread.json`:** every track, utterance and word, with start, end, confidence, tier, and whether a word was boosted. Every result carries the disclaimer.
- **`lipread.srt` and `lipread.vtt`:** captions in the form video players expect.
- **`lipread.txt`:** a plain transcript.
- **`annotated.mp4`:** the source with boxes, labels, and confidence overlaid (skip with `--no-video`).
- **`thumbs/` and `crops/`:** a face thumbnail and a mouth-crop strip per track, so you can see exactly what the model was given.
- **`source/` and `cache/`:** the downloaded input and cached model output. Delete them to free disk space once you are done.

### Known limits (desktop, v0.1)

- **Names the model never considered can't be boosted.** The roster booster fixes near-misses only (see the table above).
- **URL safety:** links to private or local network addresses are refused, but redirects after the first URL are not re-checked. The web app only accepts requests from its own page on 127.0.0.1.
- **Clip length:** windows longer than 15 minutes are refused (`LIPREAD_MAX_SECONDS`); URL downloads are capped at 30 minutes and live streams are refused.
- **Not yet run on a Mac.** Everything was built and tested in a Linux sandbox; the Mac installer and Desktop launcher are untested.
- **No sports benchmark yet.** Use the web app's "Edit / correct" button to build your own labelled set (`eval/labels_bench.py` scores it).

### Responsible use

LipRead outputs are **estimates, not quotes**. Check them against the video before you publish
anything. Do not use it to read the lips of private people in crowds, do not attach invented words
to a named person, and do not use it to harass or profile anyone. Respect the terms of the footage
you use. The full policy is in [ETHICS.md](ETHICS.md).

### Project layout

```
web/                the phone app: static PWA, runs entirely in the browser (see docs/PHONE.md)
src/lipread/        the package: ingest, shots, faces, tracker, align, quality, engine, decode, booster, export, render, pipeline, cli
src/lipread/web/    FastAPI server and the neon web UI
decoder/            Rust: word language model + context-aware beam search, compiled to WebAssembly for the phone
proofs/             Lean 4: machine-checked proof that long videos are split with no skipped frames
eval/               benchmark scripts (LRS3 and Whisper references)
tests/              unit tests (model tests are marked and need the download)
scripts/            install_mac.sh, make_mac_launcher.sh, LipRead.command, icon, build_lm.py, build_wasm.sh
docs/               ARCHITECTURE.md, PHONE.md, ROADMAP.md, PLAN.md
PRIVACY.md          privacy notes for the phone app
```

#### Rust and Lean

- **Rust (`decoder/`):** the part that weighs several candidate words and picks the most likely one from
  the words around it. Built to WebAssembly by `scripts/build_wasm.sh` (283 KB, 105 KB gzipped) and used by
  the phone app, with the identical JavaScript decoder as a fallback. It gives exactly the same output as
  the Python reference on every test fixture and runs about 2.5x faster than the JavaScript version at
  beam 16 and 3x faster at beam 64 (desktop Node 22, 20 clips). Test with `cd decoder && cargo test --release`.
- **Lean (`proofs/Windows.lean`):** a proof that the 20-second windows used for long videos cover every
  frame and stay inside the video, checked by Lean 4 on every change (`.github/workflows/proofs.yml`).
  It also shows the extra end window in `stitch_windows` is never needed.

Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the desktop pipeline, [docs/PHONE.md](docs/PHONE.md)
for the phone app, and [docs/ROADMAP.md](docs/ROADMAP.md) for what is next.

### License

- **Code:** Apache-2.0, including the phone app in `web/`. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Auto-AVSR preprocessing is ported with attribution.
- **Model weights:** not included. The desktop app downloads them at runtime under research and non-commercial terms (LRS3), and the phone app downloads the lip-reading model from Hugging Face on first run. See [MODEL_LICENSES.md](MODEL_LICENSES.md).
