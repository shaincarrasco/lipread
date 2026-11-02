# Model and asset licenses

LipRead's **code** is Apache-2.0 (see `LICENSE` and `NOTICE`). The **lip-reading
weights are not** Apache-2.0. Read this file before you use LipRead commercially,
fine-tune a model, or redistribute anything it produces.

## Summary

| Item | What it is | License / terms | In this repo? |
|---|---|---|---|
| LipRead source code (`src/lipread`, `scripts`, `eval`, `tests`) | This project | Apache-2.0 | Yes |
| Auto-AVSR code (https://github.com/mpc001/auto_avsr) | Source of the preprocessing ported in `align.py` | Apache-2.0 | Ported code only, with attribution |
| `20words_mean_face.npy` | 68-point mean face used for alignment | From Auto-AVSR (Apache-2.0) | Yes (`src/lipread/assets`) |
| `unigram5000.model` | SentencePiece unigram model for the output vocabulary | From Auto-AVSR (Apache-2.0) | Yes (`src/lipread/assets`) |
| `tokens.json` | Output token table | Comes from the ONNX export; follows the model terms below | Yes (`src/lipread/assets`) |
| Auto-AVSR LRS3 visual-only weights, int8 ONNX | The lip-reading model | **Research / non-commercial only** (LRS3 terms; see below) | **No**, downloaded at runtime |
| YuNet face detector (OpenCV Zoo; Hugging Face `opencv/face_detection_yunet`) | Face detection | MIT | **No**, downloaded at runtime |
| MediaPipe BlazeFace short-range model | Face detection and refinement | Distributed by Google MediaPipe (MediaPipe code is Apache-2.0; check the model card for the model's own terms) | **No**, downloaded at runtime |
| LRS3 dataset | Training data of the lip-reading model | See below | **No** |
| Whisper (faster-whisper) | Reference transcripts for the `eval/` suite only | MIT (upstream; not re-checked for this build) | **No**, optional `eval` extra |

## Lip-reading model (the important one)

- **Model:** Auto-AVSR, visual-only (VSR) model trained on LRS3, exported to int8 ONNX:
  https://huggingface.co/eschmechel/auto-avsr-lrs3-vsr-int8-onnx
- **Paper:** Ma et al., "Auto-AVSR: Audio-Visual Speech Recognition with Automatic Labels", ICASSP 2023.
- **Terms LipRead relies on:** the model was trained on LRS3, and its terms are for research and
  non-commercial use. The upstream Auto-AVSR README also says pre-trained models "may have their own
  licenses or terms and conditions derived from the dataset used for training."
- **What LipRead does:** `lipread models download --accept-license` fetches the weights into the
  local models folder. The weights are never committed or bundled. The download command makes you
  accept the terms before it runs.
- **Before any commercial use:** get your own rights to the weights, or replace the engine with a
  model whose license allows it. The `Engine` interface in `src/lipread/engine.py` is built for that swap.
- The Hugging Face model card of the int8 export states the license `lrs3-research-non-commercial`:
  the quantized copy inherits the LRS3 terms (research and non-commercial use only).

## LRS3 dataset

- LRS3 (TED and TEDx talks) is the training and test data of the lip-reading model.
- The KAIST mirror of the dataset lists **CC BY 4.0** for the download and **CC BY-NC-ND 4.0** for the
  content (https://mm.kaist.ac.kr/datasets/lip_reading/files/license.txt).
- **BY-NC-ND means no commercial use and no derivatives.** Do not fine-tune on LRS3, and do not
  distribute a fine-tuned model, until you have checked the terms with the rights holders.
- LipRead's `eval/` suite uses LRS3 test clips as a benchmark. Those clips are not in this repo.

## Other components

- **YuNet (OpenCV Zoo):** MIT. Downloaded at runtime.
- **MediaPipe BlazeFace:** distributed by Google MediaPipe (code Apache-2.0; see the model card for the model's terms). Downloaded at runtime.
- **MediaPipe (Python package):** Apache-2.0.
- **ffmpeg:** installed with Homebrew on your Mac. Its license depends on how it was built (LGPL or GPL).
  LipRead does not ship ffmpeg.
- **yt-dlp** (optional `url` extra): Unlicense (upstream; not re-checked for this build). Downloading from a site may break that site's terms. See ETHICS.md.

## Your outputs

Transcripts, captions and annotated videos LipRead writes are yours to handle, but they come from a
model that is research-licensed and from footage that may be copyrighted. See ETHICS.md before you share them.
