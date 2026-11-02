# LipRead evaluation

There is no labelled sports lip-reading set, so the engine is measured three ways:

| Script | What it measures | Reference |
|---|---|---|
| `lrs3_bench.py` | Engine accuracy on the standard LRS3 test set, speed, and the **confidence calibration** used to set thresholds | LRS3 transcripts |
| `audio_bench.py` | Real video end to end (`lipread.pipeline.run`), WER by face size | Whisper transcript of the same audio (pseudo ground truth) |
| `face_size_study.py` | How WER and detection rate fall as the face shrinks in a wide shot | Whisper transcript of the original clip |
| `labels_bench.py` | Your own hand labels (from the web UI) scored against a cached pipeline result | Your labels |

All scripts print a table. `lrs3_bench`, `audio_bench` and `face_size_study` also write a JSON report to `eval/reports/` (git-ignored, change with `--out`); `labels_bench` writes JSON only with `--out`.
Run them from the repo root with the project venv (`pip install -e ".[eval]"`).

Text is normalised the same way everywhere: lowercase, keep `a-z` and apostrophes, drop digits and punctuation.
Numbers are not converted to words, so a spoken "two" against a Whisper "2" counts as an error.

## lrs3_bench.py

```
python eval/lrs3_bench.py --n 200 --model /path/to/lipread_ctc.int8.onnx
python eval/lrs3_bench.py --parquet /path/to/lrs3_test_0.parquet --n 0    # every clip in the file
```

Prints corpus WER, mean per-clip WER, speed in x real time, and two tables:

- **Calibration**: every hypothesis word is binned by `Word.confidence` (as defined by `lipread.decode.greedy_words`).
  A word is correct when `jiwer.process_words` aligns it as `equal` to a reference word.
  Use the bins to choose the threshold for "show as confident" and the tier boundaries.
- **Threshold**: for `t` in 0.5, 0.7, 0.9, the share of words kept and the precision of the kept words.

Speed depends on machine load, so benchmark on an idle machine.

## audio_bench.py

```
python eval/audio_bench.py /path/to/clips --whisper small.en --max-seconds 20
python eval/audio_bench.py /path/to/clips --model /path/to/lipread_ctc.int8.onnx --max-seconds 8
```

For each video, Whisper transcribes the first `--max-seconds` seconds (faster-whisper, int8, CPU). The pipeline then
lip-reads the same range (`faces="largest"`, no video rendered). Prints a per-clip table and WER by face-height bucket
(`<45`, `45-80`, `80-150`, `150+` px). Clips with no decoded face track are listed and excluded from WER, since
there is nothing to read (for example a voiceover over a graphic).

Notes:
- The default engine requires accepting the model licence (`lipread models download --accept-license`). `--model` injects
  a local ONNX file instead, which is how the bench runs before that is accepted.
- Job folders go to `~/.cache/lipread-eval/audio_bench/<clip>/` by default (`--work`). Each keeps `lipread.json`, which
  `labels_bench.py` reads.
- Whisper is a proxy. Its own errors add noise, so use the numbers to compare settings rather than to quote accuracy.

## face_size_study.py

```
python eval/face_size_study.py /path/to/clips --max-seconds 8
python eval/face_size_study.py /path/to/clips --targets 120,80,40 --blur 5
python eval/face_size_study.py /path/to/clips --fps 50          # high-fps source, see below
```

Each clip is scaled so its face is `target` px tall and centred on a 1280x720 canvas. Then the package's
`FaceDetector`, `track_faces`, `mouth_crops` and the engine read it. The output is WER against the Whisper transcript of
the original clip, plus detection rate per target. A target larger than the clip's own face is not applicable and is
skipped (we never upscale).

- A clip where nothing is read counts as an empty hypothesis (100% WER), because that is what a viewer would get. The
  table also gives WER over the clips that were read.
- `--blur K` applies a horizontal motion-blur kernel K px wide to the shrunk face (canvas pixels).
- `--fps 50` reads a 50 fps source and averages each pair of frames into one 25 fps frame, which is what a 25 fps camera
  with a full-frame exposure would record. On a 25 fps source it changes nothing.
- If `lipread.faces` cannot be imported, the script uses the reference hybrid detector (YuNet + MediaPipe refinement)
  and prints `detector: reference` in its header line.
- Memory: the wide-shot grey frames for one target are held in RAM (about 0.9 MB per frame at 1280x720, so about 23 MB per second of video).

## labels_bench.py: score your own sports test set

Use this to build a test set from real footage, such as press conferences or sideline interviews where the audio is
clean enough to hear what was said.

1. Run the clip through the web UI or CLI so the job folder holds `lipread.json`.
2. In the web UI, label the tracks you can hear. Each label is one line in the job folder's `labels.jsonl`:
   `{"source": "/path/clip.mp4", "track_id": 3, "start": 12.4, "end": 15.1, "text": "we played hard tonight"}`
   Label what a human hears with the audio on, not what the lip reader produced. Keep ranges short (1 to 5 s) and skip
   lines you cannot make out, or write them as the best guess and accept the WER.
3. Score them:

```
python eval/labels_bench.py /path/to/job_folder/labels.jsonl --out eval/reports/my_labels.json
```

For each label it reads the utterances on that track that overlap `[start, end]`, keeps the words whose midpoint lies in
the range, and reports WER and the reference and hypothesis side by side. It also prints the calibration table for your
clips. Use that table to check that the thresholds still hold for sports footage, which is the point of the exercise.

Labels from a different source file than the result are flagged and still scored. Labels for a track that is not in the
result score as an empty hypothesis (100% WER), so check the `[track not in result]` note.

## Current numbers

See the accuracy table in the top-level README.md. Those are the numbers measured for this release; the scripts here reproduce them.
