# LipRead: build plan

This is the project plan as it stood on 2026-11-01, cleaned up for people. The
numbers are the ones measured during the build. Anything not listed under
"Measured" is a plan or a goal, not a result.

## Goal

Take any video and read the lips of the people in it, using an AI lip-reading model.
Work especially well on sports clips and fan footage, so you can see what players,
coaches and fans are saying.

## Approach

LipRead is a Python package (`lipread`). The pipeline:

1. Turn the video into a 25 fps frame stream.
2. Find shot cuts, so tracks never cross a cut.
3. Detect faces with YuNet, which finds small faces.
4. Refine keypoints with MediaPipe on an upscaled crop of each face. This matches the model's training.
5. Link detections into one track per person (IoU tracker, reset at cuts).
6. Align each face with Auto-AVSR preprocessing and cut 88x88 mouth crops.
7. Gate each track for readability: face size, yaw, blur, mouth motion.
8. Run the Auto-AVSR int8 ONNX engine on windows of 400 frames with a 50-frame overlap, and stitch the windows on their posteriors.
9. Decode greedy CTC words with timing and a confidence score.
10. Boost words that match a roster of names (CTC rescoring; boosted words are flagged).
11. Write JSON, SRT, WebVTT and TXT, and an annotated MP4.

The CLI and the local web UI both call one function, `pipeline.run(job)`.
Model weights are downloaded at runtime. They are never committed, because their
license is research and non-commercial only (see MODEL_LICENSES.md).

## Measured

- **LRS3 test set, 150 clips, greedy decoding:** 31.6% word error rate (WER). Speed: 7.9x real time on 2 CPU cores.
- **Confidence tracks accuracy:** the top third of clips by confidence had 8% WER; the bottom third had 64% WER.
- **Raw talking-head clips against Whisper transcripts:** 6% to 39% WER on 5 of 7 frontal clip segments (69% and 76% on the two hardest). The reference is an automatic transcript, not a human.
- **Beam search without a language model:** no gain over greedy decoding.
- **Small faces:** on one clip, a 120 px face read at 6% WER and an 80 px face at 13% WER.
- **Detection:** MediaPipe alone lost the face at 120 px and below in our test. The YuNet plus MediaPipe hybrid is the reason small faces work.

## Design rules

- Process video as a stream. Never hold all full-resolution frames of a clip in memory. The pipeline reads the video three times: detection and tracking, mouth crops, and rendering.
- Keep the model's preprocessing constants fixed. Changing them means re-running the benchmark.
- Put the engine behind one small interface, so newer models can drop in.
- Use `pyctcdecode` hotwords only with care: they duplicated words, so the booster is written in-house.
- No print in library code. Errors are `LipReadError`.

## Verification plan

- Install the package, run the test suite, and run the CLI on real clips and on a synthetic wide-shot clip.
- Compare output against Whisper references and check the rendered video frame by frame.
- Screenshot the web UI and check it at phone width.
- Check every number in the README against the measured list above.

## Considered and parked for v1

Replay fusion, scoreboard OCR as a prior, viseme output, super-resolution (it can invent
mouth shapes), jersey OCR for identity (planned for v0.3, see ROADMAP.md), KenLM (did not
install in the build environment), Gradio (a custom neon UI was chosen), and caption-first
mode (captions transcribe commentators, not the people on screen).

See ROADMAP.md for what comes next.
