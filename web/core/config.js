// LipRead on-device configuration. Everything the core needs to know lives here.
// A page can override any value before loading the core: window.LIPREAD_CONFIG = { ... }.

const overrides = (typeof self !== "undefined" && self.LIPREAD_CONFIG) || {};

export const CONFIG = {
  // Lip-reading model: Auto-AVSR (Ma et al., ICASSP 2023), LRS3 visual-only, int8 ONNX, encoder + CTC.
  // Research / non-commercial use only (LRS3 terms). Fetched once from Hugging Face, never bundled.
  modelUrl: "https://huggingface.co/eschmechel/auto-avsr-lrs3-vsr-int8-onnx/resolve/main/lipread_ctc.int8.onnx",
  modelSha256: "da02d72ee472eb3d7d50f7dd309bfe81973acb4cac8bb770dd86aa97872f195b",
  modelBytes: 203357712,
  modelName: "Auto-AVSR LRS3 visual-only (int8 ONNX)",
  modelLicense: "Research and non-commercial use only (LRS3 / Auto-AVSR terms)",
  modelCache: "lipread-model-v1",

  // Face detector shipped with the app (OpenCV Zoo YuNet 2023mar, MIT, 230 KB, fixed 640x640 input).
  yunetUrl: new URL("./assets/face_detection_yunet_2023mar.onnx", import.meta.url).href,

  // onnxruntime-web (pinned). The worker imports this module and loads its .wasm from the same folder.
  ortModule: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort.wasm.min.mjs",
  ortWasmPaths: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/",

  fps: 25,               // model frame rate
  maxSeconds: 30,        // longest window processed on a phone
  maxSide: 1920,         // frames are drawn at most this large (long side)
  detectStride: 2,       // run the face detector every Nth frame, track in between
  detectScore: 0.6,
  nmsIou: 0.3,
  maxFaces: 12,          // faces tracked at once (largest first)
  minTrackFrames: 13,    // ignore faces seen for less than ~0.5 s
  cropSize: 144,         // stored mouth-region crop (px), covers cropSpan x inter-eye distance
  cropSpan: 2.6,
  chunkFrames: 400,      // model window, frames (model limit 500)
  chunkOverlap: 50,
  ...overrides,
};

// Readability gate (measured face-size study, 2026-11-01) and confidence tiers (LRS3 calibration).
export const GATE = {
  facePxReadable: 56, facePxMarginal: 32,
  yawReadable: 30, yawMarginal: 50,
  coverageMin: 0.5, sharpMarginal: 15, motionMin: 0.01,
};
export const TIERS = { high: 0.90, medium: 0.70 };   // >=0.9: 96% correct, 0.7-0.9: 74% (LRS3, 761 words)
export const UTTERANCE_GAP = 0.6;
export const EDGE_PAD = 0.25;

export const assetUrl = (name) => new URL(`./assets/${name}`, import.meta.url).href;
