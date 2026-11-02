// Dev check (removed from the Pages build). Loaded as an external module because the app CSP
// forbids inline scripts. A module worker imports the vendored ONNX Runtime and opens the YuNet face
// detector. Result goes to #out and window.__ortTest.
const out = document.getElementById("out");
const base = new URL("../", location.href).href;            // web/
const ortModule = new URL("vendor/ort/ort.wasm.min.mjs", base).href;
const ortPaths = new URL("vendor/ort/", base).href;
const model = new URL("core/assets/face_detection_yunet_2023mar.onnx", base).href;

const workerSource = `
  import * as ort from ${JSON.stringify(ortModule)};
  self.onmessage = async ({ data }) => {
    try {
      ort.env.wasm.wasmPaths = ${JSON.stringify(ortPaths)};
      ort.env.wasm.numThreads = self.crossOriginIsolated ? 2 : 1;
      const buf = await (await fetch(data.model)).arrayBuffer();
      const s = await ort.InferenceSession.create(buf, { executionProviders: ["wasm"] });
      postMessage({ ok: true, isolated: self.crossOriginIsolated, threads: ort.env.wasm.numThreads,
                    inputs: s.inputNames, outputs: s.outputNames });
    } catch (err) {
      postMessage({ ok: false, error: String((err && err.stack) || err) });
    }
  };`;

const url = URL.createObjectURL(new Blob([workerSource], { type: "text/javascript" }));
const worker = new Worker(url, { type: "module" });
worker.onmessage = ({ data }) => {
  const result = {
    page: { isolated: self.crossOriginIsolated, controlled: !!navigator.serviceWorker.controller },
    worker: data,
  };
  window.__ortTest = result;
  out.textContent = JSON.stringify(result, null, 2);
};
worker.onerror = (e) => {
  window.__ortTest = { error: e.message || "worker error" };
  out.textContent = "worker error: " + (e.message || "unknown");
};
worker.postMessage({ model });
  
