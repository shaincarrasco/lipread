# Vendored third-party runtime

`ort/` holds the ONNX Runtime Web (WASM) files the app runs its face detector and lip model with.
They are not in git (about 15 MB). Nothing here is loaded from a CDN, so the app runs offline and no
third-party script executes next to the user's video.

## Get the files

```bash
bash scripts/vendor_ort.sh
```

The script packs `onnxruntime-web@1.31.1` from npm, checks the tarball against the registry's integrity
hash, and copies three files into `web/vendor/ort/`:

- `ort.wasm.min.mjs`: the loader the app imports (`core/config.js`, `ortModule`)
- `ort-wasm-simd-threaded.mjs`: the Emscripten glue the loader imports
- `ort-wasm-simd-threaded.wasm`: the WASM runtime

It also writes `ort/SHA256SUMS`. The script fails if the loader needs a file it did not copy.
The Pages workflow runs it before upload, so a deployed site always has these files. The service worker
precaches them as required files, so an install without them fails and the previous worker keeps serving.

## License

onnxruntime-web is MIT licensed (Microsoft). The npm package's `package.json` declares it; the
upstream repository is microsoft/onnxruntime.

## Upgrading

Change `ORT_VERSION` in `scripts/vendor_ort.sh`, then rerun it. Keep the three file names in `FILES` and in
the service worker's `REQUIRED_PATHS` (`web/sw.js`) in step with the loader.
