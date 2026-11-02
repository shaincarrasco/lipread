#!/usr/bin/env bash
# Vendors the pinned onnxruntime-web build into web/vendor/ort/ so the app loads no third-party script.
# Run by the Pages workflow before upload, and once locally. The output is git-ignored, never committed.
set -euo pipefail

ORT_VERSION="1.31.1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/web/vendor/ort"
# Plain WASM execution provider with threads and SIMD. The jsep, jspi, asyncify and webgpu variants are not used.
FILES=(ort.wasm.min.mjs ort-wasm-simd-threaded.mjs ort-wasm-simd-threaded.wasm)

command -v npm >/dev/null || { echo "vendor_ort: npm is required" >&2; exit 1; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

npm pack "onnxruntime-web@${ORT_VERSION}" --pack-destination "$TMP" --silent >/dev/null
TGZ="$TMP/onnxruntime-web-${ORT_VERSION}.tgz"

# The tarball must match the registry's integrity hash before we trust anything in it.
EXPECTED="$(npm view "onnxruntime-web@${ORT_VERSION}" dist.integrity)"
ACTUAL="sha512-$(openssl dgst -sha512 -binary "$TGZ" | base64 -w0)"
if [ "$EXPECTED" != "$ACTUAL" ]; then
  echo "vendor_ort: integrity mismatch for onnxruntime-web@${ORT_VERSION}" >&2
  exit 1
fi

mkdir -p "$TMP/pkg"
tar -xzf "$TGZ" -C "$TMP/pkg"
SRC="$TMP/pkg/package"

# Fail if the loader needs a runtime file this script does not copy.
for ref in $(grep -ohE 'ort[-A-Za-z0-9.]*\.(mjs|wasm)' "$SRC/dist/ort.wasm.min.mjs" "$SRC/dist/ort-wasm-simd-threaded.mjs" | sort -u); do
  case " ${FILES[*]} " in
    *" $ref "*) ;;
    *) echo "vendor_ort: loader references $ref, which is not in FILES" >&2; exit 1 ;;
  esac
done

rm -rf "$DEST"
mkdir -p "$DEST"
for f in "${FILES[@]}"; do
  cp "$SRC/dist/$f" "$DEST/$f"
done
(cd "$DEST" && sha256sum "${FILES[@]}" > SHA256SUMS)

echo "vendor_ort: onnxruntime-web@${ORT_VERSION} -> web/vendor/ort"
(cd "$DEST" && sha256sum -c SHA256SUMS)
