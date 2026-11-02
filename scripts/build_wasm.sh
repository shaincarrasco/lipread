#!/usr/bin/env bash
# Builds the Rust decoder (decoder/) to WebAssembly for the phone app.
#
#   scripts/build_wasm.sh
#
# Output (committed): web/core/wasm/lipread_decoder.js + web/core/wasm/lipread_decoder_bg.wasm
# Needs: rustup target add wasm32-unknown-unknown; cargo install wasm-bindgen-cli --version 0.2.104
# (the CLI version must equal the wasm-bindgen crate version pinned in decoder/Cargo.toml).
# Release profile (decoder/Cargo.toml): opt-level 3, lto, codegen-units 1, panic abort.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/web/core/wasm"
WANT="0.2.104"

have="$(wasm-bindgen --version 2>/dev/null | awk '{print $2}')" || true
if [ "$have" != "$WANT" ]; then
  echo "wasm-bindgen-cli $WANT is required (found: ${have:-none}): cargo install wasm-bindgen-cli --version $WANT" >&2
  exit 1
fi

cd "$ROOT/decoder"
cargo build --release --target wasm32-unknown-unknown --lib
wasm-bindgen --target web --no-typescript --out-dir "$OUT" \
  target/wasm32-unknown-unknown/release/lipread_decoder.wasm

for f in "$OUT"/lipread_decoder.js "$OUT"/lipread_decoder_bg.wasm; do
  printf '%8d bytes  %s\n' "$(wc -c < "$f")" "${f#$ROOT/}"
done
