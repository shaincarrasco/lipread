#!/bin/bash
# One-time setup for LipRead on macOS.
#
# Checks Homebrew, installs ffmpeg and Python 3.12 if they are missing, creates .venv,
# installs LipRead, then shows the model license and asks before downloading the model.
# Safe to run again.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

say() { printf '\n==> %s\n' "$*"; }
die() { printf '\nError: %s\n' "$*" >&2; exit 1; }
ask() {
    local reply
    read -r -p "$1 [y/N] " reply || reply=""
    [[ "$reply" =~ ^[Yy]$ ]]
}

[ "$(uname -s)" = "Darwin" ] || die "This script is for macOS."
ARCH="$(uname -m)"
if [ "$ARCH" != "arm64" ]; then
    echo "Note: this Mac is $ARCH. LipRead was developed on Apple Silicon, so expect it to be slower."
fi

# ---- Homebrew -----------------------------------------------------------------
say "Checking Homebrew"
if ! command -v brew >/dev/null 2>&1; then
    for b in /opt/homebrew/bin/brew /usr/local/bin/brew; do
        if [ -x "$b" ]; then eval "$("$b" shellenv)"; break; fi
    done
fi
command -v brew >/dev/null 2>&1 || die "Homebrew is not installed. Install it from https://brew.sh, then run this script again."
echo "Homebrew: $(command -v brew)"

# ---- ffmpeg -------------------------------------------------------------------
say "Checking ffmpeg"
if command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg already installed."
else
    brew install ffmpeg
fi

# ---- Python 3.11+ ---------------------------------------------------------------
say "Checking Python 3.11 or newer"
PY=""
for cand in python3.12 python3.11 python3.13; do
    if command -v "$cand" >/dev/null 2>&1; then PY="$(command -v "$cand")"; break; fi
done
if [ -z "$PY" ]; then
    brew install python@3.12
    PY="$(brew --prefix python@3.12)/bin/python3.12"
fi
[ -x "$PY" ] || die "Could not find a Python 3.11+ interpreter."
echo "Using $PY ($("$PY" --version))"

# ---- Virtual environment and package --------------------------------------------
say "Creating .venv and installing LipRead"
if [ ! -f ".venv/bin/activate" ]; then
    "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[url]"

# ---- Model -------------------------------------------------------------------
say "Lip-reading model"
cat <<'NOTE'
The default lip-reading model (Auto-AVSR, trained on LRS3) is licensed for
RESEARCH AND NON-COMMERCIAL USE ONLY. It is downloaded to this Mac and is never
bundled with LipRead. Full details: MODEL_LICENSES.md
NOTE
if ask "Download the model now?"; then
    lipread models download --accept-license
else
    echo "Skipped. Download later with: lipread models download --accept-license"
fi

# ---- iTerm2 (used by the Desktop launcher) ---------------------------------------
say "Terminal for the Desktop launcher"
if [ -d "/Applications/iTerm.app" ] || [ -d "$HOME/Applications/iTerm.app" ]; then
    echo "iTerm2 already installed."
elif ask "Install iTerm2 (the launcher opens it; without it the launcher falls back to Terminal.app)?"; then
    brew install --cask iterm2
fi

# ---- Desktop launcher -------------------------------------------------------------
say "Desktop launcher"
if ask "Create LipRead.app on your Desktop (one-click start, custom icon)?"; then
    bash "$REPO/scripts/make_mac_launcher.sh"
fi

say "Done. Start the web UI with: lipread serve, then open http://127.0.0.1:8765"
