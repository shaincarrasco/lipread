#!/bin/bash
# LipRead launcher.
#
# Launcher mode (default): opens iTerm2, starts `lipread serve` in it, waits until the
# web UI answers, then opens http://127.0.0.1:8765 in the default browser.
# Serve mode (--serve): runs inside iTerm2; activates .venv and runs the server.
#
# Use LipRead.app on the Desktop (made by make_mac_launcher.sh). Double-clicking this
# file makes macOS open it in Terminal.app, which is not what we want.

set -u

PORT="${LIPREAD_PORT:-8765}"
URL="http://127.0.0.1:${PORT}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${LIPREAD_REPO:-$(cd "$HERE/.." && pwd)}"
SELF="$REPO/scripts/LipRead.command"

alert() {
    osascript -e "display dialog \"$1\" with title \"LipRead\" buttons {\"OK\"} default button \"OK\" with icon stop" >/dev/null 2>&1 \
        || echo "$1" >&2
}

as_escape() {
    # Escape a string for use inside an AppleScript "..." literal.
    printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'
}

# ---- serve mode: runs inside iTerm2 -------------------------------------------
if [ "${1:-}" = "--serve" ]; then
    cd "$REPO" || { echo "Cannot cd to $REPO"; read -r -p "Press return to close. " _; exit 1; }
    if [ ! -f ".venv/bin/activate" ]; then
        echo "No .venv in $REPO. Run scripts/install_mac.sh first."
        read -r -p "Press return to close. " _
        exit 1
    fi
    # shellcheck disable=SC1091
    . .venv/bin/activate
    exec lipread serve --port "$PORT"
fi

# ---- launcher mode ------------------------------------------------------------
# Already running? Just open the page.
if curl -fsS --max-time 2 -o /dev/null "$URL/" 2>/dev/null; then
    open "$URL"
    exit 0
fi

if [ ! -f "$REPO/.venv/bin/activate" ]; then
    alert "LipRead is not installed yet. Run scripts/install_mac.sh in $REPO first."
    exit 1
fi

SERVE_CMD="cd $(printf %q "$REPO") && LIPREAD_REPO=$(printf %q "$REPO") /bin/bash $(printf %q "$SELF") --serve"
CMD_AS="$(as_escape "$SERVE_CMD")"

if [ -d "/Applications/iTerm.app" ] || [ -d "$HOME/Applications/iTerm.app" ]; then
    osascript <<OSA >/dev/null
tell application "iTerm"
    activate
    set newWindow to (create window with default profile)
    tell current session of newWindow
        write text "$CMD_AS"
    end tell
end tell
OSA
else
    # iTerm2 is not installed: fall back to Terminal.app so the server still starts.
    osascript <<OSA >/dev/null
tell application "Terminal"
    activate
    do script "$CMD_AS"
end tell
OSA
fi

# Wait for the server (model loading can take a few seconds).
for _ in $(seq 1 90); do
    if curl -fsS --max-time 2 -o /dev/null "$URL/" 2>/dev/null; then
        open "$URL"
        exit 0
    fi
    sleep 1
done

alert "LipRead did not start within 90 seconds. Check the iTerm2 window for the error."
exit 1
