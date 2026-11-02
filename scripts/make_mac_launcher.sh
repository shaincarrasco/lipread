#!/bin/bash
# Builds ~/Desktop/LipRead.app, a double-clickable launcher with a custom icon.
# The app runs scripts/LipRead.command from this repo. Re-run this script if you move the repo.
#
# Optional: set LIPREAD_APP_DIR to build the app somewhere other than the Desktop.
set -euo pipefail

say() { printf '==> %s\n' "$*"; }
die() { printf 'Error: %s\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = "Darwin" ] || die "This builds a macOS app. Run it on a Mac."

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
APP_DIR="${LIPREAD_APP_DIR:-$HOME/Desktop}"
APP="$APP_DIR/LipRead.app"
ICON_PNG="$HERE/icon_1024.png"
ICON_SVG="$HERE/icon.svg"

[ -f "$REPO/scripts/LipRead.command" ] || die "Missing scripts/LipRead.command in $REPO."
[ -f "$ICON_PNG" ] || die "Missing $ICON_PNG."
[ -d "$APP_DIR" ] || die "$APP_DIR does not exist."

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# 1. Icon source: re-render the SVG if rsvg-convert is installed, else use the committed PNG.
SRC_PNG="$ICON_PNG"
if command -v rsvg-convert >/dev/null 2>&1 && [ -f "$ICON_SVG" ]; then
    say "Rendering icon from icon.svg"
    rsvg-convert -w 1024 -h 1024 "$ICON_SVG" -o "$WORK/icon_1024.png"
    SRC_PNG="$WORK/icon_1024.png"
fi

# 2. PNG sizes -> .iconset -> .icns
say "Building AppIcon.icns"
ICONSET="$WORK/AppIcon.iconset"
mkdir -p "$ICONSET"
while read -r size name; do
    sips -z "$size" "$size" "$SRC_PNG" --out "$ICONSET/$name" >/dev/null
done <<'SIZES'
16 icon_16x16.png
32 icon_16x16@2x.png
32 icon_32x32.png
64 icon_32x32@2x.png
128 icon_128x128.png
256 icon_128x128@2x.png
256 icon_256x256.png
512 icon_256x256@2x.png
512 icon_512x512.png
1024 icon_512x512@2x.png
SIZES
iconutil -c icns "$ICONSET" -o "$WORK/AppIcon.icns"

# 3. The bundle
say "Writing $APP"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$WORK/AppIcon.icns" "$APP/Contents/Resources/AppIcon.icns"

cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>LipRead</string>
    <key>CFBundleDisplayName</key><string>LipRead</string>
    <key>CFBundleIdentifier</key><string>dev.liplread.launcher</string>
    <key>CFBundleExecutable</key><string>LipRead</string>
    <key>CFBundleIconFile</key><string>AppIcon</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>0.1.0</string>
    <key>CFBundleVersion</key><string>1</string>
    <key>LSMinimumSystemVersion</key><string>12.0</string>
    <key>LSUIElement</key><true/>
    <key>NSHighResolutionCapable</key><true/>
    <key>NSAppleEventsUsageDescription</key><string>LipRead uses iTerm2 to start its local server on this Mac.</string>
</dict>
</plist>
PLIST

{
    echo '#!/bin/bash'
    printf 'export LIPREAD_REPO=%q\n' "$REPO"
    echo 'exec /bin/bash "$LIPREAD_REPO/scripts/LipRead.command"'
} > "$APP/Contents/MacOS/LipRead"
chmod +x "$APP/Contents/MacOS/LipRead" "$REPO/scripts/LipRead.command"

touch "$APP"
say "Done. Double-click LipRead on your Desktop. If the old icon still shows, wait a moment or log out and back in."
