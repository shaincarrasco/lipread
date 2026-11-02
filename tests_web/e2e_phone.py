"""End-to-end test of the phone web app in headless Chromium with iPhone-size emulation.

It serves web/ locally, uses a local copy of the lip-reading model (no 203 MB download), walks the
real UI (first-run sheet -> download -> choose a video -> results) and checks the outcome.

    LIPREAD_MODEL_DIR=~/.cache/lipread/models python tests_web/e2e_phone.py path/to/clip.mp4

Needs: Python Playwright with Chromium. NOTE: this is not a substitute for a real iPhone (WebKit) test.
"""
from __future__ import annotations

import functools
import http.server
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

WEB = Path(__file__).resolve().parent.parent / "web"
MODEL_DIR = Path(os.environ.get("LIPREAD_MODEL_DIR", "~/.cache/lipread/models")).expanduser()


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):  # noqa: D401
        pass


def main(clip: str) -> int:
    root = Path(tempfile.mkdtemp(prefix="lipread-e2e-"))
    shutil.copytree(WEB, root / "web", symlinks=True, ignore=shutil.ignore_patterns("dev-model", "dev-media"))
    (root / "web" / "dev-model").symlink_to(MODEL_DIR)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(root / "web")))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    failures = []
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            str(root / "profile"), viewport={"width": 393, "height": 852}, device_scale_factor=3,
            is_mobile=True, has_touch=True,
            user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1")
        ctx.add_init_script("self.LIPREAD_CONFIG = { modelUrl: location.origin + '/dev-model/lipread_ctc.int8.onnx' };")
        pg = ctx.new_page()
        errors: list[str] = []
        pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto(url)
        pg.wait_for_timeout(2500)   # service worker installs and reloads once for cross-origin isolation
        if not pg.evaluate("self.crossOriginIsolated"):
            failures.append("page is not cross-origin isolated (threads disabled)")
        pg.wait_for_selector("#firstRunSheet:not([hidden])", timeout=15000)
        pg.click("#dlBtn")
        pg.wait_for_selector("#frDone:not([hidden])", timeout=180000)
        pg.click("#frDone")
        t = time.time()
        pg.set_input_files("#fileInput", clip)
        pg.wait_for_selector("#screen-results:not([hidden])", timeout=900000)
        took = time.time() - t
        faces = pg.locator(".face__row").count()
        words = pg.evaluate("document.querySelectorAll('.face .word').length")
        pg.click(".face__row >> nth=0")
        pg.wait_for_timeout(500)
        words = pg.evaluate("document.querySelectorAll('.face .word').length")
        pg.screenshot(path=str(root / "results.png"), full_page=True)
        ctx.close()
    srv.shutdown()
    if faces < 1:
        failures.append("no face rows on the results screen")
    if words < 3:
        failures.append(f"only {words} transcript words rendered")
    if errors:
        failures.append(f"console errors: {errors[:5]}")
    print(f"processed in {took:.1f}s, {faces} face(s), {words} words; screenshot {root / 'results.png'}")
    for f in failures:
        print("FAIL:", f)
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
