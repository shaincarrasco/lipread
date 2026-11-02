# Deploying the phone web app (GitHub Pages)

The phone app lives in `web/`: plain ES modules, no build step. A GitHub Actions workflow
(`.github/workflows/pages.yml`) publishes that folder to GitHub Pages on every push to `main`.

Live URL once published: `https://<user>.github.io/lipread/` (replace `<user>` with your GitHub login).
The app is served from a subpath, so every URL inside `web/` is relative. Keep it that way.

## 1. Create the repository

From the project root (the folder that contains `web/`):

```sh
git add -A && git commit -m "Phone web app"
gh repo create lipread --public --source . --push
```

GitHub Pages is free only for public repositories. A private repository needs a paid GitHub plan
for Pages.

## 2. Turn on Pages with the workflow

Either:

- In the browser: **Settings → Pages → Build and deployment → Source: GitHub Actions**, or
- From the terminal:

```sh
gh api -X POST repos/<user>/lipread/pages -f build_type=workflow
```

The first run starts on the push in step 1. Watch it under the **Actions** tab. When the
`deploy` job finishes, the URL is printed in the job summary and in **Settings → Pages**.

## 3. Install on the phone

- **iPhone (Safari):** open the URL, tap Share, then **Add to Home Screen**. Open it once while
  online so the app shell and the on-device runtime are cached. After that it works offline.
- **Android (Chrome):** open the URL, tap the menu, then **Install app** (or **Add to Home screen**).

The lip-reading model (about 203 MB) is downloaded by the app on first use. It is stored in the
browser's Cache Storage under `lipread-model-v1`. Deploys do not touch it.

## 4. Update the app

Commit and push to `main`:

```sh
git add -A && git commit -m "Describe the change" && git push
```

The workflow stamps `web/sw.js` with the commit SHA before upload, so every deploy gets a new
service-worker shell cache. Phones pick up the new files on their next visit and drop the old
shell cache. You do not need to bump a version by hand.

Test the app locally before pushing (service workers need `localhost` or HTTPS):

```sh
cd web && python3 -m http.server 8000
# open http://localhost:8000/
```

## 5. Custom domain (optional)

1. In **Settings → Pages → Custom domain**, enter your domain and save. Add the DNS record GitHub shows.
2. Add `web/CNAME` containing only the domain (for example `lipread.example.com`) so each deploy
   keeps it.

With a custom domain the app lives at the domain root. Nothing else changes.

## Troubleshooting

- **The phone shows an old version:** close the app fully, reopen it while online, then reload.
  If it persists, delete the home-screen icon and add it again.
- **Service worker status:** on desktop Chrome, open DevTools → Application → Service Workers.
- **The 404 page is shown for a valid URL:** check that the file is under `web/` and that the
  workflow's `upload-pages-artifact` step ran with `path: web`.
