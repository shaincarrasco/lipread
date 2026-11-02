# Privacy

**Last updated: 2026-11-02.** This is a plain-language summary for the LipRead phone app. It is
a draft and not legal advice. Have it reviewed before any paid or ad-supported release.

## The short version

- **Your video stays on your phone.** The app reads the frames on your device. The video is not
  uploaded to us or to anyone else.
- **No accounts, no analytics, no tracking.** You do not sign up, and the app does not measure you.
- **The app downloads one large file.** The lip-reading model (about 203 MB) comes from Hugging Face.
  Hugging Face sees that request, as any website host would.

## What the app does with your video

- It reads your video on the phone. It reads the picture frame by frame, finds faces, and cuts out a
  small grayscale image of each mouth (144 by 144 pixels) for the lip-reading model.
- The app does not send the video, the snapshots, the mouth images or the words it reads to any server.
- The app does not keep the video or the mouth images in storage. They exist in your phone's memory
  while a job runs. (Confirm this in the code before release.)
- The words the app reads are shown on your screen. They leave your phone only if you send them with
  the share sheet.

## What is stored on your phone

| What | Why | How to remove it |
|---|---|---|
| The lip-reading model (about 203 MB) | So you do not download it every time | Use the app's option to remove the model, or clear the site's data (see below) |
| The app's own files | So the app opens without a connection | Clear the site's data, or remove the Home Screen icon |
| Roster names you type | So the app can look for those names | Clear the roster in the app, or clear the site's data |

The browser keeps this data for the app's web address. Nothing is kept for us, because we do not
receive it.

## Who else sees what

- **Hugging Face** (where the model is downloaded from): receives the download request, which includes
  your IP address and browser details. Its privacy policy applies.
- **GitHub Pages** (where the app is hosted): serves the app's files and keeps standard server logs,
  such as IP address and time. GitHub's privacy statement applies.
- **This project:** receives nothing from the app. It runs no server that gets your video, names or
  results.
- **Anyone you share text with:** if you send a transcript or caption file with the share sheet, the app
  you choose receives it, and that app's policy applies.

## How to delete your data

- **Remove the model:** use the app's option to remove the model. It will download again the next time
  you need it.
- **Remove roster names:** clear the roster in the app.
- **Remove everything the app stored:** clear the site's data in your browser.
  - iPhone: Settings, then Safari, then Advanced, then Website Data. Find the app's address and remove it.
  - Android (Chrome): the Chrome menu, then Settings, then Site settings, then Storage. Find the app's
    address and clear it.
  - Paths can change between versions. These steps have not been tested on a device.
- **Remove the app icon:** press and hold the icon, then remove it.
- **Your video:** the app does not keep a copy. Delete your own copies as you normally would.

## Your rights

We hold no personal data about you, so there is little for us to give back or delete. For data held by
Hugging Face or GitHub, contact them directly, using their privacy pages.

## Changes

We will update this page and its date when the app changes how it handles data.

## Contact

Open an issue on the project's GitHub repository.
