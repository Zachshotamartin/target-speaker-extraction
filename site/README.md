# One Voice product site

A public listening and extraction interface. It does not expose or import the training dashboard, trainer, pause/resume controls, or training configuration. The private dashboard on port 8000 is unchanged.

## Local use

Run `npm ci`, `npm run build`, and `npm run preview` in this directory. The site opens on http://127.0.0.1:5295. Its proxy expects the inference-only `tse.public_api:create_public_app` on loopback port 5294. Start that process separately with `PYTHONPATH=src` and `TSE_CHECKPOINT` pointing at a frozen checkpoint copy. Do not point the website at `tse.api`, which serves private training tools. Never stop the trainer to start this site.

## Public deployment

Deploy this directory as a Vite project. Set `ONE_VOICE_SERVICE_URL` to an HTTPS inference host, `ONE_VOICE_SERVICE_TOKEN` to its bearer token, and `ONE_VOICE_PUBLIC_ORIGIN` to the exact public site origin. The service must use the same frozen model as the prepared samples. Do not expose the local training server or this machine through a tunnel. Without an inference host, prepared examples work and uploads explicitly show unavailable.

For inference, build `Dockerfile.inference` from the parent repository. Mount a frozen checkpoint read-only, set `TSE_CHECKPOINT` and `TSE_API_TOKEN`, add provider rate limits, and permit at most one extraction at once. Uploaded WAV/FLAC audio is processed in memory; no audio storage or training is performed. Limits: 30-second mixture, 3–10-second reference, 4 MiB combined. Measure latency and memory on the selected host before opening public uploads.

All four audio inputs support microphone recording and browser-side WAV conversion. Browser permission is required, and recording does not automatically submit audio. The Speech to text workspace uses the local-only service documented in `poc/README.md`. Public deployments display that requirement, allow recording previews, and do not attempt to reach the visitor's localhost or claim cloud transcription is available. Deploying the Vite site does not deploy the Python models or the local worker.

## Identity and assets

The current product uses warm white #FAFAF8, black #191919, Arial, and reusable spacing and typography tokens. See `UI_DESIGN.md` for navigation, responsive layout, and motion rules. The mark represents overlapping signals resolving into one voice. `public/assets/one-voice/brand/identity-board.png` preserves the original brand exploration; the SVG mark and cover are reusable code-native assets. Actual product copy avoids promising perfect noise removal.

The prepared examples share the portfolio's checked snapshot. `src/snapshot.json` carries checkpoint and audio hashes. Keep the copied listening components synchronized with the portfolio when changing playback behavior. Audio attribution is visible in the app; sources are LibriSpeech / Libri2Mix under CC BY 4.0. Training and eval reports stay in the main repository, outside this site's public directory.
