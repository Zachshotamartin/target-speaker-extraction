# One Voice product site

A public listening and extraction interface. It does not expose or import the training dashboard, trainer, pause/resume controls, or training configuration. The private dashboard on port 8000 is unchanged.

## Local use

Run `npm ci`, `npm run build`, and `npm run preview` in this directory. The site opens on http://127.0.0.1:5295. Its proxy expects the inference-only `tse.public_api:create_public_app` on loopback port 5294. Start that process separately with `PYTHONPATH=src` and `TSE_CHECKPOINT` pointing at a frozen checkpoint copy. Do not point the website at `tse.api`, which serves private training tools. Never stop the trainer to start this site.

## Public deployment

Deploy this directory as a Vite project. Set `ONE_VOICE_SERVICE_URL` to an HTTPS inference host, `ONE_VOICE_SERVICE_TOKEN` to its bearer token, and `ONE_VOICE_PUBLIC_ORIGIN` to the exact public site origin. Hosted extraction and transcription use the same frozen checkpoint. Prepared listening examples retain their own recorded checkpoint metadata; they are not regenerated when the hosted model changes. Do not expose the local training server or this machine through a tunnel. Without an inference host, prepared examples work and uploads explicitly show unavailable.

For extraction and transcription together, follow [`poc/HOSTING.md`](../poc/HOSTING.md) and deploy `Dockerfile.public` to a protected CPU Basic Hugging Face Space. It requires `TSE_API_TOKEN` and a packaged frozen model bundle. Vercel uses the Space's `/voice` URL for extraction and `/speech` URL for transcription. Set `ONE_VOICE_TRANSCRIPTION_URL` and `ONE_VOICE_TRANSCRIPTION_TOKEN` in addition to the extraction variables above. Tokens remain server-side. The extraction-only `Dockerfile.inference` is still available for deployments that do not need transcription.

All four audio inputs support microphone recording and browser-side WAV conversion. Browser permission is required, and recording does not automatically submit audio. On the public website, transcription uses the same-origin `/api/poc` proxy with a signed browser-session cookie. Local development keeps the loopback service documented in `poc/README.md`. Hosted requests never attempt to reach a visitor's localhost. Deploying the Vite site does not deploy the Python models; both services must be deployed and verified.

## Identity and assets

The footer links to `#terms` (Terms and licensing) and `#privacy`. Original project code and original model weights have no general reuse/redistribution license. Third-party code, models, metadata and CC BY 4.0 audio retain their own terms. The terms page is a scoped usage/licensing notice, not an MIT grant or a claim that the full model/data pipeline is commercially cleared. `#terms-*` section links use the shared route and scroll handling, preserving active jobs when switching pages.

The current product uses warm white #FAFAF8, black #191919, Arial, and reusable spacing and typography tokens. See `UI_DESIGN.md` for navigation, responsive layout, and motion rules. The mark represents overlapping signals resolving into one voice. `public/assets/one-voice/brand/identity-board.png` preserves the original brand exploration; the SVG mark and cover are reusable code-native assets. Actual product copy avoids promising perfect noise removal.

The prepared examples share the portfolio's checked snapshot. `src/snapshot.json` carries checkpoint and audio hashes. Keep the copied listening components synchronized with the portfolio when changing playback behavior. Audio attribution is visible in the app; sources are LibriSpeech / Libri2Mix under CC BY 4.0. Training and eval reports stay in the main repository, outside this site's public directory.
