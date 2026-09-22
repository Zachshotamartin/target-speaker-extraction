# Recording workspace

OneVoice's recording workflow accepts audio/video up to 10 minutes and 128 MiB. References remain 3–10 seconds, with at least 1.5 seconds of speech. There is no just-vibe integration.

## Workflow

1. Import audio or video, record a short microphone sample, or load a public example. Projects automatically save in this browser's IndexedDB. Reopen them in **Saved projects**. Download important work before clearing browser data.
2. Find candidate voice passages using speech detection and ECAPA embedding clustering. Listen before selecting: these are proposed samples, not verified identities or a reliable speaker count. Continuous overlap can produce a mixed sample. Select a clean solo passage or upload a separate reference when needed. Choose up to four references.
3. Each selected voice produces its own isolated waveform, transcript, and evidence report. Correct text and speaker labels without changing the sound. Make word cuts, adjust waveform boundaries, and preview optional filler, pause, or repeated-phrase suggestions. Edits are reversible; current decisions survive reload, while undo history remains session-local.
4. Export edited WAV, retimed SRT, and text for a track, or a ZIP for all tracks. Video exports retain exactly the chosen source passages, replace the soundtrack with the edited isolated voice, and burn in corrected captions. Video renders at up to 1280×720 / 30 fps, preserving aspect ratio. Frame boundaries are approximate within one frame.

## Inference, limits and isolation

The existing frozen OneVoice checkpoint, ECAPA, Silero and Whisper small.en models are reused. No training files, processes, weights or optimizer states are changed. One low-priority CPU worker runs at a time, with one queued job, a 4 GiB worker memory guard and a 60-minute workspace processing limit. Long recordings with several voices can take substantially longer than real time on shared CPU hardware; no completion-time promise is made.

The separator uses 24-second context windows with normalized overlap-add and preserves the exact input sample count. Transcription and speaker evidence retain the original timeline. Each voice is extracted independently; this is not a jointly trained four-speaker separator.

Uploads and downloads travel in chunks of at most 1 MiB, below serverless transport limits. Upload offsets and content hashes allow safe retry/resume. Submitted jobs have owner-scoped idempotency keys so a lost submission response does not launch duplicate inference. The proxy permits only explicit routes, sizes, methods and MIME types, and forwards a signed browser-session identity to the authenticated inference service. No service token reaches the browser.

Uploads expire after one hour. Server results expire 15 minutes after completion. Projects retain downloaded copies locally. A server restart does not resume an inference process; users can rerun saved inputs. Cancelling a job stops its process group, including video rendering. Media input permits known audio/video containers only; playlist and network demuxers are excluded. Video uses a streaming time-remapping filter instead of buffering one decoder branch per cut.

## Validation, 22 September 2026

- Real frozen-model inference on a 43.2-second public recording returned exactly 43.2 seconds and 113 timed words.
- A small overlap stress test used clean public demo sources and known references. Two simultaneous speakers improved SI-SDR by 13.39 and 8.11 dB (mean 10.75 dB). Three simultaneous speakers improved by 3.78, 2.81 and 2.45 dB (mean 3.01 dB).
- These few reused public clips are a smoke test, **not** a held-out quality benchmark. Three/four-speaker overlap remains explicitly experimental. No broad accuracy claim follows from these numbers.
- Regression checks cover owner isolation, resumable chunks, duplicate content, bounded decoding, full-length overlap-add, captions, video/audio synchronization, project persistence, text/audio separation, undo/redo and cut padding.
- Browser checks exercise real extraction, editing, reload/reopen, voice discovery, file import and captioned video export.

## Local development

Run `poc.server.create_app` through uvicorn with a dedicated jobs directory and frozen model directory, using the POC environment. Do not run it against an existing service's jobs directory. Point Vite at the isolated service with `ONE_VOICE_LOCAL_API=http://127.0.0.1:5300 npm run dev -- --port 5301`. Production requires updating the Python service before releasing the new frontend; `/health` must advertise `workspace.version: 1`. Install FFmpeg with libass and DejaVu fonts (included in `Dockerfile.public`). Existing overview extraction endpoints remain compatible.
