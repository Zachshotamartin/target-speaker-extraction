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

The memory budget includes the worker's process group, including video rendering. ONNX Runtime telemetry is disabled before library initialization, in addition to the Hugging Face telemetry settings.

Uploads and downloads travel in chunks of at most 1 MiB, below serverless transport limits. Upload offsets and content hashes allow safe retry/resume. Submitted jobs have owner-scoped idempotency keys so a lost submission response does not launch duplicate inference. The proxy permits only explicit routes, sizes, methods and MIME types, and forwards a signed browser-session identity to the authenticated inference service. No service token reaches the browser.

Uploads expire after one hour. Server results expire 15 minutes after completion. Projects retain downloaded copies locally. A server restart does not resume an inference process; users can rerun saved inputs. Cancelling a job stops its process group, including video rendering. Media input permits known audio/video containers only; playlist and network demuxers are excluded. Video uses a streaming time-remapping filter instead of buffering one decoder branch per cut.

## Validation, 22 September 2026

- Real frozen-model inference on a 43.2-second public recording returned exactly 43.2 seconds and 113 timed words.
- A small overlap stress test used clean public demo sources and known references. Two simultaneous speakers improved SI-SDR by 13.39 and 8.11 dB (mean 10.75 dB). Three simultaneous speakers improved by 3.78, 2.81 and 2.45 dB (mean 3.01 dB).
- These few reused public clips are a smoke test, **not** a held-out quality benchmark. Three/four-speaker overlap remains explicitly experimental. No broad accuracy claim follows from these numbers.
- Regression checks cover owner isolation, resumable chunks, duplicate content, bounded decoding, full-length overlap-add, captions, video/audio synchronization, project persistence, text/audio separation, undo/redo and cut padding.
- Browser checks exercise real extraction, editing, reload/reopen, voice discovery, file import and captioned video export.

### Pre-release review follow-up

The review corrected the following issues before deployment:

- Delayed autosaves could recreate a deleted project. Serialized writes now suppress saves for deleted project IDs, with a regression covering a late autosave.
- Quota calculations could race with worker cleanup and ignored space reserved for incomplete uploads when admitting jobs. Both paths now account for concurrent cleanup and reserved capacity.
- Video subprocess memory was outside the worker's memory accounting. The complete isolated process group is now counted, and cancellation tolerates process-exit races while terminating remaining children.
- Undo restored transcript text but left the correction input stale. The field now follows the selected word after undo/redo.
- A recovered video request could refer to a different selected speaker with identical timings/text. The recovery signature now includes the speaker track.
- Long uploaded reference filenames could exceed the server's label limit; audio-only WebM recordings could be mistaken for video. Labels are bounded and an explicit audio MIME type takes precedence over its extension.
- A real macOS inference worker crashed inside ONNX Runtime's native telemetry uploader (SIGSEGV). `ORT_DISABLE_TELEMETRY=1` is set before initialization and the API opt-out is applied as well, following [ONNX Runtime's privacy documentation](https://github.com/microsoft/onnxruntime/blob/main/docs/Privacy.md). Worker exit codes are retained for diagnosing any future native failures. This does not change model weights or training.

After the fixes, all 32 targeted workspace/transcription tests passed, including the new regression cases. A real two-reference job completed with both speaker tracks and reports downloadable, each preserving 57,600 samples (3.6 seconds), with peak worker RSS of 1.30 GiB. Browser review confirmed correction/undo synchronization and unchanged audio duration. These checks do not establish three/four-speaker separation quality or long-recording throughput on the hosted CPU; those limitations remain visible in the product. Release still requires the backend-first deployment check below.

## Local development

Run `poc.server.create_app` through uvicorn with a dedicated jobs directory and frozen model directory, using the POC environment. Do not run it against an existing service's jobs directory. Point Vite at the isolated service with `ONE_VOICE_LOCAL_API=http://127.0.0.1:5300 npm run dev -- --port 5301`. Production requires updating the Python service before releasing the new frontend; `/health` must advertise `workspace.version: 1`. Install FFmpeg with libass and DejaVu fonts (included in `Dockerfile.public`). Existing overview extraction endpoints remain compatible.
