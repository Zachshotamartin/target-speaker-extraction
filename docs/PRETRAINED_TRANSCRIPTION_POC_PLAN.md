# One Voice: pretrained transcription proof of concept

Status: implemented proof of concept, September 21, 2026. See `../poc/README.md`
for setup and `TRANSCRIPTION_POC_RESULTS.md` for evidence and the final pipeline.
The implementation decodes the complete extracted clip and attributes its words
afterwards, preserving context and timestamps rather than concatenating accepted
snippets. This document preserves the original design plan and supersedes
the training routes in VOICE_IDENTITY_PLAN.md for the current milestone.

## Scope and constraints

Build an end-to-end local application that selects a voice using a reference,
extracts it from a short recording, and produces a transcript with timestamps and
audio comparison. Use existing trained weights for every neural component.

- No new training, fine-tuning, custom PVAD network, or user-provided training data.
- Leave the ongoing One Voice training process, its environment, source snapshot,
  dataset manifest, checkpoint schedule and dashboard untouched.
- Reference audio is inference input only. Users can reuse an existing clean
  recording; no new recording is required to try the public demo.
- Use public recordings for all evaluation and threshold selection. Setting a
  deterministic acceptance threshold does not update any neural network weights;
  there is no learned calibration model in this milestone.
- Local processing and local storage; no cloud speech API, paid GPU, public upload
  deployment or tunnel is part of the proof of concept.
- English, mono speech, recordings up to 30 seconds, 3–10-second reference and
  4 MiB combined encoded input initially. Two-speaker overlap is the main case.
- Record-and-stop microphone capture is included; continuously streaming captions,
  automatic identification with no reference, and arbitrary meeting lengths are
  outside this milestone.

The model does not know who the user is from their name or account. A saved voice
profile supplies that identity reference. With no personal reference, the app can
demonstrate selection using public speakers, but cannot label one of them as the
user automatically.

## Selected components

| Component | Selection | Purpose |
| --- | --- | --- |
| UI | Existing React/Vite product site | Enrollment, audio input, job state, transcript and comparisons |
| API and worker | FastAPI, isolated Python environment, one inference worker | Bounded jobs, cancellation and temporary results |
| Speaker extraction | Frozen evaluated One Voice checkpoint | Reuse the model already trained; no dependency on future epochs |
| Speaker matching | [SpeechBrain ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) | Compare enrollment and speech embeddings; its card lists Apache 2.0 |
| Generic speech detection | [Silero VAD](https://github.com/snakers4/silero-vad), ONNX CPU | Locate speech/silence; MIT; does not determine speaker identity |
| Transcription | [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [small.en](https://huggingface.co/Systran/faster-whisper-small.en), CPU int8 | English transcription with word timestamps using existing weights |
| Decoding | PyAV/FFmpeg libraries plus the existing audio helpers | Decode uploads and browser capture, normalize to 16 kHz mono |
| Profiles | Browser IndexedDB | Persist chosen reference only on this device, with explicit save/delete |
| Exports | Plain text, SRT and JSON; extracted WAV | Usable result and reproducibility |

faster-whisper documents CPU int8 execution, word timestamps, and PyAV audio
decoding. [CTranslate2 lists macOS ARM64 support](https://opennmt.net/CTranslate2/installation.html).
These establish candidate compatibility, not latency on this particular Mac.

Use immutable model revisions, checksums and pinned compatible dependency
versions. Record license/attribution for weights, libraries and bundled public
audio separately. Provision once with visible download size/progress; subsequent
inference should work with the models cached locally. Missing model files should
produce an actionable setup error, not an unexpected upload or automatic download
inside an inference request.

Do not initialize a second live copy of the main training weights for updates.
At implementation start, select and copy a completed best-full checkpoint with
matching provenance, verifying its hash before/after copying. Existing saved
gallery outputs retain their own model labels until rerendered. Jobs pin one
checkpoint and one complete model manifest for their entire lifetime.

## User experience

1. Open the existing product site and select the transcription workspace. Keep
   the existing listening gallery and extraction-only flow working.
2. Choose a saved profile or select a clean reference file. The existing public
   voices are available as presets, so no personal files are needed for the demo.
   Show reference playback and explain whose voice will be selected. Offer
   explicit "Save on this device"; do not silently persist temporary input.
3. Upload a recording, choose a public example, or press Record and Stop. The
   microphone permission request occurs only after the Record action. Browser
   microphone data is inference input, never training material.
4. Press "Transcribe selected voice". Show actual stages: queued, checking audio,
   finding speech, extracting voice, checking speaker, transcribing, ready. Use
   indeterminate progress where no meaningful percentage exists.
5. Show transcript segments with timestamps. Clicking a segment seeks the audio.
   Offer original/extracted playback, copy text, TXT/SRT/JSON and WAV downloads.
6. Clearly distinguish accepted, uncertain and excluded regions. Uncertain text
   may be requested for review, but must not silently enter the attributed
   transcript or its default export. Never show a raw cosine score as a percent
   probability or suggest that accepted text is guaranteed to be the user's.
7. Allow cancellation, retry and reference replacement. Distinguish "no speech"
   from "could not confidently match the selected speaker" and service errors.

Deleting a saved profile removes its stored audio and derived local cache entries.
Transcript/result storage is temporary unless the user explicitly downloads it.
Include keyboard-operable controls, accessible status announcements and audio
labels that identify the selected speaker and playback source.

## Audio and identity pipeline

### A. Validate and enroll

Bound request size before parsing. Decode incrementally with explicit duration,
channel/sample-count and memory limits; compressed size alone is insufficient.
Support WAV/FLAC initially, plus MP3/M4A/WebM where the pinned decoder passes
fixture checks. Browser capture must be decoded/resampled explicitly rather than
assuming the browser produces 16 kHz WAV. Reject corrupt, silent and insufficient
reference input with an explanation, preserving the existing duration contract.

Compute the ECAPA enrollment representation and separately provide the reference
waveform to One Voice. These representations are not interchangeable. Cache by
reference content hash, preprocessing version and model revision; changing the
speaker encoder invalidates its cached vectors. Profile labels do not influence
inference and no personal reference is used to fit thresholds.

### B. Detect and extract

Run generic VAD on the original timeline. If it contains no speech, return a
no-speech result without running extraction or ASR.

For speech-containing clips, retain the full recording for One Voice extraction.
Its bidirectional recurrence and global normalization require whole-clip context;
do not splice speech snippets together before passing them to the separator.
Keep the original sample count and timestamp mapping. Do not boost quiet
extraction output to force a match. Listening gain adjustments are separate from
the actual model and identity/ASR input.

### C. Match and abstain

Compute ECAPA evidence on original speech windows and corresponding extracted
windows. An initial window proposal is about three seconds with overlap; exact
length, stride, minimum voiced context and thresholds are evaluation settings.
Short replies require surrounding context and may remain uncertain.

Prototype policy:

- Strong, consistent target evidence in original context and extracted speech:
  eligible for attribution.
- Strong negative evidence, or no extracted speech: exclude.
- Conflicting or insufficient evidence: mark uncertain and keep it out of the
  default attributed transcript.

Use separate accept/reject thresholds with temporal smoothing and a context
margin. Determine their settings on public development examples, then freeze them
before final reporting. Do not train a score-fusion classifier.

This is an intentionally conservative heuristic, not a trained personalized VAD.
The original mixture can conceal a quiet target from the verifier. Conversely,
the reference-conditioned extraction can bias the extracted-audio match. Neither
evidence source proves perfect speaker isolation. Report overlap-specific misses
and leakage; no text-based guess about which words "sound like the user" is allowed.

### D. Transcribe without losing the timeline

Run the pretrained recognizer on eligible speech spans with context padding. Keep
the source-offset mapping when making ASR chunks, and deduplicate padding overlap.
Assign word timestamps back to the original recording. Words near an uncertain
boundary stay uncertain rather than inheriting an unrelated segment's label.

Use English explicitly and deterministic decoding settings for the public demo.
Evaluate no-speech/log-probability/repetition controls on silence and negative
examples. Never invoke the recognizer merely to force text for a rejected clip.
ASR confidence is not evidence of speaker identity.

An optional review action can run ASR on uncertain extracted regions, showing
those words separately. The default export includes accepted text only, with JSON
also recording abstention intervals and reasons. Do not quietly substitute raw
mixture transcription when separation or matching fails.

## Local services and resource isolation

Keep port 8000 as the existing private training dashboard. Keep the current
product/inference services working while building the feature in a separate
checkout/environment. During development, use a separate PoC API port (proposed
5296 after checking availability) and preview port, then connect the product site
when verified. All development services bind to 127.0.0.1.

Reuse a read-only, frozen copy of the extraction implementation in the isolated
runtime; never upgrade the active training environment to satisfy SpeechBrain,
ONNX or CTranslate2 dependencies. Separate service configuration must not import
or expose pause/resume controls.

Initial scheduling: one active job, one waiting job, one CPU inference thread per
component, sequential model stages, explicit low-priority worker scheduling.
Avoid MPS/MLX acceleration while the existing trainer needs the GPU. CPU-only
inference can still compete during its CPU evaluation; low priority reduces
contention but cannot promise zero slowdown. Measure both worker memory and
training progress during compatibility checks; enforce the worker's memory and
duration budgets rather than stopping the trainer.

Target a measured peak worker RSS below 4 GiB for short clips; this is a proposed
budget, not an achieved result. If the budget fails, unload idle models between
stages, shorten the supported job limit with explicit UI copy, or select a
smaller pretrained ASR after evaluating accuracy. Do not silently change limits
or models. State measured end-to-end latency instead of promising real time.

Use a dedicated child process for cancellable compute. Cancel/timeout affects
only that job worker; never send signals by broad process name. A cancelled job
must release its queue slot and delete its temporary results.

## Proposed API contract

| Endpoint | Behavior |
| --- | --- |
| GET /health | Component readiness and missing files; no model loading as a side effect |
| GET /models | Pinned model identities, limits, supported formats and device |
| POST /reference/check | Validate a reference and return quality checks; no persistent personal-data storage |
| POST /transcriptions | Multipart recording/reference plus options; return an opaque job ID |
| GET /transcriptions/{id} | Actual stage, timings, terminal status and structured segments |
| DELETE /transcriptions/{id} | Cancel active/queued work or remove a completed temporary result |
| GET /transcriptions/{id}/audio | Request's extracted WAV until expiration |

Segment schema: start/end seconds, text, words with timestamps, attribution state
and machine-readable reason. Provenance includes checkpoint/model hashes,
preprocessing/threshold versions and stage timings. Keep raw numeric identity
scores in debug JSON rather than a misleading user-facing confidence percentage.

Uploads/results stay in bounded memory or private temporary storage with a
15-minute expiry after completion; cancellation deletes them promptly. Restarted
services report interrupted/expired jobs clearly. Do not log audio, reference
embeddings or transcripts. Restrict browser origins and do not accept arbitrary
remote audio URLs or caller-supplied filesystem paths. Reuse the existing site's
proxy shape so future hosting does not expose the training API.

Persistent service-side profiles, account authentication, billing and public
deployment are outside the local proof of concept.

## Public evaluation and acceptance

Use already available LibriSpeech speech and distinct enrollment utterances; its
[official corpus includes transcripts and public downloads](https://www.openslr.org/12/).
No personal recording collection is required. Create small deterministic cases
for target only, overlap, other-speaker only and silence. Add a modest public-noise
stress set if needed without acquiring a new training corpus.

Initial allocation: 32 development cases for deterministic threshold/settings
selection and 32 reporting cases, covering all four conditions. Separate speakers
and utterances between those groups, subject to checking available data. Keep the
existing separator's evaluation untouched. State that these are public
proof-of-concept cases, not a pristine benchmark for all pretrained components.

For word error rate, choose complete source utterances within the duration limit
and preserve their whole transcripts. Do not score an arbitrarily cropped audio
segment against the full utterance transcript. Include clean-target ASR, raw
mixture ASR and extraction without speaker rejection as comparisons.

Report per condition: target word error rate, non-target insertions, missed target
speech, accepted non-target speech duration, abstention coverage, silence/absence
false transcripts, runtime and peak memory. Overall averages must not hide a
system that rejects everything. The small sample does not establish rare-error
rates; show individual failures and counts.

Functional completion requires:

- Public preset, uploaded reference, saved profile and record-and-stop paths work.
- Transcript, seek-to-audio, original/extracted comparison and exports work.
- Silence/absence paths can return no attributed text; uncertainty is visible.
- Input bounds, busy queue, cancellation, timeout, expired jobs and missing models
  have exercised error paths and meaningful messages.
- Automated tests cover timeline mapping, threshold boundaries, profile/model
  cache isolation and exports; actual pretrained-model integration checks cover
  all four acoustic conditions. UI checks include microphone-denied behavior.
- The existing training process and its source identity remain unchanged.

To call speaker filtering beneficial, it must reduce non-target transcript
insertions while retaining useful target coverage. Compare methods at matched
coverage or report the miss/leakage tradeoff explicitly. If the speaker gate
fails this test, deliver it as an experimental review aid and document the failure;
do not claim dependable "only me" transcription or start training to rescue it.

## Build order and deliverables

1. Environment/model readiness: pin dependencies and weights, verify CPU support,
   hash the frozen separator and run bounded latency/memory probes.
2. Reference profiles and public presets: validation, local save/delete, playback
   and model-version cache rules.
3. Backend pipeline: speech detection, extraction, speaker evidence/abstention,
   transcription, timeline reconstruction and cancellable jobs.
4. Product workspace: file/capture input, real progress, transcript review,
   comparison playback and exports.
5. Evaluation and polish: public cases, error-path tests, resource checks and
   local startup documentation.

Deliver a locally runnable complete application, a pinned model manifest,
reproducible public demo/evaluation assets, an evaluation report, tests and a
one-command local launcher that manages only the PoC services.

There are zero model-training hours in this plan. A reasonable provisional
engineering allowance is several focused days (roughly 3–5) for integration and
verification, with dependency compatibility and identity performance as the main
uncertainties. Model downloads and measured inference tests are additional elapsed
time. This is an estimate, not a guaranteed completion deadline.

## Alternatives only if the selected component blocks the PoC

Speaker encoder: [NVIDIA TitaNet](https://huggingface.co/nvidia/speakerverification_en_titanet_large)
is a pretrained comparison candidate, subject to runtime and data-exposure checks.
ASR: whisper.cpp or MLX Whisper can replace the recognizer behind the same adapter,
after validation; GPU acceleration is not enabled by default alongside training.
No custom detector, encoder fine-tuning or new separation training is a fallback
within this milestone.
