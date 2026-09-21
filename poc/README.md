# Local voice-selected transcription

An inference-only extension to the existing One Voice product. Nothing here
starts, stops, resumes or trains a model. Keep this environment separate from
the running trainer; the tested setup uses a separate checkout on the SSD.

## What each model does

| Stage | Component | Receives the reference? | Changes the waveform? |
| --- | --- | --- | --- |
| Speech/silence intervals | Silero VAD, ONNX CPU | No | No |
| Target speaker extraction | Frozen One Voice checkpoint | Yes | **Yes — the only separator** |
| Identity evidence | SpeechBrain ECAPA-TDNN | Yes, to compare embeddings | No |
| Speech recognition | faster-whisper small.en, CTranslate2 CPU int8 | **No** | No; outputs words and timestamps |
| Attribution | Explicit interval rules | Similarity scores only | No; marks words accepted/uncertain/excluded |

Whisper is naturally robust to some interference. That is not disabled. Both
comparison paths call the same reference-free ASR adapter with the same model,
beam size, language, timestamps, prompt and decoding options. There is no
external source separator, denoising network, target-speaker ASR, WhisperX,
diarization model, LLM rewrite or cloud speech service hidden in this pipeline.
Silero detects speech but does not isolate a speaker. ECAPA produces vectors,
not cleaned audio. Browser microphone capture requests echo cancellation,
noise suppression and automatic gain control off, though device/OS processing
may still vary.

The UI shows raw → Whisper, One Voice → Whisper, and the additional attributed
transcript separately. A raw result that is already correct remains visible.
Extraction can improve target selection yet distort words; listen and check both.

## Setup

Use Python 3.12 and Node 22. In the repository root:

```sh
uv venv --python 3.12 .venv-poc
uv pip install --python .venv-poc/bin/python -r poc/requirements.lock.txt
```

Provision published weights and copy an evaluated checkpoint. The two input
files must describe the same completed validation. The original checkpoint is
never modified; the copied SHA-256 is verified before and after copying.

```sh
.venv-poc/bin/python -m poc.provision \
  --checkpoint /path/to/run/best-full.pt \
  --validation /path/to/run/best-full-validation.json
cd site
npm ci
npm run build
cd ..
.venv-poc/bin/python scripts/run_transcription_poc.py --detach
```

Open `http://127.0.0.1:5295/#transcribe`. The inference API binds loopback on
5296. Use `--ui-port 5297` for an isolated preview. The launcher refuses occupied
UI ports and will only reuse an API with the exact same model manifest. It does
not kill other processes. Port 8000, the training environment, dataset, source
snapshot and run checkpoints are outside its scope. The existing extraction-only
form still uses its original API on 5294; it is not started by this launcher.

Provisioning downloads approximately 0.6 GB of model weights plus a copy of the
existing checkpoint. The Python environment is additional disk space. Exact
download revisions and file hashes are saved in the ignored
`artifacts/poc/models/manifest.json`. Runtime is offline (`HF_HUB_OFFLINE=1`),
uses local paths, verifies hashes, and disables model telemetry. Model downloads
occur only through the explicit provisioning command. Model files, audio,
profiles and virtual environments must not be committed.

## Use

Start with **Public examples**, select a voice, and try overlapping speech,
the target alone, the other speaker alone, or silence. No personal data is needed.

For your own files, provide a clean 3–10 second sample of the desired speaker and
a recording of up to 30 seconds. WAV/FLAC/MP3/M4A/WebM decode through PyAV. Use
**Record from microphone** for an optional record-and-stop capture; no streaming
captions. The browser stops at 29 seconds to leave room for the encoder's final
frame. Total encoded upload is bounded to 4 MiB, with a separate decoded duration
limit. A reference needs at least 1.5 seconds of detected speech and must not be
heavily clipped. A mixed reference may still pass; use one person speaking alone.

Save a reference explicitly to browser IndexedDB if you want to reuse it. The
app cannot identify “you” without a reference, account names do not select voices,
and embeddings are not model training. Profiles belong to that browser origin;
changing localhost ports uses another storage origin. Delete a saved profile to
remove its reference. The API has no persistent identity database.

Click a transcript segment to seek playback. Switch original/extracted audio,
copy the attributed text, or export TXT, SRT, JSON and the extracted WAV. TXT/SRT
contain only accepted words. **JSON includes all evidence, comparison transcripts,
uncertain and excluded words** for inspection. WAV is the complete One Voice
output; identity gating does not modify it. Word timings refer to the original
recording timeline. Playback attenuates sample peaks above 0.98; it never boosts
quiet audio or performs a second denoising stage.

The job API keeps one worker active and at most one job waiting. Workers run on
CPU with one numerical thread and reduced process priority; no MPS/GPU use.
Cancellation terminates only the owned worker process and removes its files.
The 4 GiB resident-memory and 10-minute runtime limits stop only the POC job.
Results expire 15 minutes after completion. Original/reference uploads are removed
when the worker exits; normalized original audio and extracted audio remain with
the result until expiry or explicit deletion. API shutdown deletes jobs; stale
jobs from a crash are deleted at the next startup. If the API never restarts,
leftover files remain in the ignored, owner-only `artifacts/poc/jobs` directory.
No audio, reference, filename or transcript is written to application logs.

## Identity limitations and tests

The gate is a heuristic, not speaker authentication. Each 0.75 second decision
uses up to 3 seconds of surrounding audio. It requires sufficient speech,
similarity to the reference in the extracted audio, and evidence in the original.
Words crossing unclear boundaries are withheld from the default transcript.
The original score may be weak when another speaker dominates. The extracted
score is reference-conditioned and cannot independently prove identity. Silent,
short, similar-voice and out-of-domain recordings remain important failure cases.

The recognizer decodes the complete One Voice output with original timestamps;
the identity rules label its words afterwards. We chose this over concatenating
accepted snippets to preserve ASR context, consistent comparisons and timing.
This means withheld speech may still influence ASR context. There is no
unconditional fallback to transcribing the raw input as the selected speaker.

```sh
PYTHONPATH=src:. .venv-poc/bin/python -m pytest tests/test_transcription_poc.py -q
PYTHONPATH=src:. .venv-poc/bin/python -m poc.evaluate \
  --dataset-root /path/to/reference-baseline \
  --manifest /path/to/full-training/manifest.json \
  --split dev --output artifacts/poc/evaluation/dev
```

Inspect dev results before freezing threshold defaults. Then use `--split test`
with another output directory for reporting. The evaluator uses complete public
utterances and their actual transcripts, zero-padded to the longer source; it
never compares a truncated mixture to a full target transcript. Eight disjoint
speaker pairs per split produce 32 cases covering four conditions. Report
micro-averaged word error rate (including all deleted target words), absent-target
false attribution, silence output and both ASR paths. A low WER on a heavily
filtered subset is not reported as full recognition quality. This small clean
read-speech report is not a claim of broad microphone or meeting robustness.

See `docs/TRANSCRIPTION_POC_RESULTS.md` for measured outcomes and known limitations.

## Sources and licenses

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper), MIT;
  [CTranslate2](https://github.com/OpenNMT/CTranslate2), MIT;
  [converted small.en weights](https://huggingface.co/Systran/faster-whisper-small.en),
  based on [OpenAI Whisper](https://github.com/openai/whisper), MIT.
- [SpeechBrain ECAPA-TDNN weights and model card](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb),
  Apache 2.0; [SpeechBrain](https://github.com/speechbrain/speechbrain), Apache 2.0.
  Published weights trained on VoxCeleb; no VoxCeleb download or training here.
- [Silero VAD](https://github.com/snakers4/silero-vad), MIT, bundled as ONNX by faster-whisper.
- [PyAV](https://github.com/PyAV-Org/PyAV), BSD 3-Clause; FFmpeg libraries have their
  own license/build terms. This app uses installed packages; it does not redistribute binaries.
- [LibriSpeech](https://www.openslr.org/12/), CC BY 4.0, and
  [LibriMix](https://github.com/JorisCos/LibriMix), public mixtures and recipe.
  Demo audio was mixed/model-processed; see the existing gallery attribution and
  `docs/SOURCES.md`. New evaluation files contain only public corpus text/metrics.

Component notices do not change this repository's existing code license.
