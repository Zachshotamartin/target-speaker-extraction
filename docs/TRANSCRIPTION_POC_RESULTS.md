# Voice-selected transcription POC — September 21, 2026

The local end-to-end system works without new training. One Voice is the **only
component that separates or cleans a waveform**. SpeechBrain ECAPA compares
voice embeddings; Silero marks speech/silence; faster-whisper small.en transcribes
audio without receiving a reference, speaker embedding or prompt. No external
separator, diarization/extraction package, LLM text rewrite or cloud ASR is used.

The fair comparison uses the same CPU int8 ASR model and decoding options on
the raw mixture and One Voice output. Both share a generic no-speech guard;
Whisper's internal VAD is disabled for actual decoding. Ordinary ASR robustness
to noise and overlap is retained. These measurements show the effect of the
pipeline on this sample; they do not imply that Whisper cannot handle any overlap.

## Frozen model and protocol

- One Voice: completed best-full validation at epoch **41**, step **142,475**;
  full-development SI-SDR improvement **10.7077 dB**. That separation metric is
  separate from transcription WER. SHA-256:
  `533b283daff607d06d7ff000a95e31b0354ac6edbcb86f55850e6a7f8d1f8d6e`.
- ECAPA: `speechbrain/spkrec-ecapa-voxceleb`, revision
  `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286`.
- ASR: `Systran/faster-whisper-small.en`, revision
  `d1d751a5f8271d482d14ca55d9e2deeebbae577f`.
- Exact model hashes and package versions:
  [manifest](../reports/transcription-poc/model-manifest.json).
- **32 development cases**, then **32 reporting cases**: eight distinct speaker
  pairs in each partition, with one alternating target side per pair and four
  conditions: overlap, target only, other speaker only, silence. Speakers are
  disjoint between the LibriSpeech dev-clean and test-clean partitions.
- Deterministic selection: first qualifying pairs in published Libri2Mix CSV
  order, excluding repeated speakers and requiring both complete sources to be
  3–9 seconds. Separate enrollment utterances come from the existing manifest.
  References are truncated to at most 10 seconds; scored target utterances are
  **never truncated**. Mixtures use official gains, zero-padding to the longer
  source, then common attenuation if necessary to prevent clipping.
- This is an independent full-utterance transcription protocol, **not** the
  official Libri2Mix min-duration separation benchmark. Test-clean contains
  identities used historically in this project; it is a reporting partition,
  not a newly untouched research test set.
- WER is micro-averaged against all target words, including omissions caused by
  filtering. Normalization lowercases alphanumeric words and removes apostrophes.
  It is not the full Whisper English text normalizer. WER can exceed 100% on
  an individual example with many insertions. Empty-reference cases use word
  counts and false-attribution jobs, not WER.

## Reporting results, after thresholds were frozen

Eight cases per row; target-present rows each contain 120 reference words.

| Condition | Raw → Whisper WER | One Voice → Whisper WER | Additional identity filter WER | Jobs with attributed text |
| --- | ---: | ---: | ---: | ---: |
| Overlapping speakers | 65.0% | **15.8%** | 20.8% | 8/8 |
| Selected speaker alone | **1.7%** | 24.2% | 22.5% | 7/8 |
| Other speaker alone | Not defined | Not defined | Not defined | **0/8** |
| Silence | Not defined | Not defined | Not defined | **0/8** |

For the absent-target recordings, raw ASR produced 117 words and ASR after
extraction still produced 83 words. The identity filter withheld all of them.
That distinction matters: the separator is not a reliable target-presence detector.
Zero false-attribution jobs in eight negatives does not establish a low real-world
false-accept rate, especially for similar voices or microphone recordings.

The largest clean-speech failure was case
`1188-133604-0025_4992-23283-0016:1`: raw ASR was correct, extraction led to
19 word edits against a 10-word target, and the identity filter withheld the result.
Keep the original transcript available for review; do not treat unfiltered raw
ASR as automatically belonging to the selected voice.

All case IDs, ground-truth text, hypotheses, edit counts and input hashes are
included in [reporting.json](../reports/transcription-poc/reporting.json), including
failures. Only public corpus text and metrics are committed, not new audio.

## Development threshold selection

The initial 0.50 extracted / 0.20 original cutoffs rejected too much correct
speech. A fixed 25-pair grid was evaluated using the 32 development cases only.
The rule first minimizes negative cases with any attributed text, then target-
present word edits; ties prefer stricter cutoffs. It selected:

- extracted cosine similarity ≥ **0.30** and original similarity ≥ **0.10**;
- at least **0.60 seconds** of detected speech in the comparison context;
- uncertain band beginning at extracted similarity **0.20**;
- at least **80%** word overlap with accepted intervals and detected speech.

These numbers are not probabilities or security thresholds. Each 0.75 second
decision uses up to 3 seconds of context. The extracted score depends on the
reference-conditioned separator, so it is not independent identity confirmation.
The original score is also unreliable when the target is quiet under overlap.

With those rules replayed on the stored development evidence, overlap WER is
65.4% raw / 11.3% after extraction / 28.6% after attribution. For target-only it is
5.3% / 16.5% / 22.6%. All eight absent-target and eight silence cases yield no
attributed words. No neural weights were trained or fine-tuned.

[Development cases](../reports/transcription-poc/development.json),
[complete threshold grid](../reports/transcription-poc/calibration.json),
[protocol and timing](../reports/transcription-poc/protocol.json).

## Local performance and functional checks

Apple M3 Pro, 18 GiB RAM, one CPU numerical thread, reduced process priority,
while the original MPS training/evaluation job continued running. No GPU inference.

- The 24 non-silent reporting jobs had median model-pipeline time **11.4 seconds**
  (range **6.2–21.3 seconds**), with models reused for evaluation. This excludes
  initial model loading and HTTP/UI overhead; it is not a streaming claim.
- The reporting process peaked at **1.54 GiB RSS**. An independent first browser
  job completed in 14.7 seconds of pipeline time; the initial standalone job used
  approximately 1.22 GiB. Production workers are discarded after each job.
- Decoder checks pass for WAV, FLAC, WebM/Opus, M4A/AAC and MP3, including
  resampling, mono conversion, compressed duration limits and corrupt input.
- Automated checks cover separate ASR inputs, no identity argument to ASR,
  silence bypass, boundary attribution, exact export offsets, exclusion of
  uncertain text from TXT/SRT, upload limits, queue bounds, cancellation of only
  the owned worker, expiry, origin restrictions and duplicate-server exclusion.
- Browser checks cover public voice selection, real end-to-end results, uncertain
  text review, timestamp playback, saved-reference persistence across reload,
  deletion of the test profile, and desktop/narrow layouts without page overflow.
  Actual microphone capture requires the user's permission and has not been
  evaluated with personal recordings; its encoded formats are tested.
- The existing training source fingerprint remained
  `9013b04cf14355c88032e404efd222bda0c5d7be20f771ceaa72658c7086805d`.
  The original process remained alive and its epoch-45 full validation advanced.
  No training source, checkpoint, schedule, environment or pause state was changed.

## Remaining limitations

This establishes a useful proof of concept for overlap, not production-grade
speaker identity or universal transcription improvement. Clean speech can be
damaged; cautious attribution can omit target words; similar voices, reverberant
rooms, noise, language changes and microphones need broader public-data tests.
The UI exposes original audio, extracted audio and both ASR paths so those
failures remain inspectable. No new personal recordings or training data were used.

Implementation and reproduction commands: [poc/README.md](../poc/README.md).
