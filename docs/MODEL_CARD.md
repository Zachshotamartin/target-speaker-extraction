# Model card · One voice 0.1

An experimental, independently implemented reference-conditioned speech extractor. The selected artifact is **augmented**, chosen using development data before opening the reserved test. All weights were trained from random initialization; no SpeakerBeam code or checkpoints were used.

## Identity and intended use

- Parameters: 1,223,296; 16 kHz mono; float32; noncausal.
- Selected training update: 5,000; matched experiment budget: 5,000 updates per model; seed 42.
- Export SHA-256: `f3271f1decf7ec0e8e9b0f1fab578a693f51c4a1d1778adeeb9109a5e64a5c20`.
- Source training checkpoint: `44346f83254cd4cec293478340b3ef5933a49278f1c019574c95b9f0c4a908fa`.
- Test case manifest: `a06c84af1b766391c65b82789d85616159831c4c65e8eef2e426b7887fe51a60`.
- Artifact location: `artifacts/releases/model.pt`; accompanying config/provenance: `artifacts/releases/model.json`.
- Intended use: local research, engineering demonstration, and exploratory target-present two-speaker recordings with a separate 3–10 second reference. Maximum mixture duration: 60 seconds.

The model is an audio estimator, not an identity verifier, presence detector, or speech-transcription system. Do not interpret retained audio as proof that a particular person spoke. The app labels it as experimental.

## Data and training

Public [LibriSpeech](https://www.openslr.org/12), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), attributed to Panayotov, Chen, Povey and Khudanpur (2015). A bounded archive-order selection provides 60 training speakers and 10.784 hours in the acquired usable inventory. Development/test each use 40 different speakers from their official clean partitions. Crop eligibility further filters the pool. English read speech and synthetic complete overlap limit generalization to spontaneous conversation, accents, languages, and recording conditions.

Training uses on-demand 2-second mixtures, 5-second distinct-utterance references, levels from −5 to +5 dB, AdamW, batch 2 with four-step accumulation, and negative SI-SDR plus 0.1 normalized L1. See [implementation](IMPLEMENTATION.md) for initialization, augmentation severity, and acquisition integrity limits. Data identities and recipes are retained in [metadata](../metadata/).

## Frozen test results

Each condition has 1,000 extraction requests (500 shared-mixture target pairs) from 40 target speakers. B0 returns the mixture and has zero SI-SDR improvement by definition. Outputs are scored against the requested target, without permutation matching.

| Reference | Control SI-SDRi | Augmented SI-SDRi | Paired gain [approx. 95% CI] | Confusion, control / augmented |
| --- | ---: | ---: | ---: | ---: |
| Clean | 1.53 dB | 1.73 dB | +0.20 dB [0.03, 0.39] | 15.3% / 14.6% |
| Noise | 1.53 dB | 1.73 dB | +0.20 dB [0.03, 0.38] | 15.0% / 14.9% |
| Channel | 1.48 dB | 1.74 dB | +0.25 dB [0.09, 0.44] | 15.6% / 14.3% |
| Reverb | 1.49 dB | 1.69 dB | +0.21 dB [0.05, 0.39] | 15.8% / 14.8% |
| Combined | 1.44 dB | 1.67 dB | +0.23 dB [0.09, 0.40] | 15.8% / 14.8% |

Equal-weight augmentation gain across the four mismatch conditions: **+0.22 dB**, approximate joint interval **[0.07, 0.39]**. Conditions are averaged within each paired case before jointly resampling target-speaker clusters. This is one paired training seed. The intervals approximate target-speaker sampling uncertainty and do not include training-seed variability or fully account for shared interferers. Neither positive point estimates nor model selection alone establish a robust augmentation benefit.

Selected model, clean condition: mean **1.73 dB**, median 1.79 dB, 10th percentile -1.80 dB, mean interval [1.12, 2.28]; 29.0% of cases have negative improvement and 14.6% meet the 3 dB wrong-speaker confusion proxy. Near-silent outputs: 0.0%. Normalized waveform L1: 0.470.

The app's exact shared WAV processing path achieves **1.73 dB** on the same clean test cases, with 14.6% confusion and 29.1% negative-improvement cases. It includes decoding, normalization, model execution, inverse gain, WAV encoding, and decoding for scoring; HTTP behavior is separately integration-tested. Raw results: [control](../reports/control-test.json), [augmentation](../reports/augmented-test.json), [paired comparison](../reports/test-comparison.json), [delivered path](../reports/delivered-test.json).

![Frozen reference-condition results](../reports/figures/test-conditions.png)

## Failure diagnostics

The delivered clean test path, grouped by target level:

| Target level relative to interferer | Cases | SI-SDRi | Model confusion | Mixture-baseline confusion |
| --- | ---: | ---: | ---: | ---: |
| Below −2 dB | 300 | 2.52 dB | 34.0% | 100.0% |
| −2 to +2 dB | 400 | 1.86 dB | 8.5% | 12.2% |
| Above +2 dB | 300 | 0.78 dB | 3.3% | 0.0% |

The mixture-baseline confusion proxy is recovered from the two target scores for each identical mixture. It exposes the proxy's dependence on relative level. Across 500 complete mixture pairs, both requested targets improve in 55.0%; at least one target triggers confusion in 28.6%. These are descriptive slices, not independently randomized comparisons. Chapter and individual-speaker summaries are in [failure analysis](../reports/selected-test-failures.json).

On 80 development requests with a third, absent speaker's reference, output energy exceeds −20 dB relative to mixture in **100.0%** of cases. Mean output/mixture energy is -5.84 dB. Target absence is unsupported; the system can emit another voice. No SI-SDR against a zero target is reported.

Replacing the selected model's reference embedding with zeros yields -1.72 dB on the 80-case development diagnostic. This is an inference intervention outside training distribution, not a trained baseline.

Reference duration diagnostic:

- 1 second: 1.44 dB across 80 eligible cases; 0 excluded.
- 3 seconds: 1.90 dB across 80 eligible cases; 0 excluded.
- 5 seconds: 1.76 dB across 80 eligible cases; 0 excluded.
- 10 seconds: 1.48 dB across 4 eligible cases; 76 excluded.

Longer-reference subsets can differ. Eligible IDs and per-case values are preserved; these figures alone are not causal duration comparisons. Synthetic noise/channel/reverb robustness does not establish real-phone or real-room robustness. No formal listening study, word-preservation measure, or multilingual/three-speaker evaluation was performed.

## Runtime on the project Mac

Apple M3 Pro, 18 GiB unified memory, PyTorch 2.14.0. Five warm repetitions per length, following one first request. Audio is a repeated development clip. Processing includes decode, shared inference and WAV encode, excluding browser/network/HTTP parsing. No training runs concurrently with these measurements.

| Device | Audio | Warm median / p95 | Median real-time factor | Exact samples |
| --- | ---: | ---: | ---: | ---: |
| CPU | 10 s | 0.401 / 0.435 s | 0.0401 | 160,000 |
| CPU | 30 s | 1.258 / 1.394 s | 0.0419 | 480,000 |
| CPU | 60 s | 2.679 / 2.937 s | 0.0446 | 960,000 |
| MPS | 10 s | 0.070 / 0.072 s | 0.0070 | 160,000 |
| MPS | 30 s | 0.221 / 0.240 s | 0.0074 | 480,000 |
| MPS | 60 s | 0.448 / 0.457 s | 0.0075 | 960,000 |

Model-load time: CPU 0.038 s; MPS 0.139 s. Process peak RSS, including whole-file comparison: CPU 0.453 GiB; MPS 0.479 GiB. The MPS driver allocation at the end of profiling is 1.082 GiB, an end counter rather than a peak. Unified-memory counters overlap and must not be summed. Maximum absolute chunk/whole difference across tested lengths: CPU 1.49e-07; MPS 1.79e-07. These checks establish numeric alignment for this artifact, not natural long-conversation quality or live latency.

Full runtime protocols and samples: [CPU](../reports/delivery-cpu.json), [MPS](../reports/delivery-mps.json). [Reproduce the project](REPRODUCING.md).

## Distribution and limitations

Code, manifests and reports are public. Raw recordings and model weights remain in the local workspace and are not committed. A license for original code/model redistribution has not been selected. Public visibility is not an open-source license; data and dependencies retain their own licenses. Generated local speech examples include source attribution and describe their cropping/mixing transformations.

The first release is a complete local research pipeline with measured limitations. The proposed 5 dB quality target was not reached. Further development should use fresh held-out data after test-driven changes and replicate across training seeds.
