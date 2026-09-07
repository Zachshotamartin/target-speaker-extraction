# One voice — v0.2.0 model card

Experimental, independently implemented target-speaker extraction for two overlapping voices. A separate recording identifies the desired speaker. The updated model achieves **5.887 dB mean SI-SDR improvement** on the frozen fresh test. It still worsens 13.90% of requests and triggers the speaker-confusion proxy on 7.60%. These results do not establish production speech quality.

## Artifact and intended use

- Exported SHA-256: `c814d937b5876007a299e8861f29522cfac41130560bd6c678595ee03a80433e`.
- Architecture: `reference_conditioned_stft_tcn`; separator normalization: `per_frame`.
- Parameters loaded: 2,260,586, including any training speaker head retained in the checkpoint. Inference selects speech from the reference embedding, not the head's identity labels.
- Uniformly averaged project checkpoints at expanded-data steps 16000, 18000, 20000. Source hashes are retained in the exported provenance. Initialization includes earlier training within this project; the displayed update count is not the complete training history.
- No external pretrained speaker or separator weights. Original model: [v0.1.0 card](MODEL_CARD_V0_1.md).

Inputs are WAV/FLAC, up to 60 seconds of mixture and 3–10 seconds of separate reference. Output is mono 16 kHz with the exact mixture timeline. The requested speaker must be present. The system is offline and noncausal; faster-than-duration processing is not live-call latency.

## Fresh held-out evaluation

The fresh test contains 1,000 extraction requests from 20 reserved identities, paired by swapping targets in the same mixture. These identities were withheld before expanded training and never trained in the original model. The original opened test-clean evaluation remains historical. This is a custom LibriSpeech split, not the official Libri2Mix benchmark.

Selection used development data, then bound exact checkpoint, source-inventory and test-recipe hashes before test scoring. [Selection record](../reports/quality-v2-selection.json). Both models use the same file decoding, preprocessing, peak guard and float-WAV delivery path.

| Metric | Original model | Updated model |
| --- | ---: | ---: |
| Mean SI-SDR improvement | 1.453 dB | 5.887 dB |
| Median SI-SDR improvement | 1.656 dB | 6.270 dB |
| ESTOI intelligibility proxy | 0.5457 | 0.6534 |
| Requests worse than mixture | 30.80% | 13.90% |
| Speaker confusion proxy | 17.30% | 7.60% |
| Scalar distortion proxy | 11.58% | 11.20% |

The paired mean separation gain is +4.434 dB, with approximate 95% target-speaker-cluster interval [3.455, 5.529]. The updated mean's corresponding interval is [5.072, 6.741] dB. Resampling uses 1,000 replicates and seed 42; shared-interferer dependence remains. These are one-run comparisons with combined model/data/training changes, not a causal estimate of any single change.

Both targets improve in 76.20% of complete mixture pairs. See [failure slices](../reports/quality-v2-test-failures.json), [per-request results](../reports/quality-v2-selected-test.json), and [paired comparison](../reports/quality-v2-test-comparison.json).

ESTOI predicts intelligibility; it is not word accuracy or a human rating. The scalar distortion proxy includes filtering, phase and envelope errors and is not perceived static. Confusion means the estimate's SI-SDR against the interferer exceeds its SI-SDR against the target by more than three dB. None of these measures proves that every voice sounds clean.

## Training and research alignment

The expanded pool contains 231 speakers and 90.582 eligible hours of clean LibriSpeech source recordings. Mixtures are generated on demand, with distinct speakers, complete overlap, ratios between −5 and +5 dB, and a different reference utterance, preferably another chapter. Forty dev-clean identities supply development data. Source hours are not mixture hours or training epochs.

The spectral separator uses a 512-sample Hann STFT with a 128-sample hop, reference-conditioned temporal convolutions and a real sigmoid mask. It retains mixture phase. Its own convolutional reference encoder learns jointly with extraction. Losses combine target-specific SI-SDR, normalized waveform L1, multi-resolution spectral reconstruction and training-speaker classification. Full settings and provenance accompany the export.

This custom system is informed by SpeakerBeam, SpEx, Conv-TasNet and enrollment-augmentation research. It does not reproduce their published recipes. The [research audit](RESEARCH_AUDIT.md) documents substitutions, normalization experiments and source-informed mask diagnostics. Oracle diagnostics use clean targets and are never model estimates.

## Delivery and runtime

The service removes DC, normalizes inputs, predicts and restores mixture gain. Outputs exceeding sample peak 0.98 receive uniform attenuation. Quiet outputs are never boosted and samples are never hard-clipped. This guard does not remove interference. Model information and response metadata identify the checkpoint and processing version.

| Device | Recording | Warm median / p95 | Exact sample count |
| --- | ---: | ---: | --- |
| CPU | 10 s | 1.131 / 1.290 s | True |
| CPU | 30 s | 3.202 / 3.393 s | True |
| CPU | 60 s | 5.948 / 6.416 s | True |
| MPS | 10 s | 0.122 / 0.149 s | True |
| MPS | 30 s | 0.307 / 0.339 s | True |
| MPS | 60 s | 0.465 / 0.511 s | True |

Measured on this Apple M3 Pro Mac without concurrent training. Each device runs in its own process. End-to-end time includes file decode, inference and WAV encode, excluding browser/network/HTTP overhead. The profiler repeats a development clip to 10/30/60 seconds; this is throughput and alignment evidence, not natural long-conversation quality. [CPU protocol](../reports/quality-v2-delivery-cpu.json), [MPS protocol](../reports/quality-v2-delivery-mps.json).

## Limits, use and origin

The test covers clean, read English speech mixed synthetically. Microphones, noisy/reverberant rooms, spontaneous conversation, music, more than two speakers and out-of-domain languages are not established capabilities. Target absence has no calibrated detector or confidence score. The selected run uses clean references; the original augmentation study's robustness findings cannot simply be transferred to this model. No claim is made that static is eliminated or that output is suitable for forensic conclusions.

The [local gallery](http://127.0.0.1:8000/gallery/) uses the first 20 development requests, chosen independently of scores. A [matched-volume comparison](http://127.0.0.1:8000/gallery/quality-progress/) retains the original model for comparison. Playback matching is clearly labeled and does not alter normal inference.

Speech is from [LibriSpeech / OpenSLR 12](https://www.openslr.org/12), Panayotov et al. (2015), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Local examples are cropped, normalized and mixed derivatives with attribution. Per-file hashes and source identities are retained; the streaming acquisition did not verify the publisher's complete archive checksum. Raw audio and weights stay outside Git. Original project code has no selected reuse license.

See [quality reproduction](REPRODUCING_QUALITY.md) and the [development worklog](QUALITY_WORKLOG.md). The test freeze must remain immutable; subsequent test-informed changes need a new generalization protocol.
