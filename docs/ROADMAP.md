# Delivery roadmap

The first experimental release delivered independent implementation, two trained models, frozen evaluation, a local app, a listening gallery, reproducible records, and measured CPU/MPS delivery. Its historical results are in the [v0.1.0 model card](MODEL_CARD_V0_1.md). The [quality worklog](QUALITY_WORKLOG.md) records the subsequent model improvements and the [current model card](MODEL_CARD.md) identifies the released artifact.

| Milestone | State | Evidence |
| --- | --- | --- |
| M0 · Plan and public repository | Complete | Original planning documents and Git history |
| M1 · Native compute feasibility | Complete | CPU/MPS operator benchmarks; M3 Pro, 18 GiB; explicit device paths |
| M2 · Data and metrics | Complete | Bounded public acquisition, source hashes, speaker audit, deterministic paired recipes, signal tests |
| M3 · Model and tiny-set learning | Complete | Independent 1.22M-parameter model; 11.85 dB improvement and zero confusion on 16 fixed training cases |
| M4 · Clean-reference control | Complete | 5,000-update run; 400-case development report |
| M5 · Controlled robustness study | Complete | Two 5,000-update runs; paired test gain +0.22 dB across mismatch conditions, with one-seed limitations |
| M6 · Local product | Complete | Real browser uploads, validated API, 20-request gallery, 10/30/60-second CPU/MPS measurements, CPU container and clean wheel |
| M7 · Measured experimental release | Complete | 1,000 frozen test requests, delivered-path evaluation, source/artifact hashes, figures, model card and case study |
| M8 · Measured separation improvement | Complete | Frozen fresh test: 5.887 versus 1.453 dB on 1,000 requests; selected v0.2.0 export served locally; updated gallery, package/API checks and CPU/MPS profiles |

## Implemented acceptance checks

- Native Python/PyTorch package and pinned dependency lock; CPU CI.
- Random-tensor forward/backward timing with 20 warmup and 100 measured steps.
- Safe bounded official audio acquisition with member hashes and explicit partial-archive integrity limits.
- Cross-split speaker/utterance/hash rejection, reference independence, valid offsets, and source checksum checks.
- Exact mixture sums/levels, paired target switching, fixed corruption seeds, and a zero-improvement mixture baseline.
- Exact output lengths, masked reference pooling, gradients reaching reference parameters, and tiny speech-set learning.
- Atomic best/latest checkpoints, optimizer and RNG restoration, deterministic sample schedules, and an exact CPU resume test.
- Five reference conditions, paired target-speaker bootstrap intervals, gain-sensitive error, silent-output and confusion diagnostics.
- Duration/constant-reference/absent-reference diagnostics with explicit interpretation limits.
- WAV/FLAC input, resampling, validation, bounded context, inverse gain, and aligned float WAV output.
- Readiness warmup, model metadata, upload/origin limits, busy and invalid-input errors, and temporary upload cleanup.
- Local waveform preview, real example loading, synchronized comparison, error/loading states, and output download.
- CPU container recipe, noneditable wheel verification, code checks and Linux CI.

## Release evidence

The first study uses one paired seed at 5,000 updates per model, 80 development selection cases, 400 development reporting cases, and 1,000 reserved test cases. Evidence covers both learned models, the exact delivered file-processing path, 10/30/60-second CPU/MPS timing, chunk/whole agreement, artifact hashes, a model card and a technical case study. The selected artifact scores 1.73 dB mean improvement; the delivered path has 29.1% negative-improvement cases and 14.6% confusion.

The proposed 5 dB quality target was not reached in v0.1.0. The v0.2.0 fresh test exceeds that numerical target under a different, explicitly documented protocol, while 13.9% of requests still worsen. Broad real-microphone robustness, three-seed replication and subjective listening evaluation remain research work. Completing the pipeline and experiment does not establish those capabilities.

## Research extensions

The quality follow-up implements output peak protection, spectral reconstruction loss, joint speaker supervision, an independent STFT-mask separator, expanded data, a completed 20,000-update run, checkpoint averaging and a matched 2,000-update normalization pilot. [Development evidence](QUALITY_WORKLOG.md), [research alignment](RESEARCH_AUDIT.md). The selected averaged model scores 5.887 dB and 0.6534 ESTOI on the frozen fresh test; the [current model card](MODEL_CARD.md) details remaining failures and uncertainty.

- Improve unseen-speaker generalization with more speaker diversity and a measured longer training budget.
- Repeat the clean/augmented comparison across three or more paired seeds.
- Test real microphone/room recordings and separately held-out corruption severities.
- Evaluate partial overlap, target pauses, more than two speakers, accents and languages beyond this clean English read-speech pool.
- Add a calibrated target-presence objective before suppressing or labeling absent targets.
- Compare the implemented auxiliary speaker objective with stronger reference encoders under a matched training budget.
- Perform a documented human listening study alongside the implemented ESTOI intelligibility proxy.
- Design causal streaming state and evaluate live latency separately.
- Add an official Libri2Mix adapter if external benchmark comparison becomes a goal.
- Select explicit code/model redistribution licenses before an open-source or downloadable weight release.

These extensions are not claimed as current capabilities. The original detailed ticket plan remains available in Git history.
