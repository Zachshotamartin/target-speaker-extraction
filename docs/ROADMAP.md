# Delivery roadmap

The core system is implemented. The first paired training study and final release measurements are being completed; the model card will record their measured outcome.

| Milestone | State | Evidence |
| --- | --- | --- |
| M0 · Plan and public repository | Complete | Original planning documents and Git history |
| M1 · Native compute feasibility | Complete | CPU/MPS operator benchmarks; M3 Pro, 18 GiB; explicit device paths |
| M2 · Data and metrics | Complete | Bounded public acquisition, source hashes, speaker audit, deterministic paired recipes, signal tests |
| M3 · Model and tiny-set learning | Complete | Independent 1.22M-parameter model; 11.85 dB improvement and zero confusion on 16 fixed training cases |
| M4 · Clean-reference control | Complete | 5,000-update run; 400-case development report |
| M5 · Controlled robustness study | Running | Matched augmentation training, paired comparison and reference diagnostics |
| M6 · Local product | Implemented | Real-speech browser workflow, validated API, context-based long-file inference, CPU container and clean wheel installation |
| M7 · Measured release | Running | Frozen-test runner, report generator, model-card generation and final device profiling |

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

## Release evidence still being collected

The first study uses one paired seed at 5,000 updates per model, 80 development selection cases, 400 development reporting cases, and 1,000 reserved test cases. Final evidence includes both learned models, the exact delivered file-processing path, 10/30/60-second CPU/MPS timing, chunk/whole agreement, artifact hashes, a model card and a technical case study.

The original proposed 5 dB quality target, broad real-microphone robustness, three training seeds, and subjective/intelligibility evaluation are not assumed achieved. A complete research project can produce a negative or limited result; its documentation must say so.

## Research extensions

- Improve unseen-speaker generalization with more speaker diversity and a measured longer training budget.
- Repeat the clean/augmented comparison across three or more paired seeds.
- Test real microphone/room recordings and separately held-out corruption severities.
- Evaluate partial overlap, target pauses, more than two speakers, accents and languages beyond this clean English read-speech pool.
- Add a calibrated target-presence objective before suppressing or labeling absent targets.
- Investigate an auxiliary speaker objective or a stronger reference encoder as a controlled ablation.
- Perform a documented listening/intelligibility study.
- Design causal streaming state and evaluate live latency separately.
- Add an official Libri2Mix adapter if external benchmark comparison becomes a goal.
- Select explicit code/model redistribution licenses before an open-source or downloadable weight release.

These extensions are not claimed as current capabilities. The original detailed ticket plan remains available in Git history.
