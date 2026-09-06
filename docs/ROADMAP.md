# Roadmap and implementation backlog

The project is in the planning/foundation phase. Every model, data, API and evaluation ticket below is still open. Estimates are focused engineering effort, excluding unattended training and unexpected environment failures.

## Milestones

| Milestone | Deliverable | Depends on | Rough effort |
| --- | --- | --- | --- |
| M0 | Detailed plan, local Git foundation and public GitHub repository | Current task | Foundation session |
| M1 | Native environment and CPU/MPS feasibility report | M0 | 4–8 hours |
| M2 | Audited data pipeline and fixed pilot cases | M1 | 10–16 hours |
| M3 | Independently implemented model that learns a tiny set | M2 | 12–20 hours |
| M4 | Reproducible clean-reference control | M3 | 10–16 hours plus training |
| M5 | Robustness comparison and failure analysis | M4 | 16–24 hours plus training |
| M6 | Local inference, API and usable audio demo | M4, with M5 model selection | 16–24 hours |
| M7 | Final evaluation, model card and portfolio release | M5, M6 | 8–12 hours |

The total is approximately 80–120 focused hours after allowing for overlap and iteration. At part-time availability, plan around 8–12 weeks and revise after M1/M3. Milestone completion is determined by evidence, not by a date.

## M0 — Foundation

- [x] Establish the independent-implementation boundary.
- [x] Inspect the local machine, disk and workspace.
- [x] Initialize a local repository on `main`.
- [x] Document product scope, data, model, evaluation and implementation sequence.
- [ ] Validate repository contents and create the first commit.
- [ ] Create the requested public GitHub repository and verify the pushed commit.

## M1 — Environment and compute feasibility

### ENV-01: Create the native environment

Use verified ARM Python 3.12, add `pyproject.toml`, resolve minimal numerical/audio/PyTorch dependencies and commit the lockfile. Keep the environment ignored.

Acceptance: interpreter reports ARM architecture; WAV/FLAC round-trip works; PyTorch import and MPS availability are recorded; CPU execution works. Document exact versions and any unsupported operations.

### ENV-02: Benchmark proposed model operations

Exercise convolution, transposed convolution, normalization, reference pooling, masked loss and backward propagation on representative shapes. Later rerun the same protocol on the implemented model.

Acceptance: finite values and gradients; 20 warmup steps followed by 100 measured steps; CPU/MPS timing and memory report; estimated time for 1,000 updates with the planned accumulation. No full training launch is required for this check.

### ENV-03: Set a measured run budget

Confirm microbatch size, crop length, model width and storage headroom. Record the selected configuration and estimate pilot/main training time from measured throughput.

Acceptance: the pilot fits the declared resource budget. If it does not, reduce model/crop size and repeat only the relevant benchmark.

## M2 — Data and metrics

### DATA-01: Inventory a bounded public speech selection

Implement source registration, checksum handling, safe extraction, metadata inventory and duration statistics. Start with the training-only pilot selection, then add development sources.

Acceptance: reproducible source identifiers, attribution, hashes, actual storage footprint and a rejection report. No automatic full LibriMix download.

### DATA-02: Enforce splits and reference selection

Implement manifests, speaker-disjoint assertions, distinct enrollment utterances, valid crop selection and deterministic development cases.

Acceptance: deliberate speaker leakage, identical target/reference utterances and invalid offsets are rejected. The same recipe reconstructs the same case.

### DATA-03: Implement the mixture generator

Implement level ratios, common gain, crop/padding masks, target assignment and paired reference-switch cases. Keep augmentation and mixture RNGs independent.

Acceptance: numerical tests verify mixture sums, level ratios and timing; an audited gallery of 20 cases sounds correct.

### METRIC-01: Implement loss and baseline metrics

Implement masked SI-SDR, improvement over mixture and degeneracy handling. Add a gain-sensitive measure and confusion diagnostics.

Acceptance: known signals validate scale behavior, wrong-speaker behavior and silent-target handling. B0 produces exactly zero SI-SDR improvement within numerical tolerance.

### DATA-04: Freeze the pilot/development protocol

Record case counts, split hashes and initial validation criteria. Keep final test cases reserved for the later frozen comparison.

Acceptance: all audits pass and no normalization or vocabulary fitting uses development/test speakers.

## M3 — Model and tiny-set learning

### MODEL-01: Implement both encoders and length handling

Write mixture encoding/decoding and reference encoding/pooling from the documented design using general PyTorch operations.

Acceptance: varying lengths round-trip to the correct output shape; padded samples do not affect reference summaries; gradients reach both encoders.

### MODEL-02: Implement conditioning and the separator

Write temporal blocks, reference-derived affine modulation, skip aggregation and waveform reconstruction.

Acceptance: actual parameter count is recorded; CPU/MPS forward/backward passes are finite; no dependency on SpeakerBeam source or weights exists.

### TRAIN-01: Implement the minimal trainer

Add configuration validation, optimizer, gradient accumulation/clipping, validation, structured metrics and atomic checkpoints.

Acceptance: resume restores optimizer and RNG/sampler state; checkpoint reload reproduces predictions; incomplete writes preserve a valid prior checkpoint.

### TRAIN-02: Overfit a tiny speech set

Train on 16 fixed cases and inspect outputs. Include paired requests for each speaker in the same mixture.

Acceptance: substantial improvement over initialization, correct target switching and no collapsed reference encoder. If this fails, debug before adding training data or model depth.

## M4 — Clean-reference control

### EXP-01: Run the 1,000-update pilot

Train the clean-reference model and compare development performance against B0. Record quality, confusion, reference diagnostics, memory and time.

Acceptance: a complete report with positive held-out learning or a specific failure diagnosis. More training is not the automatic response to a failed reference-switch test.

### EXP-02: Train the main clean-reference control

Expand the data under the measured disk budget. Choose and record a feasible update budget, using development data for checkpoint selection.

Acceptance: immutable B1 artifact, complete provenance and a stable development evaluation. Freeze the core architecture for the first treatment comparison.

### INF-01: Implement short-file extraction

Add validated WAV/FLAC input, fixed resampling, reference encoding and exact-length output using the same preprocessing as evaluation.

Acceptance: local inference matches evaluation outputs for the same inputs and model; invalid references receive explicit errors.

## M5 — Robustness study

### AUG-01: Implement reference corruption

Add independent noise, channel and reverb transforms with bounded severity and saved parameters. Audit samples by listening.

Acceptance: transforms do not change identity labels, leak split/target position or accidentally modify the supervision. Validation reconstructions are deterministic.

### EXP-03: Run the paired B1/B2 comparison

Train the augmented-reference model with the same architecture, mixture schedule and update budget as B1.

Acceptance: paired clean/mismatch results, confusion rates and efficiency measures. Report regressions and negative findings as well as gains.

### EXP-04: Run targeted ablations

Start with augmentation families. Add reference duration, auxiliary speaker supervision or reduced width only when a concrete question remains.

Acceptance: one controlled change per comparison, fixed data protocol and an explanation of what the experiment resolves. Avoid a large hyperparameter grid without evidence it is needed.

### EVAL-01: Freeze the final protocol

Finalize thresholds on development data, hash the test manifest and freeze the selected comparison. Target three paired training seeds if resources permit. Use development cases for ongoing inference and product choices; reserve the final test run for M7 after the delivered inference path is fixed.

Acceptance: complete development analysis, fixed aggregation rules and exclusions, a frozen test manifest, and an explicit record of selected artifacts. No final-test feedback has been used to optimize the product.

## M6 — Product and serving

### INF-02: Implement bounded-memory long-file inference

Cache the reference representation, preserve gain/timing and evaluate chunk context and blending.

Acceptance: 10/30/60-second output lengths are exact; boundary artifacts and reference consistency are assessed; quality relative to whole-clip inference is reported.

### API-01: Implement the local extraction API

Add health/readiness/model routes, limited multipart uploads, a single in-flight extraction, temporary-file cleanup and useful errors.

Acceptance: success, unsupported audio, silent reference, over-limit input, busy service and missing-model paths behave predictably. No raw audio is retained or logged by default.

### UI-01: Build the audio comparison interface

Implement upload, validation, processing state, synchronized playback, output download and model/runtime details.

Acceptance: the full workflow succeeds with real model output. The interface describes the trained operating envelope and has usable failure states.

### PERF-01: Profile the exact delivered system

Measure cold/warm end-to-end latency and memory. Compare MPS and CPU. Consider optimization only after profiling identifies a meaningful bottleneck.

Acceptance: repeatable timing report for the served artifact. Any compressed/exported model must pass quality comparisons before replacing it.

## M7 — Portfolio release

### EVAL-02: Run the final frozen comparison

Evaluate the selected B1/B2 artifacts and the exact delivered inference path on the reserved cases. Produce per-case results, approximate clustered confidence intervals, seed variation, exclusion counts and a listening gallery.

Acceptance: report the full result, including regressions, without tuning on it. Any subsequent test-driven changes are explicitly marked as further exploratory work or require fresh reserved evaluation data.

### RELEASE-01: Produce the model card and reproduction guide

Document training protocol, dataset origin, configuration, hardware, quality, limitations, artifact hashes and exact reproduction commands that actually run.

Acceptance: every claim points to a report or artifact. No placeholder performance numbers remain.

### RELEASE-02: Produce the technical case study

Explain the user problem, independent implementation, principal hypothesis, failure cases, controlled comparisons and deployment tradeoffs. Include an architecture figure and attributed audio examples where distribution is allowed.

Acceptance: a reviewer can distinguish the learned contribution from interface work and understand why each major decision was made.

### RELEASE-03: Verify a clean installation and artifact

Run documented setup and inference from a clean environment, verify checksums and ensure the public repository contains only intended files.

Acceptance: the demonstrated version can be reproduced; code and artifact licensing decisions are documented before distributing a model release.

## Deferred backlog

Target absence/presence calibration; streaming with causal state; multilingual and three-speaker extraction; measured room/microphone data; optional official Libri2Mix adapter; portable CPU container; optional external model comparisons.

These are not required to finish the first complete project. Promote one only when a measured failure or a deliberate product decision justifies the additional scope.

## Next implementation session

Complete ENV-01 through ENV-03 first. Then implement DATA-01 and METRIC-01 on a bounded speech selection. Do not begin a large training run until M2 audits and the M3 tiny-set gate pass.
