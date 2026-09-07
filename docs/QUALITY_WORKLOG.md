# Quality development after the first release

The requested outcome is substantially cleaner target speech with stronger rejection of the competing voice. The v0.1.0 model and original final test are retained as historical evidence. Development experiments may fail; no experimental checkpoint replaces the served model merely because training finished.

## Completed diagnostics and pilots

- Fixed output sample overflow in v0.1.1. This preserves waveform shape and cannot improve voice selection.
- Two fixed spectral postfilters yield only modest development changes; neither is enabled in serving. See [the quality audit](QUALITY_IMPROVEMENT.md).
- Added multi-resolution spectral loss, a supervised speaker objective, configurable classifier logit scale, checkpoint initialization, learning-rate reduction and exact resume checks.
- Compared 1,000 additional updates of the existing model with/without spectral loss. The 400-case development means are 2.353 / 2.364 dB, and ESTOI is 0.5450 / 0.5468. The difference is insufficient to claim a substantial quality improvement. [Continuation](../reports/quality-continuation-pilot.json), [spectral loss](../reports/quality-spectral-pilot.json).
- Implemented an independent STFT-mask model: fixed 32 ms analysis and inverse transform, a bounded real mask, conditioned temporal blocks, and our existing reference encoder. Its 2,000-update pilot obtains 2.213 dB and ESTOI 0.5604 on 400 development requests. All original weights are from this project's training; no external speaker/extraction checkpoint is used.
- A clean-source reference-matching diagnostic is correct in 91.5% of 200 development comparisons, versus 92.0% in training. It is not an extraction score; it helps distinguish a weak reference representation from weak mask estimation. [Diagnostic](../reports/reference-quality-diagnostic.json).

The architecture experiments change multiple factors and are exploratory. Their scores are not controlled estimates of a single architectural effect. The matched continuation/spectral pilots differ in spectral-loss weight.

## Expanded corpus

The full official train-clean-100 download contains 28,539 FLAC recordings from 251 speakers. A deterministic, predeclared rule reserves 20 identities never used in v0.1.0 or the pilots. The resulting training pool has 231 speakers and 90.582 eligible hours in 24,377 utterances. Original dev-clean audio and mixture recipes remain the development set; original test-clean is excluded from the expanded manifest.

The fresh reserved set has 1,000 paired recipes from the 20 reserved identities. It remained unscored through model selection; scoring begins only after the hash-bound freeze. This is a custom split of public LibriSpeech, not an official Libri2Mix benchmark. [Predeclared plan](../reports/expanded-data-plan.json), [prepared data](../reports/expanded-data.json).

## Expanded training protocol

The expanded STFT run starts from the pilot backbone, creates a fresh classifier for 231 labels, and uses four-second mixtures, five-second references, effective batch eight, SI-SDR plus waveform/spectral losses and speaker classification. Checkpoint selection uses the existing 80 development requests; broader comparison uses 400 development requests and listening examples chosen independently of scores.

Two learning rates, 0.0003 and 0.001, were compared through 2,000 updates with the same initialization/data schedule. Their 400-request results were close: 3.443 / 3.484 dB SI-SDRi and 0.5867 / 0.5852 ESTOI. Neither clearly dominates across metrics. The higher-rate run was continued through 20,000 updates with a validation-driven scheduler; this was exploratory budget allocation, not a demonstrated learning-rate advantage. See [the comparison decision](../reports/expanded-learning-rate-comparison.json). The latter is also the initial rate in the [Conv-TasNet paper](https://arxiv.org/abs/1809.07454), but that paper uses a different dataset and model; this motivated a local experiment rather than establishing that it would improve this system. The lower-rate run remains retained separately.

Commands and current artifacts:

```sh
uv run python scripts/prepare_expanded_data.py
uv run tse train --config configs/quality-stft-expanded.json --root data/expanded/raw --manifest data/expanded/manifests/inventory.json --dev-cases data/expanded/manifests/dev-cases.json --run artifacts/runs/quality-stft-expanded --initialize-from artifacts/runs/quality-stft-pilot/best.pt --device mps
uv run python scripts/evaluate_quality.py --checkpoint artifacts/runs/quality-stft-expanded/best.pt --output reports/quality-stft-expanded.json --device cpu
```

For an existing run, use `--resume` and omit initialization. Do not run concurrent writers to the same directory. Learning curves, intelligibility, confusion, audio comparisons and failure slices guide selection. Final test scoring waits until selection is frozen. The [current model card](MODEL_CARD.md) identifies the promoted artifact; intermediate checkpoints are not automatically served.

## Delivery and evaluation checks

The local [listening comparison](http://127.0.0.1:8000/gallery/quality-progress/) includes the first six development requests with both original amplitude and documented matched-RMS playback. Model hashes and audio transformations accompany each comparison. It began as a training progress view and now identifies the promoted v0.2.0 release.

The quality evaluator now requires a hash-bound, pre-test selection record before accepting test cases. It rejects unlisted checkpoints, changed recipes and changed source inventories; it also checks speaker disjointness. Paired comparisons align requests by identity and record clustered uncertainty. These controls help avoid accidental test reuse; they do not make repeatedly inspected development data an independent test.

A synthetic MPS batch check measured only a small difference between microbatch four/accumulation two and batch eight: 0.164 / 0.157 seconds for forward/backward, excluding optimizer and audio preparation. The larger batch used more driver memory, so the existing effective batch and partition remain. An optional `--prefetch` reader prepares one optimizer batch on a CPU thread; per-example seeds and CPU resume regression tests verify identical training values. It does not alter the model, loss, sample schedule or effective batch. Reported final serving benchmarks must run without training active.

## Later checkpoints and acceleration

The best checkpoint available through 5,000 expanded-data updates was selected at step 3,500. Its 400-request development result is 3.825 dB SI-SDRi, 0.5995 ESTOI and 11.25% confusion; the original model scores 2.062 dB, 0.5387 and 13.25%. [Checkpoint evaluation](../reports/quality-stft-expanded-fastlr-5000.json). These are development measurements during continued tuning.

A fixed mask-power pilot on the first 80 development-report requests compares powers 1, 1.25, 1.5 and 2. Stronger masks suppress the interfering source more, but ESTOI drops and the distortion proxy rises. Power 1.25 adds only 0.008 dB; power 2 loses 0.221 dB and 0.0161 ESTOI. No mask-power option is enabled in serving. This demonstrates why lower background volume alone is insufficient. [Pilot](../reports/mask-strength-pilot.json).

Whole-extractor compilation failed in the installed PyTorch Metal backend. Compiling just the real-valued temporal layers works while STFT/ISTFT remain eager. In an isolated real-batch check, warm training compute was 0.188 seconds eager versus 0.112 seconds compiled, excluding data preparation. Initial predictions differ by at most 1.20e-7 and parameter-gradient relative L2 error is 3.19e-6. Compiler arithmetic is not bit-identical; unused final residual parameters may receive zero gradients where eager uses None. [Reproducible benchmark](../scripts/benchmark_temporal_compile.py), [results](../reports/compiled-training-benchmark.json).

An initial compiled continuation hit the default eight-entry graph cache during validation. It was restarted from the preceding atomic checkpoint at step 5,247, so the intervening uncheckpointed steps were repeated. The compiler now allows 32 entries for the six dilation patterns in training/evaluation modes. The log retains the failed segment; repeated step numbers do not represent additional retained optimizer updates. Compiled training is optional (`--compile-blocks`, validated with locked PyTorch 2.14 on MPS); saved weights load through ordinary eager inference on CPU or MPS.

PyTorch documents [compilation and recompilation controls](https://docs.pytorch.org/docs/stable/generated/torch.compile). Its [MPS code generator](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/codegen/mps.py) remains incomplete; local feasibility and numerical checks determine which parts are used here.

## Predeclared checkpoint averaging

The trajectory collector preserves fixed steps 8,000 through 20,000 every 2,000 updates. Two fixed arithmetic averages will be compared on development: the final three and final five snapshots. These are a low-cost hypothesis, not guaranteed improvements or a reproduction of the stochastic-weight-averaging paper's optimizer schedule. The [plan](../reports/checkpoint-averaging-plan.json) was written before the first of these snapshots. The approach is motivated by [Izmailov et al.](https://arxiv.org/abs/1803.05407); all averaged weights remain from this project.

The collector verifies exact checkpoint steps and preserves hashes. Averaging requires matching configurations, training labels, initializations and tensor shapes, and drops optimizer/RNG state. A regression test verifies the arithmetic, retained source files and mismatch rejection. No averaged model is eligible for final testing until it is scored and selected on development.

## Research cross-check and 10,000-update measurement

The fixed 10,000-update checkpoint reaches 4.890 dB SI-SDRi, 0.6223 ESTOI, 9.75% confusion and 15.75% negative improvements on 400 development requests. These remain exploratory development results. [Report](../reports/quality-stft-expanded-fastlr-10000.json).

The [research alignment audit](RESEARCH_AUDIT.md) records important differences from SpeakerBeam, SpEx and the enrollment-augmentation guide. In particular, per-frame channel normalization was a custom inference convenience, not a validated substitute for the global normalization in noncausal Conv-TasNet. A controlled 2,000-update global-normalization pilot is planned after the current run, using identical pilot initialization/data/optimizer to the existing 2,000-update expanded control. The reference encoder is unchanged. Whole-clip inference is required for the variant; current served models preserve their existing behavior.

A clean-source mask diagnostic reaches 14.559 dB with the same bounded real-mask representation. This is unavailable-ground-truth reconstruction, never a learned result or application path. It demonstrates representational headroom but cannot promise that a network will learn those masks. [Diagnostic](../reports/mask-capacity-development.json).

## Completed main run, normalization comparison and frozen selection

The main run completed 20,000 expanded-data updates. Its best single checkpoint, selected at 19,500, scores 5.755 dB SI-SDRi, 0.6440 ESTOI and 7.25% confusion on 400 development requests. Uniform averages of the final three and final five fixed snapshots score 6.046 / 5.954 dB and 0.6510 / 0.6490 ESTOI, both with 6.50% confusion. Their distortion proxies are 10.73% / 10.63%, versus 11.48% for the single checkpoint. The final-three average was selected for its stronger separation and intelligibility. [Single](../reports/quality-final-single-development.json), [three](../reports/quality-final-last3-development.json), [five](../reports/quality-final-last5-development.json).

The global-normalization adaptation completed its matched 2,000-update budget. It scores 3.673 dB and 0.5916 ESTOI, versus 3.484 dB and 0.5852 for the matched per-frame control. The paired gain is +0.189 dB with approximate cluster interval [0.009, 0.356]; the distortion proxy rises by 0.80 percentage points and confusion by 1.00 point. This is a modest tradeoff, not a clear solution to audible artifacts. The short adaptation cannot establish which normalization would perform better under full from-scratch training. [Comparison](../reports/global-normalization-comparison.json).

The final-three averaged export is frozen as SHA-256 `c814d937b5876007a299e8861f29522cfac41130560bd6c678595ee03a80433e`. Its parameters were verified identical to the selected source weights after export. The [selection record](../reports/quality-v2-selection.json) was committed before either model was scored on the fresh test. No additional training is part of this release cycle. Original test results, models, gallery audio and metadata remain separately preserved.

## Fresh test and v0.2.0 promotion

The frozen 1,000-request test from 20 reserved identities is now scored. The selected export reaches 5.887 dB mean SI-SDRi, compared with 1.453 dB for the original model on the same requests. The paired gain is +4.434 dB with approximate 95% target-speaker-cluster interval [3.455, 5.529]. ESTOI increases from 0.5457 to 0.6534, confusion decreases from 17.3% to 7.6%, and negative improvements decrease from 30.8% to 13.9%. [Paired results](../reports/quality-v2-test-comparison.json), [failure slices](../reports/quality-v2-test-failures.json).

The distortion-proxy difference is −0.00378 with interval [−0.01312, +0.00544]. It does not establish reduced perceived static. No human listening endorsement is recorded, and the model remains experimental. These test results were not used to tune this release.

The selected export now serves through the local API and regenerated 20-request gallery. The original gallery remains archived locally under `artifacts/gallery/v0.1.0/`. The comparison gallery uses the selected export and is labeled as a release comparison. Both target references passed real multipart extraction with exact 64,000-sample float WAV output and the expected model identity.

Real browser checks exercised both reference choices, extraction, playback to the end and WAV download. The downloaded waveform exactly matches the API example's decoded samples. The browser automation's download-event wait timed out, but the newly saved file was independently verified. The refreshed gallery identifies the new checkpoint and uses versioned audio URLs. [Browser evidence](../reports/quality-v2-browser-verification.json).

CPU/MPS delivery profiles verify exact lengths and chunk/whole agreement at 10, 30 and 60 seconds. For the repeated 60-second development clip, warm median end-to-end processing is 5.948 seconds on CPU and 0.465 seconds on MPS. No training ran concurrently. The separately installed v0.2.0 wheel runs the selected export on both devices; maximum CPU/MPS sample difference across the two four-second examples is 3.58e-7. The wheel uses the existing locked dependency environment. [CPU](../reports/quality-v2-delivery-cpu.json), [MPS](../reports/quality-v2-delivery-mps.json), [package checks](../reports/quality-v2-package-verification.json), [API checks](../reports/quality-v2-api-verification.json).
