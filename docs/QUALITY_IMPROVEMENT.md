# Audio quality diagnosis and next experiments

Listening feedback identified weak suppression of the competing voice and substantial static. The pipeline is operational, but the model's audible extraction quality needs further work. This follow-up separates a delivery bug from the learning problem. It uses development data only; the original final test and v0.1.0 reports remain frozen.

## What was measured

The selected network is unchanged: exported SHA-256 `f3271f1decf7ec0e8e9b0f1fab578a693f51c4a1d1778adeeb9109a5e64a5c20`.

- The first 20 existing gallery outputs stay below full scale (maximum absolute sample 0.986). Their reported static therefore is not explained by output sample overflow.
- On the broader 400-request development set, 58 raw outputs exceed full scale; the largest sample magnitude is 1.279. Browser audio has a nominal −1 to +1 range, so this creates a playback-clipping risk. See [MDN's AudioBuffer documentation](https://developer.mozilla.org/en-US/docs/Web/API/AudioBuffer).
- These four-second cases bypass chunking entirely. On the first development-report case, CPU and MPS outputs differ by at most 2.69e-7, and float WAV encode/decode changes no samples. These checks do not suggest chunking, file quantization or device arithmetic as the explanation for that case's audible artifacts.
- The mean residual energy after fitting the two clean sources to raw output is 11.61% of output energy. This is a **distortion proxy**, not a perceptual noise measure: phase, filtering and time-varying gain errors also count. It supports investigating waveform reconstruction but does not establish a single cause for perceived static.

Full protocol and per-case values: [development audit](../reports/quality-audit-development.json). Reproduce with `uv run python scripts/audit_audio_quality.py`. The script rejects test manifests.

## Implemented playback correction in v0.1.1

The shared file inference path uniformly attenuates an output when its sample peak exceeds 0.98. It never amplifies quiet outputs and never hard-clips samples. The original waveform shape, duration and relative speaker levels are preserved. This is a sample-peak guard, not a true-peak limiter or denoiser.

The API records `processing_version: sample-peak-guard-v1`, `raw_output_peak`, `playback_gain` and delivered `output_peak`. The interface explains when volume was reduced. Low-level `extract_array` retains unscaled model predictions for research.

On all 400 development requests, the largest delivered peak is 0.98000002 (float32 rounding), with attenuation on 68 cases. Mean SI-SDR improvement remains 2.062 dB and confusion remains 13.25%, as expected for uniform gain. Mean normalized L1 changes from 0.4529 to 0.4472. [Corrected delivery report](../reports/delivered-development-v0.1.1.json). API regression tests verify the download avoids overflow, preserves waveform shape and leaves quieter outputs unchanged.

The v0.1.0 gallery files and final-test/runtime reports are preserved. They describe the original delivery path, not a newly evaluated final release. The two models were not retrained during this audit.

## Why a cleanup filter is insufficient

Two fixed, non-oracle candidates use 32 ms Hann windows with an 8 ms hop. Candidate A projects the estimate onto a real attenuation mask of the mixture's spectrum, preserving mixture phase. Candidate B caps predicted magnitude at mixture magnitude while retaining predicted phase. Neither uses the clean target during inference. Both can remove desired speech when voices interfere destructively.

| Method | Mean development SI-SDRi | Confusion | Mean distortion proxy | Outputs above full scale |
| --- | ---: | ---: | ---: | ---: |
| Raw network output | 2.062 dB | 13.25% | 11.61% | 58 / 400 |
| Candidate A: mixture phase | 2.171 dB | 12.75% | 9.58% | 0 / 400 |
| Candidate B: spectral limit | 2.105 dB | 12.75% | 10.41% | 0 / 400 |

Candidate A's +0.109 dB mean gain is modest and exploratory. These are the same development requests, not new final-test results. The distortion proxy reduction does not establish a corresponding perceived-noise reduction. No candidate is promoted into the main app on these metrics alone.

[Listen to the comparison locally](http://127.0.0.1:8000/gallery/quality-audit/): the first six original gallery requests, including paired target switches, with current output, both candidates, mixture and clean target. Selection is independent of scores. Audio amplitudes are retained; lower volume alone is not better extraction. Files include LibriSpeech/CC BY 4.0 attribution and source identities.

## Recommended training sequence

These are proposed experiments, not implemented improvements or guaranteed gains. Keep a separate run directory and retain the current checkpoint as a baseline. Change one factor at a time before combining successful treatments.

1. **Train for cleaner waveform reconstruction.** Add a multi-resolution spectral reconstruction loss alongside the existing target-specific SI-SDR and amplitude loss. First verify loss gradients on CPU/MPS and reconstruction of clean single-speaker audio. Then compare a bounded additional-training control against the same budget with spectral loss, using identical mixture schedules. Listen for hiss, buzzing and damaged consonants at matched playback loudness. Multi-resolution STFT loss has supported high-fidelity waveform training in [Parallel WaveGAN](https://arxiv.org/abs/1910.11480); applying it to this extractor is a hypothesis that needs its own evaluation.
2. **Teach the reference encoder speaker identity explicitly.** The current classifier weight is zero; the speaker representation is learned only through extraction. Compare auxiliary training-speaker classification or same-speaker/different-speaker contrastive supervision, retaining disjoint development speakers. The classifier is a training objective, not an inference requirement that users belong to known identities. Joint reconstruction and speaker supervision is an established direction in [SpEx](https://arxiv.org/abs/2004.08326). Reusing the idea requires neither its implementation nor its checkpoints.
3. **Expand speaker diversity and continue training with validation.** The pilot has only 60 training speakers and 10.636 eligible source hours. Each 5,000-update run processes about 22.22 hours of two-second mixture crops with reuse, and its final update is the selected checkpoint. This suggests testing a longer budget; it does not prove that more updates alone will solve generalization. Grow toward the official [LibriSpeech train-clean-100 collection](https://www.openslr.org/12), measure disk/memory/time on this Mac, and compare 2-second versus 4-second crops independently. Use a learning-rate schedule and stop on development behavior, not a fixed assumption that more training is always better.

Track speaker confusion and both-target success as well as SI-SDRi. Add listening comparisons and an intelligibility measure; inspect target pauses so reduced interference does not come from deleting speech. Preserve natural volume relationships or document any playback matching. A useful next model must sound cleaner and keep the intended words, not merely improve a single average.

Select subsequent variants using development data. Because the original test has been inspected, any new final generalization claim after test-informed design decisions needs a fresh held-out evaluation protocol; do not relabel the original test as unseen.
