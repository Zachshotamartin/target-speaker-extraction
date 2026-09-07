#!/usr/bin/env python3
"""Build a quality-release model card only from matching frozen test/runtime evidence."""

import json
from pathlib import Path

from tse.quality import compare_quality
from tse.reporting import analyze_failures
from tse.utils import sha256


def read(path: str) -> dict:
    return json.loads(Path(path).read_text())


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def main() -> None:
    freeze_path = Path("reports/quality-v2-selection.json")
    frozen = read(str(freeze_path))
    if frozen["status"] != "frozen_before_fresh_test":
        raise ValueError("A pre-test selection is required")
    release = read("artifacts/releases/v0.2.0/model.json")
    reports = {
        name: read(f"reports/quality-v2-{name}-test.json") for name in ("baseline", "selected")
    }
    candidate = reports["selected"]
    selected_hash = release["checkpoint_sha256"]
    if selected_hash != frozen["checkpoints"]["candidate"]["checkpoint_sha256"]:
        raise ValueError("Export differs from frozen selection")
    for name, report in reports.items():
        expected = frozen["checkpoints"]["candidate" if name == "selected" else "baseline"]
        if (
            report["split"] != "test"
            or report["checkpoint_sha256"] != expected["checkpoint_sha256"]
            or report["selection_freeze_sha256"] != sha256(freeze_path)
            or report["case_manifest_sha256"] != frozen["case_manifest_sha256"]
            or report["source_manifest_sha256"] != frozen["source_manifest_sha256"]
        ):
            raise ValueError("Test evidence differs from the frozen artifact/protocol")
    profiles = {
        device: read(f"reports/quality-v2-delivery-{device}.json") for device in ("cpu", "mps")
    }
    if any(profile["model"]["checkpoint_sha256"] != selected_hash for profile in profiles.values()):
        raise ValueError("Runtime evidence describes another model")
    paired = compare_quality(
        Path("reports/quality-v2-baseline-test.json"),
        Path("reports/quality-v2-selected-test.json"),
        Path("reports/quality-v2-test-comparison.json"),
    )
    failures = analyze_failures(
        Path("reports/quality-v2-selected-test.json"),
        Path("data/expanded/manifests/fresh-test-cases.json"),
        Path("data/expanded/manifests/inventory.json"),
        Path("reports/quality-v2-test-failures.json"),
    )["conditions"]["clean"]
    summary = candidate["summary"]
    metrics = [
        ("Mean SI-SDR improvement", "mean_si_sdri_db", lambda value: f"{value:.3f} dB"),
        ("Median SI-SDR improvement", "median_si_sdri_db", lambda value: f"{value:.3f} dB"),
        ("ESTOI intelligibility proxy", "mean_estoi", lambda value: f"{value:.4f}"),
        ("Requests worse than mixture", "negative_improvement_fraction", pct),
        ("Speaker confusion proxy", "confusion_fraction", pct),
        ("Scalar distortion proxy", "mean_artifact_proxy_fraction", pct),
    ]
    table = ["| Metric | Original model | Updated model |", "| --- | ---: | ---: |"]
    for label, key, formatter in metrics:
        values = [formatter(reports[name]["summary"][key]) for name in ("baseline", "selected")]
        table.append(f"| {label} | {values[0]} | {values[1]} |")
    timing = [
        "| Device | Recording | Warm median / p95 | Exact sample count |",
        "| --- | ---: | ---: | --- |",
    ]
    for device, profile in profiles.items():
        for row in profile["rows"]:
            timing.append(
                f"| {device.upper()} | {row['duration_seconds']} s | {row['warm_median_seconds']:.3f} / {row['warm_p95_seconds']:.3f} s | {row['exact_length']} |"
            )
    gain = paired["metrics"]["si_sdri_db"]
    config = release["config"]
    averaging = release["provenance"].get("checkpoint_average")
    averaging_note = (
        "Uniformly averaged project checkpoints at expanded-data steps "
        + ", ".join(str(source["step"]) for source in averaging["sources"])
        + ". Source hashes are retained in the exported provenance."
        if averaging
        else f"A single project checkpoint at update {release['training_updates']:,} of its recorded run."
    )
    card = f"""# One voice — v0.2.0 model card

Experimental, independently implemented target-speaker extraction for two overlapping voices. A separate recording identifies the desired speaker. The updated model achieves **{summary["mean_si_sdri_db"]:.3f} dB mean SI-SDR improvement** on the frozen fresh test. It still worsens {pct(summary["negative_improvement_fraction"])} of requests and triggers the speaker-confusion proxy on {pct(summary["confusion_fraction"])}. These results do not establish production speech quality.

## Artifact and intended use

- Exported SHA-256: `{selected_hash}`.
- Architecture: `{config["model"]["family"]}`; separator normalization: `{config["model"].get("separation_normalization", "per_frame")}`.
- Parameters loaded: {candidate["model"]["parameters"]:,}, including any training speaker head retained in the checkpoint. Inference selects speech from the reference embedding, not the head's identity labels.
- {averaging_note} Initialization includes earlier training within this project; the displayed update count is not the complete training history.
- No external pretrained speaker or separator weights. Original model: [v0.1.0 card](MODEL_CARD_V0_1.md).

Inputs are WAV/FLAC, up to 60 seconds of mixture and 3–10 seconds of separate reference. Output is mono 16 kHz with the exact mixture timeline. The requested speaker must be present. The system is offline and noncausal; faster-than-duration processing is not live-call latency.

## Fresh held-out evaluation

The fresh test contains {summary["cases"]:,} extraction requests from {summary["target_speakers"]} reserved identities, paired by swapping targets in the same mixture. These identities were withheld before expanded training and never trained in the original model. The original opened test-clean evaluation remains historical. This is a custom LibriSpeech split, not the official Libri2Mix benchmark.

Selection used development data, then bound exact checkpoint, source-inventory and test-recipe hashes before test scoring. [Selection record](../reports/quality-v2-selection.json). Both models use the same file decoding, preprocessing, peak guard and float-WAV delivery path.

{chr(10).join(table)}

The paired mean separation gain is {gain["mean_candidate_minus_baseline"]:+.3f} dB, with approximate 95% target-speaker-cluster interval [{gain["paired_ci95"][0]:.3f}, {gain["paired_ci95"][1]:.3f}]. The updated mean's corresponding interval is [{summary["mean_ci95_db"][0]:.3f}, {summary["mean_ci95_db"][1]:.3f}] dB. Resampling uses 1,000 replicates and seed 42; shared-interferer dependence remains. These are one-run comparisons with combined model/data/training changes, not a causal estimate of any single change.

Both targets improve in {pct(failures["paired_requests"]["both_targets_improved_fraction"])} of complete mixture pairs. See [failure slices](../reports/quality-v2-test-failures.json), [per-request results](../reports/quality-v2-selected-test.json), and [paired comparison](../reports/quality-v2-test-comparison.json).

ESTOI predicts intelligibility; it is not word accuracy or a human rating. The scalar distortion proxy includes filtering, phase and envelope errors and is not perceived static. Confusion means the estimate's SI-SDR against the interferer exceeds its SI-SDR against the target by more than three dB. None of these measures proves that every voice sounds clean.

## Training and research alignment

The expanded pool contains 231 speakers and 90.582 eligible hours of clean LibriSpeech source recordings. Mixtures are generated on demand, with distinct speakers, complete overlap, ratios between −5 and +5 dB, and a different reference utterance, preferably another chapter. Forty dev-clean identities supply development data. Source hours are not mixture hours or training epochs.

The spectral separator uses a 512-sample Hann STFT with a 128-sample hop, reference-conditioned temporal convolutions and a real sigmoid mask. It retains mixture phase. Its own convolutional reference encoder learns jointly with extraction. Losses combine target-specific SI-SDR, normalized waveform L1, multi-resolution spectral reconstruction and training-speaker classification. Full settings and provenance accompany the export.

This custom system is informed by SpeakerBeam, SpEx, Conv-TasNet and enrollment-augmentation research. It does not reproduce their published recipes. The [research audit](RESEARCH_AUDIT.md) documents substitutions, normalization experiments and source-informed mask diagnostics. Oracle diagnostics use clean targets and are never model estimates.

## Delivery and runtime

The service removes DC, normalizes inputs, predicts and restores mixture gain. Outputs exceeding sample peak 0.98 receive uniform attenuation. Quiet outputs are never boosted and samples are never hard-clipped. This guard does not remove interference. Model information and response metadata identify the checkpoint and processing version.

{chr(10).join(timing)}

Measured on this Apple M3 Pro Mac without concurrent training. Each device runs in its own process. End-to-end time includes file decode, inference and WAV encode, excluding browser/network/HTTP overhead. The profiler repeats a development clip to 10/30/60 seconds; this is throughput and alignment evidence, not natural long-conversation quality. [CPU protocol](../reports/quality-v2-delivery-cpu.json), [MPS protocol](../reports/quality-v2-delivery-mps.json).

## Limits, use and origin

The test covers clean, read English speech mixed synthetically. Microphones, noisy/reverberant rooms, spontaneous conversation, music, more than two speakers and out-of-domain languages are not established capabilities. Target absence has no calibrated detector or confidence score. The selected run uses clean references; the original augmentation study's robustness findings cannot simply be transferred to this model. No claim is made that static is eliminated or that output is suitable for forensic conclusions.

The [local gallery](http://127.0.0.1:8000/gallery/) uses the first 20 development requests, chosen independently of scores. A [matched-volume comparison](http://127.0.0.1:8000/gallery/quality-progress/) retains the original model for comparison. Playback matching is clearly labeled and does not alter normal inference.

Speech is from [LibriSpeech / OpenSLR 12](https://www.openslr.org/12), Panayotov et al. (2015), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Local examples are cropped, normalized and mixed derivatives with attribution. Per-file hashes and source identities are retained; the streaming acquisition did not verify the publisher's complete archive checksum. Raw audio and weights stay outside Git. Original project code has no selected reuse license.

See [quality reproduction](REPRODUCING_QUALITY.md) and the [development worklog](QUALITY_WORKLOG.md). The test freeze must remain immutable; subsequent test-informed changes need a new generalization protocol.
"""
    Path("docs/MODEL_CARD.md").write_text(card)
    normalization = read("reports/quality-globalnorm-development.json")["summary"]
    control = read("reports/quality-stft-expanded-fastlr-2000.json")["summary"]
    study = f"""# Improving target-speaker extraction on one Mac

The first One voice release had a working training/evaluation/API pipeline, but weak separation and audible artifacts. The user heard the competing voice become quieter without being adequately removed. The quality follow-up's frozen fresh test measures **{summary["mean_si_sdri_db"]:.3f} dB mean SI-SDR improvement**, versus {reports["baseline"]["summary"]["mean_si_sdri_db"]:.3f} dB for the original model on the same requests. Remaining failures are part of the result: {pct(summary["negative_improvement_fraction"])} of requests worsen and {pct(summary["confusion_fraction"])} trigger the confusion proxy.

## Diagnose the signal path before redesigning the model

A development audit found 58 of 400 original predictions above sample magnitude one. A uniform attenuation guard fixes that playback risk without changing relative speaker levels. However, the first 20 original gallery outputs were already below full scale. Four-second gallery requests bypass chunking, float WAV round-trips preserved samples, and a CPU/MPS comparison was numerically close. Those checks did not support treating clipping or serialization as the sole explanation for the reported static. [Audio diagnosis](QUALITY_IMPROVEMENT.md).

Fixed cleanup filters and a matched additional-training comparison with spectral loss produced modest gains. Increasing mask strength suppressed more interference while damaging intelligibility. These experiments discouraged treating quieter background audio as sufficient evidence of better extraction.

## Rebuild the learning experiment and preserve an independent test

The new candidate uses independently implemented reference-conditioned temporal convolutions over an STFT, with a bounded real mask and inverse transform. Its reference encoder is learned within this project. Joint speaker classification, waveform and multi-resolution spectral objectives supplement target-specific SI-SDR. Expanded training draws from 231 speakers and 90.582 eligible source hours, using four-second mixtures and separate five-second references.

The original test had already been opened. Twenty previously unused train-clean-100 identities were therefore reserved before expanded training, while the same 40 dev-clean identities continued to guide development. Final selection binds exact model, inventory and recipe hashes before the new 1,000-request test. These are custom LibriSpeech mixtures, not official Libri2Mix results.

The principal development changes combine architecture, speaker supervision, crop length, data diversity and training exposure. Their aggregate improvement cannot be attributed to one component. A model small enough for local experiments remains a hypothesis, not evidence that the published research recipe has been reproduced.

## Check the research assumptions explicitly

The [research audit](RESEARCH_AUDIT.md) compares the implementation with SpeakerBeam, SpEx, Conv-TasNet and the enrollment-augmentation study. It identifies substantial substitutions, including the small reference encoder, per-frame separator normalization, real mixture-phase masks, custom augmentation and much shorter training schedule. No external model weights or SpeakerBeam implementation enter the core.

A source-informed diagnostic reaches 14.559 dB with the same bounded-mask format. This uses unavailable clean targets, so it is not deployable performance or a guaranteed learning ceiling. It demonstrates representational headroom and helps distinguish output-format restrictions from imperfect mask prediction.

A separate 2,000-update normalization adaptation matches initialization tensors, fresh classifier, seed, data sequence, losses and optimizer. Changing only separator normalization gives {normalization["mean_si_sdri_db"]:.3f} dB / {normalization["mean_estoi"]:.4f} ESTOI, versus {control["mean_si_sdri_db"]:.3f} dB / {control["mean_estoi"]:.4f} for the matched per-frame control on 400 development requests. The starting weights were already trained with per-frame normalization, so this short comparison cannot settle which architecture is better when trained from scratch. The global variant also requires whole-clip inference because its statistics span time.

The main expanded run completed 20,000 updates. The predeclared final-three and final-five checkpoint averages were scored alongside its best single checkpoint. Final selection and all compared report identities are retained in the [freeze](../reports/quality-v2-selection.json). {averaging_note}

## Measure the delivered result

{chr(10).join(table)}

The paired separation gain is {gain["mean_candidate_minus_baseline"]:+.3f} dB, with approximate 95% interval [{gain["paired_ci95"][0]:.3f}, {gain["paired_ci95"][1]:.3f}]. The [model card](MODEL_CARD.md) specifies metrics, uncertainty, failure slices, runtime and artifact identities. The original [case study](CASE_STUDY_V0_1.md) remains available separately; its opened test was not relabeled as unseen.

The service and evaluator share decoding, normalization, prediction, peak handling and float-WAV output. Long-file checks verify duration and the applicable inference strategy. Training uses deterministic sample seeds, atomic resumable checkpoints and provenance records; compiled temporal layers were numerically checked before use on MPS. Exact CPU resume has regression coverage. Code and metadata are public, while audio and weights remain local.

The [comparison gallery](http://127.0.0.1:8000/gallery/quality-progress/) contains examples chosen by manifest order and separately labeled matched-RMS playback. ESTOI predicts intelligibility and the distortion projection is a proxy; neither constitutes a human judgment that static is gone. The model remains experimental for two target-present voices in recorded audio. Real microphones, reverberation, background noise, target absence and broader language/domain transfer require their own evidence.

[Reproduce the quality experiments](REPRODUCING_QUALITY.md).
"""
    Path("docs/CASE_STUDY.md").write_text(study)
    print("Built current model card and case study from frozen quality-release evidence")


if __name__ == "__main__":
    main()
