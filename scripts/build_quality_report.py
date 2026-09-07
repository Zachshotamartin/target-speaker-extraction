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
    print("Built docs/MODEL_CARD.md from frozen quality-release evidence")


if __name__ == "__main__":
    main()
