#!/usr/bin/env python3
"""Build the final model card, case study and figures from completed local reports."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path: str) -> dict:
    return json.loads(Path(path).read_text())


def percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def interval(values: list[float]) -> str:
    return f"[{values[0]:.2f}, {values[1]:.2f}]"


def main() -> None:
    control = read("reports/control-test.json")
    augmented = read("reports/augmented-test.json")
    paired = read("reports/test-comparison.json")
    development = read("reports/development-comparison.json")
    selection = read("reports/model-selection.json")
    freeze = read("reports/test-freeze.json")
    release = read("artifacts/releases/model.json")
    delivered = read("reports/delivered-test.json")
    diagnostics = read("reports/reference-diagnostics.json")
    profiles = {device: read(f"reports/delivery-{device}.json") for device in ("cpu", "mps")}
    inventory = read("data/manifests/inventory.json")["audit"]
    overfit = read("artifacts/runs/overfit/summary.json")
    selected = augmented if selection["selected"] == "augmented" else control
    if (
        release["source_checkpoint_sha256"] != selected["checkpoint_sha256"]
        or delivered["checkpoint_sha256"] != release["checkpoint_sha256"]
        or freeze["case_manifest_sha256"] != delivered["case_manifest_sha256"]
        or any(
            report["model"]["checkpoint_sha256"] != release["checkpoint_sha256"]
            for report in profiles.values()
        )
    ):
        raise ValueError("Report inputs do not identify one frozen delivered artifact")
    conditions = ["clean", "noise", "channel", "reverb", "combined"]
    quality_table = [
        "| Reference | Control SI-SDRi | Augmented SI-SDRi | Paired gain [approx. 95% CI] | Confusion, control / augmented |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for condition in conditions:
        left, right = (report["summaries"][condition] for report in (control, augmented))
        gain = paired["conditions"][condition]
        quality_table.append(
            f"| {condition.capitalize()} | {left['mean_si_sdri_db']:.2f} dB | {right['mean_si_sdri_db']:.2f} dB | "
            f"{gain['mean_treatment_minus_control_db']:+.2f} dB {interval(gain['paired_ci95_db'])} | "
            f"{percent(left['confusion_fraction'])} / {percent(right['confusion_fraction'])} |"
        )
    timing_table = [
        "| Device | Audio | Warm median / p95 | Median real-time factor | Exact samples |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for device, profile in profiles.items():
        for row in profile["rows"]:
            timing_table.append(
                f"| {device.upper()} | {row['duration_seconds']} s | {row['warm_median_seconds']:.3f} / "
                f"{row['warm_p95_seconds']:.3f} s | {row['warm_median_real_time_factor']:.4f} | {row['output_samples']:,} |"
            )
    figures = Path("reports/figures")
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    colors = ("#77818b", "#47634c")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for name, color, label in zip(
        ("control", "augmented"),
        colors,
        ("Clean-reference control", "Reference augmentation"),
        strict=True,
    ):
        logs = [
            json.loads(line)
            for line in (Path("artifacts/runs") / name / "metrics.jsonl").read_text().splitlines()
        ]
        points = [row for row in logs if row["event"] == "validation"]
        axes[0].plot(
            [row["step"] for row in points],
            [row["mean_si_sdri_db"] for row in points],
            color=color,
            label=label,
        )
        axes[1].plot(
            [row["step"] for row in points],
            [100 * row["confusion_fraction"] for row in points],
            color=color,
            label=label,
        )
    axes[0].set(
        xlabel="Optimizer updates",
        ylabel="Mean SI-SDR improvement (dB)",
        title="Learning on unseen development speakers",
    )
    axes[1].set(
        xlabel="Optimizer updates",
        ylabel="Confusion proxy (%)",
        title="Requested-speaker selection",
    )
    axes[0].axhline(0, color="#aaa", linewidth=0.8, linestyle="--")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Checkpoint-selection set · 80 paired extraction requests · one training seed", fontsize=11
    )
    for extension in ("png", "svg"):
        fig.savefig(figures / f"learning-curves.{extension}", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    positions = np.arange(len(conditions))
    for index, (report, color, label) in enumerate(
        zip(
            (control, augmented),
            colors,
            ("Clean-reference control", "Reference augmentation"),
            strict=True,
        )
    ):
        means = np.array([report["summaries"][name]["mean_si_sdri_db"] for name in conditions])
        bounds = np.array([report["summaries"][name]["mean_ci95_db"] for name in conditions])
        errors = np.maximum(0, np.stack([means - bounds[:, 0], bounds[:, 1] - means]))
        ax.bar(
            positions + (index - 0.5) * 0.34,
            means,
            width=0.32,
            color=color,
            label=label,
            yerr=errors,
            capsize=3,
        )
    ax.set(
        xticks=positions,
        xticklabels=[name.capitalize() for name in conditions],
        ylabel="Mean SI-SDR improvement (dB)",
        title="Frozen test · 1,000 paired extraction requests per condition",
    )
    ax.axhline(0, color="#aaa", linewidth=0.8)
    ax.legend(frameon=False, fontsize=9)
    fig.supxlabel(
        "Bars: point estimates. Whiskers: approximate target-speaker cluster 95% intervals.",
        fontsize=8,
    )
    for extension in ("png", "svg"):
        fig.savefig(figures / f"test-conditions.{extension}", dpi=170)
    plt.close(fig)
    clean = selected["summaries"]["clean"]
    delivered_clean = delivered["summary"]
    absent = diagnostics["absent_targets"]
    constant = diagnostics["constant_conditioning"]["summary"]
    duration_lines = []
    for seconds, result in diagnostics["reference_duration_seconds"].items():
        if result["eligible_cases"]:
            duration_lines.append(
                f"- {seconds} seconds: {result['summary']['mean_si_sdri_db']:.2f} dB across {result['eligible_cases']} eligible cases; {result['excluded_cases']} excluded."
            )
    card = f"""# Model card · One voice 0.1

An experimental, independently implemented reference-conditioned speech extractor. The selected artifact is **{selection["selected"]}**, chosen using development data before opening the reserved test. All weights were trained from random initialization; no SpeakerBeam code or checkpoints were used.

## Identity and intended use

- Parameters: 1,223,296; 16 kHz mono; float32; noncausal.
- Selected training update: {release["training_updates"]:,}; matched experiment budget: {selection["step_budget"]:,} updates per model; seed 42.
- Export SHA-256: `{release["checkpoint_sha256"]}`.
- Source training checkpoint: `{release["source_checkpoint_sha256"]}`.
- Test case manifest: `{freeze["case_manifest_sha256"]}`.
- Artifact location: `artifacts/releases/model.pt`; accompanying config/provenance: `artifacts/releases/model.json`.
- Intended use: local research, engineering demonstration, and exploratory target-present two-speaker recordings with a separate 3–10 second reference. Maximum mixture duration: 60 seconds.

The model is an audio estimator, not an identity verifier, presence detector, or speech-transcription system. Do not interpret retained audio as proof that a particular person spoke. The app labels it as experimental.

## Data and training

Public [LibriSpeech](https://www.openslr.org/12), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), attributed to Panayotov, Chen, Povey and Khudanpur (2015). A bounded archive-order selection provides {inventory["train"]["speakers"]} training speakers and {inventory["train"]["hours"]:.3f} hours in the acquired usable inventory. Development/test each use 40 different speakers from their official clean partitions. Crop eligibility further filters the pool. English read speech and synthetic complete overlap limit generalization to spontaneous conversation, accents, languages, and recording conditions.

Training uses on-demand 2-second mixtures, 5-second distinct-utterance references, levels from −5 to +5 dB, AdamW, batch 2 with four-step accumulation, and negative SI-SDR plus 0.1 normalized L1. See [implementation](IMPLEMENTATION.md) for initialization, augmentation severity, and acquisition integrity limits. Data identities and recipes are retained in [metadata](../metadata/).

## Frozen test results

Each condition has {clean["cases"]:,} extraction requests (500 shared-mixture target pairs) from {clean["target_speakers"]} target speakers. B0 returns the mixture and has zero SI-SDR improvement by definition. Outputs are scored against the requested target, without permutation matching.

{chr(10).join(quality_table)}

Equal-weight augmentation gain across the four mismatch conditions: **{paired["equal_weight_mismatch_gain_db"]:+.2f} dB**, approximate joint interval **{interval(paired["equal_weight_mismatch_ci95_db"])}**. Conditions are averaged within each paired case before jointly resampling target-speaker clusters. This is one paired training seed. The intervals approximate target-speaker sampling uncertainty and do not include training-seed variability or fully account for shared interferers. Neither positive point estimates nor model selection alone establish a robust augmentation benefit.

Selected model, clean condition: mean **{clean["mean_si_sdri_db"]:.2f} dB**, median {clean["median_si_sdri_db"]:.2f} dB, 10th percentile {clean["p10_si_sdri_db"]:.2f} dB, mean interval {interval(clean["mean_ci95_db"])}; {percent(clean["negative_improvement_fraction"])} of cases have negative improvement and {percent(clean["confusion_fraction"])} meet the 3 dB wrong-speaker confusion proxy. Near-silent outputs: {percent(clean["near_silent_fraction"])}. Normalized waveform L1: {clean["mean_normalized_l1"]:.3f}.

The app's exact shared WAV processing path achieves **{delivered_clean["mean_si_sdri_db"]:.2f} dB** on the same clean test cases, with {percent(delivered_clean["confusion_fraction"])} confusion. It includes decoding, normalization, model execution, inverse gain, WAV encoding, and decoding for scoring; HTTP behavior is separately integration-tested. Raw results: [control](../reports/control-test.json), [augmentation](../reports/augmented-test.json), [paired comparison](../reports/test-comparison.json), [delivered path](../reports/delivered-test.json).

![Frozen reference-condition results](../reports/figures/test-conditions.png)

## Failure diagnostics

On {absent["cases"]} development requests with a third, absent speaker's reference, output energy exceeds −20 dB relative to mixture in **{percent(absent["output_above_minus20db_fraction"])}** of cases. Mean output/mixture energy is {absent["mean_output_mixture_energy_db"]:.2f} dB. Target absence is unsupported; the system can emit another voice. No SI-SDR against a zero target is reported.

Replacing the selected model's reference embedding with zeros yields {constant["mean_si_sdri_db"]:.2f} dB on the 80-case development diagnostic. This is an inference intervention outside training distribution, not a trained baseline.

Reference duration diagnostic:

{chr(10).join(duration_lines)}

Longer-reference subsets can differ. Eligible IDs and per-case values are preserved; these figures alone are not causal duration comparisons. Synthetic noise/channel/reverb robustness does not establish real-phone or real-room robustness. No formal listening study, word-preservation measure, or multilingual/three-speaker evaluation was performed.

## Runtime on the project Mac

Apple M3 Pro, 18 GiB unified memory, PyTorch 2.14.0. Five warm repetitions per length, following one first request. Audio is a repeated development clip. Processing includes decode, shared inference and WAV encode, excluding browser/network/HTTP parsing. No training runs concurrently with these measurements.

{chr(10).join(timing_table)}

Model-load time: CPU {profiles["cpu"]["model_load_seconds"]:.3f} s; MPS {profiles["mps"]["model_load_seconds"]:.3f} s. Process peak RSS, including whole-file comparison: CPU {profiles["cpu"]["process_peak_rss_gib"]:.3f} GiB; MPS {profiles["mps"]["process_peak_rss_gib"]:.3f} GiB. MPS driver counters are reported separately in JSON; unified-memory counters must not be summed. Maximum absolute chunk/whole difference across tested lengths: CPU {max(r["chunk_whole_max_error"] for r in profiles["cpu"]["rows"]):.3g}; MPS {max(r["chunk_whole_max_error"] for r in profiles["mps"]["rows"]):.3g}. These checks establish numeric alignment for this artifact, not natural long-conversation quality or live latency.

Full runtime protocols and samples: [CPU](../reports/delivery-cpu.json), [MPS](../reports/delivery-mps.json). [Reproduce the project](REPRODUCING.md).

## Distribution and limitations

Code, manifests and reports are public. Raw recordings and model weights remain in the local workspace and are not committed. A license for original code/model redistribution has not been selected. Public visibility is not an open-source license; data and dependencies retain their own licenses. Generated local speech examples include source attribution and describe their cropping/mixing transformations.

The first release is a complete local research pipeline with measured limitations. It does not meet a production-quality guarantee. Further development should use fresh held-out data after test-driven changes and replicate across training seeds.
"""
    Path("docs/MODEL_CARD.md").write_text(card)
    study = f"""# Case study · Keeping one voice

The product accepts an overlapping recording and a separate sample of the person the user wants to hear. A useful system must both separate speech and follow that reference when the same mixture is requested twice with different speakers. This requires learned waveform processing, not a language-model wrapper.

## What was built

The project implements its own 1.22M-parameter reference encoder and conditioned temporal separator in PyTorch, trains from random initialization, and delivers an aligned audio estimate through a local API and browser comparison workspace. Data acquisition, auditable mixture recipes, metrics, checkpoint recovery, CPU/MPS execution, and packaging are part of the system. Research ideas are attributed; SpeakerBeam code and pretrained weights are not dependencies.

The main engineering question was whether independent reference corruption during training improves mismatch robustness. Clean and augmented runs use identical architecture, seed, source/crop schedule, effective batch, and {selection["step_budget"]:,}-update budget. The four mismatch families are weighted equally. Checkpoint and product selection use development data; hashes identify both frozen artifacts before final test evaluation.

## Evidence and result

The tiny 16-case paired diagnostic reaches {overfit["best_development_si_sdri_db"]:.2f} dB improvement, showing that the network can learn the task. It is not evidence of generalization. Held-out evaluation is substantially harder.

![Training curves on unseen development speakers](../reports/figures/learning-curves.png)

On the 400-case development report, reference augmentation changes equally weighted mismatch improvement by {development["equal_weight_mismatch_gain_db"]:+.2f} dB. The predeclared rule selects **{selection["selected"]}** for delivery. On the reserved test, the corresponding augmentation change is {paired["equal_weight_mismatch_gain_db"]:+.2f} dB. The selected model's clean mean is {clean["mean_si_sdri_db"]:.2f} dB, with {percent(clean["negative_improvement_fraction"])} negative-improvement cases and {percent(clean["confusion_fraction"])} confusion. The complete tables, paired intervals, and exact artifact identities are in the [model card](MODEL_CARD.md).

The difference between tiny-set learning and unseen-speaker quality is a central finding. A working loss and a convincing single example are insufficient release evidence. This pilot uses one training seed and a small archive-order speech selection. Its results support a bounded experimental system; they do not establish general state-of-the-art speech extraction or a robust augmentation gain.

## Decisions that mattered

1. **Separate reference utterances and paired requests.** These prevent direct waveform reuse and reveal a separator that consistently chooses only the easier voice.
2. **A gain-sensitive loss alongside SI-SDR.** Waveform L1 anchors amplitude; test reports retain both separation and gain-sensitive metrics.
3. **Per-frame normalization and bounded context.** Padding does not affect global statistics, and long-file processing can be checked numerically against whole-file output.
4. **Independent augmentation RNG.** Treatment changes the reference while retaining the control's mixture schedule, enabling paired comparisons.
5. **A real delivery-path evaluation.** The final artifact is scored after the same decoding, normalization and WAV output operations used by the app.
6. **Explicit target-absence failure.** Energy diagnostics expose emitted competing speech when the desired speaker is missing, without fabricating a confidence score.

## What a reviewer can run

The [reproduction guide](REPRODUCING.md) provides setup, training, evaluation, export, serving and recovery commands. The [metadata directory](../metadata/) preserves public utterance IDs, hashes, frozen case recipes and training records. Tests exercise signal invariants, reference gradients, checkpoint resume, API errors, and long-file alignment. A fresh noneditable wheel and CPU container were exercised locally; GitHub Actions checks the CPU code path.

The app's default examples are the first development mixture with both references, selected by manifest order rather than score. Per-case test scores remain available for inspecting failures. There was no formal human listening study.

## Next research decisions

The measured generalization gap motivates more speaker diversity and longer training, with multiple paired seeds before claiming an augmentation effect. Reference corruption should also be tested on separately acquired real microphone/room recordings with appropriate consent. A calibrated target-presence objective is needed before absent-speaker use. Streaming would require a causal architecture and a separate latency/quality study. These are research extensions, not concealed capabilities of this release.
"""
    Path("docs/CASE_STUDY.md").write_text(study)
    print(
        "Wrote model card, technical case study, and learning/test figures from measured reports."
    )


if __name__ == "__main__":
    main()
