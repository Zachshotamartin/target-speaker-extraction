#!/usr/bin/env python3
"""Write the v3 results document from frozen test and delivered-runtime evidence."""

import json
from pathlib import Path

from tse.acoustic_comparison import compare_acoustic
from tse.listening import summarize_listening
from tse.utils import atomic_json, sha256


def read(path):
    return json.loads(Path(path).read_text())


def main():
    freeze_path = Path("reports/v3-selection-freeze.json")
    freeze = read(freeze_path)
    if freeze["status"] != "frozen_before_fresh_test":
        raise ValueError("A frozen selection is required")
    development = read("reports/v3-development-selection.json")
    if (
        sha256(Path("reports/v3-development-selection.json"))
        != freeze["development_selection_sha256"]
    ):
        raise ValueError("Development selection changed after freezing")
    reports = {name: read(f"reports/v3-{name}-test.json") for name in ("baseline", "candidate")}
    for name, report in reports.items():
        if report["split"] != "test" or report["selection_freeze_sha256"] != sha256(freeze_path):
            raise ValueError("Test does not match the freeze")
        if report["checkpoint_sha256"] != freeze["checkpoints"][name]["checkpoint_sha256"]:
            raise ValueError("Test checkpoint differs from frozen artifact")
        for key in (
            "source_manifest_sha256",
            "case_manifest_sha256",
            "environment_manifest_sha256",
        ):
            if report[key] != freeze[key]:
                raise ValueError(f"Test data differs: {key}")
        if report["provenance"]["source_tree_sha256"] != freeze["evaluation_source_tree_sha256"]:
            raise ValueError("Test implementation differs from the freeze")
    comparison_path = Path("reports/v3-test-comparison.json")
    if not comparison_path.exists():
        compare_acoustic(
            Path("reports/v3-baseline-test.json"),
            Path("reports/v3-candidate-test.json"),
            comparison_path,
        )
    comparison = read(comparison_path)
    for name in ("baseline", "candidate"):
        if comparison[f"{name}_report_sha256"] != sha256(Path(f"reports/v3-{name}-test.json")):
            raise ValueError("Comparison refers to another report")
    runtime = {device: read(f"reports/v3-delivery-{device}.json") for device in ("cpu", "mps")}
    for profile in runtime.values():
        if (
            profile["model"]["checkpoint_sha256"]
            != freeze["checkpoints"]["candidate"]["checkpoint_sha256"]
        ):
            raise ValueError("Runtime describes another candidate")
        if not all(row["exact_length"] for row in profile["rows"]):
            raise ValueError("Delivered timeline verification failed")
    listening = {}
    for directory in sorted(Path("artifacts/listening").glob("v3-*-listening")):
        if (directory / "key.json").is_file():
            listening[directory.name] = summarize_listening(
                Path("artifacts/listening"), directory.name
            )
    atomic_json(Path("reports/v3-listening-summary.json"), listening)
    table = [
        "| Acoustic condition | v0.2.0 SI-SDRi | Candidate SI-SDRi | Paired gain, 95% interval |",
        "| --- | ---: | ---: | ---: |",
    ]
    for condition, metrics in comparison["conditions"].items():
        if condition == "target_absent":
            continue
        metric = metrics["si_sdri_db"]
        low, high = metric["paired_ci95"]
        table.append(
            f"| {condition.replace('_', ' ')} | {metric['baseline_mean']:.3f} dB | {metric['candidate_mean']:.3f} dB | {metric['mean_candidate_minus_baseline']:+.3f} [{low:+.3f}, {high:+.3f}] |"
        )
    dev_table = [
        "| Candidate | Clean SI-SDRi | Acoustic SI-SDRi | Clean ESTOI | Clean confusion | Eligible for promotion |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in development["candidates"]:
        dev_table.append(
            f"| {row['name']} | {row['clean_si_sdri_db']:.3f} | {row['acoustic_si_sdri_db']:.3f} | {row['clean_estoi']:.4f} | {100 * row['clean_confusion']:.1f}% | {'yes' if row['promotion_eligible'] else 'no'} |"
        )
    runtime_table = [
        "| Device | Recording | Warm median / p95 | Exact samples |",
        "| --- | ---: | ---: | --- |",
    ]
    for device, profile in runtime.items():
        for row in profile["rows"]:
            runtime_table.append(
                f"| {device.upper()} | {row['duration_seconds']} s | {row['warm_median_seconds']:.3f} / {row['warm_p95_seconds']:.3f} s | {row['exact_length']} |"
            )
    metric = comparison["all_present"]["si_sdri_db"]
    absent = [
        reports[name]["conditions"]["target_absent"]["absent_median_attenuation_db"]
        for name in ("baseline", "candidate")
    ]
    candidate = reports["candidate"]["model"]
    deployment_note = (
        "The candidate passed the predeclared development promotion gates. Its final deployment status is recorded by the service and release verification."
        if freeze["deployment_candidate_passed_gates"]
        else "No new model passed every predeclared development promotion gate. The default remains v0.2.0; the test candidate is a research comparison and is not promoted because of its test result."
    )
    listening_lines = []
    for study, result in listening.items():
        listening_lines.append(
            f"- `{study}`: {result['participants']} participant(s); {result['status']}."
        )
        for identity, scores in result["models"].items():
            listening_lines.append(
                f"  - Model `{identity[:12]}`: {scores['ratings']} submitted trial ratings; mean competing-speech severity {scores['competing_speech']:.2f}, target damage {scores['target_damage']:.2f}, static {scores['static']:.2f} (1 none, 5 severe)."
            )
    document = f"""# V3: time/frequency separation, phase and acoustic robustness

All four improvement tracks were implemented and evaluated: matched learning-rate continuation, three band-separator/reference pilots followed by a longer selected run, phase-aware masks, and matched clean/realistic adaptation. The fresh test candidate achieves {metric["candidate_mean"]:.3f} dB mean SI-SDR improvement across target-present acoustic conditions, versus {metric["baseline_mean"]:.3f} dB for v0.2.0. The paired gain is {metric["mean_candidate_minus_baseline"]:+.3f} dB, with approximate 95% interval [{metric["paired_ci95"][0]:+.3f}, {metric["paired_ci95"][1]:+.3f}]. This is a measured experiment result, not a claim of fully separated voices.

{deployment_note}

## What was actually trained

The two learning-rate arms start from the same full 20,000-update checkpoint, including classifier, optimizer moments and RNG state. Both process the same next 5,000 updates; one retains the plateau policy and the other decays from 0.001 to 0.00003. Cosine improves the larger clean development result but its acoustic gain against the released checkpoint average is only +0.189 dB. The reviewer reported that the conversation and both versions sounded almost the same on the early listening pair. [Continuation results](../reports/v3-continuation-results.json), [human feedback](../reports/v3-continuation-listening-feedback.json).

Three new separator pilots receive 4,000 updates each. They split the 257 STFT bins into 18 bands, then model time and band context with four conditioned bidirectional recurrent blocks. The real/complex arms share their initial weights and our own trained reference encoder. Their initial outputs match exactly; the complex arm can learn an imaginary mask component. The third pilot uses a scaled ResNet34 reference encoder with 64 log-mel bands. It starts that encoder randomly, so this is not an equal-pretraining comparison. The selected architecture continues to 10,000 total updates. These compact models and training budgets adapt research ideas to this Mac; they are not faithful paper reproductions.

Both 3,000-update adaptation arms start from the same development-selected source with fresh identical optimizers and matching labels. The clean arm retains clean mixtures. The realistic arm uses clean examples, environmental noise, simulated rooms, partial overlap, pauses, target absence and combined conditions. Both select checkpoints on the same 140-case composite development set. No external pretrained weights or research implementation source code are used.

The final candidate is `{candidate["architecture"]}` with `{candidate["reference_encoder"]}` reference encoding and `{candidate["spectral_mask"]}` masks. Export identity: `{candidate["checkpoint_sha256"]}`. The displayed training-update counter describes the last run; provenance records earlier project training and any averaging.

## Development selection

The clean regression set has 400 requests. The acoustic development set has 350 requests across seven conditions. Promotion requires at least +0.3 dB acoustic mean gain, clean SI-SDRi regression no greater than 0.5 dB, ESTOI regression no greater than 0.01, and confusion increase no greater than one percentage point. Criteria were recorded before these candidate comparisons. The fresh test never selects the model.

{chr(10).join(dev_table)}

[Development decision](../reports/v3-development-selection.json), [evaluation policy](../reports/v3-evaluation-policy.json), [pre-test freeze](../reports/v3-selection-freeze.json).

## Fresh acoustic test

The test contains 700 requests from 33 previously unscored test-other identities: 100 requests in each condition, including 100 absent-target requests. The other 600 have an audible target. Conditions reuse source mixtures; these are dependent observations. Speech identities, noise recordings and simulated room groups are separated from training and development. The targets retain aligned room reverberation, so this evaluates extraction rather than dry-source dereverberation.

{chr(10).join(table)}

Median absent-target suppression is {absent[0]:.2f} dB for v0.2.0 and {absent[1]:.2f} dB for the candidate. That measures output energy reduction; it is not calibrated target-presence detection. SI-SDR and ESTOI are undefined for a silent target and are excluded there.

Intervals use 1,000 paired target-speaker-cluster bootstrap replicates with seed 42, keeping repeated conditions together within a speaker. Dependence through shared interfering speakers and rooms remains, and these intervals do not include training-seed uncertainty. ESTOI is an intelligibility proxy, not word accuracy. Scalar residual distortion includes filtering, phase, envelopes and environmental noise; it is not perceived static. [Complete paired results](../reports/v3-test-comparison.json), [candidate requests](../reports/v3-candidate-test.json).

## Listening assessment

The early comparison deliberately remains preserved after the reviewer reported weak audible separation. Its twelve trials reused one source pair across six conditions. The final listening positions span six different development mixtures and were fixed independently of model scores. A/B identities are hidden, assignments vary by trial, and playback power is matched with common peak headroom. Ratings and browser identifiers stay local; only aggregate scores enter the repository.

{chr(10).join(listening_lines) if listening_lines else "No human ratings have been submitted."}

This is a small convenience listening panel. A missing final-candidate rating remains pending; successful playback and objective improvements do not establish that static has disappeared.

## Delivered runtime

{chr(10).join(runtime_table)}

Each device is profiled in its own process after training finishes. Times include decoding, extraction and WAV encoding, excluding browser/network overhead. The input repeats a development clip to isolate recording length; it does not establish quality on natural long conversations. Bidirectional band models use whole-clip context, so chunk-equivalence metrics are inapplicable to them. All variants are offline and noncausal; processing faster than recording duration is not streaming latency. [CPU profile](../reports/v3-delivery-cpu.json), [MPS profile](../reports/v3-delivery-mps.json).

## Scope, research and reproducibility

This is independently implemented target-speaker extraction with a learned reference encoder and separator. It is not an LLM wrapper or a copy of SpeakerBeam. It remains experimental: synthetic mixtures of read English speech, simulated room acoustics and recorded noise do not establish natural microphone, spontaneous-conversation, multilingual, musical or multi-party performance. Individual cases can retain the wrong voice or damage the desired voice. One seed per comparison is not a robust estimate of all training variability.

The band/reference experiments follow ideas discussed in the [enrollment augmentation / BSRNN study](https://arxiv.org/html/2409.09589v1); phase-aware reconstruction is also motivated by [TF-GridNet](https://arxiv.org/abs/2211.12433). Our reduced widths, GroupNorm, mask bounds and optimization schedule are explicit substitutions. Source-informed mask diagnostics use known targets only for analysis and are never presented as learned model estimates.

Speech is [LibriSpeech / OpenSLR12](https://www.openslr.org/12/), CC BY 4.0. Selected simulated room responses and public-domain-tagged noise recordings come from [OpenSLR28](https://www.openslr.org/28/). Data preparation records archive and per-file checks. Raw audio, model weights and private ratings remain outside Git. Original code has no selected reuse license.

See the [v3 worklog and commands](V3_WORKLOG.md), [original research audit](RESEARCH_AUDIT.md), and public metadata under `metadata/v3/`. The prior release and its evaluation remain preserved separately.
"""
    Path("docs/V3_RESULTS.md").write_text(document)
    print("Wrote docs/V3_RESULTS.md from matching frozen evidence")


if __name__ == "__main__":
    main()
