"""Paired experiment comparisons and reproducible local audio examples."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tse.data import SpeechCorpus, load_cases
from tse.utils import atomic_json, sha256


def analyze_failures(report_path: Path, cases_path: Path, manifest: Path, output: Path) -> dict:
    """Join exact recipes to scores without fitting thresholds or selecting a model."""
    report = json.loads(report_path.read_text())
    if report["case_manifest_sha256"] != sha256(cases_path) or report[
        "source_manifest_sha256"
    ] != sha256(manifest):
        raise ValueError("Failure analysis requires the exact evaluated case and source manifests")
    cases = {case["case_id"]: case for case in json.loads(cases_path.read_text())["cases"]}
    records = {row["id"]: row for row in json.loads(manifest.read_text())["records"]}
    margin = report.get(
        "confusion_margin_db",
        report.get("config", {}).get("evaluation", {}).get("confusion_margin_db", 3),
    )

    def describe(rows: list[dict]) -> dict:
        baseline = [
            row["mixture_baseline_confused"] for row in rows if "mixture_baseline_confused" in row
        ]
        return {
            "cases": len(rows),
            "mean_si_sdri_db": float(np.mean([row["si_sdri_db"] for row in rows])),
            "negative_improvement_fraction": float(
                np.mean([row["si_sdri_db"] < 0 for row in rows])
            ),
            "confusion_fraction": float(np.mean([row["confused"] for row in rows])),
            "mixture_baseline_confusion_fraction": float(np.mean(baseline)) if baseline else None,
            "baseline_paired_cases": len(baseline),
        }

    joined = []
    seen = set()
    for row in report["rows"]:
        identity = (row["case_id"], row["condition"])
        if identity in seen or row["case_id"] not in cases:
            raise ValueError("Duplicate or unknown scored case")
        seen.add(identity)
        case = cases[row["case_id"]]
        index = case["target_index"]
        target = records[case["sources"][index]["id"]]
        reference = records[case["references"][index]["id"]]
        if target["speaker"] != row["target_speaker"]:
            raise ValueError("Scored target speaker does not match its recipe")
        ratio = case["ratio_db"] * (1 if index == 0 else -1)
        joined.append(
            {
                **row,
                "target_index": index,
                "target_to_interferer_db": ratio,
                "level_group": "quieter_target"
                if ratio < -2
                else "louder_target"
                if ratio > 2
                else "similar_levels",
                "reference_chapter_group": "different_chapter"
                if target["chapter"] != reference["chapter"]
                else "same_chapter",
                "mixture_key": json.dumps(
                    {key: case[key] for key in ("sources", "samples", "ratio_db")}, sort_keys=True
                ),
            }
        )
    conditions = {}
    for condition in sorted({row["condition"] for row in joined}):
        rows = [row for row in joined if row["condition"] == condition]
        pairs = {}
        for row in rows:
            pairs.setdefault(row["mixture_key"], []).append(row)
        valid_pairs = [
            pair
            for pair in pairs.values()
            if len(pair) == 2 and {row["target_index"] for row in pair} == {0, 1}
        ]
        for left, right in valid_pairs:
            left["mixture_baseline_confused"] = (
                right["mixture_si_sdr_db"] > left["mixture_si_sdr_db"] + margin
            )
            right["mixture_baseline_confused"] = (
                left["mixture_si_sdr_db"] > right["mixture_si_sdr_db"] + margin
            )
        groups = {}
        for key in ("level_group", "reference_chapter_group", "target_speaker"):
            groups[key] = {
                str(value): describe([row for row in rows if row[key] == value])
                for value in sorted({row[key] for row in rows})
            }
        conditions[condition] = {
            **groups,
            "paired_requests": {
                "complete_pairs": len(valid_pairs),
                "excluded_groups": len(pairs) - len(valid_pairs),
                "both_targets_improved_fraction": float(
                    np.mean([all(row["si_sdri_db"] > 0 for row in pair) for pair in valid_pairs])
                )
                if valid_pairs
                else None,
                "either_target_confused_fraction": float(
                    np.mean([any(row["confused"] for row in pair) for pair in valid_pairs])
                )
                if valid_pairs
                else None,
            },
        }
    result = {
        "report_sha256": sha256(report_path),
        "case_manifest_sha256": sha256(cases_path),
        "source_manifest_sha256": sha256(manifest),
        "split": report["split"],
        "confusion_margin_db": margin,
        "conditions": conditions,
        "interpretation": "Descriptive post-evaluation slices. Level bins are fixed at -2/+2 dB; chapter groups use source metadata. No demographic attributes are inferred. Small groups and shared mixtures limit precision. Do not use final-test slices to retune this release.",
    }
    atomic_json(output, result)
    return result


def compare(control_path: Path, treatment_path: Path, output: Path, replicates: int = 1000) -> dict:
    control, treatment = (json.loads(path.read_text()) for path in (control_path, treatment_path))
    for key in ("case_manifest_sha256", "source_manifest_sha256", "split", "protocol"):
        if control[key] != treatment[key]:
            raise ValueError(f"Comparison requires matching {key}")
    for key in ("model", "audio", "loss", "training", "data", "evaluation", "seed"):
        if control["config"][key] != treatment["config"][key]:
            raise ValueError(f"Controlled augmentation comparison has different {key}")
    if (
        control["config"]["augmentation"]["reference_enabled"]
        or not treatment["config"]["augmentation"]["reference_enabled"]
    ):
        raise ValueError("Expected clean-reference control and augmented-reference treatment")
    left = {(row["case_id"], row["condition"]): row for row in control["rows"]}
    right = {(row["case_id"], row["condition"]): row for row in treatment["rows"]}
    if len(left) != len(control["rows"]) or len(right) != len(treatment["rows"]):
        raise ValueError("Duplicate evaluation case/condition pairs")
    if left.keys() != right.keys():
        raise ValueError("Comparison cases or conditions differ")
    conditions = sorted({key[1] for key in left})
    groups = {}
    for condition in conditions:
        pairs = [(a, right[key]) for key, a in left.items() if key[1] == condition]
        speakers = sorted({a["target_speaker"] for a, _ in pairs})
        if any(a["target_speaker"] != b["target_speaker"] for a, b in pairs):
            raise ValueError("Paired target identities differ")
        values = np.array([b["si_sdri_db"] - a["si_sdri_db"] for a, b in pairs])
        clusters = [
            np.array(
                [
                    b["si_sdri_db"] - a["si_sdri_db"]
                    for a, b in pairs
                    if a["target_speaker"] == speaker
                ]
            )
            for speaker in speakers
        ]
        rng = np.random.default_rng(42)
        means = [
            float(
                np.concatenate(
                    [clusters[index] for index in rng.integers(len(clusters), size=len(clusters))]
                ).mean()
            )
            for _ in range(replicates)
        ]
        groups[condition] = {
            "cases": len(values),
            "mean_treatment_minus_control_db": float(values.mean()),
            "median_difference_db": float(np.median(values)),
            "paired_ci95_db": [float(v) for v in np.percentile(means, [2.5, 97.5])],
            "control_confusion_fraction": float(np.mean([a["confused"] for a, _ in pairs])),
            "treatment_confusion_fraction": float(np.mean([b["confused"] for _, b in pairs])),
        }
    mismatch_names = [name for name in ("noise", "channel", "reverb", "combined") if name in groups]
    mismatch = [groups[name]["mean_treatment_minus_control_db"] for name in mismatch_names]
    mismatch_interval = None
    if mismatch_names:
        identities = {
            condition: {case_id for case_id, name in left if name == condition}
            for condition in mismatch_names
        }
        case_ids = identities[mismatch_names[0]]
        if any(ids != case_ids for ids in identities.values()):
            raise ValueError("Mismatch conditions must use the same underlying cases")
        by_speaker = {}
        for case_id in sorted(case_ids):
            speaker = left[(case_id, mismatch_names[0])]["target_speaker"]
            value = np.mean(
                [
                    right[(case_id, name)]["si_sdri_db"] - left[(case_id, name)]["si_sdri_db"]
                    for name in mismatch_names
                ]
            )
            by_speaker.setdefault(speaker, []).append(value)
        clusters = [np.array(by_speaker[key]) for key in sorted(by_speaker)]
        rng = np.random.default_rng(42)
        samples = [
            float(
                np.concatenate(
                    [clusters[index] for index in rng.integers(len(clusters), size=len(clusters))]
                ).mean()
            )
            for _ in range(replicates)
        ]
        mismatch_interval = [float(value) for value in np.percentile(samples, [2.5, 97.5])]
    result = {
        "comparison": "reference_augmentation",
        "control_report_sha256": sha256(control_path),
        "treatment_report_sha256": sha256(treatment_path),
        "split": control["split"],
        "protocol": control["protocol"],
        "case_manifest_sha256": control["case_manifest_sha256"],
        "seed": control["config"]["seed"],
        "training_update_budget": control["config"]["training"]["max_optimizer_updates"],
        "control_selected_step": control["training_step"],
        "treatment_selected_step": treatment["training_step"],
        "conditions": groups,
        "equal_weight_mismatch_gain_db": float(np.mean(mismatch)) if mismatch else None,
        "equal_weight_mismatch_ci95_db": mismatch_interval,
        "mismatch_conditions": mismatch_names,
        "mismatch_uncertainty_method": "Average condition differences within each case, then bootstrap target-speaker clusters jointly; preserves within-case condition dependence.",
        "uncertainty_note": "One training seed unless separate seed reports are supplied; target-speaker bootstrap intervals are approximate.",
    }
    atomic_json(output, result)
    return result


def make_examples(root: Path, manifest: Path, cases_path: Path, output: Path) -> dict:
    payload = json.loads(cases_path.read_text())
    if payload["split"] != "dev":
        raise ValueError("Demo selection uses development audio, leaving the final test reserved")
    corpus = SpeechCorpus(root, manifest, "dev", 4, 5)
    cases = load_cases(cases_path, corpus)
    output.mkdir(parents=True, exist_ok=True)
    items = []
    for index, case in enumerate(cases[:2]):
        row = corpus.render(case)
        identifier = f"voice-{index + 1}"
        mixture_name, reference_name = f"{identifier}-mixture.wav", f"{identifier}-reference.wav"
        sf.write(output / mixture_name, row["mixture"], 16000, subtype="FLOAT")
        sf.write(output / reference_name, row["reference"], 16000, subtype="FLOAT")
        items.append(
            {
                "id": identifier,
                "label": f"Voice {index + 1:02d}",
                "mixture": f"/example-audio/{mixture_name}",
                "reference": f"/example-audio/{reference_name}",
                "case_id": case["case_id"],
                "speaker": row["speaker"],
                "sources": case["sources"],
                "reference_source": case["references"][case["target_index"]],
            }
        )
    result = {
        "items": items,
        "selection": "First paired development mixture, both target references; no score-based selection",
        "attribution": "LibriSpeech, Panayotov et al. (2015), CC BY 4.0",
        "source": "https://www.openslr.org/12",
        "case_manifest_sha256": sha256(cases_path),
    }
    atomic_json(output / "index.json", result)
    return result
