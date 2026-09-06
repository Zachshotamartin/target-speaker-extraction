"""Paired experiment comparisons and reproducible local audio examples."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tse.data import SpeechCorpus, load_cases
from tse.utils import atomic_json, sha256


def compare(control_path: Path, treatment_path: Path, output: Path, replicates: int = 1000) -> dict:
    control, treatment = (json.loads(path.read_text()) for path in (control_path, treatment_path))
    for key in ("case_manifest_sha256", "source_manifest_sha256", "split", "protocol"):
        if control[key] != treatment[key]:
            raise ValueError(f"Comparison requires matching {key}")
    for key in ("model", "audio", "loss", "training", "data", "seed"):
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
    mismatch = [
        groups[name]["mean_treatment_minus_control_db"]
        for name in ("noise", "channel", "reverb", "combined")
        if name in groups
    ]
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
