#!/usr/bin/env python3
"""Delivered-audio diagnostics with explicit, hash-bound access to a frozen test."""

from __future__ import annotations

import io
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from pystoi import stoi

from tse.audio import read_audio, wav_bytes
from tse.data import SpeechCorpus, load_cases
from tse.inference import Extractor
from tse.metrics import measure, summarize
from tse.utils import atomic_json, git_state, sha256, source_digest


def source_projection(estimate: np.ndarray, target: np.ndarray, other: np.ndarray) -> dict:
    basis = np.stack([target, other], axis=1).astype(np.float64)
    basis -= basis.mean(axis=0)
    output = estimate.astype(np.float64) - float(estimate.mean())
    coefficients = np.linalg.lstsq(basis, output, rcond=None)[0]
    residual = output - basis @ coefficients
    return {
        "artifact_proxy_fraction": float(np.sum(residual**2) / max(np.sum(output**2), 1e-12)),
        "relative_interferer_attenuation_db": float(
            20 * np.log10((abs(coefficients[0]) + 1e-8) / (abs(coefficients[1]) + 1e-8))
        ),
        "target_coefficient": float(coefficients[0]),
        "interferer_coefficient": float(coefficients[1]),
    }


def validate_evaluation(
    checkpoint: Path, cases_path: Path, manifest: Path, freeze: Path | None
) -> str:
    split = json.loads(cases_path.read_text())["split"]
    if split == "dev" and freeze is None:
        return split
    if split != "test" or freeze is None:
        raise ValueError("Exploration accepts development only; test requires a frozen selection")
    frozen = json.loads(freeze.read_text())
    if frozen.get("status") != "frozen_before_fresh_test":
        raise ValueError("Expected a selection frozen before fresh-test scoring")
    for key, path in (("case_manifest_sha256", cases_path), ("source_manifest_sha256", manifest)):
        if frozen.get(key) != sha256(path):
            raise ValueError(f"Frozen evaluation differs from {key}")
    allowed = {item["checkpoint_sha256"] for item in frozen["checkpoints"].values()}
    if sha256(checkpoint) not in allowed:
        raise ValueError("Checkpoint is not in the frozen selection")
    return split


def evaluate(
    checkpoint: Path,
    cases_path: Path,
    output: Path,
    device: str,
    root: Path = Path("data/raw"),
    manifest: Path = Path("data/manifests/inventory.json"),
    freeze: Path | None = None,
) -> dict:
    split = validate_evaluation(checkpoint, cases_path, manifest, freeze)
    corpus = SpeechCorpus(root, manifest, split, 4, 5)
    cases = load_cases(cases_path, corpus)
    extractor = Extractor(checkpoint, device)
    trained = set(extractor.payload.get("provenance", {}).get("train_speakers", []))
    if trained & set(corpus.speakers):
        raise ValueError("Checkpoint training speakers overlap evaluation speakers")
    rows = []
    for index, case in enumerate(cases, 1):
        signals = corpus.render(case)
        encoded, processing = extractor.extract_files(
            io.BytesIO(wav_bytes(signals["mixture"])),
            io.BytesIO(wav_bytes(signals["reference"])),
        )
        estimate = read_audio(io.BytesIO(encoded))
        measured = measure(
            *[
                torch.from_numpy(wave)[None, None]
                for wave in (estimate, signals["mixture"], signals["target"], signals["interferer"])
            ],
            margin_db=extractor.config.evaluation.confusion_margin_db,
        )[0]
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter("always")
            intelligibility = float(stoi(signals["target"], estimate, 16000, extended=True))
            mixture_intelligibility = float(
                stoi(signals["target"], signals["mixture"], 16000, extended=True)
            )
        estoi_valid = not notices and np.isfinite(intelligibility + mixture_intelligibility)
        rows.append(
            {
                "case_id": case["case_id"],
                "target_speaker": signals["speaker"],
                "condition": "clean",
                **measured,
                **source_projection(estimate, signals["target"], signals["interferer"]),
                "estoi": intelligibility if estoi_valid else None,
                "mixture_estoi": mixture_intelligibility if estoi_valid else None,
                "estoi_warnings": [str(notice.message) for notice in notices],
                "playback_gain": processing["playback_gain"],
                "output_peak": processing["output_peak"],
            }
        )
        if index % 100 == 0:
            print(f"Scored {index}/{len(cases)} {split} requests", flush=True)
    valid = [row for row in rows if row["estoi"] is not None]
    summary = {
        **summarize(rows),
        "estoi_valid_cases": len(valid),
        "estoi_excluded_cases": len(rows) - len(valid),
        "mean_estoi": float(np.mean([row["estoi"] for row in valid])) if valid else None,
        "mean_mixture_estoi": float(np.mean([row["mixture_estoi"] for row in valid]))
        if valid
        else None,
        "mean_estoi_improvement": float(
            np.mean([row["estoi"] - row["mixture_estoi"] for row in valid])
        )
        if valid
        else None,
        "mean_artifact_proxy_fraction": float(
            np.mean([row["artifact_proxy_fraction"] for row in rows])
        ),
        "median_relative_interferer_attenuation_db": float(
            np.median([row["relative_interferer_attenuation_db"] for row in rows])
        ),
        "maximum_output_peak": max(row["output_peak"] for row in rows),
    }
    result = {
        "status": "Development exploration"
        if split == "dev"
        else "Frozen fresh-test evaluation; do not use for tuning this release",
        "split": split,
        "protocol": "custom-librispeech-tse-v1",
        "confusion_margin_db": extractor.config.evaluation.confusion_margin_db,
        "selection_freeze_sha256": sha256(freeze) if freeze else None,
        "checkpoint_sha256": sha256(checkpoint),
        "case_manifest_sha256": sha256(cases_path),
        "source_manifest_sha256": sha256(manifest),
        "model": extractor.info(),
        "summary": summary,
        "rows": rows,
        "metric_notes": {
            "estoi": "pystoi 0.4.1 extended=True; clean target reference, 16 kHz, library resampling. Higher predicts better intelligibility; not word accuracy or a human rating.",
            "artifact_proxy": "Energy remaining after scalar least-squares projection onto both clean sources divided by output energy; includes phase, filtering and envelope distortion.",
            "relative_interferer_attenuation": "20 log10(abs(target coefficient)/abs(interferer coefficient)); scalar source projection, gain-invariant. Not a perceptual suppression rating.",
            "source": "https://github.com/mpariente/pystoi",
        },
        "provenance": {**git_state(), "source_tree_sha256": source_digest()},
    }
    atomic_json(output, result)
    print(json.dumps(summary, indent=2), flush=True)
    return result


def compare_quality(baseline_path: Path, candidate_path: Path, output: Path) -> dict:
    """Paired descriptive comparison; this does not claim a controlled ablation."""
    baseline, candidate = [json.loads(path.read_text()) for path in (baseline_path, candidate_path)]
    for key in ("case_manifest_sha256", "source_manifest_sha256"):
        if baseline[key] != candidate[key]:
            raise ValueError(f"Quality comparison requires matching {key}")
    if baseline.get("confusion_margin_db", 3) != candidate.get("confusion_margin_db", 3):
        raise ValueError("Confusion thresholds differ")
    tables = [{row["case_id"]: row for row in report["rows"]} for report in (baseline, candidate)]
    if any(
        len(table) != len(report["rows"])
        for table, report in zip(tables, (baseline, candidate), strict=True)
    ):
        raise ValueError("Duplicate quality case")
    if tables[0].keys() != tables[1].keys():
        raise ValueError("Quality cases differ")
    pairs = [(tables[0][key], tables[1][key]) for key in sorted(tables[0])]
    if any(left["target_speaker"] != right["target_speaker"] for left, right in pairs):
        raise ValueError("Paired target speakers differ")
    speakers = sorted({left["target_speaker"] for left, _ in pairs})
    metrics = {}
    for metric in ("si_sdri_db", "estoi", "artifact_proxy_fraction", "confused"):
        groups = [
            np.array(
                [
                    float(right[metric]) - float(left[metric])
                    for left, right in pairs
                    if left["target_speaker"] == speaker
                    and left.get(metric) is not None
                    and right.get(metric) is not None
                ]
            )
            for speaker in speakers
        ]
        groups = [group for group in groups if len(group)]
        if not groups:
            metrics[metric] = None
            continue
        rng = np.random.default_rng(42)
        samples = [
            np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))]).mean()
            for _ in range(1000)
        ]
        differences = np.concatenate(groups)
        metrics[metric] = {
            "paired_cases": len(differences),
            "mean_candidate_minus_baseline": float(differences.mean()),
            "paired_ci95": np.percentile(samples, [2.5, 97.5]).tolist(),
        }
    result = {
        "baseline_report_sha256": sha256(baseline_path),
        "candidate_report_sha256": sha256(candidate_path),
        "case_manifest_sha256": baseline["case_manifest_sha256"],
        "source_manifest_sha256": baseline["source_manifest_sha256"],
        "cases": len(pairs),
        "target_speakers": len(speakers),
        "metrics": metrics,
        "interpretation": "Paired candidate-minus-baseline differences. Higher SI-SDRi/ESTOI and lower distortion proxy/confusion are favorable. Approximate target-speaker cluster bootstrap, 1000 replicates, seed 42; shared-interferer dependence remains. Model/data/training changes are combined, not a single-factor causal ablation. ESTOI and scalar distortion are proxies, not subjective listening ratings.",
    }
    atomic_json(output, result)
    return result
