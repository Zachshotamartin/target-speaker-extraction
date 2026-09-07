"""Shared delivered-WAV evaluation of acoustic conditions and target absence."""

import io
import json
import warnings

import numpy as np
import torch
from pystoi import stoi

from tse.audio import read_audio, wav_bytes
from tse.data import load_cases
from tse.inference import Extractor
from tse.metrics import measure, summarize
from tse.quality import source_projection, validate_evaluation
from tse.realistic import SCENARIOS, RealisticCorpus
from tse.utils import atomic_json, git_state, sha256, source_digest


def summarize_condition(rows):
    present = [r for r in rows if r["target_present"]]
    valid = [r for r in present if r["estoi"] is not None]
    result = {
        "cases": len(rows),
        "present_cases": len(present),
        "absent_cases": len(rows) - len(present),
        "maximum_output_peak": max(r["output_peak"] for r in rows),
    }
    if present:
        result.update(summarize(present))
        result["mean_estoi"] = float(np.mean([r["estoi"] for r in valid])) if valid else None
        result["estoi_valid_cases"] = len(valid)
        result["mean_mixture_estoi"] = (
            float(np.mean([r["mixture_estoi"] for r in valid])) if valid else None
        )
        result["mean_artifact_proxy_fraction"] = float(
            np.mean([r["artifact_proxy_fraction"] for r in present])
        )
    absent = [r for r in rows if not r["target_present"]]
    if absent:
        result["absent_median_attenuation_db"] = float(
            np.median([r["attenuation_db"] for r in absent])
        )
        result["absent_at_least_20db_suppression_fraction"] = float(
            np.mean([r["attenuation_db"] >= 20 for r in absent])
        )
        result["absent_note"] = (
            "Output energy suppression, not calibrated target-presence accuracy; SI-SDR and ESTOI are undefined for a silent target."
        )
    return result


def evaluate_realistic(
    checkpoint,
    root,
    manifest,
    cases_path,
    environment_root,
    environment_manifest,
    output,
    device="cpu",
    freeze=None,
):
    if output.exists():
        raise FileExistsError("Preserve existing evaluation reports; use a new output path")
    split = validate_evaluation(checkpoint, cases_path, manifest, freeze)
    if freeze and json.loads(freeze.read_text()).get("environment_manifest_sha256") != sha256(
        environment_manifest
    ):
        raise ValueError("Frozen acoustic inventory differs from the evaluation")
    case_payload = json.loads(cases_path.read_text())
    if case_payload.get("rendering_protocol") != "realistic-tse-v1":
        raise ValueError("Expected the declared realistic rendering protocol")
    corpus = RealisticCorpus(root, manifest, split, 4, 5, environment_root, environment_manifest)
    cases = load_cases(cases_path, corpus)
    extractor = Extractor(checkpoint, device)
    trained = set(extractor.payload.get("provenance", {}).get("train_speakers", []))
    if trained & set(corpus.speakers):
        raise ValueError("Training identities overlap the evaluation")
    rows = []
    for index, case in enumerate(cases, 1):
        signals = corpus.render(case)
        encoded, processing = extractor.extract_files(
            io.BytesIO(wav_bytes(signals["mixture"])), io.BytesIO(wav_bytes(signals["reference"]))
        )
        estimate = read_audio(io.BytesIO(encoded))
        assert len(estimate) == len(signals["mixture"])
        row = {
            "case_id": case["case_id"],
            "target_speaker": signals["speaker"],
            "condition": case["scenario"],
            "target_present": signals["target_present"],
            "output_peak": processing["output_peak"],
            "playback_gain": processing["playback_gain"],
            "output_samples": len(estimate),
            "estoi": None,
        }
        if signals["target_present"]:
            metric = measure(
                *[
                    torch.from_numpy(wave)[None, None]
                    for wave in (
                        estimate,
                        signals["mixture"],
                        signals["target"],
                        signals["interferer"],
                    )
                ]
            )[0]
            row.update(metric)
            row.update(source_projection(estimate, signals["target"], signals["interferer"]))
            with warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter("always")
                score = float(stoi(signals["target"], estimate, 16000, extended=True))
                mixture_score = float(
                    stoi(signals["target"], signals["mixture"], 16000, extended=True)
                )
            valid = not notices and np.isfinite(score + mixture_score)
            row.update(
                estoi=score if valid else None,
                mixture_estoi=mixture_score if valid else None,
                estoi_warnings=[str(w.message) for w in notices],
            )
        else:
            row["attenuation_db"] = float(
                10
                * np.log10(
                    (np.mean(signals["mixture"] ** 2) + 1e-10) / (np.mean(estimate**2) + 1e-10)
                )
            )
        rows.append(row)
        if index % 50 == 0:
            print(f"Scored {index}/{len(cases)} realistic {split} requests", flush=True)
    result = {
        "status": "Development exploration"
        if split == "dev"
        else "Frozen fresh-test evaluation; not used for tuning this release",
        "split": split,
        "protocol": "realistic-tse-v1",
        "checkpoint_sha256": sha256(checkpoint),
        "source_manifest_sha256": sha256(manifest),
        "case_manifest_sha256": sha256(cases_path),
        "environment_manifest_sha256": sha256(environment_manifest),
        "selection_freeze_sha256": sha256(freeze) if freeze else None,
        "model": extractor.info(),
        "conditions": {
            condition: summarize_condition([r for r in rows if r["condition"] == condition])
            for condition in SCENARIOS
        },
        "present_macro_mean_si_sdri_db": float(
            np.mean([r["si_sdri_db"] for r in rows if r["target_present"]])
        ),
        "provenance": {**git_state(), "source_tree_sha256": source_digest()},
        "notes": [
            "Conditions reuse source pairs; observations are dependent.",
            "Artifact projection includes unmodeled environmental noise and is not a perceived-static score.",
            "Room targets retain reverberation; no dry-source recovery is claimed.",
            "Synthetic transformations of public read speech do not establish natural microphone/conversation performance.",
        ],
        "rows": rows,
    }
    atomic_json(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "mean_present_si_sdri_db": result["present_macro_mean_si_sdri_db"],
            }
        ),
        flush=True,
    )
    return result
