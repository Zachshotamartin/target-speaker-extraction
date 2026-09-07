#!/usr/bin/env python3
"""Development-only delivered-audio quality assessment with ESTOI and distortion diagnostics."""

from __future__ import annotations

import argparse
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


def evaluate(checkpoint: Path, cases_path: Path, output: Path, device: str) -> dict:
    if json.loads(cases_path.read_text())["split"] != "dev":
        raise ValueError("Exploratory quality evaluation accepts development cases only")
    root, manifest = Path("data/raw"), Path("data/manifests/inventory.json")
    corpus = SpeechCorpus(root, manifest, "dev", 4, 5)
    cases = load_cases(cases_path, corpus)
    extractor = Extractor(checkpoint, device)
    trained = set(extractor.payload.get("provenance", {}).get("train_speakers", []))
    if trained & set(corpus.speakers):
        raise ValueError("Checkpoint training speakers overlap development")
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
            ]
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
            print(f"Scored {index}/{len(cases)} development requests", flush=True)
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
        "status": "Development only; not final-test evidence or a subjective listening score.",
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=Path("data/manifests/dev-report-cases.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.cases, args.output, args.device)


if __name__ == "__main__":
    main()
