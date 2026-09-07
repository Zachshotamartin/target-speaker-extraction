#!/usr/bin/env python3
"""Measure source-informed mask diagnostics on development only, never inference."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from pystoi import stoi

from tse.data import SpeechCorpus, load_cases
from tse.metrics import measure
from tse.utils import atomic_json, sha256


def diagnostic_masks(mixture: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-bin least-squares real mask and ideal magnitude ratio; both use truth."""
    energy = mixture.abs().square()
    phase_sensitive = (target * mixture.conj()).real / energy.clamp_min(1e-12)
    phase_sensitive = torch.where(energy > 1e-12, phase_sensitive, 0)
    other = mixture - target
    ratio = target.abs() / (target.abs() + other.abs()).clamp_min(1e-12)
    return {
        "ideal_magnitude_ratio": ratio,
        "clipped_phase_sensitive": phase_sensitive.clamp(0, 1),
        "unbounded_real_phase_sensitive": phase_sensitive,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path("data/manifests/dev-report-cases.json"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/mask-capacity-development.json")
    )
    args = parser.parse_args()
    if json.loads(args.cases.read_text())["split"] != "dev":
        raise ValueError("Source-informed model diagnosis is restricted to development data")
    torch.set_num_threads(2)
    manifest = Path("data/manifests/inventory.json")
    corpus = SpeechCorpus(Path("data/raw"), manifest, "dev", 4, 5)
    cases = load_cases(args.cases, corpus)
    options = dict(n_fft=512, hop_length=128, window=torch.hann_window(512), center=True)
    rows = []
    with torch.inference_mode():
        for index, case in enumerate(cases, 1):
            signals = corpus.render(case)
            waves = {
                key: torch.from_numpy(signals[key]) for key in ("mixture", "target", "interferer")
            }
            spectra = {
                key: torch.stft(value, **options, pad_mode="constant", return_complex=True)
                for key, value in waves.items()
            }
            predictions = {
                name: torch.istft(spectra["mixture"] * mask, **options, length=len(waves["target"]))
                for name, mask in diagnostic_masks(spectra["mixture"], spectra["target"]).items()
            }
            reconstruction = torch.istft(spectra["target"], **options, length=len(waves["target"]))
            error = float((reconstruction - waves["target"]).abs().max())
            if error > 1e-6:
                raise ValueError("Analysis/synthesis reconstruction failed")
            for name, estimate in predictions.items():
                measured = measure(
                    *[
                        wave[None, None]
                        for wave in (
                            estimate,
                            waves["mixture"],
                            waves["target"],
                            waves["interferer"],
                        )
                    ]
                )[0]
                intelligibility = float(
                    stoi(signals["target"], estimate.numpy(), 16000, extended=True)
                )
                if not np.isfinite(intelligibility):
                    raise ValueError("Non-finite diagnostic intelligibility")
                rows.append(
                    {
                        "case_id": case["case_id"],
                        "target_speaker": signals["speaker"],
                        "method": name,
                        **measured,
                        "estoi": intelligibility,
                        "clean_reconstruction_max_error": error,
                    }
                )
            if index % 100 == 0:
                print(f"Diagnosed {index}/{len(cases)} development requests", flush=True)
    summary = {}
    for name in predictions:
        selected = [row for row in rows if row["method"] == name]
        summary[name] = {
            "cases": len(selected),
            "mean_si_sdri_db": float(np.mean([row["si_sdri_db"] for row in selected])),
            "mean_estoi": float(np.mean([row["estoi"] for row in selected])),
            "confusion_fraction": float(np.mean([row["confused"] for row in selected])),
        }
    result = {
        "status": "Source-informed development diagnostic; not a learned model or deployable result",
        "case_manifest_sha256": sha256(args.cases),
        "source_manifest_sha256": sha256(manifest),
        "transform": "512-sample Hann STFT, 128-sample hop, centered constant padding, 16 kHz",
        "interpretation": "All masks use the clean target. The clipped phase-sensitive mask minimizes per-bin complex squared error over real masks in [0,1], not waveform SI-SDR globally. These are feasible reconstruction examples, not strict SI-SDR ceilings or guaranteed learnable performance. Unbounded real masks can amplify bins and reverse phase by pi, but cannot make arbitrary phase corrections. Clean STFT/ISTFT reconstruction verifies the transform itself.",
        "summary": summary,
        "rows": rows,
    }
    atomic_json(args.output, result)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
