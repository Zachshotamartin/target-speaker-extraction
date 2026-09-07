#!/usr/bin/env python3
"""Check representable acoustic reconstructions using truth; never model inference."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tse.data import load_cases
from tse.metrics import si_sdr
from tse.realistic import RealisticCorpus
from tse.utils import atomic_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing source-informed diagnostics")
    torch.set_num_threads(2)
    manifest = Path("data/v3/manifests/inventory.json")
    environment = Path("data/v3/manifests/environments.json")
    cases_path = Path("data/v3/manifests/dev-realistic-cases.json")
    corpus = RealisticCorpus(
        Path("data/expanded/raw"), manifest, "dev", 4, 5, args.environment_root, environment
    )
    cases = load_cases(cases_path, corpus)
    options = dict(n_fft=512, hop_length=128, window=torch.hann_window(512), center=True)
    rows = []
    with torch.inference_mode():
        for case in cases:
            if case["scenario"] == "target_absent":
                continue
            signals = corpus.render(case)
            target, mixture = [torch.from_numpy(signals[key]) for key in ("target", "mixture")]
            truth, mixed = [
                torch.stft(wave, **options, return_complex=True, pad_mode="constant")
                for wave in (target, mixture)
            ]
            ratio = truth * mixed.conj() / mixed.abs().square().clamp_min(1e-12)
            real = ratio.real.clamp(0, 1)
            masks = {
                "bounded_real_truth_mask": real,
                "bounded_complex_truth_mask": torch.complex(real, ratio.imag.clamp(-1, 1)),
            }
            input_score = si_sdr(mixture[None], target[None]).item()
            for name, mask in masks.items():
                estimate = torch.istft(mask * mixed, **options, length=len(target))
                rows.append(
                    {
                        "case_id": case["case_id"],
                        "condition": case["scenario"],
                        "method": name,
                        "si_sdri_db": si_sdr(estimate[None], target[None]).item() - input_score,
                    }
                )
    conditions = {}
    for condition in sorted({row["condition"] for row in rows}):
        conditions[condition] = {
            name: float(
                np.mean(
                    [
                        row["si_sdri_db"]
                        for row in rows
                        if row["condition"] == condition and row["method"] == name
                    ]
                )
            )
            for name in masks
        }
    result = {
        "status": "Source-informed development diagnostic; NOT a trained model estimate",
        "case_manifest_sha256": sha256(cases_path),
        "source_manifest_sha256": sha256(manifest),
        "environment_manifest_sha256": sha256(environment),
        "conditions": conditions,
        "interpretation": "These reconstructions use the known target spectrum. They minimize per-bin complex squared error inside each mask's allowed rectangle, not waveform SI-SDR globally. They are neither strict SI-SDR upper bounds nor guaranteed learnable performance. The complex variant permits real [0,1] and imaginary [-1,1], matching the experiment; it cannot represent arbitrary complex ratios. Target-absent requests are excluded from SI-SDR. No source-informed mask is used by the app or listening candidates.",
        "rows": rows,
    }
    atomic_json(args.output, result)
    print(json.dumps(conditions, indent=2))


if __name__ == "__main__":
    main()
