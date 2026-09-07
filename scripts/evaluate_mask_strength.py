#!/usr/bin/env python3
"""Development-only suppression-strength experiment with one model pass per case."""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from pystoi import stoi

from tse.data import SpeechCorpus, load_cases
from tse.inference import Extractor
from tse.metrics import measure, summarize
from tse.quality import source_projection
from tse.utils import atomic_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=Path("data/manifests/dev-report-cases.json"))
    parser.add_argument("--count", type=int, default=400)
    args = parser.parse_args()
    if json.loads(args.cases.read_text())["split"] != "dev":
        raise ValueError("Strength selection accepts development data only")
    corpus = SpeechCorpus(Path("data/raw"), Path("data/manifests/inventory.json"), "dev", 4, 5)
    cases = load_cases(args.cases, corpus)[: args.count]
    if not cases or args.count < 1:
        raise ValueError("Choose a positive, nonempty development subset")
    extractor = Extractor(args.checkpoint, "cpu")
    if extractor.config.model.family != "reference_conditioned_stft_tcn":
        raise ValueError("Requires the spectral-mask architecture")
    if set(extractor.payload["provenance"]["train_speakers"]) & set(corpus.speakers):
        raise ValueError("Training and development speakers overlap")
    captured = {}
    handle = extractor.model.mask.register_forward_hook(
        lambda module, inputs, output: captured.update(mask=output.detach())
    )
    powers = (1.0, 1.25, 1.5, 2.0)
    rows = {str(power): [] for power in powers}
    config = extractor.config.model
    options = {
        "n_fft": config.stft_fft_samples,
        "hop_length": config.stft_hop_samples,
        "window": extractor.model.window,
        "center": True,
    }
    max_parity_error = 0.0
    try:
        for number, case in enumerate(cases, 1):
            signals = corpus.render(case)
            raw = extractor.extract_array(signals["mixture"], signals["reference"])
            original = signals["mixture"].astype(np.float32) - float(signals["mixture"].mean())
            gain = 0.14 / max(float(np.sqrt(np.mean(original**2))), 1e-4)
            spectrum = torch.stft(
                torch.from_numpy(original * gain)[None],
                **options,
                pad_mode="constant",
                return_complex=True,
            )
            for power in powers:
                estimate = (
                    torch.istft(
                        spectrum * captured["mask"].pow(power), **options, length=len(original)
                    )[0].numpy()
                    / gain
                )
                if power == 1:
                    error = float(np.max(np.abs(estimate - raw)))
                    max_parity_error = max(max_parity_error, error)
                    if error > 1e-6:
                        raise ValueError("Strength reconstruction differs from shared inference")
                playback_gain = min(1.0, 0.98 / max(float(np.max(np.abs(estimate))), 1e-8))
                estimate *= playback_gain
                measured = measure(
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
                with warnings.catch_warnings(record=True) as notices:
                    intelligibility = float(stoi(signals["target"], estimate, 16000, extended=True))
                rows[str(power)].append(
                    {
                        "case_id": case["case_id"],
                        "target_speaker": signals["speaker"],
                        **measured,
                        **source_projection(estimate, signals["target"], signals["interferer"]),
                        "estoi": intelligibility if not notices else None,
                    }
                )
            if number % 100 == 0:
                print(f"Scored {number}/{len(cases)} requests at four strengths", flush=True)
    finally:
        handle.remove()
    summaries = {}
    for power, values in rows.items():
        valid = [row["estoi"] for row in values if row["estoi"] is not None]
        summaries[power] = {
            **summarize(values),
            "mean_estoi": float(np.mean(valid)) if valid else None,
            "estoi_valid_cases": len(valid),
            "mean_artifact_proxy_fraction": float(
                np.mean([row["artifact_proxy_fraction"] for row in values])
            ),
            "median_relative_interferer_attenuation_db": float(
                np.median([row["relative_interferer_attenuation_db"] for row in values])
            ),
        }
    atomic_json(
        args.output,
        {
            "checkpoint_sha256": sha256(args.checkpoint),
            "case_manifest_sha256": sha256(args.cases),
            "source_manifest_sha256": corpus.manifest_hash,
            "count": len(cases),
            "selection": "First count requests in manifest order; development only",
            "max_shared_path_parity_error": max_parity_error,
            "protocol": "Raise each predicted attenuation mask to fixed powers, retain mixture phase, invert STFT and apply shared 0.98 sample-peak guard. No clean source enters the prediction; clean sources are evaluation references only. Greater suppression can damage wanted speech. This is an experiment, not an enabled serving option.",
            "summaries": summaries,
            "rows": rows,
        },
    )
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
