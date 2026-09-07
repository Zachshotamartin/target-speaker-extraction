#!/usr/bin/env python3
"""Measure independent filterbank agreement with an isolated standard audio library."""

import argparse
import sys
from pathlib import Path

import torch

from tse.librimix import LibriMixCorpus
from tse.reference_model import EnrollmentFbank
from tse.utils import atomic_json, sha256

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--torchaudio-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/reference-fbank-parity.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve the previous measurement")
    sys.path.insert(0, str(args.torchaudio_root))
    import torchaudio
    from torchaudio.compliance.kaldi import fbank

    torch.set_num_threads(2)
    corpus = LibriMixCorpus(args.root, Path("data/reference/manifest.json"), "train")
    rows = []
    for dither in (0, 1):
        frontend = EnrollmentFbank().train(bool(dither))
        for index in range(8):
            waveform = corpus.request(index, 17)["reference"]
            torch.manual_seed(19)
            expected = fbank(
                waveform[0] * 32768,
                num_mel_bins=80,
                frame_length=25,
                frame_shift=10,
                dither=dither,
                sample_frequency=16000,
                window_type="hamming",
                use_energy=False,
            )
            expected = (expected - expected.mean(0)).T
            torch.manual_seed(19)
            actual = frontend(waveform)[0]
            error = (actual - expected).abs()
            if actual.shape != expected.shape or not torch.isfinite(error).all():
                raise ValueError("Filterbank dimensions or finite-value contract differs")
            rows.append(
                {
                    "case": index,
                    "dither": dither,
                    "shape": list(actual.shape),
                    "max_abs_error_log_units": float(error.max()),
                    "mean_abs_error_log_units": float(error.mean()),
                }
            )
    atomic_json(
        args.output,
        {
            "status": "Measured numerical agreement; not bit-identical",
            "reference_library": "torchaudio " + torchaudio.__version__,
            "reference_file_sha256": sha256(
                args.torchaudio_root / "torchaudio/compliance/kaldi.py"
            ),
            "cases": rows,
            "manifest_sha256": corpus.manifest_hash,
            "scope": "Standard signal-processing verification only. The external verification library is not a runtime dependency or speaker-extraction model implementation.",
        },
    )
