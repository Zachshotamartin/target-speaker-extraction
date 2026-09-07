#!/usr/bin/env python3
"""Compare actual delivered CPU/MPS audio on fixed development conditions."""

import argparse
import io
import json
from pathlib import Path

import numpy as np
import torch

from tse.audio import read_audio, wav_bytes
from tse.inference import Extractor
from tse.realistic import SCENARIOS, RealisticCorpus
from tse.utils import atomic_json, sha256, source_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--environment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/v3-delivery-numerical.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing numerical verification")
    torch.set_num_threads(2)
    manifest = Path("data/v3/manifests/inventory.json")
    environment = Path("data/v3/manifests/environments.json")
    cases_path = Path("data/v3/manifests/dev-realistic-cases.json")
    corpus = RealisticCorpus(
        Path("data/expanded/raw"), manifest, "dev", 4, 5, args.environment_root, environment
    )
    cases = json.loads(cases_path.read_text())["cases"]
    models = {device: Extractor(args.checkpoint, device) for device in ("cpu", "mps")}
    rows = []
    for condition in SCENARIOS:
        case = next(case for case in cases if case["scenario"] == condition)
        signals = corpus.render(case)
        predictions = {}
        for device, extractor in models.items():
            encoded, _ = extractor.extract_files(
                io.BytesIO(wav_bytes(signals["mixture"])),
                io.BytesIO(wav_bytes(signals["reference"])),
            )
            predictions[device] = read_audio(io.BytesIO(encoded))
        left, right = predictions["cpu"], predictions["mps"]
        exact_length = len(left) == len(right) == len(signals["mixture"])
        if not exact_length or not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError("Delivered samples are invalid")
        difference = right.astype(np.float64) - left
        maximum = float(np.max(np.abs(difference)))
        relative = float(np.linalg.norm(difference) / max(np.linalg.norm(left), 1e-6))
        rows.append(
            {
                "case_id": case["case_id"],
                "condition": condition,
                "exact_length": exact_length,
                "maximum_absolute_error": maximum,
                "relative_l2_error": relative,
                "passed": maximum <= 1e-4 and relative <= 1e-3,
            }
        )
    result = {
        "checkpoint_sha256": sha256(args.checkpoint),
        "passed": all(row["passed"] for row in rows),
        "source_tree_sha256": source_digest(),
        "case_manifest_sha256": sha256(cases_path),
        "environment_manifest_sha256": sha256(environment),
        "rows": rows,
        "thresholds": {"maximum_absolute_error": 1e-4, "relative_l2_error": 1e-3},
        "protocol": "First development request in each acoustic condition; delivered float WAV decoded after identical preprocessing and peak guarding on CPU and MPS. Tests numerical agreement, not perceived quality.",
    }
    atomic_json(args.output, result)
    if not result["passed"]:
        raise ValueError("CPU/MPS delivered waveforms differ beyond the declared tolerances")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
