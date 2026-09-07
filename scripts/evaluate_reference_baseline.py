#!/usr/bin/env python3
"""Score the official clean benchmark using fixed enrollment, with a test freeze."""

import argparse
import json
from pathlib import Path

import torch

from tse.engine import load_model
from tse.librimix import LibriMixCorpus
from tse.reference_training import evaluate_reference
from tse.utils import atomic_json, sha256, source_digest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--manifest", type=Path, default=Path("data/reference/manifest.json"))
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing benchmark results")
    if args.split == "test":
        if args.freeze is None:
            raise ValueError("Freeze the exact artifact before benchmark test scoring")
        freeze = json.loads(args.freeze.read_text())
        if (
            freeze.get("checkpoint_sha256") != sha256(args.checkpoint)
            or freeze.get("manifest_sha256") != sha256(args.manifest)
            or freeze.get("source_tree_sha256") != source_digest()
            or freeze.get("status") != "frozen_before_benchmark_test"
        ):
            raise ValueError(
                "Benchmark test does not match its frozen artifact/data/implementation"
            )
    torch.set_num_threads(2)
    if args.device == "mps":
        torch.mps.set_per_process_memory_fraction(0.6)
    corpus = LibriMixCorpus(args.root, args.manifest, args.split)
    model, payload = load_model(args.checkpoint, args.device)
    if set(corpus.speakers) & set(payload["provenance"]["train_speakers"]):
        raise ValueError("Training and benchmark speakers overlap")
    result = evaluate_reference(model, corpus, list(range(len(corpus))))
    atomic_json(
        args.output,
        {
            "status": "Official benchmark evaluation; historical test identities, not a fresh test",
            "protocol": "libri2mix-16k-min-clean",
            "split": args.split,
            "checkpoint_sha256": sha256(args.checkpoint),
            "manifest_sha256": sha256(args.manifest),
            "source_tree_sha256": source_digest(),
            "freeze_sha256": sha256(args.freeze) if args.freeze else None,
            "summary": {k: v for k, v in result.items() if k != "rows"},
            "rows": result["rows"],
            "metric_note": "Raw model estimates with whole-utterance inference. Report absolute SI-SDR and improvement separately; success is SI-SDRi > 1 dB. App-delivered audio requires separate validation.",
        },
    )
