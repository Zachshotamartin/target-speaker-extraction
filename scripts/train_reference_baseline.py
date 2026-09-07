#!/usr/bin/env python3
"""Train the full reference baseline or an explicitly separate learning diagnostic."""

import argparse
import json
from pathlib import Path

from tse.config import ExperimentConfig
from tse.reference_training import train_reference
from tse.utils import sha256, source_digest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/reference-libri2mix.json"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/reference/manifest.json"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fixed-indices", type=int, nargs="+")
    parser.add_argument("--stop-after-updates", type=int)
    parser.add_argument(
        "--learning-gate", type=Path, default=Path("reports/reference-learning-gate.json")
    )
    args = parser.parse_args()
    if args.fixed_indices is None:
        gate = json.loads(args.learning_gate.read_text())
        config = ExperimentConfig.load(args.config)
        if (
            gate.get("status") != "passed"
            or gate.get("manifest_sha256") != sha256(args.manifest)
            or gate.get("model_config") != config.model.model_dump()
            or gate.get("training_source_tree_sha256") != source_digest()
        ):
            raise ValueError(
                "Full training requires a passed learning gate for this implementation, model and dataset"
            )
    train_reference(
        args.config,
        args.root,
        args.manifest,
        args.run,
        args.device,
        args.resume,
        args.fixed_indices,
        args.stop_after_updates,
    )
