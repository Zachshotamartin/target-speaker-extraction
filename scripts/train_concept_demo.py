#!/usr/bin/env python3
"""Run a short concept session; save and stop at its time limit."""

import argparse
from pathlib import Path

from tse.concept_training import train_concept

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/concept-demo.json"))
    parser.add_argument(
        "--source-manifest", type=Path, default=Path("data/reference/manifest.json")
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/concept/manifest.json"))
    parser.add_argument("--minutes", type=float, default=30)
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-updates", type=int)
    parser.add_argument("--gallery", type=Path, default=Path("artifacts/gallery/concept-demo"))
    args = parser.parse_args()
    train_concept(
        args.config,
        args.root,
        args.source_manifest,
        args.manifest,
        args.checkpoint,
        args.run,
        args.minutes,
        args.device,
        args.resume,
        args.gallery,
        args.stop_after_updates,
    )
