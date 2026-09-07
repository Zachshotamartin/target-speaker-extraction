#!/usr/bin/env python3
"""Train complete Libri2Mix epochs in user-started sessions of at most eight hours."""

import argparse
from pathlib import Path

from tse.full_training import train_full

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/full-training/manifest.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/full-data-efficient.json"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    parser.add_argument("--minutes", type=float, default=480)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train_full(
        args.config, args.root, args.manifest, args.run, args.device, args.resume, args.minutes
    )
