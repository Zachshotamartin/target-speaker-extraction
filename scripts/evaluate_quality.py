#!/usr/bin/env python3
"""Assess delivered audio on development data or an explicitly frozen final test."""

import argparse
from pathlib import Path

import torch

from tse.quality import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=Path("data/manifests/dev-report-cases.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/inventory.json"))
    parser.add_argument("--freeze", type=Path, help="Required hash-bound selection for final test")
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    evaluate(
        args.checkpoint, args.cases, args.output, args.device, args.root, args.manifest, args.freeze
    )


if __name__ == "__main__":
    main()
