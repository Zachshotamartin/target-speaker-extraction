#!/usr/bin/env python3
"""Evaluate a model on predeclared acoustic conditions with a frozen-test guard."""

import argparse
from pathlib import Path

import torch

from tse.realistic_evaluation import evaluate_realistic

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("data/expanded/raw"))
    parser.add_argument("--manifest", type=Path, default=Path("data/v3/manifests/inventory.json"))
    parser.add_argument(
        "--cases", type=Path, default=Path("data/v3/manifests/dev-realistic-cases.json")
    )
    parser.add_argument("--environment-root", type=Path, required=True)
    parser.add_argument(
        "--environment-manifest", type=Path, default=Path("data/v3/manifests/environments.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    evaluate_realistic(
        args.checkpoint,
        args.root,
        args.manifest,
        args.cases,
        args.environment_root,
        args.environment_manifest,
        args.output,
        args.device,
        args.freeze,
    )
