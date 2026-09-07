#!/usr/bin/env python3
"""Capture fixed training steps or average model states before development evaluation."""

import argparse
from pathlib import Path

from tse.checkpoints import average, capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collector = commands.add_parser("capture")
    collector.add_argument("--run", type=Path, required=True)
    collector.add_argument("--steps", type=int, nargs="+", required=True)
    collector.add_argument("--output", type=Path, required=True)
    averager = commands.add_parser("average")
    averager.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    averager.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "capture":
        capture(args.run, args.steps, args.output)
    else:
        average(args.checkpoints, args.output)


if __name__ == "__main__":
    main()
