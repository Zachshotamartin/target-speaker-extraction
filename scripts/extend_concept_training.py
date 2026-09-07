#!/usr/bin/env python3
"""Preserve a completed concept run and prepare an explicit bounded continuation."""

import argparse
import json
from pathlib import Path

from tse.concept_training import fork_concept

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/concept/manifest.json"))
    parser.add_argument("--minutes", type=float, default=480)
    args = parser.parse_args()
    print(
        json.dumps(
            fork_concept(args.parent_run, args.config, args.manifest, args.run, args.minutes),
            indent=2,
        )
    )
