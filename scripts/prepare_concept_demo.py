#!/usr/bin/env python3
"""Reserve new recordings for a small known-voice concept demonstration."""

import argparse
import json
from pathlib import Path

from tse.concept_data import prepare_concept

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--source-manifest", type=Path, default=Path("data/reference/manifest.json")
    )
    parser.add_argument("--gate", type=Path, default=Path("reports/reference-learning-gate.json"))
    parser.add_argument("--output", type=Path, default=Path("data/concept/manifest.json"))
    args = parser.parse_args()
    result = prepare_concept(
        args.root, args.source_manifest, args.checkpoint, args.gate, args.output
    )
    print(
        json.dumps(
            {
                "speakers": len(result["speakers"]),
                "utterances": {
                    split: sum(len(rows) for rows in pools.values())
                    for split, pools in result["splits"].items()
                },
            }
        )
    )
