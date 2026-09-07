#!/usr/bin/env python3
"""Publish reproducible v3 identities and logs, excluding audio, weights and private ratings."""

import argparse
import shutil
from pathlib import Path

from tse.utils import atomic_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("metadata/v3"))
    args = parser.parse_args()
    files = {}

    def copy(source, relative):
        destination = args.output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        files[str(relative)] = sha256(destination)

    for path in sorted(Path("data/v3/manifests").glob("*.json")):
        copy(path, Path("corpus") / path.name)
    for run in sorted(Path("artifacts/runs").glob("v3-*")):
        for name in ("config.json", "provenance.json", "metrics.jsonl", "summary.json"):
            path = run / name
            if path.is_file():
                copy(path, Path("training") / run.name / name)
    copy(args.release, Path("candidate-model.json"))
    atomic_json(
        args.output / "checksums.json",
        {
            "files": files,
            "note": "Public speech/acoustic identities, experiment configurations and training logs. No audio, model weights, browser identifiers, individual ratings, or hidden listening-study keys. Resumed timing segments and initialization histories must be interpreted from provenance rather than the last step count alone.",
        },
    )
    print(f"Snapshotted {len(files)} metadata files")


if __name__ == "__main__":
    main()
