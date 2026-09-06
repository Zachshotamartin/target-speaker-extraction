#!/usr/bin/env python3
"""Copy public source identities, frozen recipes and training records for review."""

import shutil
from pathlib import Path

from tse.utils import atomic_json, sha256


def main() -> None:
    destination = Path("metadata")
    index = {}
    sources = [
        *sorted(Path("data/manifests").glob("*.json")),
        *sorted(Path("data/raw").glob("*.acquisition.json")),
    ]
    for source in sources:
        target = destination / "data" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        index[str(target)] = sha256(target)
    for name in ("overfit", "control", "augmented"):
        for filename in ("config.json", "provenance.json", "summary.json", "metrics.jsonl"):
            source = Path("artifacts/runs") / name / filename
            if source.is_file():
                target = destination / "training" / name / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                index[str(target)] = sha256(target)
    release = Path("artifacts/releases/model.json")
    if release.exists():
        shutil.copyfile(release, destination / "model.json")
        index[str(destination / "model.json")] = sha256(destination / "model.json")
    atomic_json(
        destination / "checksums.json",
        {
            "files": index,
            "note": "Public LibriSpeech identities and relative paths only. No recordings or weights are copied. Data manifests retain CC BY 4.0 source attribution. Training durations may contain resumed segments; see implementation notes.",
        },
    )


if __name__ == "__main__":
    main()
