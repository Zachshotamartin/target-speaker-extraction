#!/usr/bin/env python3
"""Copy public source identities, frozen recipes and training records for review."""

import json
import shutil
from pathlib import Path

from tse.utils import atomic_json, sha256


def main() -> None:
    release = Path("artifacts/releases/v0.1.0/model.json")
    if not release.exists():
        release = Path("artifacts/releases/model.json")
    gallery = Path("artifacts/gallery/v0.1.0/index.json")
    if not gallery.exists():
        gallery = Path("artifacts/gallery/index.json")
    if release.exists():
        identity = json.loads(release.read_text())
        if identity["config"]["model"]["family"] != "reference_conditioned_tcn":
            raise ValueError("Use snapshot_quality_metadata.py for the newer spectral study")
        if (
            gallery.exists()
            and json.loads(gallery.read_text())["checkpoint_sha256"]
            != identity["checkpoint_sha256"]
        ):
            raise ValueError("Original-study gallery and release identities differ")
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
    if release.exists():
        shutil.copyfile(release, destination / "model.json")
        index[str(destination / "model.json")] = sha256(destination / "model.json")
    if gallery.exists():
        shutil.copyfile(gallery, destination / "gallery.json")
        index[str(destination / "gallery.json")] = sha256(destination / "gallery.json")
    atomic_json(
        destination / "checksums.json",
        {
            "files": index,
            "note": "Public LibriSpeech identities and relative paths only. No recordings or weights are copied. Data manifests retain CC BY 4.0 source attribution. Training durations may contain resumed segments; see implementation notes.",
        },
    )


if __name__ == "__main__":
    main()
