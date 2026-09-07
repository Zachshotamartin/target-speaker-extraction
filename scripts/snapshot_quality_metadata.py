#!/usr/bin/env python3
"""Snapshot the expanded-data experiment without replacing v0.1.0 metadata."""

import argparse
import shutil
from pathlib import Path

from tse.utils import atomic_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True, help="Exported model JSON")
    parser.add_argument("--gallery", type=Path, default=Path("artifacts/gallery/index.json"))
    parser.add_argument("--output", type=Path, default=Path("metadata/quality-v2"))
    args = parser.parse_args()
    files = {}

    def copy(source, relative):
        target = args.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        files[str(target)] = sha256(target)

    for name in (
        "inventory.json",
        "dev-cases.json",
        "dev-report-cases.json",
        "fresh-test-cases.json",
    ):
        copy(Path("data/expanded/manifests") / name, Path("corpus") / name)
    copy(
        Path("data/expanded/raw/train-clean-100.acquisition.json"), Path("corpus/acquisition.json")
    )
    for run in sorted(Path("artifacts/runs").glob("quality-*")):
        for name in ("config.json", "provenance.json", "metrics.jsonl", "summary.json"):
            if (run / name).is_file():
                copy(run / name, Path("training") / run.name / name)
    copy(args.release, Path("model.json"))
    copy(args.gallery, Path("gallery.json"))
    atomic_json(
        args.output / "checksums.json",
        {
            "files": files,
            "note": "Public LibriSpeech source identities, hashes and custom recipes; no audio or model weights. Original v0.1.0 metadata remains separate. Training logs can contain resumed timing segments. A completed pilot summary is not a summary of a later interrupted continuation; use the selected checkpoint's recorded step and the full metric stream.",
        },
    )
    print(f"Copied {len(files)} metadata files to {args.output}")


if __name__ == "__main__":
    main()
