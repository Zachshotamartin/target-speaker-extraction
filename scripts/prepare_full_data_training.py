#!/usr/bin/env python3
"""Freeze a training manifest with distinct-utterance development references."""

import argparse
import hashlib
import json
from pathlib import Path

from tse.utils import atomic_json, sha256


def prepare(source, output, report):
    if output.exists():
        raise FileExistsError("The full-data manifest is immutable; choose a new output path")
    payload = json.loads(source.read_text())
    rows = payload["splits"]["dev"]
    known = {record["path"]: record for row in rows for record in row["sources"]}
    repairs = []
    for row in rows:
        for side, target in enumerate(row["sources"]):
            key = f"{row['id']}:{side}"
            previous = payload["enrollments"]["dev"][key]
            reference = known[previous]
            if reference["speaker"] != target["speaker"]:
                raise ValueError("Unexpected wrong-speaker reference; inspect source mapping")
            if reference["utterance"] != target["utterance"]:
                continue
            candidates = [
                r
                for r in known.values()
                if r["speaker"] == target["speaker"] and r["utterance"] != target["utterance"]
            ]
            replacement = min(
                candidates,
                key=lambda r: (
                    hashlib.sha256(f"distinct-dev-v1:{key}:{r['utterance']}".encode()).hexdigest(),
                    r["path"],
                ),
            )
            payload["enrollments"]["dev"][key] = replacement["path"]
            repairs.append(
                {"request": key, "previous": previous, "replacement": replacement["path"]}
            )
    payload["full_training_enrollment_revision"] = {
        "source_manifest_sha256": sha256(source),
        "method": "distinct-dev-v1: hash-ranked different utterance from the same development speaker",
        "repaired_development_references": len(repairs),
        "training_and_test_unchanged": True,
    }
    atomic_json(output, payload)
    # This detailed mapping stays alongside the local manifest, outside version control.
    atomic_json(output.with_name(output.stem + "-enrollment-repairs.json"), repairs)
    summary = {
        **payload["full_training_enrollment_revision"],
        "manifest_sha256": sha256(output),
        "development_requests": len(rows) * 2,
        "changed_training_mixtures": 0,
        "changed_development_mixtures": 0,
        "note": "Development references differ from the source mapping for the repaired cases. No test audio was evaluated.",
    }
    atomic_json(report, summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/reference/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("data/full-training/manifest.json"))
    parser.add_argument(
        "--report", type=Path, default=Path("reports/full-data-enrollment-audit.json")
    )
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output, args.report), indent=2))
