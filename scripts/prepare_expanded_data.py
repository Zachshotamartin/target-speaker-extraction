#!/usr/bin/env python3
"""Expand training speech while reserving previously untrained speaker identities."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from tse.data import SpeechCorpus, audit_records, build_cases, inventory, load_cases
from tse.utils import atomic_json, sha256


def main() -> None:
    root = Path("data/expanded/raw")
    manifests = Path("data/expanded/manifests")
    manifests.mkdir(parents=True, exist_ok=True)
    acquisition = json.loads((root / "train-clean-100.acquisition.json").read_text())
    if len(acquisition["speakers"]) < 200:
        raise ValueError("Expanded acquisition has not collected the intended speaker diversity")
    old = json.loads(Path("data/manifests/inventory.json").read_text())
    # Hard links stay inside the new corpus root without duplicating public audio.
    for row in old["records"]:
        if row["split"] == "dev":
            source, target = Path("data/raw") / row["path"], root / row["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                os.link(source, target)
            if sha256(target) != row["sha256"]:
                raise ValueError("Development audio differs from its frozen source identity")
    expanded = inventory(root, manifests / "source-inventory.json")
    prior_train = {r["speaker"] for r in old["records"] if r["split"] == "train"}
    all_train = {r["speaker"] for r in expanded["records"] if r["split"] == "train"}
    candidates = sorted(
        all_train - prior_train,
        key=lambda speaker: hashlib.sha256(
            f"quality-v2-holdout-20260906:{speaker}".encode()
        ).hexdigest(),
    )
    if len(candidates) < 20:
        raise ValueError("Need 20 previously untrained speakers for the fresh reserved evaluation")
    reserved = set(candidates[:20])
    for row in expanded["records"]:
        if row["speaker"] in reserved:
            row["split"] = "test"
    expanded["audit"] = audit_records(expanded["records"])
    expanded["split_note"] = (
        "Expanded custom protocol: 20 train-clean-100 identities unused by v0.1.0 are reserved "
        "as a fresh test; remaining train-clean-100 identities train the model. Original "
        "dev-clean development audio is retained. Original test-clean is excluded. "
        "The existing v1 mixture recipe format is reused; this is not an official benchmark split."
    )
    manifest = manifests / "inventory.json"
    atomic_json(manifest, expanded)
    development = SpeechCorpus(root, manifest, "dev", 4, 5)
    for filename in ("dev-cases.json", "dev-report-cases.json"):
        payload = json.loads((Path("data/manifests") / filename).read_text())
        payload["source_manifest_sha256"] = sha256(manifest)
        atomic_json(manifests / filename, payload)
        load_cases(manifests / filename, development)
    test = SpeechCorpus(root, manifest, "test", 4, 5)
    build_cases(test, 1000, 2026090600, manifests / "fresh-test-cases.json")
    training = SpeechCorpus(root, manifest, "train", 4, 5)
    report = {
        "status": "Prepared; fresh test recipes created but not evaluated.",
        "holdout_rule": "20 new identities with smallest SHA256 of quality-v2-holdout-20260906:<speaker>.",
        "prior_training_speakers": sorted(prior_train),
        "reserved_speakers": sorted(reserved),
        "training_speakers": training.speakers,
        "split_note": expanded["split_note"],
        "audit": expanded["audit"],
        "eligible_training_utterances": sum(len(rows) for rows in training.by_speaker.values()),
        "eligible_training_hours": sum(
            row["samples"] for rows in training.by_speaker.values() for row in rows
        )
        / 16000
        / 3600,
        "manifest_sha256": sha256(manifest),
        "fresh_test_manifest_sha256": sha256(manifests / "fresh-test-cases.json"),
        "source_acquisition_sha256": sha256(root / "train-clean-100.acquisition.json"),
        "acquisition_note": "Requested caps are 1000 speakers and 10000 utterances each. No cap was reached; all encountered speech members were selected. Publisher archive checksum remains unverified; per-file hashes identify the data.",
    }
    atomic_json(Path("reports/expanded-data.json"), report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("audit", "eligible_training_utterances", "eligible_training_hours")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
