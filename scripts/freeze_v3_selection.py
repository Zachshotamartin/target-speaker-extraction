#!/usr/bin/env python3
"""Bind the development decision, verified export and untouched test protocol."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import torch

from tse.utils import atomic_json, git_state, sha256, source_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-export", type=Path, required=True)
    args = parser.parse_args()
    output = Path("reports/v3-selection-freeze.json")
    if output.exists() or any(Path("reports").glob("v3-*-test.json")):
        raise FileExistsError("Do not replace a frozen selection or reopen its test")
    decision_path = Path("reports/v3-development-selection.json")
    decision = json.loads(decision_path.read_text())
    if decision["status"] != "development_complete_pending_final_test_and_runtime_checks":
        raise ValueError("Complete the predeclared development experiments first")
    promoted = decision["selected"]["name"] != "baseline"
    chosen = (
        decision["selected"]
        if promoted
        else max(
            (row for row in decision["candidates"] if row["name"] != "baseline"),
            key=lambda row: row["acoustic_si_sdri_db"],
        )
    )
    source_path = Path(chosen["checkpoint"])
    if sha256(source_path) != chosen["checkpoint_sha256"]:
        raise ValueError("Development candidate weights changed")
    original, exported = [
        torch.load(path, map_location="cpu", weights_only=True)
        for path in (source_path, args.candidate_export)
    ]
    if (
        original["model"].keys() != exported["model"].keys()
        or original["config"] != exported["config"]
    ):
        raise ValueError("Export structure differs from selected source")
    for name, tensor in original["model"].items():
        if not torch.equal(tensor, exported["model"][name]):
            raise ValueError(f"Export tensor differs: {name}")
    exported_hash, code_hash = sha256(args.candidate_export), source_digest()
    runtime_reports = {}
    for device in ("cpu", "mps"):
        path = Path(f"reports/v3-delivery-{device}.json")
        profile = json.loads(path.read_text())
        if profile["model"]["checkpoint_sha256"] != exported_hash:
            raise ValueError("Runtime checkpoint differs")
        if profile["provenance"]["source_tree_sha256"] != code_hash:
            raise ValueError("Runtime implementation changed")
        if not all(
            row["exact_length"] and row["delivered_peak"] <= 0.980001 for row in profile["rows"]
        ):
            raise ValueError("Delivered waveform check failed")
        runtime_reports[device] = {"path": str(path), "sha256": sha256(path)}
    numerical_path = Path("reports/v3-delivery-numerical.json")
    numerical = json.loads(numerical_path.read_text())
    if (
        numerical["checkpoint_sha256"] != exported_hash
        or not numerical["passed"]
        or numerical["source_tree_sha256"] != code_hash
    ):
        raise ValueError("CPU/MPS numerical agreement has not passed")
    baseline = Path("artifacts/releases/v0.2.0/model.pt")
    expected_baseline = json.loads(Path("reports/v3-evaluation-policy.json").read_text())[
        "baseline_sha256"
    ]
    if sha256(baseline) != expected_baseline:
        raise ValueError("Baseline was modified")
    report_paths = sorted(Path("reports").glob("v3-*-development.json"))
    atomic_json(
        output,
        {
            "status": "frozen_before_fresh_test",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "development_selection_sha256": sha256(decision_path),
            "deployment_candidate_passed_gates": promoted,
            "research_candidate_name": chosen["name"],
            "selection": "Promotable development choice"
            if promoted
            else "Highest acoustic development experimental scorer; default remains v0.2.0 because promotion gates failed",
            "case_manifest_sha256": sha256(Path("data/v3/manifests/test-realistic-cases.json")),
            "source_manifest_sha256": sha256(Path("data/v3/manifests/inventory.json")),
            "environment_manifest_sha256": sha256(Path("data/v3/manifests/environments.json")),
            "evaluation_source_tree_sha256": code_hash,
            "checkpoints": {
                "baseline": {"path": str(baseline), "checkpoint_sha256": expected_baseline},
                "candidate": {
                    "path": str(args.candidate_export),
                    "checkpoint_sha256": exported_hash,
                    "source_checkpoint_sha256": sha256(source_path),
                    "weights_identical_after_export": True,
                },
            },
            "development_reports": {str(path): sha256(path) for path in report_paths},
            "runtime_reports": runtime_reports,
            "numerical_report_sha256": sha256(numerical_path),
            "final_listening_plan_sha256": sha256(Path("reports/v3-final-listening-plan.json")),
            "source": git_state(),
            "test_policy": "Neither these results nor failure slices may change this release's selected model or thresholds. The deployment decision was made on development data.",
        },
    )
    print(f"Frozen {chosen['name']} as {exported_hash}; commit this record before scoring test")


if __name__ == "__main__":
    main()
