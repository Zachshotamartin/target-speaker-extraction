#!/usr/bin/env python3
"""Gate a full training run on actual fixed-set extraction and reference switching."""

import argparse
import json
from pathlib import Path

import torch

from tse.config import ExperimentConfig
from tse.engine import load_model
from tse.librimix import LibriMixCorpus
from tse.reference_training import evaluate_reference
from tse.utils import atomic_json, sha256


def validate(run, root, output, device="mps"):
    if output.exists():
        raise FileExistsError("Preserve the existing learning gate report")
    summary = json.loads((run / "summary.json").read_text())
    if summary["status"] != "complete":
        raise ValueError("Learning diagnostic has not completed")
    torch.set_num_threads(2)
    if device == "mps":
        torch.mps.set_per_process_memory_fraction(0.6)
    checkpoint = run / "best.pt"
    model, payload = load_model(checkpoint, device)
    indices = payload["provenance"]["fixed_indices"]
    if indices != list(range(16)):
        raise ValueError(
            "The declared diagnostic includes both targets of the first eight mixtures"
        )
    config = ExperimentConfig.model_validate(payload["config"])
    corpus = LibriMixCorpus(root, Path("data/reference/manifest.json"), "train")
    if payload["manifest_sha256"] != corpus.manifest_hash:
        raise ValueError("Learning corpus changed")
    result = evaluate_reference(model, corpus, indices, config.seed, 48000)
    switches = [row["si_sdr_db"] > row["interferer_si_sdr_db"] for row in result["rows"]]
    passed = result["mean_si_sdri_db"] >= 8 and all(switches)
    report = {
        "status": "passed" if passed else "failed",
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_update": payload["step"],
        "diagnostic_completed_updates": summary["steps"],
        "evaluation_device": str(next(model.parameters()).device),
        "torch_version": str(torch.__version__),
        "training_source_tree_sha256": payload["provenance"]["source_tree_sha256"],
        "manifest_sha256": corpus.manifest_hash,
        "model_config": config.model.model_dump(),
        "thresholds": {"mean_si_sdri_db_at_least": 8, "correct_target_choices": 16},
        "correct_target_choices": sum(switches),
        "result": result,
        "interpretation": "Fixed training-set learning only; no held-out quality claim. These diagnostic weights must not initialize or replace the full baseline.",
    }
    atomic_json(output, report)
    print(
        json.dumps(
            {key: report[key] for key in ("status", "correct_target_choices", "checkpoint_sha256")}
        ),
        flush=True,
    )
    if not passed:
        raise RuntimeError("The diagnostic did not pass; debug learning before full training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/reference-learning-gate.json"))
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    args = parser.parse_args()
    validate(args.run, args.root, args.output, args.device)
