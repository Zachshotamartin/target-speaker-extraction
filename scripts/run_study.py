#!/usr/bin/env python3
"""Reproduce the bounded local study, with final test evaluation explicitly selected."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from tse.config import ExperimentConfig
from tse.data import SpeechCorpus, build_cases, inventory
from tse.utils import atomic_json


def run(*arguments: str) -> None:
    subprocess.run([sys.executable, "-m", "tse.cli", *arguments], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    if args.steps < 1000:
        parser.error("The study budget must be at least 1,000 updates")
    root = Path("data/raw")
    manifest = Path("data/manifests/inventory.json")
    if args.download:
        for split, speakers, utterances in [
            ("train-clean-100", 60, 50),
            ("dev-clean", 40, 30),
            ("test-clean", 40, 30),
        ]:
            run(
                "data",
                "acquire",
                split,
                "--speakers",
                str(speakers),
                "--utterances",
                str(utterances),
            )
    if not manifest.exists():
        inventory(root, manifest)
    for split, count, seed, seconds, name in [
        ("train", 16, 44000, 2, "train-cases"),
        ("dev", 80, 88000, 4, "dev-cases"),
        ("dev", 400, 188000, 4, "dev-report-cases"),
        ("test", 1000, 99000, 4, "test-cases"),
    ]:
        destination = Path("data/manifests") / f"{name}.json"
        if not destination.exists():
            build_cases(SpeechCorpus(root, manifest, split, seconds, 5), count, seed, destination)

    overfit_summary = Path("artifacts/runs/overfit/summary.json")
    if not overfit_summary.exists():
        run(
            "train",
            "--config",
            "configs/overfit.json",
            "--run",
            "artifacts/runs/overfit",
            "--device",
            args.device,
            "--fixed-cases",
            "data/manifests/train-cases.json",
        )
    if json.loads(overfit_summary.read_text())["best_development_si_sdri_db"] < 5:
        raise RuntimeError("Tiny-set learning gate failed; inspect that run before scaling")
    config = ExperimentConfig.load(Path("configs/control.json"))
    config.training.max_optimizer_updates = args.steps
    destinations = []
    for name, augmented in [("control", False), ("augmented", True)]:
        selected = config.model_copy(deep=True)
        if augmented:
            selected.experiment = "E04-reference-augmentation"
        selected.augmentation.reference_enabled = augmented
        destination = Path("configs") / f"{name}-study.json"
        atomic_json(destination, selected.model_dump())
        destinations.append(destination)
        folder = Path("artifacts/runs") / name
        summary = folder / "summary.json"
        finished = json.loads(summary.read_text())["steps"] if summary.exists() else 0
        if finished > args.steps:
            raise ValueError(
                "Existing run exceeds requested budget; use a fresh workspace for a smaller study"
            )
        if finished < args.steps:
            arguments = [
                "train",
                "--config",
                str(destination),
                "--run",
                str(folder),
                "--device",
                args.device,
            ]
            if (folder / "latest.pt").exists():
                arguments.append("--resume")
            run(*arguments)
        run(
            "evaluate",
            "--checkpoint",
            str(folder / "best.pt"),
            "--cases",
            "data/manifests/dev-report-cases.json",
            "--output",
            f"reports/{name}-development.json",
            "--device",
            args.device,
        )
    run(
        "compare",
        "--control",
        "reports/control-development.json",
        "--treatment",
        "reports/augmented-development.json",
        "--output",
        "reports/development-comparison.json",
    )
    comparison = json.loads(Path("reports/development-comparison.json").read_text())
    clean = comparison["conditions"]["clean"]
    use_augmented = (
        comparison["equal_weight_mismatch_gain_db"] > 0
        and clean["mean_treatment_minus_control_db"] >= -0.5
        and clean["treatment_confusion_fraction"] <= clean["control_confusion_fraction"]
    )
    selected_name = "augmented" if use_augmented else "control"
    selection = {
        "selected": selected_name,
        "rule": "Prefer augmentation only with positive average mismatch gain, <=0.5 dB clean degradation, and no clean confusion regression",
        "selection_split": "dev",
        "test_used_for_selection": False,
        "step_budget": args.steps,
    }
    atomic_json(Path("reports/model-selection.json"), selection)
    checkpoint = f"artifacts/runs/{selected_name}/best.pt"
    run("export", "--checkpoint", checkpoint)
    run("examples")
    run(
        "diagnose",
        "--checkpoint",
        checkpoint,
        "--output",
        "reports/reference-diagnostics.json",
        "--device",
        args.device,
    )
    if args.evaluate_test:
        freeze = Path("reports/test-freeze.json")
        if freeze.exists():
            raise ValueError(
                "Final test was already opened; do not silently re-use it for another model selection"
            )
        from tse.utils import sha256

        atomic_json(
            freeze,
            {
                "selection": selection,
                "case_manifest_sha256": sha256(Path("data/manifests/test-cases.json")),
                "control_checkpoint_sha256": sha256(Path("artifacts/runs/control/best.pt")),
                "augmented_checkpoint_sha256": sha256(Path("artifacts/runs/augmented/best.pt")),
                "note": "Written before final test evaluation; checkpoint selection used development data only.",
            },
        )
        for name in ("control", "augmented"):
            run(
                "evaluate",
                "--checkpoint",
                f"artifacts/runs/{name}/best.pt",
                "--cases",
                "data/manifests/test-cases.json",
                "--output",
                f"reports/{name}-test.json",
                "--device",
                args.device,
            )
        run(
            "compare",
            "--control",
            "reports/control-test.json",
            "--treatment",
            "reports/augmented-test.json",
            "--output",
            "reports/test-comparison.json",
        )
    print(
        json.dumps(
            {
                "selected_model": selected_name,
                "final_test_evaluated": args.evaluate_test,
                "next_command": "uv run tse serve",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
