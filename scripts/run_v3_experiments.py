#!/usr/bin/env python3
"""Run the predeclared v3 development experiments serially and resume safely.

This runner never opens fresh test cases or changes the served checkpoint.
Each subprocess releases its GPU allocation before the next experiment.
"""

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

from tse.config import ExperimentConfig
from tse.utils import atomic_json, sha256

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path("data/v3/manifests/inventory.json")
ENV_MANIFEST = Path("data/v3/manifests/environments.json")
STATUS = Path("artifacts/verification/v3-suite-status.json")


def read(path):
    return json.loads(Path(path).read_text())


def write_once(path, payload):
    path = Path(path)
    if path.exists():
        if read(path) != payload:
            raise ValueError(f"Existing decision differs: {path}")
    else:
        atomic_json(path, payload)


def run(command, name):
    log = Path("artifacts/verification") / f"v3-suite-{name}.log"
    with log.open("a") as stream:
        stream.write(f"\nStarted {datetime.now(UTC).isoformat()}\n")
        stream.flush()
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        atomic_json(
            STATUS,
            {
                "status": "running",
                "stage": name,
                "pid": process.pid,
                "command": command,
                "log": str(log),
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
        print(f"Started {name}; PID {process.pid}; {log}", flush=True)
        returncode = process.wait()
        if returncode:
            atomic_json(
                STATUS,
                {"status": "failed", "stage": name, "returncode": returncode, "log": str(log)},
            )
            raise subprocess.CalledProcessError(returncode, command)


def train(name, config_path, initialization, development, budget):
    config = ExperimentConfig.model_validate(read(config_path))
    if config.training.max_optimizer_updates != budget:
        config.training.max_optimizer_updates = budget
        atomic_json(config_path, config.model_dump())
    run_dir = Path("artifacts/runs") / name
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = read(summary_path)
        if summary["steps"] >= budget and summary["status"] == "complete":
            if sha256(run_dir / "best.pt") != summary["best_checkpoint_sha256"]:
                raise ValueError(f"Completed checkpoint changed: {name}")
            return run_dir / "best.pt"
    command = [
        str(ROOT / ".venv/bin/tse"),
        "train",
        "--config",
        str(config_path),
        "--root",
        "data/expanded/raw",
        "--manifest",
        str(MANIFEST),
        "--dev-cases",
        str(development),
        "--run",
        str(run_dir),
        "--device",
        "mps",
        "--prefetch",
        "--compile-blocks",
    ]
    if (run_dir / "latest.pt").exists():
        command.append("--resume")
    else:
        command += ["--initialize-from", str(initialization)]
    run(command, name)
    summary = read(summary_path)
    if summary["steps"] != budget or summary["status"] != "complete":
        raise RuntimeError(f"Training did not complete its budget: {name}")
    return run_dir / "best.pt"


def snapshot(checkpoint, name):
    destination = Path("artifacts/v3-candidates") / f"{name}.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(destination) != sha256(checkpoint):
            raise ValueError(f"Refusing to replace frozen development candidate: {destination}")
    else:
        shutil.copy2(checkpoint, destination)
    return destination


def evaluate(checkpoint, name, environment_root):
    reports = {}
    for kind, cases in (("clean", "dev-report-cases"), ("realistic", "dev-realistic-cases")):
        output = Path("reports") / f"v3-{name}-{kind}-development.json"
        if not output.exists():
            script = "evaluate_quality.py" if kind == "clean" else "evaluate_realistic.py"
            command = [
                sys.executable,
                f"scripts/{script}",
                "--checkpoint",
                str(checkpoint),
                "--cases",
                f"data/v3/manifests/{cases}.json",
                "--root",
                "data/expanded/raw",
                "--manifest",
                str(MANIFEST),
                "--output",
                str(output),
                "--device",
                "mps",
                "--cpu-threads",
                "2",
            ]
            if kind == "realistic":
                command += ["--environment-root", str(environment_root)]
            run(command, f"{name}-{kind}-evaluation")
        report = read(output)
        if report["checkpoint_sha256"] != sha256(checkpoint):
            raise ValueError(f"Report checkpoint differs: {output}")
        if report["source_manifest_sha256"] != sha256(MANIFEST):
            raise ValueError(f"Report speech inventory differs: {output}")
        if report["case_manifest_sha256"] != sha256(Path(f"data/v3/manifests/{cases}.json")):
            raise ValueError(f"Report cases differ: {output}")
        if kind == "realistic" and report["environment_manifest_sha256"] != sha256(ENV_MANIFEST):
            raise ValueError(f"Report acoustic inventory differs: {output}")
        reports[kind] = report
    clean, acoustic = reports["clean"]["summary"], reports["realistic"]
    return {
        "name": name,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "acoustic_si_sdri_db": acoustic["present_macro_mean_si_sdri_db"],
        "clean_si_sdri_db": clean["mean_si_sdri_db"],
        "clean_estoi": clean["mean_estoi"],
        "clean_confusion": clean["confusion_fraction"],
        "absent_median_attenuation_db": acoustic["conditions"]["target_absent"][
            "absent_median_attenuation_db"
        ],
    }


def clean_eligible(candidate, baseline, gates):
    return (
        candidate["clean_si_sdri_db"]
        >= baseline["clean_si_sdri_db"] - gates["maximum_clean_si_sdri_regression_db"]
        and candidate["clean_estoi"]
        >= baseline["clean_estoi"] - gates["maximum_clean_estoi_regression"]
        and candidate["clean_confusion"]
        <= baseline["clean_confusion"] + gates["maximum_clean_confusion_increase"]
    )


def architecture_experiments(environment_root):
    pilots = []
    for variant in ("band-real", "band-complex", "band-resnet"):
        name = f"v3-{variant}"
        frozen = Path(f"artifacts/v3-candidates/{variant}-pilot.pt")
        if not frozen.exists():
            checkpoint = train(
                name,
                Path(f"configs/{name}.json"),
                Path(f"artifacts/v3-initializers/{variant}.pt"),
                Path("data/v3/manifests/dev-cases.json"),
                4000,
            )
            frozen = snapshot(checkpoint, f"{variant}-pilot")
        pilots.append(evaluate(frozen, f"{variant}-pilot", environment_root))
    chosen = max(pilots, key=lambda row: row["acoustic_si_sdri_db"])
    decision = {
        "status": "selected_on_development_only",
        "candidates": pilots,
        "selected": chosen["name"],
        "rule": "Highest target-present acoustic development SI-SDRi for the predeclared 10000-update architecture continuation; deployment has separate clean-retention gates.",
    }
    write_once("reports/v3-architecture-selection.json", decision)
    variant = chosen["name"].removesuffix("-pilot")
    checkpoint = train(
        f"v3-{variant}",
        Path(f"configs/v3-{variant}.json"),
        Path(f"artifacts/v3-initializers/{variant}.pt"),
        Path("data/v3/manifests/dev-cases.json"),
        10000,
    )
    frozen = snapshot(checkpoint, f"{variant}-extended")
    extended = evaluate(frozen, f"{variant}-extended", environment_root)
    write_once("reports/v3-architecture-extended.json", extended)
    return pilots + [extended]


def adaptation_experiments(candidates, environment_root):
    baseline = next(row for row in candidates if row["name"] == "baseline")
    gates = read("reports/v3-evaluation-policy.json")["automatic_promotion_gates"]
    eligible = [row for row in candidates if clean_eligible(row, baseline, gates)]
    source = max(eligible, key=lambda row: row["acoustic_si_sdri_db"])
    write_once(
        "reports/v3-adaptation-initialization.json",
        {
            "status": "selected_before_adaptation_training",
            "source": source,
            "rule": "Highest acoustic development SI-SDRi among candidates passing the predeclared clean-retention gates, including v0.2.0.",
            "optimizer": "Fresh identical AdamW in both arms; classifier preserved with identical labels",
            "seed": 8107,
            "updates_per_arm": 3000,
            "development_cases_sha256": sha256(Path("data/v3/manifests/dev-adaptation-cases.json")),
            "environment_manifest_sha256": sha256(ENV_MANIFEST),
        },
    )
    payload = torch.load(source["checkpoint"], map_location="cpu", weights_only=True)
    results = []
    for kind in ("clean", "realistic"):
        name = f"v3-adaptation-{kind}"
        config = ExperimentConfig.model_validate(payload["config"])
        config.experiment, config.seed = name, 8107
        config.model.weights = "project_checkpoint"
        config.data.development_source = "LibriSpeech/dev-clean+dev-other"
        config.data.test_source = "LibriSpeech/test-other"
        config.training.max_optimizer_updates = 3000
        config.training.learning_rate = 0.0003
        config.training.minimum_learning_rate = 0.00003
        config.training.learning_rate_schedule = "cosine"
        config.training.schedule_decay_updates = 3000
        config.training.validation_interval_updates = 500
        config.training.preserve_initialized_classifier = True
        config.augmentation.reference_enabled = False
        config.augmentation.environment_root = str(environment_root)
        config.augmentation.environment_manifest = str(ENV_MANIFEST)
        config.augmentation.realistic_enabled = kind == "realistic"
        config.augmentation.mixture_noise_enabled = kind == "realistic"
        config.data.overlap_fraction = "variable" if kind == "realistic" else 1.0
        config.data.target_present = "mixed" if kind == "realistic" else True
        config.evaluation.realistic_validation = True
        config.evaluation.development_cases_target = 140
        config.evaluation.final_test_cases_target = 700
        config_path = Path(f"configs/{name}.json")
        write_once(config_path, config.model_dump())
        checkpoint = train(
            name,
            config_path,
            Path(source["checkpoint"]),
            Path("data/v3/manifests/dev-adaptation-cases.json"),
            3000,
        )
        results.append(
            evaluate(
                snapshot(checkpoint, name.removeprefix("v3-")),
                name.removeprefix("v3-"),
                environment_root,
            )
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-root", type=Path, required=True)
    parser.add_argument("--wait-for-pid", type=int, help="Wait for the already-running first pilot")
    args = parser.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(2)
    Path("artifacts/verification").mkdir(parents=True, exist_ok=True)
    with Path("artifacts/verification/v3-suite.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.wait_for_pid:
            atomic_json(STATUS, {"status": "waiting_for_active_pilot", "pid": args.wait_for_pid})
            while True:
                try:
                    os.kill(args.wait_for_pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(10)
        candidates = architecture_experiments(args.environment_root)
        for name, checkpoint in [
            ("baseline", Path("artifacts/releases/v0.2.0/model.pt")),
            *[
                (f"continuation-{arm}", Path(f"artifacts/runs/v3-continuation-{arm}/best.pt"))
                for arm in ("control", "cosine")
            ],
        ]:
            candidates.append(evaluate(checkpoint, name, args.environment_root))
        candidates += adaptation_experiments(candidates, args.environment_root)
        baseline = next(row for row in candidates if row["name"] == "baseline")
        gates = read("reports/v3-evaluation-policy.json")["automatic_promotion_gates"]
        for row in candidates:
            row["clean_retention_pass"] = clean_eligible(row, baseline, gates)
            row["acoustic_gain_db"] = row["acoustic_si_sdri_db"] - baseline["acoustic_si_sdri_db"]
            row["promotion_eligible"] = (
                row["clean_retention_pass"]
                and row["acoustic_gain_db"] >= gates["minimum_acoustic_mean_si_sdri_gain_db"]
            )
        eligible = [row for row in candidates if row["promotion_eligible"]]
        chosen = max(eligible, key=lambda row: row["acoustic_si_sdri_db"]) if eligible else baseline
        report = {
            "status": "development_complete_pending_final_test_and_runtime_checks",
            "selected": chosen,
            "candidates": candidates,
            "gates": gates,
            "fresh_test_scored": False,
            "served_model_changed": False,
        }
        write_once("reports/v3-development-selection.json", report)
        atomic_json(
            STATUS,
            {
                "status": "development_complete",
                "selected": chosen["name"],
                "report": "reports/v3-development-selection.json",
            },
        )
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
