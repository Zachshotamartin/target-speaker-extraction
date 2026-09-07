"""Capture fixed training steps and evaluate-ready weight averages without audio access."""

import json
import shutil
import time
from pathlib import Path

import torch

from tse.engine import save_checkpoint
from tse.utils import atomic_json, sha256


def capture(run: Path, steps: list[int], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    pending = set(steps)
    records = []
    if (output / "index.json").exists():
        previous = json.loads((output / "index.json").read_text())
        if previous["requested_steps"] != steps:
            raise ValueError("Capture plan differs from the existing snapshot index")
        records = previous["records"]
        for record in records:
            if sha256(Path(record["path"])) != record["checkpoint_sha256"]:
                raise ValueError("A captured checkpoint has changed")
            pending.remove(record["step"])
    while pending:
        validation = json.loads((run / "latest_validation.json").read_text())["summary"]
        step = validation["step"]
        if any(wanted < step for wanted in pending):
            raise ValueError("A requested checkpoint was missed; do not substitute another step")
        if step in pending:
            target = output / f"step-{step}.pt"
            if target.exists():
                raise FileExistsError(f"Refusing to replace captured checkpoint {target}")
            staging = output / f"step-{step}.pending"
            shutil.copyfile(run / "latest.pt", staging)
            payload = torch.load(staging, map_location="cpu", weights_only=True)
            if payload["step"] != step:
                raise ValueError("Checkpoint changed during capture; snapshot not accepted")
            staging.replace(target)
            records.append(
                {
                    "step": step,
                    "checkpoint_sha256": sha256(target),
                    "path": str(target),
                    "validation": validation,
                }
            )
            atomic_json(output / "index.json", {"requested_steps": steps, "records": records})
            pending.remove(step)
            print(f"Captured step {step}", flush=True)
        if pending:
            time.sleep(5)


def average(paths: list[Path], output: Path) -> None:
    if len(paths) < 2 or len(set(paths)) != len(paths):
        raise ValueError("Average at least two distinct checkpoints")
    if output.exists():
        raise FileExistsError("Refusing to replace an averaged checkpoint")
    accumulated, records, first = {}, [], None
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        identity = {
            key: payload[key]
            for key in ("config", "speaker_classes", "manifest_sha256", "dev_cases_sha256")
        }
        identity["train_speakers"] = payload["provenance"]["train_speakers"]
        identity["initialization"] = payload["provenance"].get("initialization")
        if first is None:
            first = identity
            template = payload
        elif identity != first or payload["model"].keys() != accumulated.keys():
            raise ValueError("Averaging requires the same model, configuration and training labels")
        for name, value in payload["model"].items():
            if not value.is_floating_point():
                raise ValueError("This experiment supports floating model states only")
            if (
                value.shape != template["model"][name].shape
                or value.dtype != template["model"][name].dtype
            ):
                raise ValueError("Averaging requires matching tensor shapes and dtypes")
            if not torch.isfinite(value).all():
                raise ValueError("Cannot average non-finite model weights")
            if name not in accumulated:
                accumulated[name] = value.double().clone()
            else:
                accumulated[name].add_(value.double())
        records.append(
            {"path": str(path), "checkpoint_sha256": sha256(path), "step": payload["step"]}
        )
    if len({record["step"] for record in records}) != len(records):
        raise ValueError("Averaging checkpoints must have distinct training steps")
    for key in ("optimizer", "scheduler", "torch_rng", "mps_rng"):
        template.pop(key, None)
    template["model"] = {
        key: (value / len(paths)).to(template["model"][key].dtype)
        for key, value in accumulated.items()
    }
    template["step"] = max(record["step"] for record in records)
    template["best_score"] = None
    template["provenance"] = {
        **template["provenance"],
        "checkpoint_average": {
            "method": "uniform arithmetic mean in float64, cast to original state dtype",
            "sources": records,
            "selection": "Not scored yet; evaluate on development before any test or promotion",
        },
    }
    save_checkpoint(output, template)
    atomic_json(
        output.with_suffix(".json"),
        {"checkpoint_sha256": sha256(output), **template["provenance"]["checkpoint_average"]},
    )
    print(f"Averaged {len(paths)} checkpoints into {output}")
