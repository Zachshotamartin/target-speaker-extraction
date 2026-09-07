#!/usr/bin/env python3
"""Fork a saved training state into an explicit learning-rate comparison."""

import argparse
from pathlib import Path

import torch

from tse.config import ExperimentConfig
from tse.engine import save_checkpoint
from tse.utils import atomic_json, sha256


def fork_training(source: Path, configuration: Path, destination: Path) -> dict:
    if destination.exists():
        raise FileExistsError("A training fork must have a new destination")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not all(key in payload for key in ("optimizer", "torch_rng", "scheduler")):
        raise ValueError("Fork a full training checkpoint, not exported or averaged weights")
    config = ExperimentConfig.load(configuration)
    previous = ExperimentConfig.model_validate(payload["config"])
    before, after = previous.model_dump(), config.model_dump()
    for value in (before, after):
        value.pop("experiment")
        for name in (
            "max_optimizer_updates",
            "learning_rate",
            "learning_rate_schedule",
            "minimum_learning_rate",
            "schedule_decay_updates",
        ):
            value["training"].pop(name)
    if before != after or config.training.max_optimizer_updates <= payload["step"]:
        raise ValueError("A schedule fork preserves model, data, losses and sample schedule")
    for group in payload["optimizer"]["param_groups"]:
        group["lr"] = config.training.learning_rate
        group.pop("initial_lr", None)
    # Preserve the original plateau history in its control arm. A different
    # schedule begins at local update zero with identical optimizer moments.
    if config.training.learning_rate_schedule != previous.training.learning_rate_schedule:
        payload["scheduler"] = None
    payload["config"] = config.model_dump()
    payload["validate_fork_start"] = True
    payload["provenance"]["continuation"] = {
        "checkpoint_sha256": sha256(source),
        "source_step": payload["step"],
        "optimizer_and_rng_preserved": True,
        "classifier_preserved": True,
        "learning_rate_schedule": config.training.learning_rate_schedule,
    }
    destination.mkdir(parents=True)
    save_checkpoint(destination / "latest.pt", payload)
    atomic_json(destination / "fork.json", payload["provenance"]["continuation"])
    return payload["provenance"]["continuation"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    print(fork_training(args.source, args.config, args.run))
