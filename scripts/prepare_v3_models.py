#!/usr/bin/env python3
"""Create matched, untrained separators with explicit project-reference lineage."""

from pathlib import Path

import torch

from tse.config import ExperimentConfig
from tse.data import SpeechCorpus
from tse.engine import save_checkpoint
from tse.model import make_model
from tse.utils import atomic_json, sha256


def main():
    source = Path("artifacts/releases/v0.2.0/model.pt")
    reference = torch.load(source, map_location="cpu", weights_only=True)
    corpus = SpeechCorpus(
        Path("data/expanded/raw"), Path("data/v3/manifests/inventory.json"), "train", 4, 5
    )
    baseline = None
    record = {}
    for variant in ("band-real", "band-complex", "band-resnet"):
        config = ExperimentConfig.load(Path("configs/quality-stft-expanded-fastlr.json"))
        config.experiment = "v3-" + variant
        config.seed = 707
        config.data.development_source = "LibriSpeech/dev-clean+dev-other"
        config.data.test_source = "LibriSpeech/test-other"
        model = config.model.model_dump()
        model.update(
            family="reference_conditioned_bsrnn",
            weights="project_checkpoint",
            parameter_budget=8000000,
            spectral_mask="real" if variant == "band-real" else "complex",
            band_channels=48,
            band_hidden_channels=64,
            band_blocks=4,
        )
        if variant == "band-resnet":
            model["reference_encoder"].update(family="scaled_resnet34", resnet_base_channels=16)
        config.model = type(config.model).model_validate(model)
        config.training = type(config.training).model_validate(
            {
                **config.training.model_dump(),
                "learning_rate": 0.001,
                "learning_rate_schedule": "cosine",
                "schedule_decay_updates": 10000,
                "minimum_learning_rate": 0.00003,
                "max_optimizer_updates": 4000,
                "microbatch_size": 4,
                "gradient_accumulation": 2,
                "preserve_initialized_classifier": True,
            }
        )
        torch.manual_seed(config.seed)
        candidate = make_model(config.model, len(corpus.speakers))
        if variant == "band-real":
            candidate.reference_encoder.load_state_dict(
                {
                    name.removeprefix("reference_encoder."): value
                    for name, value in reference["model"].items()
                    if name.startswith("reference_encoder.")
                }
            )
            baseline = {name: value.clone() for name, value in candidate.state_dict().items()}
        else:
            current = candidate.state_dict()
            for name, value in baseline.items():
                if variant != "band-resnet" or not name.startswith("reference_encoder."):
                    current[name] = value.clone()
            candidate.load_state_dict(current)
        initialization = {
            "separator": "Random initial weights shared across all three arms",
            "speaker_head": "Identical random classifier with matching labels in all three arms",
            "reference": "Random scaled ResNet34"
            if variant == "band-resnet"
            else "Own v0.2.0 reference encoder",
            "reference_source_sha256": None if variant == "band-resnet" else sha256(source),
            "source_training_speakers": [] if variant == "band-resnet" else corpus.speakers,
        }
        path = Path(f"artifacts/v3-initializers/{variant}.pt")
        if path.exists():
            raise FileExistsError("Do not replace a recorded initialization")
        save_checkpoint(
            path,
            {
                "format_version": 1,
                "config": config.model_dump(),
                "model": candidate.state_dict(),
                "speaker_classes": len(corpus.speakers),
                "step": 0,
                "provenance": {
                    "train_speakers": corpus.speakers,
                    "initialization": initialization,
                    "untrained_separator": True,
                },
            },
        )
        atomic_json(Path(f"configs/v3-{variant}.json"), config.model_dump())
        record[variant] = {
            "checkpoint_sha256": sha256(path),
            "parameters": sum(p.numel() for p in candidate.parameters()),
            "initialization": initialization,
        }
    atomic_json(
        Path("reports/v3-architecture-initialization.json"),
        {
            "variants": record,
            "matching": "All separator and classifier tensors identical; real/complex arms also share reference tensors and initial predictions. ResNet arm changes only reference architecture/initialization.",
        },
    )
    print({name: row["parameters"] for name, row in record.items()})


if __name__ == "__main__":
    main()
