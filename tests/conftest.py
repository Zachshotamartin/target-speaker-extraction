from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from tse.config import ExperimentConfig
from tse.data import SPLITS

torch.set_num_threads(2)


@pytest.fixture
def tiny_config():
    config = ExperimentConfig()
    config.model.encoder_channels = 32
    config.model.bottleneck_channels = 16
    config.model.hidden_channels = 32
    config.model.skip_channels = 16
    config.model.dilations = [1, 2]
    config.model.repeats = 1
    config.model.reference_encoder.channels = 16
    config.model.reference_encoder.embedding_dim = 16
    config.model.reference_encoder.dilations = [1]
    config.audio.crop_seconds = 0.25
    config.audio.reference_seconds = 0.25
    config.evaluation.mixture_seconds = 0.25
    config.evaluation.reference_seconds = 0.25
    config.training.microbatch_size = 2
    config.training.gradient_accumulation = 1
    config.training.validation_interval_updates = 1
    config.training.max_optimizer_updates = 2
    config.resources.minimum_free_disk_gib = 1
    config.runtime.preferred_device = "cpu"
    return config


@pytest.fixture
def corpus_files(tmp_path: Path):
    root = tmp_path / "raw"
    for split_index, (_, source) in enumerate(SPLITS.items()):
        for speaker_index in range(2):
            speaker = str(10 + split_index * 2 + speaker_index)
            for utterance in range(3):
                path = (
                    root
                    / "LibriSpeech"
                    / source
                    / speaker
                    / "1"
                    / f"{speaker}-1-{utterance:04d}.flac"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                t = np.arange(16000) / 16000
                frequency = 120 + (split_index * 2 + speaker_index) * 70 + utterance * 2
                audio = 0.2 * np.sin(2 * np.pi * frequency * t) + 0.06 * np.sin(
                    2 * np.pi * frequency * 2 * t
                )
                sf.write(path, audio, 16000)
    manifest = tmp_path / "inventory.json"
    # The public-data inventory requires 2 seconds; test contracts use directly
    # generated records so this fixture can stay under one second per utterance.
    from tse.utils import atomic_json, sha256

    records = []
    for path in sorted(root.rglob("*.flac")):
        parts = path.relative_to(root).parts
        records.append(
            {
                "id": path.stem,
                "speaker": parts[2],
                "chapter": parts[3],
                "split": next(k for k, v in SPLITS.items() if v == parts[1]),
                "path": str(path.relative_to(root)),
                "samples": 16000,
                "sample_rate": 16000,
                "sha256": sha256(path),
            }
        )
    atomic_json(manifest, {"records": records})
    return root, manifest
