import copy
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from tse.concept_data import ConceptCorpus, validate_reservation
from tse.concept_training import (
    concept_schedule_lr,
    fork_concept,
    initialize_concept,
    remaining_concept_seconds,
    train_concept,
)
from tse.config import ExperimentConfig
from tse.engine import save_checkpoint
from tse.model import make_model
from tse.utils import atomic_json, sha256


@pytest.fixture
def concept_fixture(tmp_path):
    labels = [str(i) for i in range(8)]
    pools = {split: {speaker: [] for speaker in labels} for split in ("train", "dev", "test")}
    flat = []
    for part, split in enumerate(pools):
        for speaker in labels:
            for index in range(2):
                identity = f"{speaker}-{part}-{index}"
                path = tmp_path / f"{identity}.wav"
                signal = (
                    np.random.default_rng(part * 100 + int(speaker) * 2 + index)
                    .normal(0, 0.1, 65000)
                    .astype(np.float32)
                )
                sf.write(path, signal, 16000, subtype="FLOAT")
                record = {
                    "path": path.name,
                    "sha256": sha256(path),
                    "speaker": speaker,
                    "utterance": identity,
                }
                pools[split][speaker].append(record)
                flat.append(record)
    rows = [
        {
            "id": str(i),
            "samples": 65000,
            "sources": flat[i : i + 2],
            "mixture": {
                "path": flat[i]["path"],
                "sha256": flat[i]["sha256"],
                "speaker": flat[i]["speaker"],
                "utterance": flat[i]["utterance"],
            },
        }
        for i in range(0, len(flat), 2)
    ]
    source = tmp_path / "source.json"
    atomic_json(
        source,
        {"protocol": "libri2mix-16k-min-clean", "splits": {"train": rows}, "enrollments": {}},
    )
    config = ExperimentConfig.load(Path("configs/concept-demo.json"))
    config.model.band_channels = 8
    config.model.band_hidden_channels = 8
    config.model.band_widths = [64, 64, 64, 65]
    config.model.band_blocks = 1
    config.training.max_optimizer_updates = 2
    config.training.validation_interval_updates = 2
    parent = config.model_copy(deep=True)
    parent.model.band_blocks = 2
    torch.manual_seed(31)
    model = make_model(parent.model, 8)
    checkpoint = tmp_path / "parent.pt"
    save_checkpoint(
        checkpoint,
        {
            "format_version": 1,
            "config": parent.model_dump(),
            "model": model.state_dict(),
            "step": 9,
            "provenance": {"train_speakers": labels},
        },
    )
    manifest = tmp_path / "concept.json"
    atomic_json(
        manifest,
        {
            "protocol": "known-voice-concept-v1",
            "source_manifest_sha256": sha256(source),
            "initialization_sha256": sha256(checkpoint),
            "initialization_excluded_utterances": [],
            "classifier_speakers": labels,
            "speakers": labels,
            "splits": pools,
            "relative_level_db": [-3, 3],
            "development_pairs": 4,
            "test_pairs": 4,
            "scope": "Test fixture",
        },
    )
    configuration = tmp_path / "config.json"
    atomic_json(configuration, config.model_dump())
    return tmp_path, source, manifest, checkpoint, configuration


def test_concept_reserves_utterances_and_switches_the_target(concept_fixture):
    root, source, manifest, _, _ = concept_fixture
    corpus = ConceptCorpus(root, source, manifest)
    batch = corpus.training_batch(42, 0)
    assert len(batch) == 8 and {r["speaker"] for r in batch} == set(corpus.speakers)
    for left, right in zip(batch[::2], batch[1::2], strict=True):
        torch.testing.assert_close(left["mixture"], right["mixture"], atol=0, rtol=0)
        torch.testing.assert_close(left["target"], right["interferer"], atol=0, rtol=0)
        torch.testing.assert_close(
            left["mixture"], left["target"] + right["target"], atol=1e-7, rtol=1e-6
        )
        assert all(row["source"] != row["reference"] for row in left["source_records"])
    invalid = copy.deepcopy(corpus.recipe)
    invalid["splits"]["dev"]["0"][0] = invalid["splits"]["train"]["0"][0]
    with pytest.raises(ValueError, match="leakage"):
        validate_reservation(invalid, corpus.source)


def test_concept_transfer_keeps_existing_weights_and_rejects_width_change(concept_fixture):
    _, _, manifest, checkpoint, configuration = concept_fixture
    recipe = json.loads(manifest.read_text())
    config = ExperimentConfig.load(configuration)
    model, origin = initialize_concept(
        config, checkpoint, sha256(checkpoint), recipe["classifier_speakers"], "cpu"
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, saved["model"][key], atol=0, rtol=0)
    assert origin["retained_blocks"] == 1 and origin["original_blocks"] == 2
    config.model.band_hidden_channels = 16
    with pytest.raises(ValueError, match="only removes"):
        initialize_concept(
            config, checkpoint, sha256(checkpoint), recipe["classifier_speakers"], "cpu"
        )


def test_concept_resume_and_session_expiry_preserve_complete_updates(concept_fixture):
    root, source, manifest, checkpoint, configuration = concept_fixture
    whole = train_concept(
        configuration,
        root,
        source,
        manifest,
        checkpoint,
        root / "whole",
        minutes=2,
        device_name="cpu",
    )
    train_concept(
        configuration,
        root,
        source,
        manifest,
        checkpoint,
        root / "resumed",
        minutes=2,
        device_name="cpu",
        stop_after_updates=1,
    )
    resumed = train_concept(
        configuration,
        root,
        source,
        manifest,
        checkpoint,
        root / "resumed",
        minutes=2,
        device_name="cpu",
        resume=True,
    )
    for key, expected in whole.state_dict().items():
        torch.testing.assert_close(expected, resumed.state_dict()[key], atol=0, rtol=0)
    train_concept(
        configuration,
        root,
        source,
        manifest,
        checkpoint,
        root / "expired",
        minutes=1e-8,
        device_name="cpu",
    )
    status = json.loads((root / "expired/status.json").read_text())
    assert status["status"] == "paused" and status["step"] == 0
    assert (root / "expired/latest.pt").is_file()


def test_continuation_preserves_learning_and_resumes_new_schedule(concept_fixture):
    root, source, manifest, checkpoint, configuration = concept_fixture
    parent = root / "completed"
    train_concept(
        configuration, root, source, manifest, checkpoint, parent, minutes=2, device_name="cpu"
    )
    original_hash = sha256(parent / "best.pt")
    original = torch.load(parent / "best.pt", map_location="cpu", weights_only=True)
    config = ExperimentConfig.load(configuration)
    config.training.max_optimizer_updates = original["step"] + 2
    config.training.learning_rate = 0.0001
    config.training.validation_interval_updates = 2
    changed = root / "extension-config.json"
    atomic_json(changed, config.model_dump())
    for name in ("extension-whole", "extension-resumed"):
        record = fork_concept(parent, changed, manifest, root / name, minutes=2)
        assert record["source_step"] == original["step"]
    forked = torch.load(root / "extension-whole/latest.pt", map_location="cpu", weights_only=True)
    for key, value in original["model"].items():
        torch.testing.assert_close(value, forked["model"][key], atol=0, rtol=0)
    for key, state in original["optimizer"]["state"].items():
        for field, value in state.items():
            torch.testing.assert_close(
                value, forked["optimizer"]["state"][key][field], atol=0, rtol=0
            )
    torch.testing.assert_close(original["torch_rng"], forked["torch_rng"], atol=0, rtol=0)
    assert concept_schedule_lr(config, original["step"], original["step"]) == 0.0001
    assert (
        concept_schedule_lr(config, config.training.max_optimizer_updates, original["step"])
        == config.training.minimum_learning_rate
    )
    whole = train_concept(
        changed,
        root,
        source,
        manifest,
        checkpoint,
        root / "extension-whole",
        minutes=2,
        device_name="cpu",
        resume=True,
    )
    train_concept(
        changed,
        root,
        source,
        manifest,
        checkpoint,
        root / "extension-resumed",
        minutes=2,
        device_name="cpu",
        resume=True,
        stop_after_updates=original["step"] + 1,
    )
    resumed = train_concept(
        changed,
        root,
        source,
        manifest,
        checkpoint,
        root / "extension-resumed",
        minutes=2,
        device_name="cpu",
        resume=True,
    )
    for key, expected in whole.state_dict().items():
        torch.testing.assert_close(expected, resumed.state_dict()[key], atol=0, rtol=0)
    assert sha256(parent / "best.pt") == original_hash
    # Continuing a run cannot reset its total authorized compute allowance.
    provenance = forked["provenance"]
    baseline = record["source_elapsed_seconds"]
    assert remaining_concept_seconds(provenance, baseline + 30, 480) == 90
    assert remaining_concept_seconds(provenance, baseline + 120, 480) == 0
    fork_concept(parent, changed, manifest, root / "expired-extension", minutes=1e-8)
    train_concept(
        changed,
        root,
        source,
        manifest,
        checkpoint,
        root / "expired-extension",
        minutes=480,
        device_name="cpu",
        resume=True,
    )
    expired = json.loads((root / "expired-extension/status.json").read_text())
    assert expired["status"] == "time_budget_complete"
    assert expired["step"] == original["step"]
    assert (root / "expired-extension/summary.json").is_file()
    config.audio.reference_seconds = 4
    atomic_json(changed, config.model_dump())
    with pytest.raises(ValueError, match="preserve"):
        fork_concept(parent, changed, manifest, root / "invalid")
