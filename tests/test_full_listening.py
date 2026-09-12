"""Validation identity and unmodified audio are essential to honest comparisons."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from tse.utils import atomic_json, sha256

spec = importlib.util.spec_from_file_location("full_listening", "scripts/watch_full_listening.py")
listening = importlib.util.module_from_spec(spec)
spec.loader.exec_module(listening)


def test_latest_is_validation_not_arbitrary_training_step(tmp_path):
    atomic_json(tmp_path / "best-full-validation.json", {"step": 100})
    atomic_json(tmp_path / "latest-full-validation.json", {"step": 200})
    atomic_json(tmp_path / "status.json", {"status": "training", "step": 250})
    assert listening.selections(tmp_path)["latest"]["step"] == 200
    atomic_json(
        tmp_path / "status.json",
        {"status": "validating", "validation_kind": "monitor", "step": 250},
    )
    assert listening.selections(tmp_path)["latest"]["step"] == 200
    atomic_json(
        tmp_path / "status.json", {"status": "validating", "validation_kind": "full", "step": 300}
    )
    choice = listening.selections(tmp_path)["latest"]
    assert choice["step"] == 300 and choice["pending"] and choice["result"] == {}
    atomic_json(tmp_path / "latest-full-validation.json", {"step": 300})
    assert not listening.selections(tmp_path)["latest"]["pending"]


def test_same_best_and_latest_uses_retained_best_checkpoint(tmp_path):
    for role in ("best", "latest"):
        atomic_json(tmp_path / f"{role}-full-validation.json", {"step": 100})
    selected = listening.selections(tmp_path)
    assert selected["latest"]["checkpoint"] == selected["best"]["checkpoint"] == "best-full.pt"


def test_snapshot_rejects_wrong_step_data_or_hash(tmp_path):
    path = tmp_path / "latest.pt"
    torch.save({"step": 200, "manifest_sha256": "data", "optimizer": {}}, path)
    choice = {"step": 100, "result": {}}
    with pytest.raises(ValueError, match="advanced"):
        listening.load_snapshot(path, choice, "data")
    choice["step"] = 200
    with pytest.raises(ValueError, match="dataset differ"):
        listening.load_snapshot(path, choice, "other-data")
    choice["result"] = {"checkpoint_sha256": "wrong"}
    with pytest.raises(ValueError, match="changed during selection"):
        listening.load_snapshot(path, choice, "data")
    choice["result"] = {"checkpoint_sha256": sha256(path)}
    payload, digest = listening.load_snapshot(path, choice, "data")
    assert digest == sha256(path) and "optimizer" not in payload


def test_volume_matching_retains_raw_and_only_changes_gain():
    signal = np.array([-3, -0.01, 0, 0.2, 1], dtype=np.float32)
    versions, gain = listening.playback_versions(signal)
    np.testing.assert_array_equal(versions["raw"], signal)
    np.testing.assert_allclose(versions["matched"], signal * gain)
    assert np.max(np.abs(versions["matched"])) <= 0.98
    versions, _ = listening.playback_versions(np.zeros(32, dtype=np.float32))
    assert not versions["matched"].any()
    with pytest.raises(ValueError, match="Non-finite"):
        listening.playback_versions(np.array([np.nan]))


def test_render_caches_exact_case_audio_without_changing_checkpoint(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    config = listening.ExperimentConfig.load(Path("configs/full-data-efficient.json"))
    payload = {
        "config": config.model_dump(),
        "step": 3475,
        "speaker_classes": 1,
        "manifest_sha256": sha256(manifest),
        "model": {},
        "provenance": {"train_speakers": ["train"], "source_tree_sha256": "source"},
    }
    checkpoint = tmp_path / "best-full.pt"
    torch.save(payload, checkpoint)
    original_hash = sha256(checkpoint)

    class Corpus:
        speakers = ["dev"]

        def __init__(self, *args):
            pass

    class Model(torch.nn.Module):
        def forward(self, mixture, reference):
            return mixture * 0.25

    generator = torch.Generator().manual_seed(19)
    target = torch.randn(1, 1, 1600, generator=generator) * 0.1
    other = torch.randn(1, 1, 1600, generator=generator) * 0.1

    def request(corpus, index, seed, reference_samples):
        assert seed == listening.request_seed(19, 0, index)
        assert reference_samples == 48000
        return {
            "mixture": target + other,
            "target": target,
            "interferer": other,
            "reference": target,
            "case_id": f"case:{index}",
            "speaker": "dev",
        }

    monkeypatch.setattr(listening, "LibriMixCorpus", Corpus)
    monkeypatch.setattr(listening, "make_model", lambda *args: Model())
    monkeypatch.setattr(listening, "cropped_request", request)
    pointer = {"root": str(tmp_path), "manifest": str(manifest)}
    choice = {"step": 3475, "checkpoint": "best-full.pt", "result": {}}
    output = tmp_path / "output"
    record = listening.render_snapshot(tmp_path, pointer, choice, output)
    assert record["indices"] == [0, 1, 2, 3]
    assert record["checkpoint_sha256"] == original_hash == sha256(checkpoint)
    audio_path = (
        output
        / Path(record["items"][0]["tracks"]["estimate"]["raw"]).parent.name
        / "00-estimate-raw.wav"
    )
    audio, rate = sf.read(audio_path, dtype="float32")
    assert rate == 16000
    np.testing.assert_array_equal(audio, ((target + other) * 0.25).numpy().reshape(-1))
    checkpoint.unlink()  # Cached validation audio survives later checkpoint replacement.
    assert listening.render_snapshot(tmp_path, pointer, choice, output) == record
