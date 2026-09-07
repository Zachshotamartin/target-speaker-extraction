import json

import numpy as np
import pytest
import soundfile as sf
import torch

from tse.realistic import SCENARIOS, AcousticResources, RealisticCorpus, extraction_loss
from tse.utils import atomic_json, sha256


@pytest.fixture
def acoustic_files(tmp_path):
    root = tmp_path / "acoustics"
    root.mkdir()
    rows = []
    for j, split in enumerate(("train", "dev", "test")):
        rng = np.random.default_rng(300 + j)
        for kind in ("noise", "rir"):
            audio = rng.normal(0, 0.05, 12000).astype(np.float32)
            if kind == "rir":
                audio *= np.exp(-np.arange(len(audio)) / 1000)
                audio[30] = 1
            path = root / f"{split}-{kind}.wav"
            sf.write(path, audio, 16000, subtype="FLOAT")
            rows.append(
                {
                    "id": path.name,
                    "path": path.name,
                    "kind": kind,
                    "group": path.stem,
                    "split": split,
                    "sha256": sha256(path),
                }
            )
    manifest = root / "manifest.json"
    atomic_json(manifest, {"records": rows})
    return root, manifest


def test_realistic_sources_sum_and_target_switches_preserve_mixture(corpus_files, acoustic_files):
    root, manifest = corpus_files
    env_root, env_manifest = acoustic_files
    corpus = RealisticCorpus(root, manifest, "train", 0.5, 0.5, env_root, env_manifest)
    base = corpus.make_case(194)
    for scenario in SCENARIOS:
        case = {**base, "scenario": scenario, "environment_seed": 9184}
        a = corpus.render(case)
        repeated = corpus.render(case)
        np.testing.assert_array_equal(a["mixture"], repeated["mixture"])
        np.testing.assert_allclose(
            a["mixture"], a["target"] + a["interferer"] + a.get("noise", 0), atol=1e-7
        )
        assert np.isfinite(a["reference"]).all() and np.max(np.abs(a["mixture"])) <= 0.900001
        assert len(a["target"]) == 8000
        if scenario == "target_absent":
            assert not a["target_present"] and np.max(np.abs(a["target"])) == 0
        else:
            b = corpus.render({**case, "target_index": 1})
            np.testing.assert_array_equal(a["mixture"], b["mixture"])
            np.testing.assert_array_equal(a["target"], b["interferer"])


def test_acoustic_split_leakage_is_rejected(acoustic_files):
    root, manifest = acoustic_files
    payload = json.loads(manifest.read_text())
    payload["records"][2]["sha256"] = payload["records"][0]["sha256"]
    atomic_json(manifest, payload)
    with pytest.raises(ValueError, match="Identical acoustic"):
        AcousticResources(root, manifest, "train")


def test_absent_loss_is_finite_and_penalizes_retained_speech(tiny_config):
    mixture = torch.randn(2, 1, 4000) * 0.1
    target = torch.zeros_like(mixture)
    output = mixture.clone().requires_grad_()
    loss = extraction_loss(output, target, mixture, tiny_config.loss)
    assert loss > extraction_loss(output * 0.1, target, mixture, tiny_config.loss)
    loss.backward()
    assert torch.isfinite(output.grad).all() and output.grad.abs().sum() > 0
    target[0] = mixture[0]
    assert torch.isfinite(extraction_loss(output, target, mixture, tiny_config.loss))


def test_acoustic_training_resume_preserves_model_and_resource_contract(
    corpus_files, acoustic_files, tiny_config, tmp_path
):
    from tse.data import SpeechCorpus, build_cases
    from tse.engine import load_model, train

    root, manifest = corpus_files
    env_root, env_manifest = acoustic_files
    tiny_config.augmentation.environment_root = str(env_root)
    tiny_config.augmentation.environment_manifest = str(env_manifest)
    tiny_config.augmentation.realistic_enabled = True
    tiny_config.augmentation.mixture_noise_enabled = True
    tiny_config.evaluation.realistic_validation = True
    tiny_config.training.learning_rate_schedule = "cosine"
    development = SpeechCorpus(root, manifest, "dev", 0.25, 0.25)
    cases = tmp_path / "dev.json"
    build_cases(development, 2, 800, cases)
    payload = json.loads(cases.read_text())
    for row in payload["cases"]:
        row.update(scenario="combined", environment_seed=9184)
    atomic_json(cases, payload)
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    train(tiny_config, root, manifest, cases, full, "cpu", prefetch=True)
    first = tiny_config.model_copy(deep=True)
    first.training.max_optimizer_updates = 1
    train(first, root, manifest, cases, resumed, "cpu", prefetch=True)
    train(tiny_config, root, manifest, cases, resumed, "cpu", resume=True, prefetch=True)
    left, _ = load_model(full / "latest.pt")
    right, checkpoint = load_model(resumed / "latest.pt")
    assert checkpoint["provenance"]["environment_manifest_sha256"] == sha256(env_manifest)
    for a, b in zip(left.parameters(), right.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    resources = json.loads(env_manifest.read_text())
    resources["revision"] = 2
    atomic_json(env_manifest, resources)
    with pytest.raises(ValueError, match="Resume acoustic resources"):
        train(tiny_config, root, manifest, cases, resumed, "cpu", resume=True)
