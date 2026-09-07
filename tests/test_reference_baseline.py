import copy
import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from torch.nn import functional as F

from tse.config import ExperimentConfig
from tse.engine import save_checkpoint
from tse.inference import Extractor
from tse.librimix import LibriMixCorpus, epoch_order, pcm16_roundtrip, render_clean
from tse.model import make_model
from tse.reference_model import EnrollmentFbank, ReferenceBSRNN
from tse.reference_training import backward_batch, schedule_lr, separation_score, train_reference
from tse.utils import atomic_json, sha256


def reference_config():
    config = ExperimentConfig.load(Path("configs/reference-libri2mix.json"))
    config.model.band_channels = 8
    config.model.band_hidden_channels = 8
    config.model.band_blocks = 1
    config.model.band_widths = [64, 64, 64, 65]
    return config


def test_pcm16_generation_matches_audio_writer_and_mixes_before_quantization():
    values = np.array([-1.5, -1, -0.100001, 0, 0.00002, 0.99999, 1.5], dtype=np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, values, 16000, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    actual, _ = sf.read(buffer, dtype="float32")
    np.testing.assert_array_equal(actual, pcm16_roundtrip(values))
    sources, mixture = render_clean(
        np.array([0.00002, 0.2], np.float32), np.array([0.00002, 0.1, 0.4], np.float32), [1.0, 1.0]
    )
    assert len(mixture) == 2
    assert mixture[0] != sources[0][0] + sources[1][0]


def test_epoch_coverage_and_learning_rate_endpoints():
    first = epoch_order(27800, 42, 0)
    assert sorted(first) == list(range(27800))
    assert first == epoch_order(27800, 42, 0)
    assert first != epoch_order(27800, 42, 1)
    config = reference_config()
    assert schedule_lr(config, 0, 100) == config.training.learning_rate
    assert abs(schedule_lr(config, 100, 100) - config.training.minimum_learning_rate) < 1e-12


def test_frontend_full_utterance_cmn_and_deterministic_evaluation():
    frontend = EnrollmentFbank().eval()
    waveform = torch.randn(2, 1, 8000)
    features = frontend(waveform)
    assert features.shape == (2, 80, 48)
    torch.testing.assert_close(features.mean(-1), torch.zeros(2, 80), atol=1e-5, rtol=0)
    torch.testing.assert_close(features, frontend(waveform), rtol=0, atol=0)
    torch.testing.assert_close(features, frontend(waveform * 2), atol=2e-5, rtol=0)


def test_complex_coefficients_can_amplify_and_reverse_polarity():
    logits = torch.tensor([[[4.0], [-4.0], [10.0], [10.0]]])
    mask = ReferenceBSRNN.complex_mask(logits, 1)
    assert mask.real.item() > 3.9 and mask.imag.item() < -3.9


def test_reference_delivery_preserves_benchmark_preprocessing_and_whole_sequence(tmp_path):
    config = reference_config()
    torch.manual_seed(41)
    model = make_model(config.model, 2).eval()
    checkpoint = tmp_path / "reference.pt"
    save_checkpoint(
        checkpoint,
        {
            "format_version": 1,
            "config": config.model_dump(),
            "model": model.state_dict(),
            "speaker_classes": 2,
            "step": 0,
        },
    )
    rng = np.random.default_rng(41)
    mixture = (rng.normal(0, 0.07, 66000) + 0.025).astype(np.float32)
    reference = (rng.normal(0, 0.12, 53000) + 0.015).astype(np.float32)
    with torch.inference_mode():
        expected = model(
            torch.from_numpy(mixture)[None, None], torch.from_numpy(reference)[None, None]
        )[0, 0].numpy()
    extractor = Extractor(checkpoint)
    actual = extractor.extract_array(mixture, reference, chunked=True)
    np.testing.assert_array_equal(actual, expected)
    assert extractor.info()["inference_strategy"] == "whole_clip_sequence_model"


@pytest.mark.parametrize("separator_microbatch", [1, 2])
def test_recomputed_joint_training_matches_gradients_and_batchnorm_buffers(separator_microbatch):
    torch.manual_seed(19)
    config = reference_config()
    config.training.microbatch_size = separator_microbatch
    ordinary = make_model(config.model, 2).double().train()
    recomputed = copy.deepcopy(ordinary)
    recomputed.activation_checkpointing = True
    recomputed.reference_encoder.activation_checkpointing = True
    features = torch.randn(2, 80, 32, dtype=torch.float64)
    mixture = torch.randn(2, 1, 1024, dtype=torch.float64) * 0.1
    embedding = ordinary.reference_encoder.encode_features(features)
    output = ordinary.extract(mixture, embedding)
    # Float64 isolates mathematical equivalence from batch-kernel float32 rounding.
    target = output.detach() + torch.randn_like(output) * output.detach().std() * 0.1
    loss = -0.9 * separation_score(output, target).mean() + 0.1 * F.cross_entropy(
        ordinary.speaker_head(embedding), torch.tensor([0, 1])
    )
    loss.backward()
    requests = [
        {"mixture": mixture[i : i + 1], "target": target[i : i + 1], "label": i} for i in range(2)
    ]
    metrics = backward_batch(recomputed, requests, features, config)
    assert abs(metrics["loss"] - float(loss.detach())) < 1e-4
    for (name, expected), (_, actual) in zip(
        ordinary.named_parameters(), recomputed.named_parameters(), strict=True
    ):
        assert actual.grad is not None, name
        torch.testing.assert_close(expected.grad, actual.grad, atol=1e-10, rtol=1e-10, msg=name)
    for (name, expected), (_, actual) in zip(
        ordinary.named_buffers(), recomputed.named_buffers(), strict=True
    ):
        torch.testing.assert_close(expected, actual, atol=0, rtol=0, msg=name)
    ordinary.eval()
    reference = torch.randn(1, 1, 6400, dtype=torch.float64)
    with torch.no_grad():
        result = ordinary(mixture[:1], reference)
        swapped = ordinary(mixture[:1], -reference.flip(-1))
    assert result.shape == mixture[:1].shape and torch.isfinite(result).all()
    assert not torch.allclose(result, swapped)


def test_reference_training_resume_replays_exact_samples_and_state(tmp_path):
    splits, enrollments = {}, {}
    for split in ("train", "dev"):
        rows, mapping = [], {}
        for pair in range(2):
            signals = []
            files = []
            for side in range(2):
                relative = f"{split}/{pair}-s{side}.wav"
                signal = (
                    np.random.default_rng(pair + side * 30).normal(0, 0.1, 4800).astype(np.float32)
                )
                path = tmp_path / relative
                path.parent.mkdir(exist_ok=True)
                sf.write(path, signal, 16000)
                files.append(
                    {
                        "path": relative,
                        "sha256": sha256(path),
                        "speaker": f"{split}-{side}",
                        "utterance": f"{split}-{side}-{pair}",
                    }
                )
                signals.append(signal)
                mapping[f"{pair}:{side}"] = f"{split}/{1 - pair}-s{side}.wav"
            path = tmp_path / f"{split}/{pair}-mix.wav"
            sf.write(path, signals[0] + signals[1], 16000)
            rows.append(
                {
                    "id": str(pair),
                    "sources": files,
                    "samples": 4800,
                    "mixture": {"path": f"{split}/{pair}-mix.wav", "sha256": sha256(path)},
                }
            )
        splits[split], enrollments[split] = rows, mapping
    manifest = tmp_path / "manifest.json"
    atomic_json(
        manifest,
        {"protocol": "libri2mix-16k-min-clean", "splits": splits, "enrollments": enrollments},
    )
    config = reference_config()
    config.audio.crop_seconds = 0.25
    config.training.epochs = 2
    config.training.microbatch_size = 1
    config.training.gradient_accumulation = 2
    config.training.max_optimizer_updates = 4
    config.training.activation_checkpointing = False
    config_path = tmp_path / "config.json"
    atomic_json(config_path, config.model_dump())
    complete = train_reference(config_path, tmp_path, manifest, tmp_path / "complete", "cpu")
    train_reference(
        config_path, tmp_path, manifest, tmp_path / "resumed", "cpu", stop_after_updates=1
    )
    resumed = train_reference(
        config_path, tmp_path, manifest, tmp_path / "resumed", "cpu", resume=True
    )
    for name, expected in complete.state_dict().items():
        torch.testing.assert_close(expected, resumed.state_dict()[name], atol=0, rtol=0, msg=name)
    corpus = LibriMixCorpus(tmp_path, manifest, "dev")
    request = corpus.request(0)
    assert request["reference_path"] == "dev/1-s0.wav"
    # Changing an enrollment file must be detected, not silently accepted on resume/evaluation.
    (tmp_path / "dev/1-s0.wav").write_bytes(b"changed")
    with pytest.raises(ValueError, match="audio changed"):
        LibriMixCorpus(tmp_path, manifest, "dev").request(0)
