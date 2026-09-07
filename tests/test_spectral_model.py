import numpy as np
import torch

from tse.engine import save_checkpoint
from tse.inference import Extractor
from tse.model import make_model


def test_global_normalization_is_per_example_and_spans_time():
    from tse.model import GlobalNorm

    norm = GlobalNorm(3)
    x = torch.randn(2, 3, 19)
    actual = norm(x)
    expected = torch.nn.functional.layer_norm(x, x.shape[1:], eps=1e-5)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(norm(x[:1]), actual[:1])
    assert not torch.allclose(norm(x[..., :9]), actual[..., :9])


def spectral_config(config, **overrides):
    config.model = type(config.model).model_validate(
        {
            **config.model.model_dump(),
            "family": "reference_conditioned_stft_tcn",
            "encoder_channels": 257,
            "mask_activation": "sigmoid",
            **overrides,
        }
    )
    return config


def test_constant_spectral_mask_preserves_waveform_and_reference_receives_gradients(tiny_config):
    config = spectral_config(tiny_config)
    model = make_model(config.model)
    mixture, reference = torch.randn(2, 1, 4017) * 0.1, torch.randn(2, 1, 4800) * 0.1
    predicted = model(mixture, reference)
    (predicted - mixture * 0.7).square().mean().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.reference_encoder.parameters()
    )
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    with torch.no_grad():
        model.mask[1].weight.zero_()
        model.mask[1].bias.zero_()
        half = model(mixture, reference)
    torch.testing.assert_close(half, mixture * 0.5, atol=1e-6, rtol=1e-5)


def test_spectral_chunks_include_full_receptive_context(tiny_config, tmp_path):
    config = spectral_config(tiny_config, dilations=[1, 2, 4, 8, 16, 32, 64, 128])
    model = make_model(config.model)
    assert model.context_samples > 16000
    checkpoint = tmp_path / "spectral.pt"
    save_checkpoint(
        checkpoint,
        {
            "format_version": 1,
            "config": config.model_dump(),
            "model": model.state_dict(),
            "speaker_classes": 0,
            "step": 0,
            "provenance": {"experiment_type": "unit_test_only"},
        },
    )
    extractor = Extractor(checkpoint, "cpu")
    rng = np.random.default_rng(94)
    mixture = rng.normal(0, 0.1, 112017).astype(np.float32)
    reference = rng.normal(0, 0.1, 48000).astype(np.float32)
    whole = extractor.extract_array(mixture, reference, chunked=False)
    chunked = extractor.extract_array(mixture, reference)
    np.testing.assert_allclose(whole, chunked, atol=1e-6, rtol=1e-4)


def test_global_normalization_uses_whole_clip_and_retains_reference_padding(tiny_config, tmp_path):
    config = spectral_config(tiny_config, separation_normalization="global")
    model = make_model(config.model)
    reference = torch.randn(1, 1, 4800)
    padded = torch.nn.functional.pad(reference, (0, 800))
    torch.testing.assert_close(
        model.reference_encoder(reference),
        model.reference_encoder(padded, torch.tensor([4800])),
    )
    checkpoint = tmp_path / "global.pt"
    save_checkpoint(
        checkpoint,
        {
            "format_version": 1,
            "config": config.model_dump(),
            "model": model.state_dict(),
            "speaker_classes": 0,
            "step": 0,
            "provenance": {"experiment_type": "unit_test_only"},
        },
    )
    extractor = Extractor(checkpoint, "cpu")
    rng = np.random.default_rng(95)
    mixture = rng.normal(0, 0.1, 112017).astype(np.float32)
    ref = rng.normal(0, 0.1, 48000).astype(np.float32)
    whole = extractor.extract_array(mixture, ref, chunked=False)
    default = extractor.extract_array(mixture, ref)
    np.testing.assert_array_equal(whole, default)
    assert extractor.info()["inference_strategy"] == "whole_clip_global_normalization"
