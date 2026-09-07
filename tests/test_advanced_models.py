import copy

import pytest
import torch

from tse.model import make_model, make_reference_encoder


def band_config(config):
    data = config.model.model_dump()
    data.update(
        family="reference_conditioned_bsrnn",
        encoder_channels=257,
        mask_activation="sigmoid",
        band_channels=8,
        band_hidden_channels=8,
        band_blocks=1,
        parameter_budget=8000000,
    )
    config.model = type(config.model).model_validate(data)
    return config


def test_phase_arm_starts_identically_and_can_learn_imaginary_corrections(tiny_config):
    config = band_config(tiny_config)
    real = make_model(config.model)
    complex_config = copy.deepcopy(config.model)
    complex_config.spectral_mask = "complex"
    complex_model = make_model(complex_config)
    complex_model.load_state_dict(real.state_dict())
    mixture = torch.randn(2, 1, 4017) * 0.1
    reference = torch.randn(2, 1, 4800) * 0.1
    target = torch.roll(mixture, 3, -1)
    real_output = real(mixture, reference)
    complex_output = complex_model(mixture, reference)
    torch.testing.assert_close(real_output, complex_output, atol=0, rtol=0)
    for model, output in [(real, real_output), (complex_model, complex_output)]:
        (output - target).square().mean().backward()
        assert output.shape == mixture.shape
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.reference_encoder.parameters()
        )
    assert all(
        head[-1].weight.grad[width:].abs().sum() == 0
        for head, width in zip(real.mask_heads, real.config.band_widths, strict=True)
    )
    assert any(
        head[-1].weight.grad[width:].abs().sum() > 0
        for head, width in zip(complex_model.mask_heads, real.config.band_widths, strict=True)
    )
    with torch.no_grad():
        for head in real.mask_heads:
            head[-1].weight.zero_()
            head[-1].bias.zero_()
        torch.testing.assert_close(real(mixture, reference), mixture * 0.5, atol=1e-6, rtol=1e-5)
    assert real.requires_whole_clip


def test_residual_reference_ignores_padding_and_backpropagates(tiny_config):
    tiny_config.model.reference_encoder.family = "scaled_resnet34"
    tiny_config.model.reference_encoder.resnet_base_channels = 4
    encoder = make_reference_encoder(tiny_config.model)
    reference = torch.randn(1, 1, 4000) * 0.1
    padded = torch.nn.functional.pad(reference, (0, 960))
    expected = encoder(reference)
    torch.testing.assert_close(expected, encoder(padded, torch.tensor([4000])))
    expected[:, 0].sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.blocks.parameters())


def test_band_partition_rejects_missing_bins(tiny_config):
    config = band_config(tiny_config)
    with pytest.raises(ValueError, match="partition"):
        type(config.model).model_validate({**config.model.model_dump(), "band_widths": [128, 128]})


def test_cosine_schedule_stops_at_floor_after_declared_horizon(tiny_config):
    from tse.engine import make_scheduler

    tiny_config.training.learning_rate_schedule = "cosine"
    tiny_config.training.schedule_decay_updates = 4
    tiny_config.training.minimum_learning_rate = 3e-5
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=tiny_config.training.learning_rate)
    scheduler = make_scheduler(optimizer, tiny_config)
    rates = []
    for _ in range(9):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])
    assert rates == sorted(rates, reverse=True)
    assert rates[3:] == pytest.approx([3e-5] * 6)
