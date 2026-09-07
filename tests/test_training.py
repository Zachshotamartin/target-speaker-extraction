import pytest
import torch

from tse.data import SpeechCorpus, build_cases
from tse.engine import load_model, train


@pytest.mark.parametrize("schedule", ["none", "plateau", "cosine"])
@pytest.mark.parametrize("prefetch", [False, True])
def test_resume_matches_uninterrupted_training(
    corpus_files, tiny_config, tmp_path, schedule, prefetch
):
    tiny_config.training.learning_rate_schedule = schedule
    root, manifest = corpus_files
    development = SpeechCorpus(root, manifest, "dev", 0.25, 0.25)
    dev_cases = tmp_path / "dev.json"
    build_cases(development, 2, 800, dev_cases)
    full = tmp_path / "full"
    resumed = tmp_path / "resumed"
    train(tiny_config, root, manifest, dev_cases, full, "cpu")
    first = tiny_config.model_copy(deep=True)
    first.training.max_optimizer_updates = 1
    train(first, root, manifest, dev_cases, resumed, "cpu")
    train(tiny_config, root, manifest, dev_cases, resumed, "cpu", resume=True, prefetch=prefetch)
    a, _ = load_model(full / "latest.pt")
    b, _ = load_model(resumed / "latest.pt")
    for left, right in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize("scope", ["whole", "reference", "normalization"])
def test_new_run_initializes_backbone_with_a_fresh_classifier(
    corpus_files, tiny_config, tmp_path, scope
):
    root, manifest = corpus_files
    development = SpeechCorpus(root, manifest, "dev", 0.25, 0.25)
    dev_cases = tmp_path / "dev.json"
    build_cases(development, 2, 800, dev_cases)
    source = tmp_path / "source"
    train(tiny_config, root, manifest, dev_cases, source, "cpu")
    initial_model, _ = load_model(source / "best.pt")
    config = tiny_config.model_copy(deep=True)
    config.seed = 999
    if scope in {"whole", "normalization"}:
        config.model.weights = "project_checkpoint"
    else:
        config.model = type(config.model).model_validate(
            {
                **config.model.model_dump(),
                "weights": "project_reference",
                "family": "reference_conditioned_stft_tcn",
                "encoder_channels": 257,
                "mask_activation": "sigmoid",
            }
        )
    config.loss.speaker_classification_weight = 0.2
    config.loss.spectral_weight = 0.5
    config.training.learning_rate = 1e-8
    config.training.max_optimizer_updates = 1
    destination = tmp_path / "new"
    initialization = {
        "initialize_reference_from" if scope == "reference" else "initialize_from": source
        / "best.pt"
    }
    if scope == "normalization":
        config.model.separation_normalization = "global"
        with pytest.raises(ValueError, match="identical extraction/reference architecture"):
            train(config, root, manifest, dev_cases, destination, "cpu", **initialization)
        initialization["initialize_normalization_transfer"] = True
    train(config, root, manifest, dev_cases, destination, "cpu", **initialization)
    changed, payload = load_model(destination / "latest.pt")
    assert changed.speaker_head is not None and payload["step"] == 1
    assert payload["provenance"]["initialization"]["source_step"] > 0
    for name, before in initial_model.state_dict().items():
        if scope != "reference" or name.startswith("reference_encoder."):
            torch.testing.assert_close(before, changed.state_dict()[name], rtol=0, atol=1e-6)
    if scope == "normalization":
        assert payload["provenance"]["initialization"]["normalization_transfer"] == {
            "from": "per_frame",
            "to": "global",
        }


def test_spectral_objective_penalizes_gain_and_noise_with_finite_gradients():
    from tse.metrics import spectral_loss

    time = torch.arange(4000) / 16000
    target = (torch.sin(2 * torch.pi * 220 * time) * 0.1)[None, None]
    assert spectral_loss(target, target, [256, 512]).item() == 0
    assert spectral_loss(target * 0.5, target, [256, 512]).item() > 0.5
    estimate = (target + 0.01 * torch.sin(2 * torch.pi * 3000 * time)).requires_grad_()
    loss = spectral_loss(estimate, target, [256, 512])
    loss.backward()
    assert loss.item() > 0 and torch.isfinite(estimate.grad).all()
    assert estimate.grad.abs().sum().item() > 0
