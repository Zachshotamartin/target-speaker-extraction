import copy
import io
import json

import numpy as np
import pytest
import soundfile as sf
import torch
from pydantic import ValidationError

from tse.audio import read_audio, speech_check, wav_bytes
from tse.config import ExperimentConfig
from tse.data import SpeechCorpus, audit_records, augment_reference, build_cases, load_cases
from tse.metrics import measure, si_sdr
from tse.model import TargetExtractor


def test_strict_configuration():
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"made_up_option": 1})
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"model": {"temporal_kernel": 4}})
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"training": {"microbatch_size": 0}})


def test_si_sdr_scale_padding_and_silence():
    torch.manual_seed(9)
    target = torch.randn(2, 1, 1000)
    estimate = target * 2 + torch.randn_like(target) * 0.05
    score = si_sdr(estimate, target)
    scaled = si_sdr(estimate * 3, target)
    torch.testing.assert_close(score, scaled, atol=1e-4, rtol=1e-4)
    lengths = torch.tensor([1000, 1000])
    extended_target = torch.cat([target, torch.ones(2, 1, 300) * 99], dim=-1)
    extended_estimate = torch.cat([estimate, torch.ones(2, 1, 300) * -22], dim=-1)
    torch.testing.assert_close(score, si_sdr(extended_estimate, extended_target, lengths))
    assert (si_sdr(torch.zeros_like(target), target) == -80).all()
    with pytest.raises(ValueError, match="non-silent"):
        si_sdr(target, torch.zeros_like(target))


def test_mixture_baseline_is_zero():
    target = torch.randn(2, 1, 4000)
    other = torch.randn_like(target)
    mixture = target + other
    rows = measure(mixture, mixture, target, other)
    assert all(row["si_sdri_db"] == 0 for row in rows)


def test_changed_source_is_rejected(corpus_files):
    root, manifest = corpus_files
    corpus = SpeechCorpus(root, manifest, "train", 0.25, 0.25)
    record = next(row for row in corpus.records.values() if row["split"] == "train")
    (root / record["path"]).write_bytes(b"changed after inventory")
    with pytest.raises(ValueError, match="checksum differs"):
        corpus.read(record["id"])


@pytest.mark.parametrize("length", [1, 31, 32, 33, 257, 4000])
def test_model_length_and_reference_gradients(tiny_config, length):
    model = TargetExtractor(tiny_config.model)
    mixture = torch.randn(2, 1, length)
    reference = torch.randn(2, 1, 4000)
    output = model(mixture, reference)
    assert output.shape == mixture.shape
    output.square().mean().backward()
    gradients = [p.grad for p in model.reference_encoder.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert sum(g.abs().sum() for g in gradients) > 0


def test_reference_padding_and_conditioning(tiny_config):
    torch.manual_seed(4)
    model = TargetExtractor(tiny_config.model).eval()
    reference = torch.randn(1, 1, 1000)
    padded = torch.cat([reference, torch.randn(1, 1, 1000) * 99], -1)
    lengths = torch.tensor([1000])
    torch.testing.assert_close(
        model.reference_encoder(reference),
        model.reference_encoder(padded, lengths),
        atol=1e-5,
        rtol=1e-5,
    )
    mixture = torch.randn(1, 1, 4000)
    assert not torch.equal(model(mixture, reference), model(mixture, torch.randn_like(reference)))


def test_data_audits_and_deterministic_mixing(corpus_files, tmp_path):
    root, manifest = corpus_files
    corpus = SpeechCorpus(root, manifest, "train", 0.25, 0.25)
    case = corpus.make_case(42)
    a, b = corpus.render(case), corpus.render(case)
    np.testing.assert_array_equal(a["mixture"], b["mixture"])
    np.testing.assert_allclose(a["mixture"], a["target"] + a["interferer"], atol=1e-7)
    ratio = 20 * np.log10(np.linalg.norm(a["target"]) / np.linalg.norm(a["interferer"]))
    assert ratio == pytest.approx(case["ratio_db"], abs=1e-4)
    swapped = dict(case, target_index=1)
    np.testing.assert_array_equal(corpus.render(swapped)["mixture"], a["mixture"])
    np.testing.assert_array_equal(corpus.render(swapped)["target"], a["interferer"])
    invalid = copy.deepcopy(case)
    invalid["references"][0] = invalid["sources"][0]
    with pytest.raises(ValueError, match="Reference identity"):
        corpus.validate_case(invalid)
    records = json.loads(manifest.read_text())["records"]
    next(row for row in records if row["split"] == "train")["split"] = "dev"
    with pytest.raises(ValueError, match="Speaker leakage"):
        audit_records(records)
    cases_path = tmp_path / "cases.json"
    build_cases(corpus, 4, 100, cases_path)
    assert len(load_cases(cases_path, corpus)) == 4


def test_audio_roundtrip_resample_limits():
    t = np.arange(8000) / 8000
    source = (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    stream = io.BytesIO()
    sf.write(stream, source, 8000, format="WAV")
    stream.seek(0)
    resampled = read_audio(stream)
    assert len(resampled) == 16000
    speech_check(resampled)
    roundtrip = read_audio(io.BytesIO(wav_bytes(resampled)))
    np.testing.assert_array_equal(resampled, roundtrip)
    with pytest.raises(ValueError, match="silent"):
        speech_check(np.zeros(16000, dtype=np.float32))
    with pytest.raises(ValueError, match="at most"):
        read_audio(io.BytesIO(wav_bytes(resampled)), max_seconds=0.5)


def test_augmentation_is_deterministic():
    source = np.random.default_rng(5).normal(0, 0.1, 8000).astype(np.float32)
    for condition in ("clean", "noise", "channel", "reverb", "combined"):
        one = augment_reference(source, condition, 22)
        np.testing.assert_array_equal(one, augment_reference(source, condition, 22))
        assert one.shape == source.shape and np.isfinite(one).all()
