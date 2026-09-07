import io
import json

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from tse.api import create_app
from tse.audio import read_audio, wav_bytes
from tse.engine import save_checkpoint
from tse.inference import Extractor
from tse.model import TargetExtractor


@pytest.fixture
def checkpoint(tiny_config, tmp_path):
    torch.manual_seed(13)
    path = tmp_path / "model.pt"
    model = TargetExtractor(tiny_config.model)
    save_checkpoint(
        path,
        {
            "format_version": 1,
            "config": tiny_config.model_dump(),
            "model": model.state_dict(),
            "step": 0,
            "speaker_classes": 0,
            "provenance": {"experiment_type": "unit_test_only"},
        },
    )
    return path


def sound(seconds=3, frequency=220):
    t = np.arange(round(seconds * 16000)) / 16000
    return (0.15 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def uploads(reference=None):
    return {
        "mixture": ("conversation.wav", wav_bytes(sound(1)), "audio/wav"),
        "reference": (
            "voice.wav",
            wav_bytes(sound() if reference is None else reference),
            "audio/wav",
        ),
    }


def test_ready_interface_and_successful_extraction(checkpoint):
    with TestClient(create_app(checkpoint, "cpu")) as client:
        assert client.get("/health").json() == {"status": "alive"}
        assert client.get("/ready").status_code == 200
        info = client.get("/model").json()
        assert info["ready"] and info["training_updates"] == 0
        assert "Keep one voice" in client.get("/").text
        response = client.post("/extract", files=uploads())
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "audio/wav"
        assert response.headers["cache-control"] == "no-store"
        output = read_audio(io.BytesIO(response.content))
        assert output.shape == (16000,) and np.isfinite(output).all()


def test_listening_http_ratings_are_local_to_the_declared_study(checkpoint, tmp_path, monkeypatch):
    from tse.utils import atomic_json

    monkeypatch.chdir(tmp_path)
    study = tmp_path / "artifacts/listening/qa-study"
    atomic_json(study / "key.json", {"trials": [{"id": "trial-01"}]})
    scores = {"competing_speech": 2, "target_damage": 3, "static": 1}
    payload = {"participant": "test-only", "trial": "trial-01", "A": scores, "B": scores}
    with TestClient(create_app(checkpoint, "cpu")) as client:
        response = client.post("/listening/qa-study/ratings", json=payload)
        assert response.status_code == 200
        assert response.json()["completed_trials"] == 1
        assert client.post("/listening/missing-study/ratings", json=payload).status_code == 404
        payload["trial"] = "unknown"
        assert client.post("/listening/qa-study/ratings", json=payload).status_code == 422
        assert client.get("/experiments/v3/status").json()["job_process_alive"] is False
    assert len(json.loads((study / "ratings/test-only.json").read_text())["ratings"]) == 1


def test_missing_model_and_invalid_reference(checkpoint, tmp_path):
    with TestClient(create_app(tmp_path / "missing.pt", "cpu")) as client:
        assert client.get("/ready").status_code == 503
        assert client.post("/extract", files=uploads()).status_code == 503
    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"invalid checkpoint")
    with TestClient(create_app(corrupt, "cpu")) as client:
        assert client.get("/ready").status_code == 503
        assert client.get("/").status_code == 200
    with TestClient(create_app(checkpoint, "cpu")) as client:
        assert (
            client.post("/extract", files=uploads(np.zeros(48000, dtype=np.float32))).status_code
            == 422
        )
        assert client.post("/extract", files=uploads(sound(1))).status_code == 422
        response = client.post(
            "/extract",
            files={
                "mixture": ("invalid.wav", b"not audio"),
                "reference": ("voice.wav", wav_bytes(sound())),
            },
        )
        assert response.status_code == 422


def test_upload_origin_size_and_busy(checkpoint, monkeypatch):
    app = create_app(checkpoint, "cpu")
    with TestClient(app) as client:
        assert (
            client.post(
                "/extract", files=uploads(), headers={"Origin": "https://unrelated.example"}
            ).status_code
            == 403
        )
        app.state.gate.acquire()
        try:
            assert client.post("/extract", files=uploads()).status_code == 429
        finally:
            app.state.gate.release()
        monkeypatch.setattr("tse.api.MAX_UPLOAD", 128)
        assert client.post("/extract", content=b"x" * 129).status_code == 413


def test_chunked_inference_preserves_length_and_context(checkpoint):
    extractor = Extractor(checkpoint, "cpu")
    mixture = sound(7.001) + sound(7.001, 370) * 0.7
    reference = sound(3)
    whole = extractor.extract_array(mixture, reference, chunked=False)
    chunked = extractor.extract_array(mixture, reference, chunked=True)
    assert whole.shape == chunked.shape == mixture.shape
    np.testing.assert_allclose(whole, chunked, atol=2e-6, rtol=2e-4)


@pytest.mark.parametrize("amplitude", [0.4, 1.4])
def test_download_preserves_shape_without_playback_overflow(checkpoint, monkeypatch, amplitude):
    raw = sound(1) * (amplitude / 0.15)
    before = raw.copy()
    with TestClient(create_app(checkpoint, "cpu")) as client:
        # Isolate the delivery contract from stochastic model predictions.
        monkeypatch.setattr(client.app.state.extractor, "extract_array", lambda *args: raw)
        response = client.post("/extract", files=uploads())
        assert response.status_code == 200
        delivered = read_audio(io.BytesIO(response.content))
        metadata = json.loads(response.headers["X-TSE-Metadata"])
        assert metadata["processing_version"] == "sample-peak-guard-v1"
        assert np.max(np.abs(delivered)) <= 0.980001
        assert 0 < metadata["playback_gain"] <= 1
        np.testing.assert_allclose(delivered, before * metadata["playback_gain"], atol=1e-7)
        np.testing.assert_array_equal(raw, before)
        if amplitude < 0.98:
            np.testing.assert_array_equal(delivered, before)
        else:
            assert metadata["playback_gain"] < 1 and metadata["raw_output_peak"] > 1
