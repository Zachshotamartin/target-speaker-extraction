"""Boundary, attribution and process-isolation checks for the local POC."""

import io
import subprocess
import sys
import time

import av
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from poc.common import MAX_BYTES, RATE, decode, export_text, overlap, wav
from poc.models import ASR_SETTINGS, Runtime
from poc.pipeline import attribute, classify, run
from poc.server import Jobs, create_app


def test_decoder_rejects_empty_corrupt_oversized_and_long_audio():
    for data in [b"", b"bad audio", b"x" * (MAX_BYTES + 1), wav(np.zeros(31 * RATE))]:
        with pytest.raises(ValueError):
            decode(data)


def test_decoder_downmixes_resamples_and_rejects_nonfinite():
    buffer = io.BytesIO()
    sf.write(
        buffer,
        np.column_stack([np.full(44100, 0.2), np.zeros(44100)]),
        44100,
        format="WAV",
        subtype="FLOAT",
    )
    signal = decode(buffer.getvalue())
    assert len(signal) == RATE
    assert np.mean(signal[100:-100]) == pytest.approx(0.1, abs=0.05)
    invalid = io.BytesIO()
    sf.write(invalid, np.full(RATE, np.nan), RATE, format="WAV", subtype="FLOAT")
    with pytest.raises(ValueError):
        decode(invalid.getvalue())


@pytest.mark.parametrize(
    "format_name,codec,rate",
    [
        ("webm", "libopus", 48000),
        ("ipod", "aac", 16000),
        ("mp3", "libmp3lame", 16000),
        ("flac", "flac", 16000),
    ],
)
def test_browser_and_upload_formats(format_name, codec, rate):
    buffer = io.BytesIO()
    with av.open(buffer, "w", format=format_name) as container:
        stream = container.add_stream(codec, rate=rate)
        stream.layout = "mono"
        signal = (0.1 * np.sin(2 * np.pi * 220 * np.arange(3 * RATE) / RATE)).astype(np.float32)
        frame = av.AudioFrame.from_ndarray(signal[None], format="flt", layout="mono")
        frame.sample_rate = RATE
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    result = decode(buffer.getvalue())
    assert 2.95 < len(result) / RATE < 3.1
    assert np.isfinite(result).all()


def test_compressed_duration_is_bounded_after_decode(tmp_path):
    path = tmp_path / "long.flac"
    sf.write(path, np.zeros(31 * RATE), RATE)
    assert path.stat().st_size < MAX_BYTES
    with pytest.raises(ValueError, match="at most"):
        decode(path.read_bytes())


def test_word_boundary_is_not_silently_accepted_and_offsets_stay_original():
    windows = [
        {"start": 5, "end": 6, "attribution": "accepted"},
        {"start": 6, "end": 7, "attribution": "uncertain"},
    ]
    words = [
        {"start": 5.1, "end": 5.8, "text": " yes"},
        {"start": 5.9, "end": 6.3, "text": " maybe"},
        {"start": 7.3, "end": 7.8, "text": " other"},
    ]
    segments = attribute(words, windows, [[5, 8]], 9)
    assert [s["attribution"] for s in segments] == ["accepted", "uncertain", "excluded"]
    assert segments[0]["start"] == 5.1
    assert "maybe" not in export_text({"segments": segments}, "txt")
    assert "other" not in export_text({"segments": segments}, "srt")
    assert "00:00:05,100" in export_text({"segments": segments}, "srt")
    assert "uncertain" in export_text({"segments": segments}, "json")
    assert overlap(0, 2, [[0, 1.5], [1, 2]]) == 2


def test_identity_requires_original_evidence_and_enough_speech():
    assert classify(0.9, 0.05, 2) != "accepted"
    assert classify(0.9, 0.9, 0.3) != "accepted"
    assert classify(0.9, 0.9, 2) == "accepted"
    assert classify(0.1, 0.8, 2) == "excluded"


class FakeRuntime:
    manifest = {"one_voice": {"sha256": "test"}}

    def __init__(self):
        self.asr_inputs = []
        self.extractions = 0

    def speech(self, audio):
        return [[0, len(audio) / RATE]] if np.any(audio) else []

    def extract(self, mixture, reference):
        self.extractions += 1
        return mixture / 2

    def release_separator(self):
        pass

    def embedding(self, audio):
        return np.array([1.0, 0.0])

    def transcribe(self, audio):
        self.asr_inputs.append(audio.copy())
        return [{"start": 0.2, "end": 1.0, "text": " Hello"}]


def test_silence_never_invokes_separator_or_asr():
    runtime = FakeRuntime()
    result, estimate = run(runtime, np.zeros(4 * RATE), np.ones(3 * RATE) * 0.1)
    assert result["outcome"] == "no_speech"
    assert result["segments"] == [] and not np.any(estimate)
    assert runtime.extractions == 0 and runtime.asr_inputs == []


def test_only_one_voice_changes_waveform_and_asr_receives_no_reference():
    import inspect

    assert list(inspect.signature(Runtime.transcribe).parameters) == ["self", "audio"]
    assert ASR_SETTINGS["vad_filter"] is False
    assert ASR_SETTINGS["initial_prompt"] is None
    runtime = FakeRuntime()
    mixture = np.full(4 * RATE, 0.2, np.float32)
    result, _ = run(runtime, mixture, np.full(3 * RATE, 0.1, np.float32))
    assert runtime.extractions == 1
    assert len(runtime.asr_inputs) == 2
    np.testing.assert_array_equal(runtime.asr_inputs[0], mixture / 2)
    np.testing.assert_array_equal(runtime.asr_inputs[1], mixture)
    assert result["comparison"]["same_asr_settings"]
    assert not result["comparison"]["reference_supplied_to_asr"]


@pytest.fixture
def slow_worker(monkeypatch):
    original = subprocess.Popen

    def spawn(command, **kwargs):
        if "poc.worker" in command:
            command = [sys.executable, "-c", "import time;time.sleep(30)"]
        return original(command, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", spawn)


def until(predicate):
    deadline = time.monotonic() + 4
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.03)


def test_queue_cancel_only_owned_worker_and_cleanup(tmp_path, slow_worker):
    other = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"])
    jobs = Jobs(tmp_path / "jobs", tmp_path / "models")
    try:
        first = jobs.submit(b"mix", b"ref", True)
        second = jobs.submit(b"mix", b"ref", False)
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as error:
            jobs.submit(b"mix", b"ref", True)
        assert error.value.status_code == 429
        until(lambda: jobs.public(first["id"])["status"] == "running")
        owned = jobs.jobs[first["id"]]["process"]
        jobs.cancel(first["id"])
        assert owned.poll() is not None
        assert other.poll() is None
        assert not (jobs.directory / first["id"]).exists()
        jobs.cancel(second["id"])
        assert jobs.public(second["id"])["status"] == "cancelled"
        jobs.jobs[first["id"]]["expires_at"] = time.time() - 1
        until(lambda: first["id"] not in jobs.jobs)
    finally:
        jobs.close()
        other.terminate()
        other.wait(timeout=2)


def test_api_limits_origin_and_no_training_routes(tmp_path, slow_worker):
    models = tmp_path / "models"
    models.mkdir()
    (models / "manifest.json").write_text("{}")
    app = create_app(tmp_path / "jobs", models)
    with TestClient(app) as client:
        assert client.get("/health").json()["ready"]
        assert client.get("/health", headers={"Origin": "https://example.com"}).status_code == 403
        assert client.post("/train").status_code == 404
        assert client.post("/pause").status_code == 404
        assert client.post("/transcriptions", content=b"a" * (MAX_BYTES + 1)).status_code == 413
        assert (
            client.post(
                "/transcriptions", content=b"{}", headers={"Content-Type": "application/json"}
            ).status_code
            == 415
        )
        assert (
            client.post("/transcriptions", files={"mixture": ("a.wav", b"mix")}).status_code == 400
        )
        assert (
            client.post(
                "/transcriptions",
                files=[
                    ("mixture", ("a.wav", b"a")),
                    ("mixture", ("b.wav", b"b")),
                    ("reference", ("c.wav", b"c")),
                ],
            ).status_code
            == 400
        )
        response = client.post(
            "/transcriptions", files={"mixture": ("a.wav", b"mix"), "reference": ("b.wav", b"ref")}
        )
        assert response.status_code == 202
        key = response.json()["id"]
        assert client.get(f"/transcriptions/{key}/audio/extracted").status_code == 409
        assert client.delete(f"/transcriptions/{key}").json()["status"] == "cancelled"


def test_second_server_cannot_delete_live_jobs(tmp_path):
    app1 = create_app(tmp_path / "jobs", tmp_path / "models")
    app2 = create_app(tmp_path / "jobs", tmp_path / "models")
    with TestClient(app1):
        with pytest.raises(BlockingIOError), TestClient(app2):
            pass
