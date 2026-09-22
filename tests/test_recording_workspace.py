import json
import subprocess
import time

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from poc.common import RATE, decode, wav
from poc.server import create_app
from poc.workspace import CHUNK_BYTES, Uploads, render_options
from poc.workspace_worker import discover, probe, render_video, subtitle_text, windowed_extract


def test_resumable_upload_checks_owner_offsets_and_duplicate_content(tmp_path):
    uploads = Uploads(tmp_path / "uploads")
    try:
        state = uploads.create(CHUNK_BYTES + 3, "alice")
        key = state["id"]
        with pytest.raises(HTTPException):
            uploads.public(key, "bob")
        with pytest.raises(HTTPException):
            uploads.write(key, 1, b"end", "alice")
        uploads.write(key, 0, b"a" * CHUNK_BYTES, "alice")
        assert uploads.write(key, 0, b"a" * CHUNK_BYTES, "alice")["received"] == CHUNK_BYTES
        with pytest.raises(HTTPException):
            uploads.write(key, 0, b"b" * CHUNK_BYTES, "alice")
        with pytest.raises(HTTPException):
            uploads.complete(key, "alice")
        uploads.write(key, 1, b"end", "alice")
        assert uploads.complete(key, "alice").read_bytes() == b"a" * CHUNK_BYTES + b"end"
        uploads.items[key]["expires_at"] = time.time() - 1
        with pytest.raises(HTTPException):
            uploads.complete(key, "alice")
        assert not (tmp_path / "uploads" / f"{key}.input").exists()
    finally:
        uploads.close()


@pytest.mark.parametrize("seconds", [24.1, 43.9, 44, 44.1, 65, 600])
def test_long_windowing_preserves_samples_including_overlap_and_final_tail(seconds):
    audio = np.random.default_rng(19).normal(0, 0.1, round(seconds * RATE)).astype(np.float32)
    actual = windowed_extract(lambda mix, ref: mix * 0.5, audio, audio[: 4 * RATE])
    np.testing.assert_allclose(actual, audio * 0.5, atol=1e-7)
    assert len(actual) == len(audio)


def test_long_decode_bounds_and_caption_validation():
    assert len(decode(wav(np.zeros(40 * RATE)), maximum=600)) == 40 * RATE
    with pytest.raises(ValueError):
        decode(wav(np.zeros(40 * RATE)))
    for options in [
        {"clips": [{"start": 4, "end": 2}]},
        {"clips": [{"start": 0, "end": 2}, {"start": 1, "end": 4}]},
        {"clips": [{"start": float("nan"), "end": 4}]},
    ]:
        with pytest.raises(HTTPException):
            render_options(options)
    clean = render_options(
        {"clips": [{"start": 1, "end": 3}], "words": [{"text": "Hello", "start": 0, "end": 1}]}
    )
    assert "00:00:00,000 --> 00:00:01,000" in subtitle_text(clean["words"])
    assert "<" not in subtitle_text([{"text": "<font>hi{\\pos(0,0)}", "start": 0, "end": 1}])


def test_workspace_upload_and_assets_are_owner_scoped(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "manifest.json").write_text("{}")
    app = create_app(directory=tmp_path / "jobs", models=models, access_token="test-secret" * 4)
    with TestClient(app) as client:
        app.state.jobs.stop.set()
        app.state.jobs.thread.join()
        alice = {
            "Authorization": "Bearer " + "test-secret" * 4,
            "X-OneVoice-Owner": "11111111-1111-4111-8111-111111111111",
        }
        bob = {**alice, "X-OneVoice-Owner": "22222222-2222-4222-8222-222222222222"}
        created = client.post("/workspace/uploads", json={"size": 3}, headers=alice)
        assert created.status_code == 201
        key = created.json()["id"]
        assert client.get(f"/workspace/uploads/{key}", headers=bob).status_code == 404
        assert (
            client.post(
                f"/workspace/uploads/{key}/chunks/0", content=b"abc", headers=alice
            ).status_code
            == 200
        )
        job = client.post(
            "/workspace/jobs", json={"kind": "discover", "recording": key}, headers=alice
        ).json()
        assert client.get(f"/workspace/jobs/{job['id']}", headers=bob).status_code == 404
        folder = tmp_path / "jobs" / job["id"]
        (folder / "original.wav").write_bytes(b"wav")
        (folder / "result.json").write_text(json.dumps({"kind": "discover"}))
        app.state.jobs.jobs[job["id"]]["status"] = "ready"
        path = f"/workspace/jobs/{job['id']}/assets/original.wav"
        assert client.get(path + "/info", headers=alice).json()["bytes"] == 3
        assert client.get(path + "/chunks/0", headers=alice).content == b"wav"
        assert client.get(path + "/chunks/0", headers=bob).status_code == 404
        assert client.get(path + "/chunks/1", headers=alice).status_code == 404


def test_discovery_returns_timed_auditionable_candidates():
    class Runtime:
        def speech(self, audio):
            return [[0, len(audio) / RATE]]

        def embedding(self, audio):
            return np.array([1.0, 0.0]) if audio.mean() > 0 else np.array([0.0, 1.0])

    audio = np.concatenate([np.full(8 * RATE, 0.1), np.full(8 * RATE, -0.1)])
    candidates = discover(Runtime(), audio)
    assert len(candidates) == 2
    assert all(3 <= c["end"] - c["start"] <= 10 for c in candidates)


def test_captioned_video_keeps_audio_and_picture_in_sync(tmp_path):
    import shutil

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg required")
    source = tmp_path / "mixture.input"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=30:duration=4",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=4",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-c:a",
            "aac",
            "-f",
            "mp4",
            str(source),
        ],
        check=True,
    )
    (tmp_path / "edited.input").write_bytes(wav(np.zeros(2 * RATE, dtype=np.float32)))
    options = render_options(
        {
            "clips": [{"start": 0, "end": 1}, {"start": 3, "end": 4}],
            "words": [
                {"start": 0.1, "end": 0.8, "text": "First"},
                {"start": 1.1, "end": 1.8, "text": "Last"},
            ],
        }
    )
    result = render_video(tmp_path, options, lambda _: None)
    metadata = probe(tmp_path / result["asset"])
    assert float(metadata["format"]["duration"]) == pytest.approx(2, abs=0.08)
    streams = {s["codec_type"]: float(s["duration"]) for s in metadata["streams"]}
    assert abs(streams["video"] - streams["audio"]) < 0.08
    assert not (tmp_path / "render.wav").exists()


def test_workspace_submission_is_idempotent_after_inputs_are_consumed(tmp_path):
    import uuid

    models = tmp_path / "models"
    models.mkdir()
    (models / "manifest.json").write_text("{}")
    app = create_app(directory=tmp_path / "jobs", models=models)
    with TestClient(app) as client:
        app.state.jobs.stop.set()
        app.state.jobs.thread.join()
        upload = client.post("/workspace/uploads", json={"size": 3}).json()["id"]
        client.post(f"/workspace/uploads/{upload}/chunks/0", content=b"abc")
        request_id = str(uuid.uuid4())
        body = {"kind": "discover", "recording": upload, "request_id": request_id}
        first = client.post("/workspace/jobs", json=body)
        retry = client.post("/workspace/jobs", json=body)
        assert first.status_code == retry.status_code == 202
        assert first.json()["id"] == retry.json()["id"]
        recovered = client.get(f"/workspace/requests/{request_id}")
        assert recovered.json()["id"] == first.json()["id"]
        assert len(app.state.jobs.jobs) == 1
