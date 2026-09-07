import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient

from tse.api import create_app
from tse.config import ExperimentConfig
from tse.full_control import control
from tse.full_training import development_plan, resumable_validation, run_lock, train_full
from tse.librimix import LibriMixCorpus
from tse.utils import atomic_json, sha256


@pytest.fixture
def full_fixture(tmp_path):
    splits, enrollments = {}, {}
    for split in ("train", "dev"):
        rows, mapping = [], {}
        for pair in range(2):
            sources, signals = [], []
            for side in range(2):
                relative = f"{split}/{pair}-s{side}.wav"
                waveform = (
                    np.random.default_rng(pair + side * 30).normal(0, 0.1, 4800).astype(np.float32)
                )
                path = tmp_path / relative
                path.parent.mkdir(exist_ok=True)
                sf.write(path, waveform, 16000)
                sources.append(
                    {
                        "path": relative,
                        "sha256": sha256(path),
                        "speaker": f"{split}-{side}",
                        "utterance": f"{split}-{side}-{pair}",
                    }
                )
                signals.append(waveform)
                mapping[f"{pair}:{side}"] = f"{split}/{1 - pair}-s{side}.wav"
            mixture = tmp_path / f"{split}/{pair}-mix.wav"
            sf.write(mixture, signals[0] + signals[1], 16000)
            rows.append(
                {
                    "id": str(pair),
                    "sources": sources,
                    "samples": 4800,
                    "mixture": {
                        "path": str(mixture.relative_to(tmp_path)),
                        "sha256": sha256(mixture),
                    },
                }
            )
        splits[split], enrollments[split] = rows, mapping
    manifest = tmp_path / "manifest.json"
    atomic_json(
        manifest,
        {"protocol": "libri2mix-16k-min-clean", "splits": splits, "enrollments": enrollments},
    )
    config = ExperimentConfig.load(Path("configs/full-data-efficient.json"))
    config.model.band_channels = 8
    config.model.band_hidden_channels = 8
    config.model.band_blocks = 1
    config.model.band_widths = [64, 64, 64, 65]
    config.model.reference_encoder.resnet_base_channels = 4
    config.audio.crop_seconds = config.audio.reference_seconds = 0.25
    config.training.epochs = 2
    config.training.microbatch_size = 1
    config.training.gradient_accumulation = 2
    config.training.max_optimizer_updates = 4
    config.training.validation_interval_updates = 2
    config.evaluation.development_cases_target = 4
    path = tmp_path / "config.json"
    atomic_json(path, config.model_dump())
    return path, tmp_path, manifest


@pytest.mark.parametrize("pause_during_validation", [False, True])
def test_full_resume_preserves_epoch_optimizer_rng_and_schedule(
    full_fixture, monkeypatch, pause_during_validation
):
    config, root, manifest = full_fixture
    train_full(config, root, manifest, root / "complete", "cpu")
    run = root / "resumed"
    if pause_during_validation:
        import tse.full_training as training

        original = training.resumable_validation

        def interrupt(*args):
            if args[5]["kind"] != "full":
                return original(*args)
            checks = 0

            def pause():
                nonlocal checks
                checks += 1
                if checks > 1:
                    (run / "pause.request").touch()
                    return True
                return False

            return original(*args[:6], pause, *args[7:])

        with monkeypatch.context() as patch:
            patch.setattr(training, "resumable_validation", interrupt)
            train_full(config, root, manifest, run, "cpu")
        state = json.loads((run / "full-validation-state.json").read_text())
        assert len(state["rows"]) == 1
        assert not (run / "best-full.pt").exists()
        (run / "pause.request").unlink()
    else:
        train_full(config, root, manifest, run, "cpu", stop_after_updates=1)
        saved = torch.load(run / "latest.pt", weights_only=True)
        assert saved["step"] == 1
        # A session pause must not force the four-update schedule to its minimum.
        assert saved["optimizer"]["param_groups"][0]["lr"] == 0.001
    train_full(config, root, manifest, run, "cpu", resume=True)
    expected = torch.load(root / "complete/latest.pt", weights_only=True)
    actual = torch.load(run / "latest.pt", weights_only=True)
    for name, tensor in expected["model"].items():
        torch.testing.assert_close(tensor, actual["model"][name], atol=0, rtol=0, msg=name)
    for key, state in expected["optimizer"]["state"].items():
        for field, value in state.items():
            torch.testing.assert_close(
                value, actual["optimizer"]["state"][key][field], atol=0, rtol=0
            )
    assert actual["optimizer"]["param_groups"] == expected["optimizer"]["param_groups"]
    assert actual["validated_steps"] == expected["validated_steps"] == {"monitor": 4, "full": 4}
    assert actual["best_scores"] == expected["best_scores"]
    torch.testing.assert_close(actual["torch_rng"], expected["torch_rng"], atol=0, rtol=0)
    assert json.loads((run / "status.json").read_text())["status"] == "complete"


def test_monitor_is_balanced_and_rejects_training_speakers(full_fixture):
    _, root, manifest = full_fixture
    train, dev = [LibriMixCorpus(root, manifest, split) for split in ("train", "dev")]
    plan = development_plan(train, dev, 4)
    assert sorted(plan["indices"]) == list(range(4))
    assert plan == development_plan(train, dev, 4)
    with pytest.raises(ValueError, match="overlap"):
        development_plan(train, train, 4)


def test_validation_resume_keeps_completed_cases(full_fixture):
    _, root, manifest = full_fixture
    corpus = LibriMixCorpus(root, manifest, "dev")

    class Identity(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.5))

        def forward(self, mixture, reference):
            return mixture * self.scale

    calls = 0

    def pause():
        nonlocal calls
        calls += 1
        return calls > 2

    path = root / "evaluation.json"
    model = Identity()
    args = model, corpus, [0, 1, 2, 3], 4000, path, {"step": 10}
    assert resumable_validation(*args, pause) is None
    assert len(json.loads(path.read_text())["rows"]) == 2
    resumed = resumable_validation(*args, lambda: False)
    complete = resumable_validation(
        model, corpus, [0, 1, 2, 3], 4000, root / "other.json", {"step": 10}, lambda: False
    )
    assert resumed == complete


def test_session_expiry_and_double_trainer_protection(full_fixture):
    config, root, manifest = full_fixture
    run = root / "expired"
    with run_lock(run), pytest.raises(RuntimeError, match="active trainer"):
        train_full(config, root, manifest, run, "cpu")
    train_full(config, root, manifest, run, "cpu", minutes=0.000001)
    status = json.loads((run / "status.json").read_text())
    assert status["status"] == "paused" and status["pause_reason"] == "session_time_limit"
    assert torch.load(run / "latest.pt", weights_only=True)["step"] == 0


@pytest.mark.parametrize("platform", ["linux", "darwin"])
@pytest.mark.parametrize("session_minutes", [None, 60])
def test_controls_are_fixed_and_duplicate_resume_is_idempotent(
    tmp_path, monkeypatch, session_minutes, platform
):
    import tse.full_control as controls

    run = tmp_path / "run"
    run.mkdir()
    atomic_json(
        tmp_path / "artifacts/full-training-active.json",
        {
            "run": str(run),
            "root": str(tmp_path),
            "manifest": "manifest.json",
            "config": "config.json",
            "device": "cpu",
            "session_minutes": session_minutes,
        },
    )
    launches = []

    class Worker:
        pid = 123456

        def __init__(self, args, **kwargs):
            launches.append(args)

        def poll(self):
            return None

    monkeypatch.setattr(controls.subprocess, "Popen", Worker)
    monkeypatch.setattr(controls.sys, "platform", platform)
    monkeypatch.setattr(controls, "_WORKERS", {})
    assert control("resume", tmp_path)["status"] == "started"
    assert control("resume", tmp_path)["status"] == "already_running"
    assert len(launches) == (2 if platform == "darwin" else 1)
    if platform == "darwin":
        assert launches[1] == ["caffeinate", "-i", "-w", "123456"]
    if session_minutes is None:
        assert "--minutes" not in launches[0]
    else:
        assert launches[0][-2:] == ["--minutes", str(session_minutes)]
    assert control("pause", tmp_path)["status"] == "pause_requested"
    assert (run / "pause.request").exists()
    monkeypatch.chdir(tmp_path)
    with TestClient(create_app(tmp_path / "missing.pt")) as client:
        assert (
            client.post("/experiments/full/control", json={"action": "delete"}).status_code == 422
        )
        assert (
            client.post(
                "/experiments/full/control", json={"action": "resume", "run": "/tmp"}
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/experiments/full/control",
                json={"action": "pause"},
                headers={"Origin": "https://example.com"},
            ).status_code
            == 403
        )


def test_latest_result_does_not_overwrite_an_earlier_best(full_fixture, monkeypatch):
    import tse.full_training as training

    config, root, manifest = full_fixture
    original = training.resumable_validation

    def scores(*args):
        result = original(*args)
        if result is not None:
            identity = args[5]
            result["mean_si_sdri_db"] = {0: -20.0, 2: 5.0, 4: 2.0}[identity["step"]]
        return result

    monkeypatch.setattr(training, "resumable_validation", scores)
    run = root / "best-retention"
    train_full(config, root, manifest, run, "cpu")
    assert torch.load(run / "latest.pt", weights_only=True)["step"] == 4
    for kind, name in [("monitor", "best.pt"), ("full", "best-full.pt")]:
        assert torch.load(run / name, weights_only=True)["step"] == 2
        assert json.loads((run / f"best-{kind}-validation.json").read_text())["step"] == 2
        assert json.loads((run / f"latest-{kind}-validation.json").read_text())["step"] == 4


def test_preflight_rejects_same_utterance_reference_and_repair_preserves_mixtures(full_fixture):
    import importlib.util

    _, root, manifest = full_fixture
    payload = json.loads(manifest.read_text())
    payload["enrollments"]["dev"]["0:0"] = "dev/0-s0.wav"
    atomic_json(manifest, payload)
    train, dev = [LibriMixCorpus(root, manifest, s) for s in ("train", "dev")]
    with pytest.raises(ValueError, match="different utterance"):
        development_plan(train, dev, 4)
    spec = importlib.util.spec_from_file_location(
        "prepare_full", "scripts/prepare_full_data_training.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = root / "repaired.json"
    summary = module.prepare(manifest, output, root / "audit.json")
    repaired = json.loads(output.read_text())
    assert summary["repaired_development_references"] == 1
    assert repaired["splits"] == payload["splits"]
    assert repaired["enrollments"]["dev"]["0:0"] == "dev/1-s0.wav"
    assert development_plan(train, LibriMixCorpus(root, output, "dev"), 4)


def test_unlimited_session_survives_more_than_eight_hours_and_still_pauses(
    full_fixture, monkeypatch
):
    from types import SimpleNamespace

    import tse.full_training as training

    calls = 0

    def clock():
        nonlocal calls
        calls += 1
        return 0.0 if calls == 1 else 9 * 3600.0 + calls

    config, root, manifest = full_fixture
    monkeypatch.setattr(training, "time", SimpleNamespace(monotonic=clock))
    run = root / "unlimited"
    train_full(config, root, manifest, run, "cpu", stop_after_updates=1)
    state = json.loads((run / "status.json").read_text())
    assert state["session_minutes"] is None
    assert state["session_elapsed_seconds"] > 8 * 3600
    assert state["status"] == "paused" and state["pause_reason"] == "requested_update_limit"
    assert torch.load(run / "latest.pt", weights_only=True)["step"] == 1


@pytest.mark.parametrize("minutes", [0, -1, float("nan"), float("inf")])
def test_explicit_time_limits_must_be_finite_and_positive(full_fixture, minutes):
    config, root, manifest = full_fixture
    with pytest.raises(ValueError, match="finite and positive"):
        train_full(config, root, manifest, root / "invalid-limit", "cpu", minutes=minutes)
