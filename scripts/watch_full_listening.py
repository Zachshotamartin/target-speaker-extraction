#!/usr/bin/env python3
"""Publish fixed full-validation listening examples without modifying the trainer.

Run with .venv/bin/python scripts/watch_full_listening.py [--once]. The existing
gallery mount serves the output. CPU only, one thread, one worker per workspace.
"""

import argparse
import fcntl
import gc
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from tse.audio import wav_bytes
from tse.config import ExperimentConfig
from tse.full_training import cropped_request
from tse.librimix import LibriMixCorpus, request_seed
from tse.metrics import measure
from tse.model import make_model
from tse.utils import atomic_json, sha256

INDICES = [0, 1, 2, 3]  # Both target voices in the first two development mixtures.
OUTPUT = Path("artifacts/gallery/full-validation")
CONTROL = Path("artifacts/full-listening")


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def selections(run):
    """Never call an arbitrary latest training checkpoint a validated model."""
    best = read(run / "best-full-validation.json")
    latest = read(run / "latest-full-validation.json")
    status = read(run / "status.json")
    result = {}
    if best:
        result["best"] = {
            "step": best["step"],
            "result": best,
            "pending": False,
            "checkpoint": "best-full.pt",
        }
    if latest:
        result["latest"] = {
            "step": latest["step"],
            "result": latest,
            "pending": False,
            "checkpoint": "latest.pt",
        }
        if best and latest["step"] == best["step"]:
            result["latest"]["checkpoint"] = "best-full.pt"
    # Capture weights while full validation runs, before latest.pt advances again.
    if (
        status.get("status") == "validating"
        and status.get("validation_kind") == "full"
        and status["step"] > latest.get("step", -1)
    ):
        result["latest"] = {
            "step": status["step"],
            "result": {},
            "pending": True,
            "checkpoint": "latest.pt",
        }
    return result


def load_snapshot(path, selection, manifest_hash):
    # The trainer atomically replaces checkpoints. An open descriptor keeps one
    # complete version even if the filename changes while we hash/read it.
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
        handle.seek(0)
        payload = torch.load(handle, map_location="cpu", weights_only=True)
    if payload["step"] != selection["step"]:
        raise ValueError("The checkpoint has advanced; waiting for an exact validation snapshot")
    if payload["manifest_sha256"] != manifest_hash:
        raise ValueError("Checkpoint and listening dataset differ")
    expected = selection["result"].get("checkpoint_sha256")
    if expected and digest != expected:
        raise ValueError("Checkpoint changed during selection; retrying next refresh")
    payload.pop("optimizer", None)
    return payload, digest


def playback_versions(signal):
    """Volume matching is playback only; preserve unmodified float WAV as raw."""
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    if not np.isfinite(signal).all():
        raise ValueError("Non-finite model audio")
    rms = float(np.sqrt(np.mean(signal.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(signal)))
    gain = min(0.08 / max(rms, 1e-8), 0.98 / max(peak, 1e-8), 100.0)
    return {"raw": signal, "matched": signal * gain}, gain


def render_snapshot(run, pointer, selection, output=OUTPUT):
    manifest_hash = sha256(Path(pointer["manifest"]))
    run_id = hashlib.sha256((str(run.resolve()) + manifest_hash).encode()).hexdigest()[:12]
    directory = output / f"step-{run_id}-{selection['step']:07d}"
    cached = read(directory / "record.json")
    if cached:
        if cached["indices"] != INDICES or cached["manifest_sha256"] != manifest_hash:
            raise ValueError("Cached listening selection differs")
        return cached
    payload, digest = load_snapshot(run / selection["checkpoint"], selection, manifest_hash)
    config = ExperimentConfig.model_validate(payload["config"])
    corpus = LibriMixCorpus(pointer["root"], pointer["manifest"], "dev")
    if set(payload["provenance"]["train_speakers"]) & set(corpus.speakers):
        raise ValueError("Listening speakers overlap training")
    model = make_model(config.model, payload["speaker_classes"]).cpu().eval()
    model.load_state_dict(payload.pop("model"))
    temporary = directory.with_name(directory.name + ".partial")
    if temporary.exists():
        shutil.rmtree(temporary)  # Only our incomplete unpublished output.
    temporary.mkdir(parents=True)
    records = []
    for index in INDICES:
        request = cropped_request(
            corpus, index, request_seed(19, 0, index), int(config.audio.reference_seconds * 16000)
        )
        with torch.inference_mode():
            prediction = model(request["mixture"], request["reference"])
        metrics = measure(prediction, request["mixture"], request["target"], request["interferer"])[
            0
        ]
        # Check the exact case against the official completed validation when available.
        official = next(
            (r for r in selection["result"].get("rows", []) if r["index"] == index), None
        )
        if official and abs(official["si_sdri_db"] - metrics["si_sdri_db"]) > 0.03:
            raise ValueError("Listening prediction does not reproduce the validation case")
        tracks = {}
        for name in ("mixture", "reference", "target", "estimate"):
            signal = prediction if name == "estimate" else request[name]
            versions, gain = playback_versions(signal.numpy())
            tracks[name] = {"playback_gain": gain}
            for mode, samples in versions.items():
                filename = f"{index:02d}-{name}-{mode}.wav"
                (temporary / filename).write_bytes(wav_bytes(samples))
                tracks[name][mode] = f"/gallery/full-validation/{directory.name}/{filename}"
        records.append(
            {
                "index": index,
                "case_id": request["case_id"],
                "speaker": request["speaker"],
                "mixture_number": index // 2 + 1,
                "duration_seconds": request["mixture"].shape[-1] / 16000,
                "metrics": metrics,
                "tracks": tracks,
            }
        )
        print(f"Rendered update {selection['step']}, case {index + 1}/{len(INDICES)}", flush=True)
    record = {
        "step": selection["step"],
        "indices": INDICES,
        "items": records,
        "manifest_sha256": manifest_hash,
        "checkpoint_sha256": digest,
        "source_tree_sha256": payload["provenance"]["source_tree_sha256"],
        "config_sha256": config.digest(),
        "created_at": time.time(),
        "epoch": selection["step"]
        / (config.training.max_optimizer_updates / config.training.epochs),
    }
    atomic_json(temporary / "record.json", record)
    temporary.rename(directory)
    del model, payload, corpus
    gc.collect()
    return record


def publish(output=OUTPUT):
    pointer = read(Path("artifacts/full-training-active.json"))
    if not pointer:
        raise ValueError("No full training run is registered")
    run = Path(pointer["run"])
    if not run.is_dir():
        raise ValueError("Reconnect the dataset SSD to refresh listening samples")
    selected = selections(run)
    models, errors = {}, []
    for role, choice in selected.items():
        atomic_json(
            output / "worker.json",
            {
                "status": "rendering",
                "role": role,
                "step": choice["step"],
                "checked_at": time.time(),
                "pid": os.getpid(),
            },
        )
        try:
            record = render_snapshot(run, pointer, choice, output)
            models[role] = {
                **record,
                "pending": choice["pending"],
                "validation": {k: v for k, v in choice["result"].items() if k != "rows"},
            }
        except (ValueError, OSError) as error:
            errors.append(f"{role.capitalize()}: {error}")
    result = {
        "models": models,
        "errors": errors,
        "updated_at": time.time(),
        "selection": "Both target voices in the first two development mixtures, fixed independently of scores.",
        "playback": "Matched playback adjusts each track toward RMS 0.08 with a 0.98 peak ceiling. No denoising or filtering. Raw output is also available.",
        "attribution": "LibriSpeech / OpenSLR 12, Panayotov et al. (2015), CC BY 4.0. Libri2Mix clean mixtures; processed derivatives.",
    }
    atomic_json(output / "index.json", result)
    atomic_json(
        output / "worker.json",
        {"status": "ready", "checked_at": time.time(), "pid": os.getpid(), "errors": errors},
    )
    # Bound disk use, retaining recent audio for open browser tabs for at least a day.
    retained = {
        Path(item["tracks"]["estimate"]["raw"]).parent.name
        for model in models.values()
        for item in model["items"]
    }
    archives = sorted(
        output.glob("step-*/record.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    for path in archives[4:]:
        if path.parent.name not in retained and time.time() - path.stat().st_mtime > 86400:
            shutil.rmtree(path.parent)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    CONTROL.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    os.nice(15)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with (CONTROL / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(CONTROL / "active.json", {"pid": os.getpid(), "started_at": time.time()})
        while True:
            try:
                publish()
            except Exception as error:
                atomic_json(
                    OUTPUT / "worker.json",
                    {
                        "status": "error",
                        "detail": str(error),
                        "checked_at": time.time(),
                        "pid": os.getpid(),
                    },
                )
                print(f"Listening refresh failed: {error}", flush=True)
                if args.once:
                    raise
            if args.once:
                break
            time.sleep(30)


if __name__ == "__main__":
    main()
