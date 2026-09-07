"""Full Libri2Mix passes with bounded sessions and recoverable development evaluation."""

import copy
import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from tse.config import ExperimentConfig
from tse.engine import save_checkpoint
from tse.librimix import LibriMixCorpus, epoch_order, request_seed
from tse.metrics import measure
from tse.model import choose_device, make_model
from tse.reference_training import backward_batch, full_reference_features, schedule_lr
from tse.utils import atomic_json, git_state, sha256, source_digest


@contextmanager
def run_lock(run):
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    with (run / "trainer.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("This run already has an active trainer") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def development_plan(train, dev, count):
    """Freeze a speaker-balanced monitor without using model scores or test audio."""
    if set(train.speakers) & set(dev.speakers):
        raise ValueError("Training and development speakers overlap")
    pools = defaultdict(list)
    for index in range(len(dev)):
        row = dev.rows[index // 2]
        target = row["sources"][index % 2]
        reference = dev.known_files[dev.enrollments[f"{row['id']}:{index % 2}"]]
        if (
            reference["speaker"] != target["speaker"]
            or reference["utterance"] == target["utterance"]
        ):
            raise ValueError(
                "Development enrollment must use a different utterance by the target speaker"
            )
        pools[row["sources"][index % 2]["speaker"]].append(index)
    if count < len(pools) or count % len(pools):
        raise ValueError("Monitor size must be a multiple of the development speaker count")
    selected = {}
    for speaker, indices in sorted(pools.items()):
        indices.sort(key=lambda i: request_seed(197, 0, i))
        selected[speaker] = indices[: count // len(pools)]
        if len(selected[speaker]) != count // len(pools):
            raise ValueError("Not enough development requests for balanced monitoring")
    # Interleave voices so the first listening examples cover different speakers.
    indices = [selected[s][i] for i in range(count // len(pools)) for s in sorted(pools)]
    return {"indices": indices, "speakers": sorted(pools), "seed": 197, "split": "dev"}


def cropped_request(corpus, index, seed, reference_samples, mixture_samples=None):
    request = corpus.request(index, seed, mixture_samples)
    reference = request["reference"]
    length = reference.shape[-1]
    if length < reference_samples:
        reference = reference.repeat(1, 1, math.ceil(reference_samples / length))
    start = int(
        np.random.default_rng(request_seed(seed, 1, index)).integers(
            reference.shape[-1] - reference_samples + 1
        )
    )
    request["reference"] = reference[..., start : start + reference_samples].clone()
    return request


@torch.inference_mode()
def resumable_validation(
    model,
    corpus,
    indices,
    reference_samples,
    path,
    identity,
    should_pause,
    progress=lambda done, total: None,
):
    """Persist only complete cases; never promote a score from an incomplete suite."""
    state = json.loads(path.read_text()) if path.exists() else {}
    if state.get("identity") != identity:
        state = {"identity": identity, "rows": []}
    rows = state["rows"]
    if [row["index"] for row in rows] != indices[: len(rows)]:
        raise ValueError("Development cursor does not match the frozen request order")
    progress(len(rows), len(indices))
    if len(rows) > len(indices):
        raise ValueError("Invalid development cursor")
    model.eval()
    device = next(model.parameters()).device
    for index in indices[len(rows) :]:
        if should_pause():
            atomic_json(path, state)
            return None
        if device.type == "mps":
            torch.mps.empty_cache()
        request = cropped_request(corpus, index, request_seed(19, 0, index), reference_samples)
        prediction = model(request["mixture"].to(device), request["reference"].to(device))
        row = measure(
            prediction.cpu(), request["mixture"], request["target"], request["interferer"]
        )[0]
        row.update(case_id=request["case_id"], target_speaker=request["speaker"], index=index)
        rows.append(row)
        if len(rows) % 25 == 0:
            atomic_json(path, state)
            progress(len(rows), len(indices))
    atomic_json(path, state)
    return {
        "cases": len(rows),
        "mean_si_sdr_db": float(np.mean([r["si_sdr_db"] for r in rows])),
        "mean_si_sdri_db": float(np.mean([r["si_sdri_db"] for r in rows])),
        "positive_improvement_fraction": float(np.mean([r["si_sdri_db"] > 0 for r in rows])),
        "confusion_fraction": float(np.mean([r["confused"] for r in rows])),
        "rows": rows,
    }


def train_full(
    config_path,
    root,
    manifest,
    run,
    device_name="mps",
    resume=False,
    minutes=480,
    stop_after_updates=None,
):
    if not 0 < minutes <= 480:
        raise ValueError("Session duration must be between zero and 480 minutes")
    run = Path(run)
    with run_lock(run):
        return _train(
            config_path, root, manifest, run, device_name, resume, minutes, stop_after_updates
        )


def _train(config_path, root, manifest, run, device_name, resume, minutes, stop_after_updates):
    started = time.monotonic()
    config = ExperimentConfig.load(config_path)
    if (
        config.model.family != "reference_bsrnn"
        or config.model.weights != "random_initialization"
        or config.data.protocol != "libri2mix-16k-min-clean"
        or config.audio.reference_mode != "crop"
        or config.training.optimizer != "adam"
        or config.training.learning_rate_schedule != "exponential"
        or config.loss.waveform_weight
        or config.loss.spectral_weight
    ):
        raise ValueError("Expected the declared efficient full-data recipe")
    if (run / "latest.pt").exists() and not resume:
        raise FileExistsError("Existing training is preserved; use --resume")
    train, dev = [LibriMixCorpus(root, manifest, split) for split in ("train", "dev")]
    plan = development_plan(train, dev, config.evaluation.development_cases_target)
    plan_hash = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    batch = config.training.microbatch_size * config.training.gradient_accumulation
    epoch_steps = math.ceil(len(train) / batch)
    total = epoch_steps * config.training.epochs
    if total != config.training.max_optimizer_updates:
        raise ValueError("The update budget must contain exactly the declared full epochs")
    if shutil.disk_usage(run).free < config.resources.minimum_free_disk_gib * 1024**3:
        raise OSError("Not enough free disk space for training checkpoints")
    device = choose_device(device_name)
    torch.set_num_threads(2)
    torch.manual_seed(config.seed)
    if device.type == "mps":
        torch.mps.manual_seed(config.seed)
        torch.mps.set_per_process_memory_fraction(config.runtime.mps_memory_fraction)
    model = make_model(config.model, len(train.speakers)).to(device)
    model.activation_checkpointing = config.training.activation_checkpointing
    model.reference_encoder.activation_checkpointing = config.training.activation_checkpointing
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    provenance = {
        **git_state(),
        "source_tree_sha256": source_digest(),
        "torch_version": str(torch.__version__),
        "device": str(device),
        "evaluation_device": "cpu",
        "initialization": "fresh_random_weights",
        "train_speakers": train.speakers,
        "development_speakers": dev.speakers,
        "parameters": sum(p.numel() for p in model.parameters()),
        "experiment_type": "full_data_efficient_bsrnn",
        "monitor_sha256": plan_hash,
        "reference_seconds": config.audio.reference_seconds,
    }
    step, elapsed_before = 0, 0.0
    best = {"monitor": None, "full": None}
    validated = {"monitor": 0, "full": 0}
    if resume:
        saved = torch.load(run / "latest.pt", map_location="cpu", weights_only=True)
        if (
            saved["config"] != config.model_dump()
            or saved["manifest_sha256"] != train.manifest_hash
            or any(
                saved["provenance"][k] != provenance[k]
                for k in ("source_tree_sha256", "torch_version", "device", "monitor_sha256")
            )
        ):
            raise ValueError("Resume requires the same recipe, data, source, PyTorch and device")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value) and key != "step":
                    state[key] = value.to(device)
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "mps":
            torch.mps.set_rng_state(saved["mps_rng"])
        step, elapsed_before = saved["step"], saved["elapsed_seconds"]
        best, validated, provenance = (
            saved["best_scores"],
            saved["validated_steps"],
            saved["provenance"],
        )
        del saved
    atomic_json(run / "config.json", config.model_dump())
    atomic_json(run / "provenance.json", provenance)
    atomic_json(run / "monitor-plan.json", plan)
    interrupted = False
    checkpoint_step = step if resume else None
    pause_reason = None

    def stop(signum, frame):
        nonlocal interrupted
        interrupted = True

    def should_pause():
        nonlocal pause_reason
        if interrupted or (run / "pause.request").exists():
            pause_reason = "requested"
        elif time.monotonic() - started >= minutes * 60:
            pause_reason = "session_time_limit"
        elif stop_after_updates is not None and step >= stop_after_updates:
            pause_reason = "requested_update_limit"
        return pause_reason is not None

    def status(state, **extra):
        atomic_json(
            run / "status.json",
            {
                "status": state,
                "pid": os.getpid(),
                "step": step,
                "checkpoint_step": checkpoint_step,
                "epoch": step / epoch_steps,
                "updates_per_epoch": epoch_steps,
                "total_epochs": config.training.epochs,
                "total_updates": total,
                "train_mixtures": len(train.rows),
                "train_speakers": len(train.speakers),
                "development_speakers": len(dev.speakers),
                "session_minutes": minutes,
                "session_elapsed_seconds": time.monotonic() - started,
                "elapsed_seconds": elapsed_before + time.monotonic() - started,
                "next_learning_rate": schedule_lr(config, step, total),
                "pause_reason": pause_reason,
                **extra,
            },
        )

    def save(name="latest.pt"):
        nonlocal checkpoint_step
        save_checkpoint(
            run / name,
            {
                "format_version": 1,
                "config": config.model_dump(),
                "model": model.state_dict(),
                "speaker_classes": len(train.speakers),
                "manifest_sha256": train.manifest_hash,
                "provenance": provenance,
                "step": step,
                "epoch": step // epoch_steps,
                "optimizer": optimizer.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "mps_rng": torch.mps.get_rng_state() if device.type == "mps" else None,
                "elapsed_seconds": elapsed_before + time.monotonic() - started,
                "best_scores": best.copy(),
                "validated_steps": validated.copy(),
            },
        )
        if name == "latest.pt":
            checkpoint_step = step

    def emit(entry):
        with (run / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(entry, allow_nan=False) + "\n")
        print(json.dumps(entry, allow_nan=False), flush=True)

    handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        status("starting")
        if not resume:
            save()
            # Persist the initial state separately to demonstrate fresh initialization.
            shutil.copy2(run / "latest.pt", run / "initial.pt")
        emit({"event": "session_start", "step": step, "resume": resume, "minutes": minutes})
        order_epoch, order = None, None
        while True:
            if should_pause():
                save()
                status("paused")
                break
            due = []
            if (
                step % config.training.validation_interval_updates == 0
                and validated["monitor"] != step
            ):
                due.append("monitor")
            if step > 0 and step % epoch_steps == 0 and validated["full"] != step:
                due.append("full")
            for kind in due:
                save()
                indices = plan["indices"] if kind == "monitor" else list(range(len(dev)))
                identity = {
                    "step": step,
                    "kind": kind,
                    "plan_sha256": plan_hash,
                    "manifest_sha256": train.manifest_hash,
                }
                # The step/config/source identity stays fixed when pause checkpoints update elapsed time.
                identity["source_sha256"] = provenance["source_tree_sha256"]
                identity["config_sha256"] = config.digest()
                status(
                    "validating",
                    validation_kind=kind,
                    validation_done=0,
                    validation_total=len(indices),
                )
                # Variable-length MPS recurrent workspaces can exceed the Mac memory cap.
                # A CPU clone preserves whole-recording evaluation without temporal chunking.
                model.eval()
                evaluation_model = copy.deepcopy(model).cpu() if device.type == "mps" else model
                if device.type == "mps":
                    torch.mps.empty_cache()
                result = resumable_validation(
                    evaluation_model,
                    dev,
                    indices,
                    int(config.audio.reference_seconds * 16000),
                    run / f"{kind}-validation-state.json",
                    identity,
                    should_pause,
                    lambda done, total, kind=kind: status(
                        "validating",
                        validation_kind=kind,
                        validation_done=done,
                        validation_total=total,
                    ),
                )
                del evaluation_model
                if result is None:
                    break
                result.update(
                    step=step,
                    epoch=step / epoch_steps,
                    kind=kind,
                    evaluation_device="cpu",
                    manifest_sha256=train.manifest_hash,
                    monitor_sha256=plan_hash,
                )
                score = result["mean_si_sdri_db"]
                improved = best[kind] is None or score > best[kind]
                validated[kind] = step
                if improved:
                    best[kind] = score
                    name = "best.pt" if kind == "monitor" else "best-full.pt"
                    save(name)
                    result["checkpoint_sha256"] = sha256(run / name)
                    atomic_json(run / f"best-{kind}-validation.json", result)
                atomic_json(run / f"latest-{kind}-validation.json", result)
                emit({"event": "validation", **{k: v for k, v in result.items() if k != "rows"}})
                save()
            if should_pause():
                continue
            if step >= total:
                save()
                status("complete")
                break
            epoch = step // epoch_steps
            if order_epoch != epoch:
                order_epoch, order = epoch, epoch_order(len(train), config.seed, epoch)
            begin = (step % epoch_steps) * batch
            tick = time.monotonic()
            requests = [
                cropped_request(
                    train,
                    i,
                    request_seed(config.seed, epoch, i),
                    int(config.audio.reference_seconds * 16000),
                    int(config.audio.crop_seconds * 16000),
                )
                for i in order[begin : begin + batch]
            ]
            model.train()
            features = full_reference_features(requests, training=True)
            lr = schedule_lr(config, step, total)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            metrics = backward_batch(model, requests, features, config, clear_mps_cache=False)
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.training.gradient_clip_norm, error_if_nonfinite=True
            )
            optimizer.step()
            step += 1
            if step == 1 or step % 10 == 0:
                emit(
                    {
                        "event": "train",
                        "step": step,
                        "epoch": step / epoch_steps,
                        "learning_rate": lr,
                        "gradient_norm": float(norm),
                        "update_seconds": time.monotonic() - tick,
                        **metrics,
                    }
                )
                status("training", update_seconds=time.monotonic() - tick)
            if step % config.training.checkpoint_interval_updates == 0 or step % epoch_steps == 0:
                save()
            if step % epoch_steps == 0 and step // epoch_steps > config.training.epochs - 5:
                # Model-only snapshots bound disk use; latest.pt retains the full optimizer.
                save_checkpoint(
                    run / f"epoch-{step // epoch_steps:03d}.pt",
                    {
                        "config": config.model_dump(),
                        "model": model.state_dict(),
                        "step": step,
                        "epoch": step // epoch_steps,
                        "speaker_classes": len(train.speakers),
                        "manifest_sha256": train.manifest_hash,
                        "provenance": provenance,
                    },
                )
    except BaseException as error:
        # Do not save a possibly partial optimizer update over the last complete state.
        status("failed", detail=f"{type(error).__name__}: {error}")
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return model
