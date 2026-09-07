"""Epoch-based training and resumable evaluation for the independent reference baseline."""

import json
import math
import signal
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tse.config import ExperimentConfig
from tse.engine import save_checkpoint
from tse.librimix import LibriMixCorpus, epoch_order, request_seed
from tse.metrics import measure
from tse.model import choose_device, make_model
from tse.reference_model import EnrollmentFbank
from tse.utils import atomic_json, git_state, sha256, source_digest


def separation_score(estimate, target):
    """Unclamped zero-mean SI-SDR with the reference recipe's 1e-8 safeguards."""
    estimate = estimate.flatten(1)
    target = target.flatten(1)
    estimate = estimate - estimate.mean(-1, keepdim=True)
    target = target - target.mean(-1, keepdim=True)
    projection = (
        (estimate * target).sum(-1, keepdim=True)
        * target
        / (target.square().sum(-1, keepdim=True) + 1e-8)
    )
    return 10 * torch.log10(
        projection.square().sum(-1) / ((estimate - projection).square().sum(-1) + 1e-8) + 1e-8
    )


@contextmanager
def frozen_running_statistics(module):
    """Recompute a training forward without counting BatchNorm observations twice."""
    modules = [
        child for child in module.modules() if isinstance(child, nn.modules.batchnorm._BatchNorm)
    ]
    states = [child.track_running_stats for child in modules]
    try:
        for child in modules:
            child.track_running_stats = False
        yield
    finally:
        for child, tracking in zip(modules, states, strict=True):
            child.track_running_stats = tracking


def full_reference_features(requests, training):
    frontend = EnrollmentFbank().train(training)
    features = [frontend(request["reference"])[0] for request in requests]
    longest = max(feature.shape[-1] for feature in features)
    # Padding occurs after per-utterance CMN, matching the feature-batch convention.
    return torch.stack([F.pad(feature, (0, longest - feature.shape[-1])) for feature in features])


def backward_batch(model, requests, features, config, clear_mps_cache=True):
    """Exact full reference batch statistics with one separator microbatch at a time.

    Reference embeddings are recomputed once for their backward pass. This retains
    the full reference batch's BatchNorm behavior without keeping its activations
    alive during recurrent separation. Gradients and running buffers are tested
    against ordinary joint-batch backpropagation.
    """
    device = next(model.parameters()).device
    if device.type == "mps" and clear_mps_cache:
        torch.mps.empty_cache()
    features = features.to(device)
    with torch.no_grad():
        vectors = model.reference_encoder.encode_features(features)
    vectors = vectors.detach().requires_grad_()
    if device.type == "mps" and clear_mps_cache:
        torch.mps.empty_cache()
    labels = torch.tensor([r["label"] for r in requests], device=device)
    logits = model.speaker_head(vectors)
    classification = F.cross_entropy(logits, labels)
    (config.loss.speaker_classification_weight * classification).backward()
    total_loss = float(classification.detach()) * config.loss.speaker_classification_weight
    total_score = 0.0
    microbatch = config.training.microbatch_size
    for start in range(0, len(requests), microbatch):
        subset = requests[start : start + microbatch]
        mixture = torch.cat([r["mixture"] for r in subset]).to(device)
        target = torch.cat([r["target"] for r in subset]).to(device)
        prediction = model.extract(mixture, vectors[start : start + len(subset)])
        scores = separation_score(prediction, target)
        objective = -config.loss.separation_weight * scores.sum() / len(requests)
        if not torch.isfinite(objective):
            raise FloatingPointError("Reference separation objective became non-finite")
        objective.backward()
        total_loss += float(objective.detach())
        total_score += float(scores.detach().sum()) / len(requests)
    if vectors.grad is None or not torch.isfinite(vectors.grad).all():
        raise FloatingPointError("Reference conditioning received invalid gradients")
    if device.type == "mps" and clear_mps_cache:
        # Recurrent workspace caches otherwise compete with the full enrollment
        # CNN's recomputation, especially for longer reference utterances.
        torch.mps.empty_cache()
    with frozen_running_statistics(model.reference_encoder):
        recomputed = model.reference_encoder.encode_features(features)
        recomputed.backward(vectors.grad)
    return {
        "loss": total_loss,
        "si_sdr_db": total_score,
        "speaker_accuracy": float((logits.detach().argmax(-1) == labels).float().mean()),
    }


@torch.inference_mode()
def evaluate_reference(model, corpus, indices, fixed_training_seed=None, crop_samples=None):
    model.eval()
    device = next(model.parameters()).device
    rows = []
    for index in indices:
        if device.type == "mps":
            torch.mps.empty_cache()
        request = corpus.request(
            index,
            seed=request_seed(19 if fixed_training_seed is None else fixed_training_seed, 0, index),
            crop_samples=crop_samples,
        )
        prediction = model(request["mixture"].to(device), request["reference"].to(device))
        result = measure(
            prediction.cpu(), request["mixture"], request["target"], request["interferer"]
        )[0]
        result.update(case_id=request["case_id"], target_speaker=request["speaker"])
        rows.append(result)
    return {
        "cases": len(rows),
        "mean_si_sdr_db": float(np.mean([r["si_sdr_db"] for r in rows])),
        "mean_si_sdri_db": float(np.mean([r["si_sdri_db"] for r in rows])),
        "accuracy_si_sdri_above_1db": float(np.mean([r["si_sdri_db"] > 1 for r in rows])),
        "confusion_fraction": float(np.mean([r["confused"] for r in rows])),
        "rows": rows,
    }


def schedule_lr(config, step, total):
    ratio = config.training.minimum_learning_rate / config.training.learning_rate
    return config.training.learning_rate * ratio ** min(step / total, 1)


def average_final_epochs(paths, output):
    payloads = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
    base = payloads[-1]
    for payload in payloads:
        if (
            payload["config"] != base["config"]
            or payload["manifest_sha256"] != base["manifest_sha256"]
            or payload["provenance"]["source_tree_sha256"]
            != base["provenance"]["source_tree_sha256"]
        ):
            raise ValueError("Final averaging requires identical recipe, implementation and data")
    state = {}
    for key, value in base["model"].items():
        # Average floating weights and BN buffers; use the final BN observation counter.
        state[key] = (
            torch.stack([p["model"][key].double() for p in payloads]).mean(0).to(value.dtype)
            if value.is_floating_point()
            else value.clone()
        )
    result = {k: v for k, v in base.items() if k not in {"optimizer", "torch_rng", "mps_rng"}}
    result["model"] = state
    result["provenance"] = {
        **base["provenance"],
        "checkpoint_average": {
            "method": "Final five epoch floating states averaged; final integer BatchNorm counters retained",
            "sources": [
                {"path": str(path), "checkpoint_sha256": sha256(path), "epoch": p["epoch"]}
                for path, p in zip(paths, payloads, strict=True)
            ],
        },
    }
    save_checkpoint(output, result)


def train_reference(
    config_path,
    root,
    manifest,
    run,
    device_name="mps",
    resume=False,
    fixed_indices=None,
    stop_after_updates=None,
):
    config = ExperimentConfig.load(config_path)
    if (
        config.model.family != "reference_bsrnn"
        or config.data.protocol != "libri2mix-16k-min-clean"
    ):
        raise ValueError("Use the explicit reference model and data recipe")
    if (
        config.training.optimizer != "adam"
        or config.training.learning_rate_schedule != "exponential"
        or config.loss.waveform_weight
        or config.loss.spectral_weight
    ):
        raise ValueError(
            "Reference training requires Adam, exponential decay and no extra reconstruction losses"
        )
    run = Path(run)
    if run.exists() and any(run.iterdir()) and not resume:
        raise FileExistsError("Use --resume for an existing run")
    run.mkdir(parents=True, exist_ok=True)
    train = LibriMixCorpus(root, manifest, "train")
    dev = LibriMixCorpus(root, manifest, "dev")
    batch = config.training.microbatch_size * config.training.gradient_accumulation
    count = len(fixed_indices) if fixed_indices is not None else len(train)
    epoch_steps = math.ceil(count / batch)
    total_steps = config.training.epochs * epoch_steps
    if fixed_indices is None and config.training.max_optimizer_updates != total_steps:
        raise ValueError("Update budget must equal full data passes, including both targets")
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
        "train_speakers": train.speakers,
        "device": str(device),
        "initialization": None,
        "parameters": sum(p.numel() for p in model.parameters()),
        "experiment_type": "tiny_set_diagnostic"
        if fixed_indices is not None
        else "official_recipe_training",
        "fixed_indices": fixed_indices,
        "reference_batch": batch,
        "separator_microbatch": config.training.microbatch_size,
        "backward": "Full reference batch recomputation with single BatchNorm running-state update",
    }
    step, best, previous_elapsed = 0, -float("inf"), 0.0
    if resume:
        saved = torch.load(run / "latest.pt", map_location="cpu", weights_only=True)
        if (
            saved["config"] != config.model_dump()
            or saved["manifest_sha256"] != train.manifest_hash
            or saved["provenance"]["source_tree_sha256"] != provenance["source_tree_sha256"]
            or saved["provenance"]["fixed_indices"] != fixed_indices
        ):
            raise ValueError("Resume requires the exact saved recipe, source code and data")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value) and key != "step":
                    state[key] = value.to(device)
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "mps":
            torch.mps.set_rng_state(saved["mps_rng"])
        step, best = saved["step"], saved["best_score"]
        previous_elapsed = saved["elapsed_seconds"]
        provenance = saved["provenance"]
    atomic_json(run / "config.json", config.model_dump())
    atomic_json(run / "provenance.json", provenance)
    started = time.monotonic()
    interrupted = False

    def stop(signum, frame):
        nonlocal interrupted
        interrupted = True

    previous_handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}

    def save(name):
        save_checkpoint(
            run / name,
            {
                "format_version": 1,
                "config": config.model_dump(),
                "model": model.state_dict(),
                "speaker_classes": len(train.speakers),
                "manifest_sha256": train.manifest_hash,
                "dev_cases_sha256": train.manifest_hash,
                "provenance": provenance,
                "step": step,
                "epoch": step // epoch_steps,
                "best_score": best,
                "optimizer": optimizer.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "mps_rng": torch.mps.get_rng_state() if device.type == "mps" else None,
                "elapsed_seconds": previous_elapsed + time.monotonic() - started,
            },
        )

    def emit(entry):
        with (run / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)

    try:
        while step < total_steps and not interrupted:
            epoch = step // epoch_steps
            order = epoch_order(count, config.seed, epoch)
            for batch_index in range(step % epoch_steps, epoch_steps):
                indices = order[batch_index * batch : (batch_index + 1) * batch]
                if fixed_indices is not None:
                    indices = [fixed_indices[i] for i in indices]
                requests = [
                    train.request(
                        i,
                        request_seed(config.seed, 0 if fixed_indices else epoch, i),
                        int(config.audio.crop_seconds * 16000),
                    )
                    for i in indices
                ]
                model.train()
                features = full_reference_features(requests, training=True)
                lr = schedule_lr(config, step, total_steps)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                metrics = backward_batch(model, requests, features, config)
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.training.gradient_clip_norm, error_if_nonfinite=True
                )
                optimizer.step()
                step += 1
                elapsed = previous_elapsed + time.monotonic() - started
                if step == 1 or step % 10 == 0:
                    entry = {
                        "event": "train",
                        "step": step,
                        "epoch": step / epoch_steps,
                        "target_requests_seen": min(step * batch, count * config.training.epochs),
                        "learning_rate": lr,
                        "gradient_norm": float(norm),
                        "elapsed_seconds": elapsed,
                        **metrics,
                    }
                    emit(entry)
                    atomic_json(
                        run / "status.json",
                        {
                            "status": "training",
                            "total_updates": total_steps,
                            "total_epochs": config.training.epochs,
                            **entry,
                        },
                    )
                end_epoch = step % epoch_steps == 0
                validate_now = end_epoch and (
                    fixed_indices is None
                    or step % config.training.validation_interval_updates == 0
                    or step == total_steps
                )
                if validate_now:
                    eval_corpus = train if fixed_indices is not None else dev
                    # Full development evaluation is mandatory at the end of the full recipe.
                    indices_eval = (
                        fixed_indices if fixed_indices is not None else list(range(len(dev)))
                    )
                    result = evaluate_reference(
                        model,
                        eval_corpus,
                        indices_eval,
                        config.seed if fixed_indices is not None else None,
                        int(config.audio.crop_seconds * 16000)
                        if fixed_indices is not None
                        else None,
                    )
                    best = max(best, result["mean_si_sdr_db"])
                    emit(
                        {
                            "event": "validation",
                            "step": step,
                            "epoch": step // epoch_steps,
                            "split": "train_diagnostic" if fixed_indices is not None else "dev",
                            **{k: v for k, v in result.items() if k != "rows"},
                        }
                    )
                    atomic_json(run / "latest-validation.json", result)
                    if result["mean_si_sdr_db"] == best:
                        save("best.pt")
                if end_epoch and step // epoch_steps > config.training.epochs - 5:
                    save(f"epoch-{step // epoch_steps:03d}.pt")
                if (
                    step % config.training.checkpoint_interval_updates == 0
                    or end_epoch
                    or interrupted
                ):
                    save("latest.pt")
                if interrupted or (stop_after_updates is not None and step >= stop_after_updates):
                    interrupted = step < total_steps
                    break
        save("latest.pt")
        if not interrupted:
            if config.training.epochs >= 5:
                paths = [
                    run / f"epoch-{epoch:03d}.pt"
                    for epoch in range(config.training.epochs - 4, config.training.epochs + 1)
                ]
                average_final_epochs(paths, run / "average-last5.pt")
            atomic_json(
                run / "summary.json",
                {
                    "status": "complete",
                    "steps": step,
                    "epochs": config.training.epochs,
                    "best_dev_si_sdr_db": best,
                },
            )
        atomic_json(
            run / "status.json",
            {
                "status": "paused" if interrupted else "complete",
                "step": step,
                "epoch": step / epoch_steps,
                "total_updates": total_steps,
            },
        )
    except BaseException as error:
        atomic_json(
            run / "status.json",
            {
                "status": "failed",
                "step": step,
                "error": type(error).__name__,
                "detail": str(error),
                "resume": "Use the last atomically saved complete checkpoint; partial update state is discarded.",
            },
        )
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    return model
