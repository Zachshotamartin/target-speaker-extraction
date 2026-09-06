"""Training, resumable artifacts, evaluation and device profiling."""

from __future__ import annotations

import json
import os
import resource
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from tse.config import ExperimentConfig
from tse.data import CONDITIONS, SpeechCorpus, load_cases
from tse.metrics import measure, si_sdr, summarize, waveform_loss
from tse.model import TargetExtractor, choose_device
from tse.utils import atomic_json, git_state, sha256, source_digest


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as output:
        torch.save(payload, output)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def load_model(path: Path, device_name: str = "cpu") -> tuple[TargetExtractor, dict]:
    device = choose_device(device_name)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    config = ExperimentConfig.model_validate(payload["config"])
    model = TargetExtractor(config.model, payload.get("speaker_classes", 0))
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    return model, payload


@torch.no_grad()
def evaluate_model(
    model: TargetExtractor,
    corpus: SpeechCorpus,
    cases: list[dict],
    condition: str = "clean",
    batch_size: int = 4,
    margin_db: float = 3,
) -> list[dict]:
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    rows = []
    try:
        for start in range(0, len(cases), batch_size):
            current = cases[start : start + batch_size]
            batch = corpus.batch(current, device, [condition] * len(current))
            output = model(batch["mixture"], batch["reference"])
            metrics = measure(
                output, batch["mixture"], batch["target"], batch["interferer"], margin_db
            )
            for case, speaker, result in zip(current, batch["speakers"], metrics, strict=True):
                rows.append(
                    {
                        "case_id": case["case_id"],
                        "target_speaker": speaker,
                        "condition": condition,
                        **result,
                    }
                )
    finally:
        model.train(was_training)
    return rows


def train(
    config: ExperimentConfig,
    root: Path,
    manifest: Path,
    dev_cases_path: Path,
    run_dir: Path,
    device_name: str | None = None,
    resume: bool = False,
    fixed_cases_path: Path | None = None,
) -> dict:
    device = choose_device(device_name or config.runtime.preferred_device)
    run_dir.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(run_dir).free < config.resources.minimum_free_disk_gib * 1024**3:
        raise OSError("Free disk is below the configured reserve")
    if (run_dir / "latest.pt").exists() and not resume:
        raise ValueError("Run already exists; choose another directory or use --resume")
    if resume and not (run_dir / "latest.pt").exists():
        raise ValueError("Cannot resume without latest.pt")
    torch.manual_seed(config.seed)
    corpus = SpeechCorpus(
        root, manifest, "train", config.audio.crop_seconds, config.audio.reference_seconds
    )
    development = SpeechCorpus(
        root,
        manifest,
        "dev",
        config.evaluation.mixture_seconds,
        config.evaluation.reference_seconds,
    )
    dev_cases = load_cases(dev_cases_path, development)
    fixed_cases = load_cases(fixed_cases_path, corpus) if fixed_cases_path else None
    classes = len(corpus.speakers) if config.loss.speaker_classification_weight else 0
    model = TargetExtractor(config.model, classes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    best_score = -float("inf")
    start_step = 0
    if resume:
        payload = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=True)
        previous = ExperimentConfig.model_validate(payload["config"])
        old, new = previous.model_dump(), config.model_dump()
        for key in (old, new):
            key["training"].pop("max_optimizer_updates")
        if (
            old != new
            or payload["manifest_sha256"] != corpus.manifest_hash
            or payload["dev_cases_sha256"] != sha256(dev_cases_path)
        ):
            raise ValueError("Resume configuration or data differs from the saved run")
        if payload["provenance"].get("fixed_cases_sha256") != (
            sha256(fixed_cases_path) if fixed_cases_path else None
        ):
            raise ValueError("Resume fixed-case protocol differs from the saved run")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        # Optimizer moments loaded from CPU need the model's active device.
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value) and key != "step":
                    state[key] = value.to(device)
        torch.set_rng_state(payload["torch_rng"])
        if device.type == "mps" and payload.get("mps_rng") is not None:
            torch.mps.set_rng_state(payload["mps_rng"])
        start_step = payload["step"]
        best_score = payload["best_score"]
    provenance = {
        **git_state(),
        "source_tree_sha256": source_digest(),
        "torch_version": str(torch.__version__),
        "manifest_sha256": corpus.manifest_hash,
        "dev_cases_sha256": sha256(dev_cases_path),
        "fixed_cases_sha256": sha256(fixed_cases_path) if fixed_cases_path else None,
        "config_sha256": config.digest(),
        "device": str(device),
        "parameters": sum(p.numel() for p in model.parameters()),
        "train_speakers": corpus.speakers,
        "experiment_type": "tiny_set_diagnostic" if fixed_cases else "held_out_speaker_training",
    }
    atomic_json(run_dir / "provenance.json", provenance)
    atomic_json(run_dir / "config.json", config.model_dump())
    started = time.monotonic()
    last_step = start_step
    microbatch = config.training.microbatch_size
    accumulation = config.training.gradient_accumulation

    def checkpoint(step: int) -> dict:
        return {
            "format_version": 1,
            "config": config.model_dump(),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "best_score": best_score,
            "speaker_classes": classes,
            "manifest_sha256": corpus.manifest_hash,
            "dev_cases_sha256": sha256(dev_cases_path),
            "torch_rng": torch.get_rng_state(),
            "mps_rng": torch.mps.get_rng_state() if device.type == "mps" else None,
            "provenance": provenance,
        }

    print(
        json.dumps(
            {
                "event": "training_start",
                "run": run_dir.name,
                **{k: provenance[k] for k in ("device", "parameters")},
                "start_step": start_step,
            }
        ),
        flush=True,
    )
    try:
        for step in range(start_step, config.training.max_optimizer_updates):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for micro in range(accumulation):
                start_index = (step * accumulation + micro) * microbatch
                if fixed_cases:
                    cases = [
                        fixed_cases[(start_index + j) % len(fixed_cases)] for j in range(microbatch)
                    ]
                else:
                    cases = [
                        corpus.make_case(config.seed * 10000000 + start_index + j)
                        for j in range(microbatch)
                    ]
                conditions = [
                    CONDITIONS[(config.seed + start_index + j) % len(CONDITIONS)]
                    if config.augmentation.reference_enabled
                    else "clean"
                    for j in range(microbatch)
                ]
                batch = corpus.batch(cases, device, conditions)
                embedding = model.reference_encoder(batch["reference"])
                output = model.extract(batch["mixture"], embedding)
                loss = -si_sdr(output, batch["target"], epsilon=config.loss.epsilon).mean()
                loss = loss + config.loss.waveform_weight * waveform_loss(output, batch["target"])
                if classes:
                    loss = loss + config.loss.speaker_classification_weight * F.cross_entropy(
                        model.speaker_head(embedding), batch["labels"]
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Training produced a non-finite loss")
                (loss / accumulation).backward()
                losses.append(float(loss.detach()))
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.training.gradient_clip_norm, error_if_nonfinite=True
            )
            optimizer.step()
            last_step = step + 1
            if last_step == 1 or last_step % 25 == 0:
                entry = {
                    "event": "train",
                    "step": last_step,
                    "loss": float(np.mean(losses)),
                    "gradient_norm": float(gradient_norm),
                    "elapsed_seconds": round(time.monotonic() - started, 2),
                }
                with (run_dir / "metrics.jsonl").open("a") as log:
                    log.write(json.dumps(entry) + "\n")
                print(json.dumps(entry), flush=True)
            if (
                last_step % config.training.validation_interval_updates == 0
                or last_step == config.training.max_optimizer_updates
            ):
                if fixed_cases:
                    rows = evaluate_model(model, corpus, fixed_cases, batch_size=microbatch)
                else:
                    rows = evaluate_model(model, development, dev_cases, batch_size=microbatch)
                score = float(np.mean([r["si_sdri_db"] for r in rows]))
                improved = score > best_score
                if improved:
                    best_score = score
                state = checkpoint(last_step)
                save_checkpoint(run_dir / "latest.pt", state)
                if improved:
                    save_checkpoint(run_dir / "best.pt", state)
                validation = {
                    "event": "validation",
                    "step": last_step,
                    "mean_si_sdri_db": score,
                    "confusion_fraction": float(np.mean([r["confused"] for r in rows])),
                    "best_score": best_score,
                    "selection_split": "train_diagnostic" if fixed_cases else "dev",
                }
                with (run_dir / "metrics.jsonl").open("a") as log:
                    log.write(json.dumps(validation) + "\n")
                atomic_json(
                    run_dir / "latest_validation.json", {"summary": validation, "rows": rows}
                )
                print(json.dumps(validation), flush=True)
    except KeyboardInterrupt:
        # Incomplete gradients are discarded; resuming regenerates the next full step.
        if last_step:
            save_checkpoint(run_dir / "latest.pt", checkpoint(last_step))
        print(json.dumps({"event": "interrupted", "last_complete_step": last_step}), flush=True)
        raise
    result = {
        "status": "complete",
        "steps": last_step,
        "best_development_si_sdri_db": best_score,
        "elapsed_seconds": time.monotonic() - started,
        "best_checkpoint_sha256": sha256(run_dir / "best.pt"),
        "provenance": provenance,
    }
    atomic_json(run_dir / "summary.json", result)
    return result


def evaluate(
    checkpoint_path: Path,
    root: Path,
    manifest: Path,
    cases_path: Path,
    output: Path,
    device: str,
    conditions: list[str] | None = None,
) -> dict:
    model, payload = load_model(checkpoint_path, device)
    config = ExperimentConfig.model_validate(payload["config"])
    protocol = json.loads(cases_path.read_text())
    corpus = SpeechCorpus(
        root,
        manifest,
        protocol["split"],
        config.evaluation.mixture_seconds,
        config.evaluation.reference_seconds,
    )
    cases = load_cases(cases_path, corpus)
    checkpoint_hash = sha256(checkpoint_path)
    rows = []
    summaries = {}
    started = time.monotonic()
    for condition in conditions or list(CONDITIONS):
        current = evaluate_model(
            model,
            corpus,
            cases,
            condition,
            config.training.microbatch_size,
            config.evaluation.confusion_margin_db,
        )
        rows.extend(current)
        summaries[condition] = summarize(
            current, config.evaluation.bootstrap_replicates, config.seed
        )
        print(
            json.dumps(
                {
                    "event": "evaluated",
                    "condition": condition,
                    "mean_si_sdri_db": summaries[condition]["mean_si_sdri_db"],
                }
            ),
            flush=True,
        )
    report = {
        "schema_version": 1,
        "protocol": protocol["protocol"],
        "split": protocol["split"],
        "checkpoint_sha256": checkpoint_hash,
        "case_manifest_sha256": sha256(cases_path),
        "source_manifest_sha256": sha256(manifest),
        "training_step": payload["step"],
        "config": config.model_dump(),
        "training_provenance": payload["provenance"],
        "evaluation_provenance": {**git_state(), "source_tree_sha256": source_digest()},
        "elapsed_seconds": time.monotonic() - started,
        "summaries": summaries,
        "rows": rows,
    }
    atomic_json(output, report)
    return report


def benchmark(config: ExperimentConfig, device_name: str, output: Path) -> dict:
    device = choose_device(device_name)
    torch.manual_seed(config.seed)
    model = TargetExtractor(config.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    mixture = torch.randn(
        config.training.microbatch_size, 1, int(config.audio.crop_seconds * 16000), device=device
    )
    reference = torch.randn(
        config.training.microbatch_size,
        1,
        int(config.audio.reference_seconds * 16000),
        device=device,
    )
    timings = []
    for index in range(
        config.runtime.benchmark_warmup_steps + config.runtime.benchmark_measured_steps
    ):
        synchronize(device)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        result = model(mixture, reference)
        loss = -si_sdr(result, mixture).mean()
        loss.backward()
        optimizer.step()
        synchronize(device)
        elapsed = time.perf_counter() - started
        if index >= config.runtime.benchmark_warmup_steps:
            timings.append(elapsed)
    result = {
        "device": device_name,
        "torch_version": str(torch.__version__),
        "parameters": sum(p.numel() for p in model.parameters()),
        "warmup_steps": config.runtime.benchmark_warmup_steps,
        "measured_steps": len(timings),
        "median_microstep_seconds": float(np.median(timings)),
        "p95_microstep_seconds": float(np.percentile(timings, 95)),
        "estimated_1000_updates_seconds": float(
            np.median(timings) * config.training.gradient_accumulation * 1000
        ),
        "rss_peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024**3 if os.uname().sysname == "Darwin" else 1024**2),
        "config": config.model_dump(),
        "note": "Random-tensor operator/training benchmark. Does not measure data loading or model quality.",
    }
    if device.type == "mps":
        result["mps_allocated_gib"] = torch.mps.current_allocated_memory() / 1024**3
        result["mps_driver_gib"] = torch.mps.driver_allocated_memory() / 1024**3
    atomic_json(output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "config"}), flush=True)
    return result
