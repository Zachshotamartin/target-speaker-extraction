"""Time-limited adaptation for a small, explicitly known-voice demonstration."""

import html
import json
import math
import signal
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from tse.concept_data import ConceptCorpus
from tse.config import ExperimentConfig
from tse.engine import save_checkpoint
from tse.metrics import measure
from tse.model import choose_device, make_model
from tse.reference_training import backward_batch, full_reference_features
from tse.utils import atomic_json, git_state, sha256, source_digest


class SessionExpired(Exception):
    pass


def development_goal(result):
    return (
        result["mean_si_sdri_db"] >= 6
        and result["p25_si_sdri_db"] >= 3
        and result["positive_improvement_fraction"] >= 0.9
        and result["correct_source_fraction"] >= 0.9
    )


def initialize_concept(config, checkpoint, expected_hash, labels, device):
    if sha256(checkpoint) != expected_hash:
        raise ValueError("Initialization changed after reserving evaluation audio")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    original = ExperimentConfig.model_validate(saved["config"])
    before, after = original.model.model_dump(), config.model.model_dump()
    for specification in (before, after):
        specification.pop("weights")
        specification.pop("band_blocks")
    if before != after or not 1 <= config.model.band_blocks <= original.model.band_blocks:
        raise ValueError("Concept transfer only removes trailing separator blocks")
    if saved["provenance"]["train_speakers"] != labels:
        raise ValueError("Classifier label identities changed")
    model = make_model(config.model, len(labels))
    retained = model.state_dict()
    model.load_state_dict({key: saved["model"][key] for key in retained}, strict=True)
    return model.to(device), {
        "kind": "project_checkpoint_with_fewer_separator_blocks",
        "checkpoint_sha256": expected_hash,
        "source_updates": saved["step"],
        "original_blocks": original.model.band_blocks,
        "retained_blocks": config.model.band_blocks,
        "external_weights": False,
    }


@torch.inference_mode()
def evaluate_concept(model, corpus, deadline, cancelled=lambda: False):
    model.eval()
    device = next(model.parameters()).device
    if device.type == "mps":
        torch.mps.empty_cache()
    rows, previews = [], []
    for index, request in enumerate(corpus.evaluation_requests("dev")):
        if cancelled() or time.monotonic() >= deadline:
            raise SessionExpired
        prediction = model(request["mixture"].to(device), request["reference"].to(device)).cpu()
        row = measure(prediction, request["mixture"], request["target"], request["interferer"])[0]
        row.update(case_id=request["case_id"], target_speaker=request["speaker"])
        rows.append(row)
        if index < 8:
            previews.append({"request": request, "estimate": prediction[0, 0].numpy().copy()})
    values = np.array([row["si_sdri_db"] for row in rows])
    result = {
        "cases": len(rows),
        "mean_si_sdri_db": float(values.mean()),
        "p25_si_sdri_db": float(np.percentile(values, 25)),
        "positive_improvement_fraction": float(np.mean(values > 0)),
        "correct_source_fraction": float(
            np.mean([row["si_sdr_db"] > row["interferer_si_sdr_db"] for row in rows])
        ),
        "rows": rows,
    }
    if device.type == "mps":
        torch.mps.empty_cache()
    return result, previews


def write_concept_gallery(previews, directory, step, checkpoint_hash, result):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    sections = []
    # Each validation gets immutable filenames. Replace the index only after all
    # its audio has been written, so an open page cannot mix checkpoints.
    for index, preview in enumerate(previews, 1):
        request = preview["request"]
        tracks = {name: request[name][0, 0].numpy() for name in ("mixture", "reference", "target")}
        tracks["estimate"] = preview["estimate"]
        matched = {
            name: wave * (0.1 / max(float(np.sqrt(np.mean(wave**2))), 1e-5))
            for name, wave in tracks.items()
        }
        gain = min(1.0, 0.98 / max(float(np.abs(wave).max()) for wave in matched.values()))
        players = []
        for name, label in (
            ("mixture", "Two voices together"),
            ("reference", "Voice to keep"),
            ("estimate", "Model estimate"),
            ("target", "Known clean target"),
        ):
            filename = f"{checkpoint_hash[:12]}-step-{step:04d}-{index:02d}-{name}.wav"
            path = directory / filename
            if not path.exists():
                sf.write(path, matched[name] * gain, 16000, subtype="FLOAT")
            players.append(
                f'<label>{label}<audio controls preload="none" src="{filename}"></audio></label>'
            )
        sections.append(
            f'<section><h2>Example {index} · voice {html.escape(request["speaker"])}</h2><div class="tracks">{"".join(players)}</div></section>'
        )
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Small voice demo · One voice</title><link rel="stylesheet" href="/assets/style.css"><style>.tracks{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}section{{border-top:1px solid #bbb;padding:24px 0}}audio{{display:block;width:100%;margin-top:8px}}@media(max-width:650px){{.tracks{{grid-template-columns:1fr}}}}</style></head><body><div class="page"><header class="topbar"><a class="brand" href="/">one voice.</a><a href="/experiments/concept/">Training progress</a></header><main><p class="eyebrow">SMALL KNOWN-VOICE DEMONSTRATION</p><h1>Keep one of two voices.</h1><p>Eight familiar voices, using recordings excluded from this candidate's training and initialization. These are clean simulated mixtures. New-speaker and noisy-room performance are not established.</p><p>Checkpoint at update {step}; development mean {result["mean_si_sdri_db"]:.2f} dB improvement. Model {checkpoint_hash[:12]}. The first eight development requests are shown, selected before scoring. Listening volume is matched so quieter output alone cannot appear to be better extraction.</p>{"".join(sections)}<p>LibriSpeech / OpenSLR12, CC BY 4.0. Cropped, mixed and processed derivatives. This candidate does not replace the default app model.</p></main></div><script>document.querySelectorAll('audio').forEach(player=>player.addEventListener('play',()=>document.querySelectorAll('audio').forEach(other=>{{if(other!==player)other.pause();}})));</script></body></html>"""
    temporary = directory / "index.partial"
    temporary.write_text(page)
    temporary.replace(directory / "index.html")


def train_concept(
    config_path,
    root,
    source_manifest,
    manifest,
    checkpoint,
    run,
    minutes=30,
    device_name="mps",
    resume=False,
    gallery=None,
    stop_after_updates=None,
):
    if not 0 < minutes <= 120:
        raise ValueError("Choose a session limit between zero and 120 minutes")
    started = time.monotonic()
    deadline = started + minutes * 60
    config = ExperimentConfig.load(config_path)
    if (
        config.data.protocol != "known-voice-concept-v1"
        or config.audio.crop_seconds != 3
        or config.audio.reference_seconds != 3
        or config.training.microbatch_size * config.training.gradient_accumulation != 8
        or config.training.activation_checkpointing
    ):
        raise ValueError("Use the fixed-shape, effective-batch-eight concept recipe")
    if config.training.learning_rate_schedule != "cosine" or config.training.optimizer != "adam":
        raise ValueError("Concept adaptation uses Adam and its declared cosine schedule")
    corpus = ConceptCorpus(root, source_manifest, manifest)
    if len(corpus.speakers) != 8:
        raise ValueError("The concept sampler requires exactly eight voices")
    run = Path(run)
    if run.exists() and any(run.iterdir()) and not resume:
        raise FileExistsError("Use --resume for the existing concept run")
    run.mkdir(parents=True, exist_ok=True)
    device = choose_device(device_name)
    torch.set_num_threads(2)
    torch.manual_seed(config.seed)
    if device.type == "mps":
        torch.mps.manual_seed(config.seed)
        torch.mps.set_per_process_memory_fraction(config.runtime.mps_memory_fraction)
    model, initialization = initialize_concept(
        config,
        checkpoint,
        corpus.recipe["initialization_sha256"],
        corpus.recipe["classifier_speakers"],
        device,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    provenance = {
        **git_state(),
        "source_tree_sha256": source_digest(),
        "torch_version": str(torch.__version__),
        "train_speakers": corpus.speakers,
        "device": str(device),
        "initialization": initialization,
        "scope": corpus.recipe["scope"],
        "fixed_shape_graph_reuse": True,
    }
    step, last_validation, best, streak, elapsed_before = 0, -1, -math.inf, 0, 0.0
    if resume:
        saved = torch.load(run / "latest.pt", map_location="cpu", weights_only=True)
        if (
            saved["config"] != config.model_dump()
            or saved["manifest_sha256"] != corpus.manifest_hash
            or saved["provenance"]["source_tree_sha256"] != provenance["source_tree_sha256"]
            or saved["provenance"]["device"] != str(device)
            or saved["provenance"]["initialization"] != initialization
        ):
            raise ValueError(
                "Resume requires the same concept recipe, source, device and reservation"
            )
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value) and key != "step":
                    state[key] = value.to(device)
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "mps":
            torch.mps.set_rng_state(saved["mps_rng"])
        step, last_validation, best, streak, elapsed_before = (
            saved[k]
            for k in (
                "step",
                "last_validation_step",
                "best_score",
                "goal_streak",
                "elapsed_seconds",
            )
        )
        provenance = saved["provenance"]
    atomic_json(run / "config.json", config.model_dump())
    atomic_json(run / "provenance.json", provenance)
    interrupted = False

    def stop(signum, frame):
        nonlocal interrupted
        interrupted = True

    handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}

    def save(name):
        save_checkpoint(
            run / name,
            {
                "format_version": 1,
                "config": config.model_dump(),
                "model": model.state_dict(),
                "speaker_classes": len(corpus.labels),
                "manifest_sha256": corpus.manifest_hash,
                "provenance": provenance,
                "step": step,
                "last_validation_step": last_validation,
                "best_score": best,
                "goal_streak": streak,
                "optimizer": optimizer.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "mps_rng": torch.mps.get_rng_state() if device.type == "mps" else None,
                "elapsed_seconds": elapsed_before + time.monotonic() - started,
            },
        )

    def status(state, **extra):
        atomic_json(
            run / "status.json",
            {
                "status": state,
                "step": step,
                "total_updates": config.training.max_optimizer_updates,
                "session_minutes": minutes,
                "session_elapsed_seconds": time.monotonic() - started,
                "total_elapsed_seconds": elapsed_before + time.monotonic() - started,
                "best_si_sdri_db": best if math.isfinite(best) else None,
                "goal_streak": streak,
                **extra,
            },
        )

    def emit(entry):
        with (run / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)

    reason = "budget_complete"
    try:
        status("starting")
        while True:
            if interrupted or time.monotonic() >= deadline:
                raise SessionExpired
            validate_now = (
                step == 0
                or step % config.training.validation_interval_updates == 0
                or step == config.training.max_optimizer_updates
            ) and last_validation != step
            if validate_now:
                status("validating")
                result, previews = evaluate_concept(model, corpus, deadline, lambda: interrupted)
                last_validation = step
                streak = streak + 1 if development_goal(result) else 0
                emit(
                    {
                        "event": "validation",
                        "step": step,
                        **{k: v for k, v in result.items() if k != "rows"},
                    }
                )
                atomic_json(run / "latest-validation.json", {"step": step, **result})
                if result["mean_si_sdri_db"] > best:
                    best = result["mean_si_sdri_db"]
                    save("best.pt")
                    atomic_json(run / "best-validation.json", {"step": step, **result})
                    if gallery is not None:
                        write_concept_gallery(
                            previews, gallery, step, sha256(run / "best.pt"), result
                        )
                save("latest.pt")
            if streak >= 2 and step >= 100:
                reason = "development_target_met"
                break
            if step >= config.training.max_optimizer_updates:
                break
            if stop_after_updates is not None and step >= stop_after_updates:
                raise SessionExpired
            model.train()
            requests = corpus.training_batch(config.seed, step)
            if any(
                r["mixture"].shape != (1, 1, 48000) or r["reference"].shape != (1, 1, 48000)
                for r in requests
            ):
                raise ValueError("Graph reuse requires the declared fixed training shapes")
            features = full_reference_features(requests, True)
            fraction = step / config.training.max_optimizer_updates
            lr = (
                config.training.minimum_learning_rate
                + (config.training.learning_rate - config.training.minimum_learning_rate)
                * (1 + math.cos(math.pi * fraction))
                / 2
            )
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
                        "learning_rate": lr,
                        "gradient_norm": float(norm),
                        "elapsed_seconds": elapsed_before + time.monotonic() - started,
                        "mps_driver_gib": torch.mps.driver_allocated_memory() / 1024**3
                        if device.type == "mps"
                        else None,
                        **metrics,
                    }
                )
                status("training")
            if step % config.training.checkpoint_interval_updates == 0:
                save("latest.pt")
        save("latest.pt")
        atomic_json(
            run / "summary.json",
            {
                "status": reason,
                "steps": step,
                "best_development_si_sdri_db": best,
                "new_speaker_quality_established": False,
                "reserved_test_opened": False,
                "human_listening": "not yet rated",
            },
        )
        status(reason)
    except SessionExpired:
        save("latest.pt")
        status("paused", reason="User stop or session/update limit; complete updates are saved")
    except BaseException as error:
        status(
            "failed",
            error=type(error).__name__,
            detail=str(error),
            resume="Use the previous complete atomic checkpoint; the partial update was discarded",
        )
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return model
