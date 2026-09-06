"""Evaluate the delivered waveform path and profile bounded long-file processing."""

from __future__ import annotations

import io
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
import torch

from tse.audio import read_audio, wav_bytes
from tse.data import SpeechCorpus, load_cases
from tse.engine import synchronize
from tse.inference import Extractor
from tse.metrics import measure, summarize
from tse.utils import atomic_json, git_state, sha256, source_digest


def evaluate_delivery(
    checkpoint: Path,
    root: Path,
    manifest: Path,
    cases_path: Path,
    output: Path,
    device: str = "cpu",
) -> dict:
    extractor = Extractor(checkpoint, device)
    protocol = json.loads(cases_path.read_text())
    corpus = SpeechCorpus(root, manifest, protocol["split"], 4, 5)
    cases = load_cases(cases_path, corpus)
    rows = []
    started = time.perf_counter()
    for case in cases:
        batch = corpus.batch([case], torch.device("cpu"))
        mixture, reference = (batch[key][0, 0].numpy() for key in ("mixture", "reference"))
        encoded, metadata = extractor.extract_files(
            io.BytesIO(wav_bytes(mixture)), io.BytesIO(wav_bytes(reference))
        )
        prediction = torch.from_numpy(read_audio(io.BytesIO(encoded)))[None, None]
        score = measure(
            prediction,
            batch["mixture"],
            batch["target"],
            batch["interferer"],
            extractor.config.evaluation.confusion_margin_db,
        )[0]
        rows.append(
            {
                "case_id": case["case_id"],
                "target_speaker": batch["speakers"][0],
                "condition": "clean",
                "processing_seconds": metadata["processing_seconds"],
                "raw_output_peak": metadata["raw_output_peak"],
                "output_peak": metadata["output_peak"],
                "playback_gain": metadata["playback_gain"],
                **score,
            }
        )
    result = {
        "checkpoint_sha256": sha256(checkpoint),
        "processing_version": extractor.info()["processing_version"],
        "source_manifest_sha256": sha256(manifest),
        "case_manifest_sha256": sha256(cases_path),
        "split": protocol["split"],
        "device": device,
        "confusion_margin_db": extractor.config.evaluation.confusion_margin_db,
        "path": "Float WAV encode -> API's shared extract_files -> Float WAV decode -> score",
        "elapsed_seconds": time.perf_counter() - started,
        "summary": summarize(rows),
        "rows": rows,
        "provenance": {**git_state(), "source_tree_sha256": source_digest()},
    }
    atomic_json(output, result)
    return result


def profile_delivery(
    checkpoint: Path,
    mixture_path: Path,
    reference_path: Path,
    output: Path,
    device: str = "cpu",
    repeats: int = 5,
) -> dict:
    if repeats < 2:
        raise ValueError("At least two warm measurements are required")
    started = time.perf_counter()
    extractor = Extractor(checkpoint, device)
    synchronize(extractor.device)
    load_seconds = time.perf_counter() - started
    mixture = read_audio(mixture_path)
    reference = read_audio(reference_path, max_seconds=10)
    reference_bytes = wav_bytes(reference)
    # Repeating a fixed development clip isolates length-dependent runtime. These
    # synthetic transitions are not evidence about real conversational continuity.
    rows = []
    for seconds in (10, 30, 60):
        waveform = np.resize(mixture, seconds * 16000)
        encoded = wav_bytes(waveform)
        elapsed = []
        model_elapsed = []
        first_seconds = None
        for iteration in range(repeats + 1):
            prediction_bytes, metadata = extractor.extract_files(
                io.BytesIO(encoded), io.BytesIO(reference_bytes)
            )
            if iteration == 0:
                first_seconds = metadata["processing_seconds"]
            else:
                elapsed.append(metadata["processing_seconds"])
                model_elapsed.append(metadata["model_seconds"])
        chunked = read_audio(io.BytesIO(prediction_bytes))
        whole = extractor.extract_array(waveform, reference, chunked=False)
        difference = chunked - whole
        boundaries = np.concatenate(
            [difference[index - 160 : index + 160] for index in range(32000, len(whole), 32000)]
        )
        rows.append(
            {
                "duration_seconds": seconds,
                "output_samples": len(chunked),
                "expected_samples": seconds * 16000,
                "exact_length": len(chunked) == seconds * 16000,
                "first_request_seconds": first_seconds,
                "warm_repeats": repeats,
                "warm_median_seconds": float(np.median(elapsed)),
                "warm_p95_seconds": float(np.percentile(elapsed, 95)),
                "warm_model_median_seconds": float(np.median(model_elapsed)),
                "warm_median_real_time_factor": float(np.median(elapsed) / seconds),
                "chunk_whole_rms_error": float(np.sqrt(np.mean(difference**2))),
                "chunk_whole_max_error": float(np.max(np.abs(difference))),
                "boundary_10ms_rms_error": float(np.sqrt(np.mean(boundaries**2))),
                "whole_output_rms": float(np.sqrt(np.mean(whole**2))),
                "timings_seconds": elapsed,
            }
        )
    result = {
        "model": extractor.info(),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "model_load_seconds": load_seconds,
        "process_peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024**3 if platform.system() == "Darwin" else 1024**2),
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "cpu_threads": torch.get_num_threads(),
        "mixture_sha256": sha256(mixture_path),
        "reference_sha256": sha256(reference_path),
        "rows": rows,
        "protocol": "One process per device; model load timed separately, first request per length followed by repeated warm requests. End-to-end includes audio decode/resample, model and WAV encode; excludes browser/network/HTTP parsing. RSS includes whole-clip diagnostic and is a process high-water mark, not isolated serving memory. Timing input repeats a development clip; no training workload should run concurrently. First request is not a fresh OS disk-cache measurement.",
        "provenance": {**git_state(), "source_tree_sha256": source_digest()},
    }
    if device == "mps":
        result["mps_allocated_gib"] = torch.mps.current_allocated_memory() / 1024**3
        result["mps_driver_gib"] = torch.mps.driver_allocated_memory() / 1024**3
    atomic_json(output, result)
    return result
