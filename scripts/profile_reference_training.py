#!/usr/bin/env python3
"""Measure full-size training on actual development-free training requests."""

import argparse
import json
import resource
import time
from pathlib import Path

import torch

from tse.config import ExperimentConfig
from tse.librimix import LibriMixCorpus, request_seed
from tse.model import make_model
from tse.reference_training import backward_batch, full_reference_features
from tse.utils import atomic_json, source_digest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/reference-libri2mix.json"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/reference-training-profile.json")
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--separator-microbatch", type=int, choices=[1, 2, 4, 8])
    parser.add_argument(
        "--longest-enrollment",
        action="store_true",
        help="Append a stress batch using the longest valid full enrollments in the corpus",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("A profile needs at least one update")
    if args.output.exists():
        raise FileExistsError("Keep the previous profile")
    config = ExperimentConfig.load(args.config)
    if args.separator_microbatch is not None:
        config.training.microbatch_size = args.separator_microbatch
        config.training.gradient_accumulation = 8 // args.separator_microbatch
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.mps.set_per_process_memory_fraction(config.runtime.mps_memory_fraction)
    corpus = LibriMixCorpus(args.root, Path("data/reference/manifest.json"), "train")
    model = make_model(config.model, len(corpus.speakers)).to("mps").train()
    model.activation_checkpointing = config.training.activation_checkpointing
    model.reference_encoder.activation_checkpointing = config.training.activation_checkpointing
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    timings, results = [], []
    longest = sorted(corpus.rows, key=lambda row: row["samples"], reverse=True)[:4]
    stress_references = [source for row in longest for source in row["sources"]]
    for repeat in range(args.repeats + int(args.longest_enrollment)):
        started = time.monotonic()
        stress = repeat == args.repeats
        if stress:
            requests = []
            for reference in stress_references:
                # Choose a different target utterance by the same speaker, then
                # supply this full enrollment. Only this discarded profile uses
                # forced long enrollments; the real sampler is unchanged.
                index = next(
                    2 * pair + side
                    for pair, row in enumerate(corpus.rows)
                    for side, source in enumerate(row["sources"])
                    if source["speaker"] == reference["speaker"]
                    and source["utterance"] != reference["utterance"]
                )
                request = corpus.request(index, request_seed(42, 0, index), 48000)
                request["reference"] = torch.from_numpy(corpus.read(reference["path"]).copy())[
                    None, None
                ]
                request["reference_path"] = reference["path"]
                requests.append(request)
        else:
            requests = [
                corpus.request(i + repeat * 8, request_seed(42, 0, i + repeat * 8), 48000)
                for i in range(8)
            ]
        features = full_reference_features(requests, True)
        optimizer.zero_grad(set_to_none=True)
        metrics = backward_batch(model, requests, features, config)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
        optimizer.step()
        torch.mps.synchronize()
        duration = time.monotonic() - started
        timings.append(duration)
        entry = {
            "repeat": repeat,
            "kind": "longest_enrollment_stress" if stress else "ordinary_training",
            "seconds": duration,
            "enrollment_seconds": [request["reference"].shape[-1] / 16000 for request in requests],
            "padded_feature_frames": features.shape[-1],
            "mps_driver_gib": torch.mps.driver_allocated_memory() / 1024**3,
            "mps_tensor_gib": torch.mps.current_allocated_memory() / 1024**3,
            **metrics,
        }
        results.append(entry)
        print(json.dumps(entry), flush=True)
    ordinary = timings[: args.repeats]
    mean = sum(ordinary[1:]) / len(ordinary[1:]) if len(ordinary) > 1 else ordinary[0]
    atomic_json(
        args.output,
        {
            "status": "finite_full_size_training",
            "source_tree_sha256": source_digest(),
            "mps_memory_fraction": config.runtime.mps_memory_fraction,
            "parameters": sum(p.numel() for p in model.parameters()),
            "batch": 8,
            "separator_microbatch": config.training.microbatch_size,
            "gradient_accumulation": config.training.gradient_accumulation,
            "seconds_per_update": timings,
            "warm_mean_seconds": mean,
            "estimated_training_only_hours_100_epochs": mean * 347500 / 3600,
            "mps_driver_gib_end": torch.mps.driver_allocated_memory() / 1024**3,
            "process_peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
            "results": results,
            "limitations": "Short sample on training audio, not convergence evidence. Estimate excludes full-development scoring/checkpoint I/O and thermal changes. No external weights.",
        },
    )
