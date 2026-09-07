#!/usr/bin/env python3
"""Validate MPS temporal compilation and measure warm training compute."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from tse.data import SpeechCorpus
from tse.engine import compile_temporal_blocks, load_model
from tse.metrics import si_sdr, spectral_loss, waveform_loss
from tse.utils import atomic_json, sha256, source_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    corpus = SpeechCorpus(
        Path("data/expanded/raw"), Path("data/expanded/manifests/inventory.json"), "train", 4, 5
    )
    cases = [corpus.make_case(401990000 + j) for j in range(8)]
    batches = [corpus.batch(cases[j : j + 4], torch.device("mps")) for j in (0, 4)]
    result = {
        "protocol": "Main training and quality evaluation paused. Same real training batch eight, microbatch four. Twenty iterations per mode, first three excluded from warm median. Includes forward, all losses, backward, clipping and optimizer; excludes data preparation. Numeric parity measured before the first update. This is training compute, not serving runtime or a quality result.",
        "checkpoint_sha256": sha256(args.checkpoint),
        "torch_version": str(torch.__version__),
        "source_tree_sha256": source_digest(),
    }
    for mode in ("eager", "compiled"):
        model, payload = load_model(args.checkpoint, "mps")
        model.train()
        if payload["provenance"]["train_speakers"] != corpus.speakers:
            raise ValueError("Benchmark classifier labels differ from the training corpus")
        keys = list(model.state_dict())
        if mode == "compiled":
            compile_temporal_blocks(model)
        if keys != list(model.state_dict()):
            raise ValueError("Compilation changed checkpoint keys")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        elapsed = []
        for iteration in range(20):
            torch.mps.synchronize()
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            outputs = []
            for batch in batches:
                embedding = model.reference_encoder(batch["reference"])
                output = model.extract(batch["mixture"], embedding)
                loss = -si_sdr(output, batch["target"]).mean()
                loss += 0.5 * waveform_loss(output, batch["target"])
                loss += 0.5 * spectral_loss(output, batch["target"], [256, 512, 1024])
                loss += 0.2 * F.cross_entropy(model.speaker_head(embedding) * 10, batch["labels"])
                (loss / 2).backward()
                outputs.append(output.detach())
            if iteration == 0:
                gradients = {
                    name: parameter.grad.detach().cpu().clone()
                    if parameter.grad is not None
                    else torch.zeros_like(parameter, device="cpu")
                    for name, parameter in model.named_parameters()
                }
                if mode == "eager":
                    original = torch.cat(outputs).cpu()
                    original_gradients = gradients
                else:
                    result["forward_max_error"] = float(
                        (torch.cat(outputs).cpu() - original).abs().max()
                    )
                    numerator = sum(
                        (value - original_gradients[name]).square().sum()
                        for name, value in gradients.items()
                    )
                    denominator = sum(value.square().sum() for value in original_gradients.values())
                    result["gradient_relative_l2_error"] = float((numerator / denominator).sqrt())
                    if (
                        result["forward_max_error"] >= 1e-4
                        or result["gradient_relative_l2_error"] >= 1e-3
                    ):
                        raise ValueError("Compiler failed numerical agreement thresholds")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            torch.mps.synchronize()
            elapsed.append(time.perf_counter() - started)
        result[mode] = {
            "times_seconds": elapsed,
            "median_warm_seconds": float(np.median(elapsed[3:])),
            "driver_gib": torch.mps.driver_allocated_memory() / 1024**3,
        }
        print(mode, result[mode], flush=True)
    result["note"] = (
        "Unused final residual parameters can receive zero gradients from the compiler where eager uses None; parity treats both as zero. These parameters do not affect the returned skip output. Compiler arithmetic is close, not bit-identical; compiler mode is recorded in training provenance."
    )
    result["status"] = "passed"
    atomic_json(args.output, result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
