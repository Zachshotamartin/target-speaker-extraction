#!/usr/bin/env python3
"""Compare a reference with clean target/interferer embeddings, never final-test audio."""

import argparse
from pathlib import Path

import numpy as np
import torch

from tse.data import SpeechCorpus
from tse.engine import load_model
from tse.utils import atomic_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/inventory.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model, payload = load_model(args.checkpoint, "cpu")
    labels = {
        speaker: index for index, speaker in enumerate(payload["provenance"]["train_speakers"])
    }
    result = {}
    with torch.inference_mode():
        for split in ("train", "dev"):
            corpus = SpeechCorpus(args.root, args.manifest, split, 4, 5)
            if split == "dev" and set(corpus.speakers) & set(labels):
                raise ValueError("Development overlaps checkpoint training speakers")
            margins, correct, labeled = [], 0, 0
            for index in range(0, 200, 4):
                cases = [corpus.make_case(94622000 + index + j) for j in range(4)]
                batch = corpus.batch(cases, torch.device("cpu"))
                reference = model.reference_encoder(batch["reference"])
                vectors = []
                for key in ("target", "interferer"):
                    signal = batch[key] - batch[key].mean(-1, keepdim=True)
                    signal = (
                        signal * 0.1 / signal.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-4)
                    )
                    vectors.append(model.reference_encoder(signal))
                margins.extend(
                    ((reference * vectors[0]).sum(-1) - (reference * vectors[1]).sum(-1)).tolist()
                )
                if split == "train" and model.speaker_head is not None:
                    predictions = model.speaker_head(reference).argmax(1).tolist()
                    for speaker, prediction in zip(batch["speakers"], predictions, strict=True):
                        if speaker in labels:
                            labeled += 1
                            correct += prediction == labels[speaker]
            result[split] = {
                "cases": 200,
                "reference_matching_fraction": float(np.mean(np.array(margins) > 0)),
                "mean_cosine_margin": float(np.mean(margins)),
                "training_classification_accuracy": correct / labeled if labeled else None,
                "classification_labeled_cases": labeled,
            }
    result["checkpoint_sha256"] = sha256(args.checkpoint)
    result["source_manifest_sha256"] = sha256(args.manifest)
    result["protocol"] = (
        "Separate reference matched to clean target versus clean interferer; 200 cases per train/dev split, seeds94622000..94622199. Clean-source diagnostic, not extraction quality. No test data."
    )
    atomic_json(args.output, result)


if __name__ == "__main__":
    main()
