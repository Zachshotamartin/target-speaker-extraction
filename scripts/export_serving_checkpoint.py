"""Export trusted project weights without training-only state; verify every tensor."""

import argparse
import json
from pathlib import Path

import torch

from tse.utils import sha256


def export(source, output):
    if source.resolve() == output.resolve():
        raise ValueError("The resumable source checkpoint must be preserved")
    original = torch.load(source, map_location="cpu", weights_only=True)
    fields = {
        "format_version",
        "config",
        "model",
        "speaker_classes",
        "manifest_sha256",
        "provenance",
        "step",
        "epoch",
    }
    serving = {k: v for k, v in original.items() if k in fields}
    serving["provenance"] = {
        **serving["provenance"],
        "source_checkpoint_sha256": sha256(source),
        "export": "inference-only; model tensors and buffers unchanged; optimizer and RNG state excluded",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(serving, output)
    loaded = torch.load(output, map_location="cpu", weights_only=True)
    assert original["model"].keys() == loaded["model"].keys()
    assert original["config"] == loaded["config"]
    assert all(torch.equal(value, loaded["model"][key]) for key, value in original["model"].items())
    return {
        "source_checkpoint_sha256": sha256(source),
        "serving_sha256": sha256(output),
        "model_tensor_count": len(loaded["model"]),
        "tensor_equality": True,
        "serving_bytes": output.stat().st_size,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.output), indent=2))
