"""Explicit, resumable provisioning of published weights and a frozen One Voice copy."""

import argparse
import importlib.metadata
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from poc.common import ROOT, atomic_json, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/poc/models")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    validation = json.loads(args.validation.read_text())
    expected = validation["checkpoint_sha256"]
    if digest(args.checkpoint) != expected:
        raise ValueError(
            "Checkpoint and completed validation do not match. Retry after evaluation."
        )
    target = output / "one-voice.pt"
    if not target.exists() or digest(target) != expected:
        temp = output / "one-voice.partial"
        shutil.copyfile(args.checkpoint, temp)
        if digest(temp) != expected or digest(args.checkpoint) != expected:
            temp.unlink(missing_ok=True)
            raise ValueError("Checkpoint changed during copy; original was not modified.")
        temp.replace(target)
    manifest = {
        "schema": 1,
        "one_voice": {
            "sha256": expected,
            "epoch": validation["epoch"],
            "step": validation["step"],
            "validation_improvement": validation["mean_si_sdri_db"],
        },
        "models": {},
        "contract": {
            "only_waveform_separator": "one_voice",
            "asr_reference_input": False,
            "external_separation": False,
            "training": False,
        },
    }
    repos = {
        "speaker": (
            "speechbrain/spkrec-ecapa-voxceleb",
            ["*.ckpt", "hyperparams.yaml", "label_encoder.txt"],
        ),
        "asr": (
            "Systran/faster-whisper-small.en",
            [
                "config.json",
                "model.bin",
                "tokenizer.json",
                "vocabulary.*",
                "preprocessor_config.json",
            ],
        ),
    }
    for name, (repo, files) in repos.items():
        previous_path = output / "download-revisions.json"
        revisions = json.loads(previous_path.read_text()) if previous_path.exists() else {}
        revision = revisions.get(repo) or HfApi().model_info(repo).sha
        revisions[repo] = revision
        atomic_json(previous_path, revisions)
        print(f"Downloading {name}: {repo}@{revision}", flush=True)
        snapshot_download(
            repo, revision=revision, local_dir=output / name, allow_patterns=files, max_workers=2
        )
        manifest["models"][name] = {
            "repo": repo,
            "revision": revision,
            "files": {
                str(p.relative_to(output)): digest(p)
                for p in sorted((output / name).iterdir())
                if p.is_file()
            },
        }
    import faster_whisper

    vad = Path(faster_whisper.__file__).parent / "assets/silero_vad_v6.onnx"
    if not vad.exists():
        candidates = list((Path(faster_whisper.__file__).parent / "assets").glob("*silero*.onnx"))
        if len(candidates) != 1:
            raise ValueError("Cannot identify the packaged Silero model")
        vad = candidates[0]
    manifest["models"]["vad"] = {
        "provider": "Silero, packaged by faster-whisper",
        "sha256": digest(vad),
    }
    manifest["packages"] = {
        name: importlib.metadata.version(name)
        for name in [
            "torch",
            "torchaudio",
            "speechbrain",
            "faster-whisper",
            "ctranslate2",
            "onnxruntime",
            "av",
        ]
    }
    atomic_json(output / "manifest.json", manifest)
    print(f"Ready: {output / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
