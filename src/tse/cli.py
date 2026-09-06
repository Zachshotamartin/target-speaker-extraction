"""Command line for dataset preparation, experiments, evaluation and local serving."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tse.config import ExperimentConfig


def main() -> int:
    parser = argparse.ArgumentParser(prog="tse", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("data", help="Acquire and audit public audio")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    acquire = data_commands.add_parser("acquire")
    acquire.add_argument("split", choices=["train-clean-100", "dev-clean", "test-clean"])
    acquire.add_argument("--root", type=Path, default=Path("data/raw"))
    acquire.add_argument("--speakers", type=int, default=60)
    acquire.add_argument("--utterances", type=int, default=50)
    inventory = data_commands.add_parser("inventory")
    inventory.add_argument("--root", type=Path, default=Path("data/raw"))
    inventory.add_argument("--output", type=Path, default=Path("data/manifests/inventory.json"))
    audit = data_commands.add_parser("audit")
    audit.add_argument("--manifest", type=Path, default=Path("data/manifests/inventory.json"))
    cases = data_commands.add_parser("build-cases")
    cases.add_argument("--root", type=Path, default=Path("data/raw"))
    cases.add_argument("--manifest", type=Path, default=Path("data/manifests/inventory.json"))
    cases.add_argument("--split", choices=["train", "dev", "test"], required=True)
    cases.add_argument("--count", type=int, default=400)
    cases.add_argument("--seed", type=int, default=88000)
    cases.add_argument("--seconds", type=float, default=4)
    cases.add_argument("--reference-seconds", type=float, default=5)
    cases.add_argument("--output", type=Path, required=True)

    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("--config", type=Path, default=Path("configs/control.json"))
    benchmark.add_argument("--device", choices=["cpu", "mps"], default="mps")
    benchmark.add_argument("--output", type=Path, required=True)

    train = commands.add_parser("train")
    train.add_argument("--config", type=Path, default=Path("configs/control.json"))
    train.add_argument("--root", type=Path, default=Path("data/raw"))
    train.add_argument("--manifest", type=Path, default=Path("data/manifests/inventory.json"))
    train.add_argument("--dev-cases", type=Path, default=Path("data/manifests/dev-cases.json"))
    train.add_argument("--run", type=Path, required=True)
    train.add_argument("--device", choices=["cpu", "mps"])
    train.add_argument("--steps", type=int)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--fixed-cases", type=Path)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--root", type=Path, default=Path("data/raw"))
    evaluate.add_argument("--manifest", type=Path, default=Path("data/manifests/inventory.json"))
    evaluate.add_argument("--cases", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--device", choices=["cpu", "mps"], default="mps")
    evaluate.add_argument(
        "--conditions", nargs="+", choices=["clean", "noise", "channel", "reverb", "combined"]
    )

    extract = commands.add_parser("extract")
    extract.add_argument("--checkpoint", type=Path, default=Path("artifacts/releases/model.pt"))
    extract.add_argument("--mixture", type=Path, required=True)
    extract.add_argument("--reference", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--device", choices=["cpu", "mps"], default="mps")

    export = commands.add_parser("export")
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--output", type=Path, default=Path("artifacts/releases/model.pt"))

    serve = commands.add_parser("serve")
    serve.add_argument("--checkpoint", type=Path, default=Path("artifacts/releases/model.pt"))
    serve.add_argument("--device", choices=["cpu", "mps"], default="mps")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    try:
        if args.command == "data":
            from tse import data as dataset

            if args.data_command == "acquire":
                from tse.acquire import acquire

                result = acquire(args.root, args.split, args.speakers, args.utterances)
                print(
                    json.dumps({"files": len(result["files"]), "speakers": len(result["speakers"])})
                )
            elif args.data_command == "inventory":
                print(json.dumps(dataset.inventory(args.root, args.output)["audit"], indent=2))
            elif args.data_command == "audit":
                print(
                    json.dumps(
                        dataset.audit_records(json.loads(args.manifest.read_text())["records"]),
                        indent=2,
                    )
                )
            else:
                corpus = dataset.SpeechCorpus(
                    args.root, args.manifest, args.split, args.seconds, args.reference_seconds
                )
                result = dataset.build_cases(corpus, args.count, args.seed, args.output)
                print(json.dumps({"cases": len(result["cases"]), "output": str(args.output)}))
        elif args.command in {"benchmark", "train", "evaluate", "export"}:
            from tse import engine

            if args.command == "benchmark":
                engine.benchmark(ExperimentConfig.load(args.config), args.device, args.output)
            elif args.command == "train":
                config = ExperimentConfig.load(args.config)
                if args.steps is not None:
                    config.training.max_optimizer_updates = args.steps
                engine.train(
                    config,
                    args.root,
                    args.manifest,
                    args.dev_cases,
                    args.run,
                    args.device,
                    args.resume,
                    args.fixed_cases,
                )
            elif args.command == "evaluate":
                engine.evaluate(
                    args.checkpoint,
                    args.root,
                    args.manifest,
                    args.cases,
                    args.output,
                    args.device,
                    args.conditions,
                )
            else:
                from tse.utils import atomic_json, sha256

                _, payload = engine.load_model(args.checkpoint)
                for key in ("optimizer", "torch_rng", "mps_rng"):
                    payload.pop(key, None)
                engine.save_checkpoint(args.output, payload)
                atomic_json(
                    args.output.with_suffix(".json"),
                    {
                        "checkpoint_sha256": sha256(args.output),
                        "source_checkpoint_sha256": sha256(args.checkpoint),
                        "training_updates": payload["step"],
                        "config": payload["config"],
                        "provenance": payload["provenance"],
                    },
                )
                print(json.dumps({"output": str(args.output), "sha256": sha256(args.output)}))
        elif args.command == "extract":
            from tse.inference import Extractor

            audio, metadata = Extractor(args.checkpoint, args.device).extract_files(
                args.mixture, args.reference
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(audio)
            print(json.dumps(metadata, indent=2))
        elif args.command == "serve":
            import uvicorn

            from tse.api import create_app

            uvicorn.run(
                create_app(args.checkpoint, args.device),
                host="127.0.0.1",
                port=args.port,
                log_level="info",
            )
    except (ValueError, OSError, RuntimeError) as error:
        print(f"tse: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
