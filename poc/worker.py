"""One low-priority, CPU-only job. No training control or network inference."""

# ruff: noqa: E402 -- thread and offline limits must precede numeric imports.
import argparse
import os

for variable in [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]:
    os.environ[variable] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import resource
from pathlib import Path

from poc.common import atomic_json, decode, wav


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    args = parser.parse_args()
    os.nice(10)

    def progress(stage):
        atomic_json(args.job / "progress.json", {"stage": stage})

    try:
        options = json.loads((args.job / "options.json").read_text())
        if options.get("kind"):
            from poc.workspace_worker import run_workspace

            run_workspace(args.job, args.models, options, progress)
            return
        from poc.models import Runtime
        from poc.pipeline import run

        progress("Checking audio")
        mixture = decode((args.job / "mixture.input").read_bytes())
        reference = decode((args.job / "reference.input").read_bytes(), maximum=10)
        options = json.loads((args.job / "options.json").read_text())
        runtime = Runtime(args.models, progress)
        result, estimate = run(runtime, mixture, reference, options["compare"], progress)
        result["peak_worker_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if os.uname().sysname == "Darwin" else 1024
        )
        (args.job / "original.wav").write_bytes(wav(mixture, playback=True))
        (args.job / "extracted.wav").write_bytes(wav(estimate, playback=True))
        atomic_json(args.job / "result.json", result)
    except Exception as error:
        # No transcript, raw audio, reference or upload filename goes into logs.
        atomic_json(args.job / "error.json", {"message": str(error)[:400]})
        raise SystemExit(1) from None
    finally:
        for path in [*args.job.glob("*.input"), args.job / "options.json"]:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
