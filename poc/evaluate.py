"""Reproducible public-data checks using complete utterances, with no training.

Use dev to inspect fixed heuristic thresholds; freeze them before running test.
LibriSpeech test-clean is a reporting set, not a new untouched model benchmark.
"""

# ruff: noqa: E402 -- set numerical limits before importing numpy/torch.
import argparse
import os

for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[key] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import csv
import json
import re
import resource
import tarfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from poc.common import RATE, ROOT, atomic_json, digest
from poc.models import Runtime
from poc.pipeline import GATE, run


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower().replace("'", ""))


def errors(reference, hypothesis):
    left, right = tokens(reference), tokens(hypothesis)
    row = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        nxt = [i]
        for j, b in enumerate(right, 1):
            nxt.append(min(row[j] + 1, nxt[-1] + 1, row[j - 1] + (a != b)))
        row = nxt
    return {
        "edits": row[-1],
        "reference_words": len(left),
        "hypothesis_words": len(right),
        "wer": row[-1] / len(left) if left else None,
    }


def transcripts(root, split):
    # Read only public transcript text from the already-acquired archive. Never extract paths.
    result = {}
    with tarfile.open(root / "archives" / f"{split}-clean.tar.gz", "r|gz") as archive:
        for entry in archive:
            if entry.isfile() and entry.name.endswith(".trans.txt") and entry.size < 1024 * 1024:
                for line in archive.extractfile(entry).read().decode().splitlines():
                    key, text = line.split(" ", 1)
                    result[key] = text
    return result


def cases(root, manifest, split, count):
    enrollment = manifest["enrollments"][split]
    used, selected = set(), []
    rows = list(csv.DictReader((root / "metadata" / f"libri2mix_{split}-clean.csv").open()))
    for row in rows:
        paths = [root / "speech/LibriSpeech" / row[f"source_{i}_path"] for i in (1, 2)]
        speakers = [p.stem.split("-")[0] for p in paths]
        if used.intersection(speakers):
            continue
        info = [sf.info(p) for p in paths]
        if any(not 3 <= i.duration <= 9 or i.samplerate != RATE for i in info):
            continue
        refs = [root / enrollment[f"{row['mixture_ID']}:{side}"] for side in (0, 1)]
        if any(sf.info(p).duration < 3 for p in refs):
            continue
        selected.append((row, paths, refs))
        used.update(speakers)
        if len(selected) == count:
            break
    if len(selected) < count:
        raise ValueError("Not enough disjoint-speaker public examples satisfy the duration limits.")
    return selected


def summarize(rows):
    result = {}
    for condition in ["overlap", "target_only", "absent", "silence"]:
        group = [r for r in rows if r["condition"] == condition]
        detail = {
            "cases": len(group),
            "jobs_with_attributed_text": sum(r["outcome"] == "matched" for r in group),
        }
        for name in ["raw", "one_voice", "attributed"]:
            total = sum(r["scores"][name]["reference_words"] for r in group)
            edits = sum(r["scores"][name]["edits"] for r in group)
            detail[name + "_wer"] = edits / total if total else None
            detail[name + "_words"] = sum(r["scores"][name]["hypothesis_words"] for r in group)
        result[condition] = detail
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--models", type=Path, default=ROOT / "artifacts/poc/models")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.nice(10)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    print("Reading complete public utterance transcripts", flush=True)
    text = transcripts(args.dataset_root, args.split)
    selected = cases(args.dataset_root, manifest, args.split, args.count)
    runtime = Runtime(args.models)
    rows, began = [], time.monotonic()
    for number, (row, paths, references) in enumerate(selected):
        side = number % 2
        sources = [
            sf.read(p, dtype="float32")[0] * float(row[f"source_{i + 1}_gain"])
            for i, p in enumerate(paths)
        ]
        length = max(map(len, sources))
        sources = [np.pad(x, (0, length - len(x))) for x in sources]
        gain = min(1.0, 0.95 / max(float(np.max(np.abs(sources[0] + sources[1]))), 1e-8))
        sources = [x * gain for x in sources]
        reference = sf.read(references[side], dtype="float32")[0][: 10 * RATE]
        if sum(b - a for a, b in runtime.speech(reference)) < 1.5:
            raise ValueError(
                "Selected enrollment failed the fixed speech requirement; review selection explicitly."
            )
        for condition, mixture in [
            ("overlap", sources[0] + sources[1]),
            ("target_only", sources[side]),
            ("absent", sources[1 - side]),
            ("silence", np.zeros(length, np.float32)),
        ]:
            file = args.output / f"{number:02}-{condition}.json"
            if file.exists():
                cached = json.loads(file.read_text())
                if cached["gate"] != GATE or cached["model_manifest"] != runtime.manifest:
                    raise ValueError(
                        "Existing report uses other thresholds or weights. Choose a new output directory."
                    )
                rows.append(cached)
                continue
            print(
                f"{args.split} {number + 1}/{len(selected)} {condition}: {row['mixture_ID']}:{side}",
                flush=True,
            )
            output, _ = run(runtime, mixture, reference, compare=True)
            gold = text[paths[side].stem] if condition in {"overlap", "target_only"} else ""
            comparison = output.get("comparison") or {
                "raw": {"text": ""},
                "one_voice": {"text": ""},
            }
            attributed = " ".join(
                s["text"].strip() for s in output["segments"] if s["attribution"] == "accepted"
            )
            output.update(
                condition=condition,
                case_id=f"{row['mixture_ID']}:{side}",
                target_speaker=paths[side].stem.split("-")[0],
                reference_text=gold,
                source_hashes=[digest(p) for p in paths],
                reference_hash=digest(references[side]),
                scores={
                    name: errors(gold, hypothesis)
                    for name, hypothesis in [
                        ("raw", comparison["raw"]["text"]),
                        ("one_voice", comparison["one_voice"]["text"]),
                        ("attributed", attributed),
                    ]
                },
            )
            atomic_json(file, output)
            rows.append(output)
    report = {
        "split": args.split,
        "cases": len(rows),
        "gate": dict(GATE),
        "model_manifest": runtime.manifest,
        "manifest_sha256": digest(args.manifest),
        "summary": summarize(rows),
        "seconds": round(time.monotonic() - began, 2),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if os.uname().sysname == "Darwin" else 1024),
        "protocol": "First qualifying disjoint-speaker pairs in published CSV order, alternating target side. Full source utterances, zero-padded to the longer duration, official gains then common peak attenuation. Not the official min-duration separation benchmark.",
        "limitations": "Small clean public read-speech set; test-clean has been used historically for this project. No claim of a new untouched test, live microphone robustness, or calibrated identity confidence. WER uses lowercase alphanumeric words with apostrophes removed; absent/silence report word counts, not WER.",
    }
    atomic_json(args.output / "summary.json", report)
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
