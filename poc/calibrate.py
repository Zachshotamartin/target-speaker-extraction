"""Inspect a fixed rule grid on public development results; no weight updates."""

import argparse
import json
from pathlib import Path

from poc.common import atomic_json
from poc.evaluate import errors
from poc.pipeline import GATE, attribute, classify


def replay(row, settings):
    windows = [
        {
            **w,
            "attribution": classify(
                w["extracted_similarity"], w["original_similarity"], w["speech_seconds"], settings
            ),
        }
        for w in row["windows"]
    ]
    words = (row.get("comparison") or {}).get("one_voice", {}).get("words", [])
    spans = [
        [max(a, c), min(b, d)]
        for a, b in row["speech"]
        for c, d in row.get("extracted_speech", [])
        if min(b, d) > max(a, c)
    ]
    segments = attribute(words, windows, spans, row["duration"])
    text = " ".join(s["text"].strip() for s in segments if s["attribution"] == "accepted")
    return segments, windows, errors(row["reference_text"], text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.dev / "summary.json").read_text())
    if summary["split"] != "dev":
        raise ValueError("Threshold selection accepts the development partition only.")
    rows = [json.loads(p.read_text()) for p in sorted(args.dev.glob("[0-9]*.json"))]
    candidates = []
    for extracted in [0.30, 0.35, 0.40, 0.45, 0.50]:
        for original in [0.10, 0.15, 0.20, 0.25, 0.30]:
            settings = {
                **GATE,
                "accepted_extracted": extracted,
                "accepted_original": original,
                "uncertain_extracted": 0.20,
                "version": "public-dev32-v1",
            }
            scores = {
                condition: {"edits": 0, "words": 0, "false_jobs": 0}
                for condition in ["overlap", "target_only", "absent", "silence"]
            }
            for row in rows:
                _, _, score = replay(row, settings)
                group = scores[row["condition"]]
                group["edits"] += score["edits"]
                group["words"] += score["reference_words"]
                group["false_jobs"] += int(
                    score["hypothesis_words"] > 0 and row["condition"] in {"absent", "silence"}
                )
            candidates.append({"settings": settings, "scores": scores})

    def order(candidate):
        scores = candidate["scores"]
        return (
            scores["absent"]["false_jobs"] + scores["silence"]["false_jobs"],
            scores["overlap"]["edits"] + scores["target_only"]["edits"],
            -candidate["settings"]["accepted_extracted"],
            -candidate["settings"]["accepted_original"],
        )

    best = min(candidates, key=order)
    atomic_json(
        args.output,
        {
            "dev_cases": len(rows),
            "selection": "Minimize false-attribution jobs first, then total target-present edits; ties prefer stricter thresholds. No neural weights updated.",
            "chosen": best,
            "grid": candidates,
        },
    )
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
