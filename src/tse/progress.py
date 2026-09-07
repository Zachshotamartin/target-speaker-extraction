"""Read compact progress for the local v3 experiment sequence."""

import json
import os
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def experiment_progress(root=Path(".")):
    suite = read_json(root / "artifacts/verification/v3-suite-status.json")
    selection = read_json(root / "reports/v3-architecture-selection.json")
    extended = selection.get("selected", "").removesuffix("-pilot")
    active = False
    if suite.get("pid"):
        try:
            os.kill(suite["pid"], 0)
            active = True
        except ProcessLookupError:
            pass
    specifications = [
        ("continuation-control", "Learning rate · plateau", 20000, 25000),
        ("continuation-cosine", "Learning rate · cosine decay", 20000, 25000),
        ("band-real", "Band separator · real mask", 0, 4000),
        ("band-complex", "Band separator · phase correction", 0, 4000),
        ("band-resnet", "Band separator · ResNet reference", 0, 4000),
        (
            extended or "architecture-not-selected",
            "Selected architecture · longer training",
            4000,
            10000,
        ),
        ("adaptation-clean", "Adaptation · clean control", 0, 3000),
        ("adaptation-realistic", "Adaptation · realistic acoustics", 0, 3000),
    ]
    rows = []
    for name, label, start, end in specifications:
        directory = root / "artifacts/runs" / f"v3-{name}"
        metrics = directory / "metrics.jsonl"
        entries = []
        if metrics.is_file():
            # A partial final write may be visible to the reader; prior entries
            # remain valid. Reading only the tail bounds every poll's work.
            with metrics.open("rb") as stream:
                stream.seek(max(0, metrics.stat().st_size - 32768))
                lines = stream.read().splitlines()
            for line in lines:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        latest_step = max((entry.get("step", 0) for entry in entries), default=0)
        validations = [entry for entry in entries if entry.get("event") == "validation"]
        complete = read_json(directory / "summary.json").get("steps", 0) >= end
        performed = min(end - start, max(0, latest_step - start))
        state = (
            "complete"
            if complete
            else "training"
            if performed and active
            else "paused"
            if performed
            else "waiting"
        )
        if latest_step == end and not complete and active:
            state = "validating"
        rows.append(
            {
                "name": label,
                "updates": performed,
                "budget": end - start,
                "status": state,
                "last_validation_si_sdri_db": validations[-1]["mean_si_sdri_db"]
                if validations
                else None,
            }
        )
    final = read_json(root / "reports/v3-development-selection.json")
    studies = []
    gallery = root / "artifacts/gallery"
    if gallery.exists():
        studies = [
            f"/gallery/{path.name}/"
            for path in sorted(gallery.glob("v3-*-listening"))
            if (path / "index.html").is_file()
        ]
    return {
        "status": suite.get("status", "not_started"),
        "job_process_alive": active,
        "stages": rows,
        "selected_candidate": final.get("selected", {}).get("name"),
        "fresh_test_complete": (root / "reports/v3-test-comparison.json").is_file(),
        "listening_studies": studies,
        "note": "These counters track training work. Validation scores use each run's declared selection set and should not be compared across different sets. Promotion also requires the larger development comparisons and delivery checks.",
    }
