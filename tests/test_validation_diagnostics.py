import importlib.util

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "validation_diagnostics", "scripts/publish_validation_diagnostics.py"
)
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


def evaluation(scores, step):
    return dict(
        rows=[
            dict(case_id=str(i), target_speaker=str(i % 2), si_sdri_db=score, confused=i == 0)
            for i, score in enumerate(scores)
        ],
        cases=len(scores),
        manifest_sha256="same-manifest",
        mean_si_sdri_db=float(np.mean(scores)),
        positive_improvement_fraction=float(np.mean(np.array(scores) > 0)),
        confusion_fraction=1 / len(scores),
        epoch=step,
        step=step,
    )


def history(*evaluations):
    return [dict(event="validation", kind="full", **r) for r in evaluations]


def test_pairing_and_threshold_boundaries():
    best = evaluation([-2, 0, 5, 10], 80)
    latest = evaluation([-1, -2, 7, 9], 81)
    latest["rows"].reverse()
    result = diagnostics.analyze(best, latest, history(best, latest))
    assert [r["delta_db"] for r in result["paired"]] == [1, -2, 2, -1]
    summary = result["summary"]
    assert summary["best"]["no_improvement"] == 2  # Includes exactly zero.
    assert summary["best"]["improved"] == 2
    assert summary["paired"] == dict(
        mean_change=0, improved_over_1db=1, within_1db=2, worsened_over_1db=1
    )
    assert sum(row["cases"] for row in result["speakers"]) == 4
    assert summary["speakers"] == 2
    # The two failure signals overlap; they are independently measured, not a partition.
    assert result["failures"][-1]["no_improvement_percent"] == 50
    assert result["failures"][-1]["wrong_speaker_percent"] == 25


@pytest.mark.parametrize("issue", ["duplicate", "missing", "speaker", "nonfinite", "manifest"])
def test_misaligned_validation_is_rejected(issue):
    best = evaluation([1, 2, 3, 4], 80)
    latest = evaluation([1, 2, 3, 4], 81)
    if issue == "duplicate":
        latest["rows"].append(latest["rows"][0])
    elif issue == "missing":
        latest["rows"].pop()
    elif issue == "speaker":
        latest["rows"][0]["target_speaker"] = "different"
    elif issue == "nonfinite":
        latest["rows"][0]["si_sdri_db"] = float("nan")
    else:
        latest["manifest_sha256"] = "different"
    with pytest.raises(ValueError):
        diagnostics.paired_rows(best, latest)


def test_history_excludes_monitor_and_newer_checkpoints():
    best = evaluation([1, 2, 3, 4], 80)
    latest = evaluation([1, 2, 3, 4], 81)
    future = evaluation([1, 2, 3, 4], 82)
    monitor = dict(event="validation", kind="monitor", **evaluation([1, 2], 81))
    records = history(latest, future, best, latest) + [monitor]
    assert [r["step"] for r in diagnostics.analyze(best, latest, records)["failures"]] == [80, 81]


def test_aggregate_mismatch_is_rejected():
    best = evaluation([1, 2, 3, 4], 80)
    latest = evaluation([1, 2, 3, 4], 81)
    best["mean_si_sdri_db"] = 100
    with pytest.raises(ValueError, match="aggregate"):
        diagnostics.analyze(best, latest, history(best, latest))
