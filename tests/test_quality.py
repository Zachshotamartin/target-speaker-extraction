import json

import numpy as np
import pytest

from tse.quality import compare_quality, source_projection, validate_evaluation
from tse.utils import atomic_json, sha256


def test_test_scoring_requires_exact_frozen_artifacts(tmp_path):
    checkpoint, cases, manifest, freeze = [
        tmp_path / name for name in ("model.pt", "cases.json", "inventory.json", "freeze.json")
    ]
    checkpoint.write_bytes(b"frozen checkpoint")
    manifest.write_text("source inventory")
    atomic_json(cases, {"split": "test"})
    with pytest.raises(ValueError, match="test requires"):
        validate_evaluation(checkpoint, cases, manifest, None)
    atomic_json(
        freeze,
        {
            "status": "frozen_before_fresh_test",
            "case_manifest_sha256": sha256(cases),
            "source_manifest_sha256": sha256(manifest),
            "checkpoints": {"candidate": {"checkpoint_sha256": sha256(checkpoint)}},
        },
    )
    assert validate_evaluation(checkpoint, cases, manifest, freeze) == "test"
    frozen = json.loads(freeze.read_text())
    atomic_json(freeze, {**frozen, "evaluation_source_tree_sha256": "different source"})
    with pytest.raises(ValueError, match="frozen implementation"):
        validate_evaluation(checkpoint, cases, manifest, freeze)
    atomic_json(freeze, frozen)
    checkpoint.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="not in the frozen"):
        validate_evaluation(checkpoint, cases, manifest, freeze)
    atomic_json(cases, {"split": "test", "changed": True})
    with pytest.raises(ValueError, match="case_manifest"):
        validate_evaluation(checkpoint, cases, manifest, freeze)
    atomic_json(cases, {"split": "dev"})
    assert validate_evaluation(checkpoint, cases, manifest, None) == "dev"
    atomic_json(cases, {"split": "train"})
    with pytest.raises(ValueError):
        validate_evaluation(checkpoint, cases, manifest, None)


def test_source_projection_distinguishes_gain_from_suppression_and_distortion():
    rng = np.random.default_rng(48)
    target, other, noise = rng.normal(size=(3, 8000))
    reduced = source_projection(0.1 * (target + other), target, other)
    separated = source_projection(0.1 * target + 0.01 * other, target, other)
    distorted = source_projection(target + 0.5 * noise, target, other)
    assert reduced["relative_interferer_attenuation_db"] == pytest.approx(0, abs=1e-5)
    assert separated["relative_interferer_attenuation_db"] == pytest.approx(20, abs=1e-4)
    assert separated["artifact_proxy_fraction"] < 1e-10
    assert distorted["artifact_proxy_fraction"] > 0.1


def test_quality_comparison_pairs_by_id_and_rejects_mismatches(tmp_path):
    paths = [tmp_path / name for name in ("baseline.json", "candidate.json", "result.json")]
    rows = [
        {
            "case_id": str(i),
            "target_speaker": str(i),
            "si_sdri_db": float(i),
            "estoi": 0.4,
            "artifact_proxy_fraction": 0.2,
            "confused": False,
        }
        for i in range(2)
    ]
    baseline = {"case_manifest_sha256": "cases", "source_manifest_sha256": "source", "rows": rows}
    candidate = {
        **baseline,
        "rows": [
            {**row, "si_sdri_db": row["si_sdri_db"] + 2, "estoi": 0.6} for row in reversed(rows)
        ],
    }
    atomic_json(paths[0], baseline)
    atomic_json(paths[1], candidate)
    result = compare_quality(*paths)
    assert result["metrics"]["si_sdri_db"]["paired_ci95"] == [2.0, 2.0]
    assert result["metrics"]["estoi"]["mean_candidate_minus_baseline"] == pytest.approx(0.2)
    candidate["rows"][0]["target_speaker"] = "different"
    atomic_json(paths[1], candidate)
    with pytest.raises(ValueError, match="speakers differ"):
        compare_quality(*paths)
