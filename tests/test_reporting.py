import copy

import pytest

from tse.reporting import compare
from tse.utils import atomic_json


def test_paired_comparison_rejects_uncontrolled_changes(tiny_config, tmp_path):
    config = tiny_config.model_dump()
    rows = [
        {
            "case_id": "a",
            "condition": "clean",
            "target_speaker": "1",
            "si_sdri_db": 2.0,
            "confused": False,
        },
        {
            "case_id": "b",
            "condition": "noise",
            "target_speaker": "2",
            "si_sdri_db": 1.0,
            "confused": False,
        },
    ]
    control = {
        "config": config,
        "rows": rows,
        "case_manifest_sha256": "cases",
        "source_manifest_sha256": "source",
        "split": "dev",
        "protocol": "custom",
        "training_step": 2,
    }
    treatment = copy.deepcopy(control)
    treatment["config"]["augmentation"]["reference_enabled"] = True
    treatment["rows"][1]["si_sdri_db"] = 2.0
    a, b, result = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "comparison.json"
    atomic_json(a, control)
    atomic_json(b, treatment)
    assert compare(a, b, result, 10)["equal_weight_mismatch_gain_db"] == 1.0
    treatment["config"]["training"]["learning_rate"] *= 2
    atomic_json(b, treatment)
    with pytest.raises(ValueError, match="different training"):
        compare(a, b, result, 10)
