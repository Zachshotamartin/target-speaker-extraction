import json

import pytest

from tse.acoustic_comparison import compare_acoustic
from tse.utils import atomic_json


def test_acoustic_comparison_pairs_conditions_and_excludes_absence_from_si_sdr(tmp_path):
    baseline, candidate, output = [tmp_path / name for name in ("a.json", "b.json", "out.json")]
    rows = [
        {
            "case_id": f"{speaker}-{condition}",
            "target_speaker": speaker,
            "condition": condition,
            "target_present": condition != "target_absent",
            "output_samples": 64000,
            **(
                {"si_sdri_db": 2.0, "estoi": 0.5, "confused": False, "artifact_proxy_fraction": 0.2}
                if condition != "target_absent"
                else {"attenuation_db": 7.0}
            ),
        }
        for speaker in ("1", "2")
        for condition in ("clean", "noise", "target_absent")
    ]
    original = {
        "rows": rows,
        "checkpoint_sha256": "original",
        "split": "dev",
        "case_manifest_sha256": "cases",
        "source_manifest_sha256": "speech",
        "environment_manifest_sha256": "acoustics",
        "protocol": "realistic-tse-v1",
    }
    changed = {
        **original,
        "checkpoint_sha256": "changed",
        "rows": [
            {**row, **({"si_sdri_db": 4.0} if row["target_present"] else {"attenuation_db": 12.0})}
            for row in reversed(rows)
        ],
    }
    atomic_json(baseline, original)
    atomic_json(candidate, changed)
    result = compare_acoustic(baseline, candidate, output)
    assert result["all_present"]["si_sdri_db"]["paired_cases"] == 4
    assert result["all_present"]["si_sdri_db"]["paired_ci95"] == [2.0, 2.0]
    assert result["conditions"]["target_absent"]["attenuation_db"]["paired_ci95"] == [5.0, 5.0]
    with pytest.raises(FileExistsError):
        compare_acoustic(baseline, candidate, output)
    changed["rows"][0]["target_speaker"] = "wrong"
    atomic_json(candidate, changed)
    with pytest.raises(ValueError, match="speakers or timelines"):
        compare_acoustic(baseline, candidate, tmp_path / "bad.json")
    changed = json.loads(baseline.read_text())
    changed["environment_manifest_sha256"] = "other rooms"
    atomic_json(candidate, changed)
    with pytest.raises(ValueError, match="environment_manifest"):
        compare_acoustic(baseline, candidate, tmp_path / "bad.json")
