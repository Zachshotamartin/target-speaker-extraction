"""Paired acoustic-condition comparisons, retaining target-speaker clusters."""

import json
from collections import defaultdict

import numpy as np

from tse.utils import atomic_json, sha256


def paired_metric(pairs, metric):
    groups = defaultdict(list)
    baseline, candidate = [], []
    for left, right in pairs:
        a, b = left.get(metric), right.get(metric)
        if a is None or b is None:
            continue
        if not np.isfinite(float(a) + float(b)):
            raise ValueError("Non-finite paired metric")
        baseline.append(float(a))
        candidate.append(float(b))
        groups[left["target_speaker"]].append(float(b) - float(a))
    if not groups:
        return None
    arrays = [np.array(groups[speaker]) for speaker in sorted(groups)]
    rng = np.random.default_rng(42)
    draws = [
        np.concatenate([arrays[j] for j in rng.integers(len(arrays), size=len(arrays))]).mean()
        for _ in range(1000)
    ]
    return {
        "paired_cases": len(baseline),
        "target_speakers": len(groups),
        "baseline_mean": float(np.mean(baseline)),
        "candidate_mean": float(np.mean(candidate)),
        "mean_candidate_minus_baseline": float(np.mean(candidate) - np.mean(baseline)),
        "paired_ci95": np.percentile(draws, [2.5, 97.5]).tolist(),
    }


def compare_acoustic(baseline_path, candidate_path, output):
    if output.exists():
        raise FileExistsError("Preserve prior comparisons")
    baseline, candidate = [json.loads(path.read_text()) for path in (baseline_path, candidate_path)]
    identities = (
        "case_manifest_sha256",
        "source_manifest_sha256",
        "environment_manifest_sha256",
        "split",
        "protocol",
        "selection_freeze_sha256",
    )
    for key in identities:
        if baseline.get(key) != candidate.get(key):
            raise ValueError(f"Acoustic comparison requires matching {key}")
    tables = [{row["case_id"]: row for row in report["rows"]} for report in (baseline, candidate)]
    if any(
        len(table) != len(report["rows"])
        for table, report in zip(tables, (baseline, candidate), strict=True)
    ):
        raise ValueError("Duplicate acoustic request")
    if tables[0].keys() != tables[1].keys():
        raise ValueError("Acoustic requests differ")
    pairs = [(tables[0][key], tables[1][key]) for key in sorted(tables[0])]
    for left, right in pairs:
        if any(
            left[key] != right[key]
            for key in ("condition", "target_speaker", "target_present", "output_samples")
        ):
            raise ValueError("Acoustic request conditions, speakers or timelines differ")
    metrics = ("si_sdri_db", "estoi", "confused", "artifact_proxy_fraction")
    conditions = {}
    for condition in sorted({left["condition"] for left, _ in pairs}):
        current = [(a, b) for a, b in pairs if a["condition"] == condition]
        present = {a["target_present"] for a, _ in current}
        if len(present) != 1:
            raise ValueError("A condition mixes present and absent requests")
        chosen_metrics = metrics if True in present else ("attenuation_db",)
        conditions[condition] = {
            metric: paired_metric(current, metric) for metric in chosen_metrics
        }
    present_pairs = [(a, b) for a, b in pairs if a["target_present"]]
    result = {
        "baseline_report_sha256": sha256(baseline_path),
        "candidate_report_sha256": sha256(candidate_path),
        **{key: baseline.get(key) for key in identities},
        "baseline_checkpoint_sha256": baseline["checkpoint_sha256"],
        "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
        "requests": len(pairs),
        "conditions": conditions,
        "all_present": {metric: paired_metric(present_pairs, metric) for metric in metrics},
        "interpretation": "Paired candidate-minus-baseline differences. Higher SI-SDRi, ESTOI and absent-target attenuation are favorable; lower confusion and distortion proxy are favorable. Approximate 95% target-speaker cluster bootstrap, 1000 replicates, seed 42. Repeated conditions stay together within target-speaker clusters; dependence through shared interferers and rooms remains. These intervals do not include training-seed uncertainty. Static is a human listening judgment, not the scalar artifact proxy.",
    }
    atomic_json(output, result)
    return result
