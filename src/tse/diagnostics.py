"""Reference-duration and target-absence diagnostics outside headline quality metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from tse.data import SpeechCorpus, augment_reference, load_cases
from tse.engine import evaluate_model, load_model
from tse.metrics import measure, summarize
from tse.utils import atomic_json, sha256


def diagnose(
    checkpoint: Path,
    root: Path,
    manifest: Path,
    cases_path: Path,
    output: Path,
    device_name: str = "mps",
    limit: int = 80,
) -> dict:
    if limit < 2:
        raise ValueError("Use at least two diagnostic cases")
    model, payload = load_model(checkpoint, device_name)
    protocol = json.loads(cases_path.read_text())
    corpus = SpeechCorpus(root, manifest, protocol["split"], 4, 5)
    cases = load_cases(cases_path, corpus)[:limit]
    device = next(model.parameters()).device
    duration_results = {}
    for seconds in (1, 3, 5, 10):
        selected = []
        for case in cases:
            if all(
                entry["offset"] + seconds * 16000 <= corpus.records[entry["id"]]["samples"]
                for entry in case["references"]
            ):
                selected.append(dict(case, reference_samples=seconds * 16000))
        if selected:
            rows = evaluate_model(model, corpus, selected, batch_size=2)
            duration_results[str(seconds)] = {
                "eligible_cases": len(selected),
                "excluded_cases": len(cases) - len(selected),
                "summary": summarize(rows),
                "case_ids": [c["case_id"] for c in selected],
                "rows": rows,
            }
        else:
            duration_results[str(seconds)] = {"eligible_cases": 0, "excluded_cases": len(cases)}
    absent = []
    constant_rows = []
    with torch.inference_mode():
        for case in cases:
            batch = corpus.batch([case], device)
            embedding = torch.zeros(1, model.config.reference_encoder.embedding_dim, device=device)
            constant_prediction = model.extract(batch["mixture"], embedding)
            constant_rows.append(
                {
                    "case_id": case["case_id"],
                    "target_speaker": batch["speakers"][0],
                    **measure(
                        constant_prediction, batch["mixture"], batch["target"], batch["interferer"]
                    )[0],
                }
            )
            speakers = {corpus.records[source["id"]]["speaker"] for source in case["sources"]}
            candidates = [speaker for speaker in corpus.speakers if speaker not in speakers]
            if not candidates:
                continue
            speaker = candidates[case["seed"] % len(candidates)]
            source = corpus.by_speaker[speaker][0]
            reference = corpus.read(source["id"])[:80000].copy()
            reference -= reference.mean()
            reference = augment_reference(reference, "clean", case["seed"])
            prediction = model(batch["mixture"], torch.from_numpy(reference[None, None]).to(device))
            ratio = float(
                10
                * torch.log10(
                    (prediction.square().mean() + 1e-10)
                    / (batch["mixture"].square().mean() + 1e-10)
                )
            )
            absent.append(
                {
                    "case_id": case["case_id"],
                    "absent_reference_speaker": speaker,
                    "output_mixture_energy_db": ratio,
                }
            )
    result = {
        "checkpoint_sha256": sha256(checkpoint),
        "case_manifest_sha256": sha256(cases_path),
        "source_manifest_sha256": sha256(manifest),
        "split": protocol["split"],
        "selected_step": payload["step"],
        "reference_duration_seconds": duration_results,
        "duration_note": "Eligible subsets differ at longer durations; compare paired common IDs for causal duration claims.",
        "constant_conditioning": {
            "summary": summarize(constant_rows),
            "rows": constant_rows,
            "interpretation": "Reference embedding replaced by all zeros at inference only. This out-of-distribution intervention diagnoses reference use; it is not an independently trained separation baseline.",
        },
        "absent_targets": {
            "cases": len(absent),
            "mean_output_mixture_energy_db": float(
                np.mean([r["output_mixture_energy_db"] for r in absent])
            )
            if absent
            else None,
            "output_above_minus20db_fraction": float(
                np.mean([r["output_mixture_energy_db"] > -20 for r in absent])
            )
            if absent
            else None,
            "rows": absent,
            "interpretation": "The model assumes its target is present. Emitting audio with an absent reference is a known failure, not a valid extraction or an SI-SDR score.",
        },
    }
    atomic_json(output, result)
    return result
