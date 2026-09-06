"""Speaker-disjoint manifests and deterministic, reference-conditioned speech mixtures."""

from __future__ import annotations

import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import fftconvolve, lfilter

from tse.audio import read_audio
from tse.utils import atomic_json, sha256

SPLITS = {"train": "train-clean-100", "dev": "dev-clean", "test": "test-clean"}
CONDITIONS = ("clean", "noise", "channel", "reverb", "combined")


def inventory(root: Path, output: Path) -> dict:
    records = []
    rejected = []
    for path in sorted((root / "LibriSpeech").rglob("*.flac")):
        parts = path.relative_to(root).parts
        if len(parts) != 5 or parts[1] not in SPLITS.values():
            continue
        info = sf.info(path)
        if info.samplerate != 16000 or info.channels != 1 or info.duration < 2:
            rejected.append(
                {"id": path.stem, "reason": "Expected mono 16 kHz and at least 2 seconds"}
            )
            continue
        records.append(
            {
                "id": path.stem,
                "speaker": parts[2],
                "chapter": parts[3],
                "split": next(key for key, value in SPLITS.items() if value == parts[1]),
                "path": str(path.relative_to(root)),
                "samples": info.frames,
                "sample_rate": info.samplerate,
                "sha256": sha256(path),
            }
        )
    if not records:
        raise ValueError("No usable LibriSpeech files found")
    report = audit_records(records)
    result = {
        "schema_version": 1,
        "source": "LibriSpeech / OpenSLR SLR12",
        "license": "CC-BY-4.0",
        "records": records,
        "rejected": rejected,
        "audit": report,
    }
    atomic_json(output, result)
    return result


def audit_records(records: list[dict]) -> dict:
    ids = set()
    speakers: dict[str, set] = defaultdict(set)
    hashes: dict[str, str] = {}
    for row in records:
        if row["id"] in ids:
            raise ValueError("Duplicate utterance identifier")
        ids.add(row["id"])
        if row["split"] not in SPLITS:
            raise ValueError("Unknown split")
        path = Path(row["path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Manifest paths must remain inside the data root")
        speakers[row["split"]].add(row["speaker"])
        previous = hashes.get(row["sha256"])
        if previous is not None and previous != row["split"]:
            raise ValueError("Identical audio crosses corpus splits")
        hashes[row["sha256"]] = row["split"]
    for left, right in [("train", "dev"), ("train", "test"), ("dev", "test")]:
        if speakers[left] & speakers[right]:
            raise ValueError(f"Speaker leakage between {left} and {right}")
    return {
        split: {
            "speakers": len(speakers[split]),
            "utterances": sum(r["split"] == split for r in records),
            "hours": round(
                sum(r["samples"] for r in records if r["split"] == split) / 16000 / 3600, 4
            ),
        }
        for split in SPLITS
    }


def augment_reference(samples: np.ndarray, condition: str, seed: int) -> np.ndarray:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown reference condition: {condition}")
    rng = np.random.default_rng(seed)
    output = samples.copy()
    if condition in {"channel", "combined"}:
        coefficient = float(rng.uniform(0.6, 0.9))
        output = lfilter([1 - coefficient], [1, -coefficient], output).astype(np.float32)
    if condition in {"reverb", "combined"}:
        impulse = np.zeros(int(0.18 * 16000), dtype=np.float32)
        impulse[0] = 1
        delays = rng.integers(80, len(impulse), size=12)
        impulse[delays] += rng.uniform(-0.35, 0.35, 12) * np.exp(-delays / 1000)
        output = fftconvolve(output, impulse)[: len(output)].astype(np.float32)
    if condition in {"noise", "combined"}:
        snr = 10.0 if condition == "noise" else 15.0
        noise = rng.standard_normal(len(output)).astype(np.float32)
        noise *= np.sqrt(np.mean(output**2) + 1e-8) / (10 ** (snr / 20))
        output += noise
    rms = np.sqrt(np.mean(output**2) + 1e-8)
    return (output * (0.1 / rms)).astype(np.float32)


class SpeechCorpus:
    def __init__(
        self,
        root: Path,
        manifest: Path,
        split: str,
        seconds: float = 2,
        reference_seconds: float = 5,
    ):
        self.root = root.resolve()
        self.manifest = manifest
        self.manifest_hash = sha256(manifest)
        self._verified_sources: set[str] = set()
        payload = json.loads(manifest.read_text())
        audit_records(payload["records"])
        self.records = {r["id"]: r for r in payload["records"]}
        self.split = split
        self.length = int(seconds * 16000)
        self.reference_length = int(reference_seconds * 16000)
        self.by_speaker: dict[str, list[dict]] = defaultdict(list)
        for row in self.records.values():
            if row["split"] == split and row["samples"] >= max(self.length, self.reference_length):
                self.by_speaker[row["speaker"]].append(row)
        self.by_speaker = {key: rows for key, rows in self.by_speaker.items() if len(rows) >= 3}
        self.speakers = sorted(self.by_speaker)
        if len(self.speakers) < 2:
            raise ValueError(
                f"Need at least two {split} speakers with distinct eligible references"
            )
        self.speaker_labels = {speaker: index for index, speaker in enumerate(self.speakers)}

    @lru_cache(maxsize=128)  # noqa: B019 -- Bounded cache; corpus lifetime is one run.
    def read(self, utterance: str) -> np.ndarray:
        row = self.records[utterance]
        path = (self.root / row["path"]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Audio path escapes the data root")
        if utterance not in self._verified_sources:
            if sha256(path) != row["sha256"]:
                raise ValueError(f"Audio checksum differs from inventory: {utterance}")
            self._verified_sources.add(utterance)
        return read_audio(path)

    def _offset(self, row: dict, count: int, rng: np.random.Generator) -> int:
        audio = self.read(row["id"])
        for _ in range(20):
            offset = int(rng.integers(max(1, len(audio) - count + 1)))
            crop = audio[offset : offset + count]
            if len(crop) == count and np.sqrt(np.mean(crop**2)) > 0.001:
                return offset
        raise ValueError(f"Could not find active speech in {row['id']}")

    def make_case(self, seed: int, target_speaker: str | None = None) -> dict:
        rng = np.random.default_rng(seed)
        speaker = target_speaker or self.speakers[int(rng.integers(len(self.speakers)))]
        others = [s for s in self.speakers if s != speaker]
        other = others[int(rng.integers(len(others)))]
        sources = []
        references = []
        for identity in (speaker, other):
            rows = self.by_speaker[identity]
            source = rows[int(rng.integers(len(rows)))]
            candidates = [
                r for r in rows if r["id"] != source["id"] and r["chapter"] != source["chapter"]
            ]
            if not candidates:
                candidates = [r for r in rows if r["id"] != source["id"]]
            reference = candidates[int(rng.integers(len(candidates)))]
            sources.append({"id": source["id"], "offset": self._offset(source, self.length, rng)})
            references.append(
                {
                    "id": reference["id"],
                    "offset": self._offset(reference, self.reference_length, rng),
                }
            )
        return {
            "case_id": f"{self.split}-{seed}",
            "split": self.split,
            "seed": seed,
            "sources": sources,
            "references": references,
            "target_index": 0,
            "samples": self.length,
            "reference_samples": self.reference_length,
            "ratio_db": float(rng.uniform(-5, 5)),
            "condition": "clean",
            "protocol": "custom-librispeech-tse-v1",
        }

    def validate_case(self, case: dict) -> None:
        if case["split"] != self.split or case["protocol"] != "custom-librispeech-tse-v1":
            raise ValueError("Case protocol or split does not match the corpus")
        if case["target_index"] not in (0, 1):
            raise ValueError("Invalid target index")
        identities = []
        source_ids = {source["id"] for source in case["sources"]}
        for source, reference in zip(case["sources"], case["references"], strict=True):
            s, r = self.records[source["id"]], self.records[reference["id"]]
            if s["split"] != self.split or r["split"] != self.split:
                raise ValueError("Case contains audio from another split")
            if s["speaker"] != r["speaker"] or r["id"] in source_ids:
                raise ValueError("Reference identity mismatch or identical source/reference")
            for entry, row, count in [
                (source, s, case["samples"]),
                (reference, r, case["reference_samples"]),
            ]:
                if entry["offset"] < 0 or count < 1 or entry["offset"] + count > row["samples"]:
                    raise ValueError("Invalid crop bounds")
            identities.append(s["speaker"])
        if len(identities) != 2 or identities[0] == identities[1]:
            raise ValueError("Mixture must contain two different speakers")

    def render(self, case: dict, condition: str | None = None) -> dict:
        self.validate_case(case)
        signals = []
        for entry in case["sources"]:
            samples = self.read(entry["id"])[
                entry["offset"] : entry["offset"] + case["samples"]
            ].copy()
            samples -= samples.mean()
            rms = np.sqrt(np.mean(samples**2))
            if rms < 1e-5:
                raise ValueError("Source crop is silent")
            signals.append(samples * (0.1 / rms))
        signals[1] *= 10 ** (-case["ratio_db"] / 20)
        mixture = signals[0] + signals[1]
        common_gain = min(1.0, 0.9 / max(float(np.max(np.abs(mixture))), 1e-6))
        mixture *= common_gain
        signals = [signal * common_gain for signal in signals]
        index = case["target_index"]
        entry = case["references"][index]
        reference = self.read(entry["id"])[
            entry["offset"] : entry["offset"] + case["reference_samples"]
        ].copy()
        reference -= reference.mean()
        selected = condition or case.get("condition", "clean")
        reference = augment_reference(reference, selected, case["seed"] + 10000019)
        speaker = self.records[case["sources"][index]["id"]]["speaker"]
        return {
            "mixture": mixture.astype(np.float32),
            "target": signals[index].astype(np.float32),
            "interferer": signals[1 - index].astype(np.float32),
            "reference": reference,
            "speaker": speaker,
            "label": self.speaker_labels.get(speaker, -1),
            "common_gain": common_gain,
        }

    def batch(
        self, cases: list[dict], device: torch.device, conditions: list[str] | None = None
    ) -> dict:
        rows = [
            self.render(case, conditions[j] if conditions else None) for j, case in enumerate(cases)
        ]
        result = {
            key: torch.from_numpy(np.stack([row[key] for row in rows])[:, None]).to(device)
            for key in ("mixture", "target", "interferer", "reference")
        }
        result["labels"] = torch.tensor(
            [row["label"] for row in rows], device=device, dtype=torch.long
        )
        result["speakers"] = [row["speaker"] for row in rows]
        return result


def build_cases(corpus: SpeechCorpus, count: int, seed: int, output: Path) -> dict:
    if count < 2 or count % 2:
        raise ValueError("Use an even case count for paired reference switches")
    cases = []
    for index in range(count // 2):
        case = corpus.make_case(seed + index, corpus.speakers[index % len(corpus.speakers)])
        corpus.validate_case(case)
        cases.append(case)
        swapped = dict(case, target_index=1, case_id=case["case_id"] + "-swap")
        cases.append(swapped)
    payload = {
        "schema_version": 1,
        "protocol": "custom-librispeech-tse-v1",
        "source_manifest_sha256": corpus.manifest_hash,
        "split": corpus.split,
        "seed": seed,
        "cases": cases,
    }
    atomic_json(output, payload)
    return payload


def load_cases(path: Path, corpus: SpeechCorpus) -> list[dict]:
    payload = json.loads(path.read_text())
    if len({case["case_id"] for case in payload["cases"]}) != len(payload["cases"]):
        raise ValueError("Duplicate evaluation case identifiers")
    if payload["source_manifest_sha256"] != corpus.manifest_hash:
        raise ValueError("Case manifest refers to a different source inventory")
    if payload["split"] != corpus.split:
        raise ValueError("Case manifest split mismatch")
    for case in payload["cases"]:
        corpus.validate_case(case)
    return payload["cases"]
