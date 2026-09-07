"""Small known-voice task with utterance-disjoint development and test audio."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from tse.librimix import LibriMixCorpus, request_seed
from tse.utils import atomic_json, sha256

PROTOCOL = "known-voice-concept-v1"


def prepare_concept(root, source_manifest, checkpoint, gate_path, output):
    """Reuse project learning, reserving audio never used by that initialization."""
    output = Path(output)
    if output.exists():
        raise FileExistsError("Keep the existing concept reservation")
    gate = json.loads(Path(gate_path).read_text())
    if gate["status"] != "passed" or gate["checkpoint_sha256"] != sha256(checkpoint):
        raise ValueError("Use the exact project checkpoint from the passed learning check")
    corpus = LibriMixCorpus(root, source_manifest, "train")
    if corpus.manifest_hash != gate["manifest_sha256"]:
        raise ValueError("Learning-check data identity differs")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["provenance"]["fixed_indices"] != list(range(16)):
        raise ValueError("Initialization exposure must be the recorded sixteen fixed requests")
    excluded, familiar = set(), set()
    for index in range(16):
        row = corpus.rows[index // 2]
        excluded.update(source["utterance"] for source in row["sources"])
        request = corpus.request(index, request_seed(payload["config"]["seed"], 0, index), 48000)
        excluded.add(corpus.known_files[request["reference_path"]]["utterance"])
        familiar.add(request["speaker"])
    speakers = sorted(familiar)[:8]
    splits = {split: {} for split in ("train", "dev", "test")}
    lengths = {source["path"]: row["samples"] for row in corpus.rows for source in row["sources"]}
    for speaker in speakers:
        eligible = [
            source
            for source in corpus.pools[speaker]
            if source["utterance"] not in excluded and lengths[source["path"]] >= 64000
        ]
        eligible.sort(
            key=lambda row: hashlib.sha256(f"concept-42:{row['utterance']}".encode()).digest()
        )
        if len(eligible) < 45:
            raise ValueError("Each voice needs at least 45 eligible, previously unused utterances")
        splits["dev"][speaker] = eligible[:10]
        splits["test"][speaker] = eligible[10:20]
        splits["train"][speaker] = eligible[20:]
    report = {
        "protocol": PROTOCOL,
        "source_manifest_sha256": corpus.manifest_hash,
        "initialization_sha256": sha256(checkpoint),
        "initialization_update": payload["step"],
        "initialization_excluded_utterances": sorted(excluded),
        "classifier_speakers": payload["provenance"]["train_speakers"],
        "speakers": speakers,
        "splits": splits,
        "mixture_seconds": 3,
        "reference_seconds": 3,
        "development_seconds": 4,
        "relative_level_db": [-3, 3],
        "development_pairs": 16,
        "test_pairs": 32,
        "scope": "Eight familiar voices, clean fully overlapping speech; development/test reserve different utterances, not different speakers. Initializer audio is excluded from every pool.",
        "license": "LibriSpeech CC-BY-4.0; sources independently rendered from LibriMix metadata",
    }
    validate_reservation(report, corpus)
    atomic_json(output, report)
    return report


def validate_reservation(payload, corpus):
    if payload["protocol"] != PROTOCOL or payload["source_manifest_sha256"] != corpus.manifest_hash:
        raise ValueError("Concept source protocol or manifest differs")
    seen = set(payload["initialization_excluded_utterances"])
    for split in ("train", "dev", "test"):
        pools = payload["splits"][split]
        if sorted(pools) != payload["speakers"]:
            raise ValueError("Every partition must contain the declared voices")
        for speaker, sources in pools.items():
            if len(sources) < 2:
                raise ValueError("Each voice needs distinct source and reference recordings")
            for source in sources:
                if source["utterance"] in seen:
                    raise ValueError(
                        "Utterance leakage across initialization or concept partitions"
                    )
                if source != corpus.known_files.get(source["path"]) or source["speaker"] != speaker:
                    raise ValueError("Concept source identity differs from the prepared corpus")
                seen.add(source["utterance"])


class ConceptCorpus:
    def __init__(self, root, source_manifest, manifest):
        self.source = LibriMixCorpus(root, source_manifest, "train")
        self.manifest_hash = sha256(Path(manifest))
        self.recipe = json.loads(Path(manifest).read_text())
        validate_reservation(self.recipe, self.source)
        self.speakers = self.recipe["speakers"]
        self.labels = {speaker: i for i, speaker in enumerate(self.recipe["classifier_speakers"])}

    def crop(self, source, count, rng):
        wave = self.source.read(source["path"])
        for _ in range(20):
            start = int(rng.integers(len(wave) - count + 1))
            result = wave[start : start + count].copy()
            result -= result.mean()
            rms = float(np.sqrt(np.mean(result**2)))
            if rms > 0.001:
                return result * (0.1 / rms), start
        raise ValueError("No active crop in the chosen recording")

    def pair(self, split, seed, speakers, seconds):
        if split not in self.recipe["splits"] or len(set(speakers)) != 2:
            raise ValueError("Expected a declared split and two different voices")
        rng = np.random.default_rng(seed)
        sources, references, records = [], [], []
        for speaker in speakers:
            pool = self.recipe["splits"][split][speaker]
            first, second = rng.choice(len(pool), 2, replace=False)
            source, source_start = self.crop(pool[first], int(seconds * 16000), rng)
            reference, ref_start = self.crop(pool[second], 48000, rng)
            sources.append(source)
            references.append(reference)
            records.append(
                {
                    "source": pool[first]["utterance"],
                    "reference": pool[second]["utterance"],
                    "source_start": source_start,
                    "reference_start": ref_start,
                }
            )
        ratio = float(rng.uniform(*self.recipe["relative_level_db"]))
        sources[0] *= 10 ** (ratio / 20)
        mixture = sources[0] + sources[1]
        gain = min(1.0, 0.9 / max(float(np.abs(mixture).max()), 1e-5))
        mixture *= gain
        sources = [source * gain for source in sources]
        requests = []
        for side in (0, 1):
            requests.append(
                {
                    "mixture": torch.from_numpy(mixture.copy())[None, None],
                    "target": torch.from_numpy(sources[side])[None, None],
                    "interferer": torch.from_numpy(sources[1 - side])[None, None],
                    "reference": torch.from_numpy(references[side])[None, None],
                    "label": self.labels[speakers[side]],
                    "speaker": speakers[side],
                    "case_id": f"concept-{split}-{seed}:{side}",
                    "source_records": records,
                }
            )
        return requests

    def training_batch(self, seed, step):
        rng = np.random.default_rng(np.random.SeedSequence([seed, step]))
        voices = rng.permutation(self.speakers).tolist()
        return [
            request
            for i in range(0, len(voices), 2)
            for request in self.pair("train", request_seed(seed, step, i), voices[i : i + 2], 3)
        ]

    def evaluation_requests(self, split):
        if split not in {"dev", "test"}:
            raise ValueError("Evaluation requires reserved development or test recordings")
        count = self.recipe["development_pairs" if split == "dev" else "test_pairs"]
        for pair in range(count):
            cycle = pair // (len(self.speakers) // 2)
            rng = np.random.default_rng(
                np.random.SeedSequence([771 if split == "dev" else 997, cycle])
            )
            voices = rng.permutation(self.speakers).tolist()
            start = (pair % (len(self.speakers) // 2)) * 2
            yield from self.pair(
                split,
                request_seed(19 if split == "dev" else 23, 0, pair),
                voices[start : start + 2],
                4,
            )
