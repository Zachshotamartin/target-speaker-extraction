"""Official Libri2Mix clean mixture identities with independent rendering/loading."""

import hashlib
import json
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from tse.utils import sha256

PROTOCOL = "libri2mix-16k-min-clean"


def pcm16_roundtrip(signal):
    """libsndfile's PCM16 write/read mapping, including clipping and negative floor."""
    return (np.floor(np.asarray(signal) * 32768).clip(-32768, 32767) / 32768).astype(np.float32)


def render_clean(left, right, gains):
    length = min(len(left), len(right))
    sources = [
        np.asarray(x[:length], dtype=np.float32) * gain
        for x, gain in zip((left, right), gains, strict=True)
    ]
    # Mix before PCM quantization, as in the published generator.
    mixture = sources[0] + sources[1]
    return [pcm16_roundtrip(x) for x in sources], pcm16_roundtrip(mixture)


class LibriMixCorpus:
    def __init__(self, root, manifest, split):
        self.root = Path(root).resolve()
        self.manifest_path = Path(manifest)
        self.manifest_hash = sha256(self.manifest_path)
        payload = json.loads(self.manifest_path.read_text())
        if payload["protocol"] != PROTOCOL or split not in {"train", "dev", "test"}:
            raise ValueError("Expected declared Libri2Mix clean protocol")
        self.rows = payload["splits"][split]
        self.split = split
        self.enrollments = payload["enrollments"].get(split, {})
        pools = defaultdict(list)
        for row in self.rows:
            for source in row["sources"]:
                pools[source["speaker"]].append(source)
        self.pools = dict(pools)
        self.speakers = sorted(self.pools)
        self.labels = {s: i for i, s in enumerate(self.speakers)}
        self.known_files = {
            file["path"]: file for row in self.rows for file in [row["mixture"], *row["sources"]]
        }
        self.verified = set()
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.rows) * 2

    def file_path(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Audio path escapes the dataset root")
        return path

    def read(self, relative):
        if relative in self.cache:
            self.cache.move_to_end(relative)
            return self.cache[relative]
        path = self.file_path(relative)
        record = self.known_files[relative]
        if relative not in self.verified:
            if sha256(path) != record["sha256"]:
                raise ValueError(f"Prepared audio changed: {relative}")
            self.verified.add(relative)
        signal, rate = sf.read(path, dtype="float32")
        if rate != 16000 or signal.ndim != 1 or not np.isfinite(signal).all():
            raise ValueError("Expected finite mono 16 kHz audio")
        self.cache[relative] = signal
        if len(self.cache) > 64:
            self.cache.popitem(last=False)
        return signal

    def request(self, index, seed=0, crop_samples=None):
        row = self.rows[index // 2]
        side = index % 2
        target, other = row["sources"][side], row["sources"][1 - side]
        rng = np.random.default_rng(seed)
        if self.split == "train":
            pool = [
                source
                for source in self.pools[target["speaker"]]
                if source["utterance"] != target["utterance"]
            ]
            reference_record = pool[int(rng.integers(len(pool)))]
            ref_path = reference_record["path"]
        else:
            ref_path = self.enrollments[f"{row['id']}:{side}"]
            reference_record = self.known_files[ref_path]
        if (
            reference_record["speaker"] != target["speaker"]
            or reference_record["utterance"] == target["utterance"]
        ):
            raise ValueError("Enrollment must be a different utterance by the requested speaker")
        signals = [self.read(record["path"]) for record in [row["mixture"], target, other]]
        length = len(signals[0])
        start = 0
        if crop_samples is not None:
            if length < crop_samples:
                signals = [
                    np.tile(x, int(np.ceil(crop_samples / length)))[:crop_samples] for x in signals
                ]
            else:
                start = int(rng.integers(length - crop_samples + 1))
                signals = [x[start : start + crop_samples] for x in signals]
        mixture, truth, interferer = [torch.from_numpy(x.copy())[None, None] for x in signals]
        return {
            "mixture": mixture,
            "target": truth,
            "interferer": interferer,
            "reference": torch.from_numpy(self.read(ref_path).copy())[None, None],
            "label": self.labels[target["speaker"]],
            "speaker": target["speaker"],
            "case_id": f"{row['id']}:{side}",
            "reference_path": ref_path,
            "crop_start": start,
        }


def epoch_order(count, seed, epoch):
    return np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(count).tolist()


def request_seed(seed, epoch, index):
    return int.from_bytes(hashlib.sha256(f"{seed}:{epoch}:{index}".encode()).digest()[:8], "big")
