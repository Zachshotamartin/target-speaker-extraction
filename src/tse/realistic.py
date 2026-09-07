"""Deterministic recorded-noise, room, overlap, pause and absence experiments."""

import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import fftconvolve

from tse.data import SpeechCorpus
from tse.utils import sha256

SCENARIOS = ("clean", "noise", "reverb", "partial_overlap", "pauses", "target_absent", "combined")
TRAIN_SCENARIOS = ("clean", *SCENARIOS)


class AcousticResources:
    def __init__(self, root: Path, manifest: Path, split: str):
        self.root = root.resolve()
        self.manifest_hash = sha256(manifest)
        payload = json.loads(manifest.read_text())
        self.records = {row["id"]: row for row in payload["records"]}
        self.noises = []
        self.rooms = defaultdict(list)
        groups, hashes = {}, {}
        for row in self.records.values():
            group = (row["kind"], row["group"])
            if groups.setdefault(group, row["split"]) != row["split"]:
                raise ValueError("Acoustic recording/room group crosses splits")
            if hashes.setdefault(row["sha256"], row["split"]) != row["split"]:
                raise ValueError("Identical acoustic audio crosses splits")
            if row["split"] != split:
                continue
            if row["kind"] == "noise":
                self.noises.append(row["id"])
            elif row["kind"] == "rir":
                self.rooms[row["group"]].append(row["id"])
        self.noises.sort()
        self.room_names = sorted(self.rooms)
        if not self.noises or not self.room_names:
            raise ValueError("Each acoustic split needs noise and room resources")

    @lru_cache(maxsize=64)  # noqa: B019 -- bounded lifetime of one corpus.
    def read(self, identity):
        row = self.records[identity]
        path = (self.root / row["path"]).resolve()
        if not path.is_relative_to(self.root) or sha256(path) != row["sha256"]:
            raise ValueError("Acoustic path or checksum differs from the manifest")
        samples, rate = sf.read(path, dtype="float32", always_2d=True)
        if rate != 16000 or not np.isfinite(samples).all():
            raise ValueError("Acoustic data must be finite 16 kHz audio")
        return samples[:, 0] if row["kind"] == "rir" else samples.mean(axis=1)

    def noise(self, count, rng):
        # Some environmental recordings have long silent regions. Deterministic
        # retries avoid amplifying digital silence or near-zero crops.
        for _ in range(20):
            identity = self.noises[int(rng.integers(len(self.noises)))]
            samples = self.read(identity)
            if len(samples) < count:
                samples = np.tile(samples, int(np.ceil(count / len(samples))))
            start = int(rng.integers(max(1, len(samples) - count + 1)))
            crop = samples[start : start + count].copy()
            crop -= crop.mean()
            if np.sqrt(np.mean(crop**2)) > 1e-5:
                return crop / np.sqrt(np.mean(crop**2))
        raise ValueError("Could not find an active environmental noise segment")

    def room_pair(self, rng):
        room = self.room_names[int(rng.integers(len(self.room_names)))]
        ids = self.rooms[room]
        chosen = rng.choice(len(ids), 2, replace=len(ids) < 2)
        return [self.impulse(ids[int(index)]) for index in chosen]

    def impulse(self, identity):
        response = self.read(identity)
        # Align the strongest arrival to the input timeline; retain up to 0.8 s
        # of its decay. This is a simulated-room suppression protocol, not a
        # claim that the model removes reverberation or recovers a dry source.
        response = response[int(np.argmax(np.abs(response))) :][:12800].copy()
        norm = np.sqrt(np.sum(response**2))
        if norm < 1e-6:
            raise ValueError("Invalid room impulse response")
        return response / norm


def tapered_interval(count, start, stop):
    mask = np.zeros(count, dtype=np.float32)
    start, stop = max(0, start), min(count, stop)
    mask[start:stop] = 1
    ramp = min(160, (stop - start) // 2)
    if ramp:
        edge = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, ramp))
        mask[start : start + ramp] = edge
        mask[stop - ramp : stop] = edge[::-1]
    return mask


class RealisticCorpus(SpeechCorpus):
    def __init__(
        self,
        root,
        manifest,
        split,
        seconds=4,
        reference_seconds=5,
        environment_root=None,
        environment_manifest=None,
        augment_training=False,
    ):
        super().__init__(root, manifest, split, seconds, reference_seconds)
        self.acoustics = AcousticResources(
            Path(environment_root), Path(environment_manifest), split
        )
        self.augment_training = augment_training

    def make_case(self, seed, target_speaker=None):
        case = super().make_case(seed, target_speaker)
        if self.augment_training:
            case["scenario"] = TRAIN_SCENARIOS[seed % len(TRAIN_SCENARIOS)]
            case["environment_seed"] = seed + 3000000000
        return case

    def render(self, case, condition=None):
        scenario = case.get("scenario", "clean")
        if scenario not in SCENARIOS:
            raise ValueError("Unknown acoustic scenario")
        original = super().render(case, "clean")
        original["target_present"] = True
        original["scenario"] = scenario
        if scenario == "clean":
            return original
        rng = np.random.default_rng(case["environment_seed"])
        ref_rng = np.random.default_rng(case["environment_seed"] + 7001 + case["target_index"])
        index = case["target_index"]
        signals = (
            [original["target"], original["interferer"]]
            if index == 0
            else [original["interferer"], original["target"]]
        )
        signals = [s.copy() for s in signals]
        count = len(signals[0])
        if scenario in {"partial_overlap", "combined"}:
            gap = int(rng.uniform(0.15, 0.35) * count)
            signals[0] *= tapered_interval(count, 0, count - gap)
            signals[1] *= tapered_interval(count, gap, count)
        if scenario in {"pauses", "combined"}:
            for j in range(2):
                start = int(rng.uniform(0.15, 0.55) * count)
                stop = min(count, start + int(rng.uniform(0.15, 0.30) * count))
                signals[j] *= 1 - tapered_interval(count, start, stop)
        if scenario in {"reverb", "combined"}:
            for j, impulse in enumerate(self.acoustics.room_pair(rng)):
                before = float(np.sqrt(np.mean(signals[j] ** 2)))
                signals[j] = fftconvolve(signals[j], impulse)[:count].astype(np.float32)
                signals[j] *= before / max(float(np.sqrt(np.mean(signals[j] ** 2))), 1e-6)
        if scenario == "target_absent":
            signals[index].fill(0)
            original["target_present"] = False
        mixture = signals[0] + signals[1]
        noise = np.zeros(count, dtype=np.float32)
        if scenario in {"noise", "combined", "target_absent"}:
            snr = float(rng.uniform(0, 20))
            noise = (
                self.acoustics.noise(count, rng) * np.sqrt(np.mean(mixture**2)) / 10 ** (snr / 20)
            )
            mixture = mixture + noise
        reference = original["reference"].copy()
        if scenario in {"reverb", "combined"}:
            impulse = self.acoustics.room_pair(ref_rng)[0]
            reference = fftconvolve(reference, impulse)[: len(reference)].astype(np.float32)
        if scenario in {"noise", "combined"}:
            reference += (
                self.acoustics.noise(len(reference), ref_rng)
                * np.sqrt(np.mean(reference**2))
                / 10 ** (float(ref_rng.uniform(5, 20)) / 20)
            )
        reference -= reference.mean()
        reference *= 0.1 / max(float(np.sqrt(np.mean(reference**2))), 1e-6)
        gain = min(1.0, 0.9 / max(float(np.max(np.abs(mixture))), 1e-6))
        return {
            **original,
            "mixture": (mixture * gain).astype(np.float32),
            "target": (signals[index] * gain).astype(np.float32),
            "interferer": (signals[1 - index] * gain).astype(np.float32),
            "noise": (noise * gain).astype(np.float32),
            "reference": reference,
            "common_gain": original["common_gain"] * gain,
        }

    def batch(self, cases, device, conditions=None):
        result = super().batch(cases, device, conditions)
        result["target_present"] = torch.tensor(
            [c.get("scenario") != "target_absent" for c in cases], device=device
        )
        return result


def extraction_loss(output, target, mixture, config):
    """SI-SDR only for present speech; normalized output energy for absent targets."""
    from tse.metrics import si_sdr, spectral_loss, waveform_loss

    present = target.square().sum(dim=(-1, -2)) > config.epsilon
    loss = output.sum() * 0
    if bool(present.any()):
        estimate, truth = output[present], target[present]
        active = -si_sdr(estimate, truth, epsilon=config.epsilon).mean()
        active = active + config.waveform_weight * waveform_loss(estimate, truth)
        if config.spectral_weight:
            active = active + config.spectral_weight * spectral_loss(
                estimate, truth, config.spectral_fft_sizes
            )
        loss = loss + present.float().mean() * active
    if bool((~present).any()):
        energy = output[~present].square().mean(dim=(-1, -2))
        input_energy = mixture[~present].square().mean(dim=(-1, -2)).clamp_min(1e-6)
        loss = (
            loss
            + (~present).float().mean()
            * config.absent_target_weight
            * (energy / input_energy).mean()
        )
    return loss
