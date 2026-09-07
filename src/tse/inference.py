"""Local waveform extraction with shared preprocessing and bounded context."""

from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import torch

from tse.audio import read_audio, speech_check, wav_bytes
from tse.config import ExperimentConfig
from tse.engine import load_model, synchronize
from tse.utils import sha256

PROCESSING_VERSION = "sample-peak-guard-v1"


class Extractor:
    def __init__(self, checkpoint: Path, device: str = "cpu"):
        self.model, self.payload = load_model(checkpoint, device)
        self.config = ExperimentConfig.model_validate(self.payload["config"])
        self.device = next(self.model.parameters()).device
        self.checkpoint_hash = sha256(checkpoint)

    def warmup(self) -> None:
        """Exercise the complete numeric path before the API declares readiness."""
        time_axis = np.arange(48000, dtype=np.float32) / 16000
        reference = 0.1 * np.sin(2 * np.pi * 220 * time_axis)
        mixture = reference[:16000] + 0.08 * np.sin(2 * np.pi * 370 * time_axis[:16000])
        self.extract_array(mixture, reference)
        synchronize(self.device)

    def info(self) -> dict:
        return {
            "ready": True,
            "model_id": self.checkpoint_hash[:12],
            "checkpoint_sha256": self.checkpoint_hash,
            "processing_version": PROCESSING_VERSION,
            "training_updates": self.payload["step"],
            "training_experiment": self.config.experiment,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "device": str(self.device),
            "sample_rate": 16000,
            "operating_envelope": "Experimental, target-present, two-speaker offline extraction. Quality varies with voice and recording conditions.",
            "limits": {
                "mixture_seconds": self.config.inference.maximum_mixture_seconds,
                "reference_seconds": [
                    self.config.inference.minimum_reference_seconds,
                    self.config.inference.maximum_reference_seconds,
                ],
                "combined_upload_mib": 24,
            },
            "confidence_available": False,
        }

    @torch.inference_mode()
    def extract_array(
        self, mixture: np.ndarray, reference: np.ndarray, chunked: bool = True
    ) -> np.ndarray:
        if mixture.ndim != 1 or reference.ndim != 1:
            raise ValueError("Expected mono waveform arrays")
        speech_check(mixture)
        speech_check(reference)
        original = mixture.astype(np.float32) - float(mixture.mean())
        gain = 0.14 / max(float(np.sqrt(np.mean(original**2))), 1e-4)
        normalized = original * gain
        ref = reference.astype(np.float32) - float(reference.mean())
        ref *= 0.1 / max(float(np.sqrt(np.mean(ref**2))), 1e-4)
        tensor = torch.from_numpy(ref[None, None]).to(self.device)
        embedding = self.model.reference_encoder(tensor)
        core = 32000
        context = getattr(self.model, "context_samples", 16000)
        output = np.empty(len(mixture), dtype=np.float32)
        if not chunked or len(mixture) <= 64000:
            x = torch.from_numpy(normalized[None, None]).to(self.device)
            output[:] = self.model.extract(x, embedding)[0, 0].cpu().numpy()
        else:
            for start in range(0, len(mixture), core):
                stop = min(start + core, len(mixture))
                left, right = max(0, start - context), min(len(mixture), stop + context)
                x = torch.from_numpy(normalized[None, None, left:right]).to(self.device)
                prediction = self.model.extract(x, embedding)[0, 0]
                output[start:stop] = prediction[start - left : stop - left].cpu().numpy()
        output /= gain
        if not np.isfinite(output).all() or len(output) != len(mixture):
            raise RuntimeError("Model produced invalid audio")
        return output

    def extract_files(
        self, mixture_source: Path | io.BytesIO, reference_source: Path | io.BytesIO
    ) -> tuple[bytes, dict]:
        started = time.perf_counter()
        settings = self.config.inference
        mixture = read_audio(mixture_source, max_seconds=settings.maximum_mixture_seconds)
        reference = read_audio(reference_source, max_seconds=settings.maximum_reference_seconds)
        speech_check(reference, settings.minimum_reference_seconds)
        synchronize(self.device)
        model_started = time.perf_counter()
        output = self.extract_array(mixture, reference)
        synchronize(self.device)
        model_seconds = time.perf_counter() - model_started
        raw_peak = float(np.max(np.abs(output)))
        # Uniform attenuation prevents sample overflow in playback while preserving
        # the waveform shape and relative speaker levels. Never boost quiet output.
        playback_gain = min(1.0, 0.98 / max(raw_peak, 1e-8))
        output = output * playback_gain
        encoded = wav_bytes(output)
        elapsed = time.perf_counter() - started
        duration = len(mixture) / 16000
        metadata = {
            "model_id": self.checkpoint_hash[:12],
            "processing_version": PROCESSING_VERSION,
            "sample_rate": 16000,
            "duration_seconds": duration,
            "processing_seconds": round(elapsed, 4),
            "model_seconds": round(model_seconds, 4),
            "real_time_factor": round(elapsed / duration, 4),
            "output_peak": float(np.max(np.abs(output))),
            "raw_output_peak": raw_peak,
            "playback_gain": playback_gain,
            "training_updates": self.payload["step"],
        }
        return encoded, metadata
