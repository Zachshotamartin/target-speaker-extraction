"""Consistent audio decoding, resampling and validation."""

from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def read_audio(
    source: Path | io.BytesIO, sample_rate: int = 16000, max_seconds: float = 600
) -> np.ndarray:
    try:
        with sf.SoundFile(source) as handle:
            if handle.format not in {"WAV", "WAVEX", "FLAC"}:
                raise ValueError("Use a WAV or FLAC recording")
            if handle.samplerate < 8000 or handle.samplerate > 192000:
                raise ValueError("Unsupported sample rate")
            if handle.channels not in {1, 2}:
                raise ValueError("Use a mono or stereo recording")
            if handle.frames == 0 or handle.frames / handle.samplerate > max_seconds:
                raise ValueError(f"Recording must be nonempty and at most {max_seconds:g} seconds")
            original_rate = handle.samplerate
            samples = handle.read(dtype="float32", always_2d=True).mean(axis=1)
    except (sf.LibsndfileError, RuntimeError) as error:
        raise ValueError("The recording could not be decoded as WAV or FLAC") from error
    if not np.isfinite(samples).all():
        raise ValueError("Audio contains non-finite samples")
    if original_rate != sample_rate:
        divisor = math.gcd(original_rate, sample_rate)
        samples = resample_poly(samples, sample_rate // divisor, original_rate // divisor).astype(
            np.float32
        )
    return samples


def speech_check(samples: np.ndarray, minimum_seconds: float = 0, sample_rate: int = 16000) -> None:
    if len(samples) < int(minimum_seconds * sample_rate):
        raise ValueError(f"Reference must contain at least {minimum_seconds:g} seconds")
    if not np.isfinite(samples).all() or np.sqrt(np.mean(samples**2)) < 1e-4:
        raise ValueError("Recording is silent or too quiet")
    frame = max(1, int(sample_rate * 0.02))
    usable = samples[: len(samples) // frame * frame]
    if usable.size:
        energies = np.sqrt(np.mean(usable.reshape(-1, frame) ** 2, axis=1))
        if np.count_nonzero(energies > max(1e-4, energies.max() * 0.03)) < min(10, len(energies)):
            raise ValueError("Recording contains too little usable audio")


def wav_bytes(samples: np.ndarray, sample_rate: int = 16000) -> bytes:
    if not np.isfinite(samples).all():
        raise ValueError("Model output contains non-finite audio")
    output = io.BytesIO()
    sf.write(output, samples, sample_rate, format="WAV", subtype="FLOAT")
    return output.getvalue()
