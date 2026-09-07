"""Target-specific waveform metrics, with explicit silent-output handling."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def si_sdr(
    estimate: Tensor, target: Tensor, lengths: Tensor | None = None, epsilon: float = 1e-8
) -> Tensor:
    if estimate.shape != target.shape:
        raise ValueError("Estimate and target shapes must match")
    if target.ndim == 3:
        estimate, target = estimate[:, 0], target[:, 0]
    if target.ndim != 2:
        raise ValueError("Expected batch by time, optionally with one channel")
    if lengths is None:
        mask = torch.ones_like(target)
    else:
        mask = (torch.arange(target.shape[-1], device=target.device)[None] < lengths[:, None]).to(
            target.dtype
        )
    count = mask.sum(-1, keepdim=True).clamp_min(1)
    s = (target - (target * mask).sum(-1, keepdim=True) / count) * mask
    e = (estimate - (estimate * mask).sum(-1, keepdim=True) / count) * mask
    target_energy = s.square().sum(-1)
    if torch.any(target_energy <= epsilon):
        raise ValueError("SI-SDR requires a non-silent target")
    alpha = (e * s).sum(-1) / target_energy.clamp_min(epsilon)
    projection = alpha[:, None] * s
    residual = e - projection
    score = 10 * torch.log10(
        (projection.square().sum(-1) + epsilon) / (residual.square().sum(-1) + epsilon)
    )
    return torch.where(e.square().sum(-1) <= epsilon, torch.full_like(score, -80), score).clamp(
        -80, 80
    )


def waveform_loss(estimate: Tensor, target: Tensor) -> Tensor:
    scale = target.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-4)
    return ((estimate - target).abs() / scale).mean()


def spectral_loss(estimate: Tensor, target: Tensor, fft_sizes: list[int]) -> Tensor:
    """Multi-resolution spectral convergence and log-magnitude reconstruction.

    Normalize both waveforms by target RMS, retaining sensitivity to predicted
    gain. A magnitude floor bounds log loss in silent frequency bins.
    """
    scale = target.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-4)
    prediction = (estimate / scale).reshape(-1, estimate.shape[-1])
    truth = (target / scale).reshape(-1, target.shape[-1])
    losses = []
    for size in fft_sizes:
        window = torch.hann_window(size, dtype=estimate.dtype, device=estimate.device)
        options = dict(
            n_fft=size,
            hop_length=size // 4,
            window=window,
            center=True,
            pad_mode="constant",
            return_complex=True,
        )
        predicted = torch.stft(prediction, **options).abs().clamp_min(1e-4)
        actual = torch.stft(truth, **options).abs().clamp_min(1e-4)
        convergence = torch.linalg.vector_norm(
            predicted - actual, dim=(-2, -1)
        ) / torch.linalg.vector_norm(actual, dim=(-2, -1)).clamp_min(1e-4)
        logarithmic = (predicted.log() - actual.log()).abs().mean(dim=(-2, -1))
        losses.append((convergence + logarithmic).mean())
    return torch.stack(losses).mean()


def measure(
    estimate: Tensor, mixture: Tensor, target: Tensor, interferer: Tensor, margin_db: float = 3
) -> list[dict]:
    with torch.no_grad():
        target_score = si_sdr(estimate, target)
        input_score = si_sdr(mixture, target)
        other_score = si_sdr(estimate, interferer)
        energy = estimate.square().mean(dim=(-1, -2))
        normalized_error = (estimate - target).abs().mean(dim=(-1, -2)) / target.square().mean(
            dim=(-1, -2)
        ).sqrt().clamp_min(1e-4)
        gain_db = 10 * torch.log10((energy + 1e-10) / (mixture.square().mean(dim=(-1, -2)) + 1e-10))
    return [
        {
            "si_sdr_db": float(target_score[j]),
            "mixture_si_sdr_db": float(input_score[j]),
            "si_sdri_db": float(target_score[j] - input_score[j]),
            "interferer_si_sdr_db": float(other_score[j]),
            "confused": bool(other_score[j] > target_score[j] + margin_db),
            "near_silent": bool(energy[j] < 1e-8),
            "normalized_l1": float(normalized_error[j]),
            "output_mixture_energy_db": float(gain_db[j]),
        }
        for j in range(estimate.shape[0])
    ]


def summarize(rows: list[dict], replicates: int = 1000, seed: int = 42) -> dict:
    if not rows:
        raise ValueError("Cannot summarize an empty evaluation")
    values = np.array([r["si_sdri_db"] for r in rows], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite evaluation result")
    speakers = sorted({r["target_speaker"] for r in rows})
    groups = [
        np.array([r["si_sdri_db"] for r in rows if r["target_speaker"] == speaker])
        for speaker in speakers
    ]
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(replicates):
        chosen = rng.integers(len(groups), size=len(groups))
        means.append(float(np.concatenate([groups[i] for i in chosen]).mean()))
    return {
        "cases": len(rows),
        "target_speakers": len(speakers),
        "mean_si_sdri_db": float(values.mean()),
        "median_si_sdri_db": float(np.median(values)),
        "p10_si_sdri_db": float(np.percentile(values, 10)),
        "negative_improvement_fraction": float((values < 0).mean()),
        "confusion_fraction": float(np.mean([r["confused"] for r in rows])),
        "near_silent_fraction": float(np.mean([r["near_silent"] for r in rows])),
        "mean_normalized_l1": float(np.mean([r["normalized_l1"] for r in rows])),
        "mean_ci95_db": [float(x) for x in np.percentile(means, [2.5, 97.5])],
        "confidence_method": "Approximate target-speaker cluster bootstrap; shared-interferer dependence remains",
        "bootstrap_seed": seed,
        "bootstrap_replicates": replicates,
    }
