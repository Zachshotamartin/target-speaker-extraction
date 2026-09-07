"""Independent implementation of the published-scale BSRNN/ResNet34 method.

Specifications and remaining reproduction differences are recorded in
docs/REFERENCE_BASELINE.md. No research package or checkpoint is imported.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class EnrollmentFbank(nn.Module):
    """80 log filterbanks: 25 ms Hamming frames, 10 ms hop, preemphasis, CMN.

    Implements the Kaldi-style signal processing equations directly in PyTorch.
    Dither is one PCM unit during training; evaluation is deterministic.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.hamming_window(400, periodic=False), persistent=False)
        edges = torch.linspace(
            1127 * torch.log1p(torch.tensor(20 / 700)),
            1127 * torch.log1p(torch.tensor(8000 / 700)),
            82,
        )
        frequencies = 1127 * torch.log1p(torch.arange(257) * (16000 / 512 / 700))
        rising = (frequencies[None] - edges[:-2, None]) / (edges[1:-1] - edges[:-2])[:, None]
        falling = (edges[2:, None] - frequencies[None]) / (edges[2:] - edges[1:-1])[:, None]
        bank = torch.minimum(rising, falling).clamp_min(0)
        bank[:, -1] = 0
        self.register_buffer("bank", bank, persistent=False)

    @torch.no_grad()
    def forward(self, waveform):
        if waveform.ndim == 3:
            waveform = waveform[:, 0]
        if waveform.shape[-1] < 400:
            raise ValueError("Enrollment needs at least one 25 ms frame")
        frames = (waveform * 32768).unfold(-1, 400, 160).clone()
        if self.training:
            frames = frames + torch.randn_like(frames)
        frames = frames - frames.mean(-1, keepdim=True)
        previous = F.pad(frames, (1, 0), mode="replicate")[..., :-1]
        frames = (frames - 0.97 * previous) * self.window
        power = torch.fft.rfft(frames, n=512).abs().square()
        features = (power @ self.bank.T).clamp_min(torch.finfo(power.dtype).eps).log()
        return (features - features.mean(1, keepdim=True)).transpose(1, 2)


class SpeakerResidual(nn.Module):
    def __init__(self, inputs, outputs, stride):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(),
            nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
            nn.BatchNorm2d(outputs),
        )
        self.skip = (
            nn.Identity()
            if inputs == outputs and stride == 1
            else nn.Sequential(
                nn.Conv2d(inputs, outputs, 1, stride=stride, bias=False), nn.BatchNorm2d(outputs)
            )
        )

    def forward(self, signal):
        return F.relu(self.body(signal) + self.skip(signal))


class EnrollmentResNet34(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation_checkpointing = False
        self.fbank = EnrollmentFbank()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )
        stages, inputs = [], 32
        for stage, count in enumerate((3, 4, 6, 3)):
            outputs = 32 * 2**stage
            for index in range(count):
                stages.append(SpeakerResidual(inputs, outputs, 2 if stage and index == 0 else 1))
                inputs = outputs
        self.stages = nn.Sequential(*stages)
        self.project = nn.Linear(256 * 10 * 2, 256)

    def encode_features(self, features):
        hidden = self.stem(features[:, None])
        for stage in self.stages:
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                hidden = checkpoint(stage, hidden, use_reentrant=False)
            else:
                hidden = stage(hidden)
        variance, mean = torch.var_mean(hidden, dim=-1, correction=1)
        statistics = torch.cat([mean.flatten(1), (variance + 1e-7).sqrt().flatten(1)], dim=1)
        # Preserve embedding magnitude, as in the reference fusion/classifier recipe.
        return self.project(statistics)

    def forward(self, reference, lengths=None):
        if lengths is not None and not bool(torch.all(lengths == reference.shape[-1])):
            if self.training:
                raise ValueError(
                    "Training batches must explicitly pad precomputed full-utterance features"
                )
            return torch.cat(
                [self(x[None, :, : int(n)]) for x, n in zip(reference, lengths, strict=True)]
            )
        return self.encode_features(self.fbank(reference))


class SequenceResidual(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.normalize = nn.GroupNorm(1, width, eps=torch.finfo(torch.float32).eps)
        self.recurrent = nn.LSTM(width, hidden, batch_first=True, bidirectional=True)
        self.project = nn.Linear(hidden * 2, width)

    def forward(self, signal):
        sequence, _ = self.recurrent(self.normalize(signal).transpose(1, 2))
        return signal + self.project(sequence).transpose(1, 2)


class TimeFrequencyBlock(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.over_time = SequenceResidual(width, hidden)
        self.over_bands = SequenceResidual(width, hidden)

    def forward(self, signal):
        batch, bands, width, frames = signal.shape
        temporal = self.over_time(signal.reshape(batch * bands, width, frames))
        temporal = temporal.reshape(batch, bands, width, frames)
        frequency = temporal.permute(0, 3, 2, 1).reshape(batch * frames, width, bands)
        return self.over_bands(frequency).reshape(batch, frames, width, bands).permute(0, 3, 2, 1)


class ReferenceBSRNN(nn.Module):
    requires_whole_clip = True
    preserve_input_scale = True

    def __init__(self, config, speaker_classes=0):
        super().__init__()
        self.config = config
        self.activation_checkpointing = False
        self.reference_encoder = EnrollmentResNet34()
        self.register_buffer("window", torch.hann_window(512), persistent=False)
        width = config.band_channels
        eps = torch.finfo(torch.float32).eps
        self.analysis = nn.ModuleList(
            [
                nn.Sequential(nn.GroupNorm(1, 2 * bins, eps=eps), nn.Conv1d(2 * bins, width, 1))
                for bins in config.band_widths
            ]
        )
        self.condition = nn.Linear(256, width)
        self.separation = nn.ModuleList(
            [
                TimeFrequencyBlock(width, config.band_hidden_channels)
                for _ in range(config.band_blocks)
            ]
        )
        self.synthesis = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(1, width, eps=eps),
                    nn.Conv1d(width, 4 * width, 1),
                    nn.Tanh(),
                    nn.Conv1d(4 * width, 4 * width, 1),
                    nn.Tanh(),
                    nn.Conv1d(4 * width, 4 * bins, 1),
                )
                for bins in config.band_widths
            ]
        )
        self.speaker_head = nn.Linear(256, speaker_classes) if speaker_classes else None
        if sum(p.numel() for p in self.parameters()) > config.parameter_budget:
            raise ValueError("Reference model exceeds its declared parameter budget")

    @staticmethod
    def complex_mask(logits, bins):
        coefficients, gates = logits.reshape(logits.shape[0], 2, 2, bins, -1).unbind(1)
        components = coefficients * gates.sigmoid()
        return torch.complex(components[:, 0], components[:, 1])

    def extract(self, mixture, embedding):
        samples = mixture.shape[-1]
        signal = mixture[:, 0]
        # Reflect centering matches the reference STFT; tiny inputs are padded explicitly.
        if samples <= 256:
            signal = F.pad(signal, (0, 257 - samples))
        spectrum = torch.stft(signal, 512, 128, window=self.window, return_complex=True)
        parts = spectrum.split(self.config.band_widths, dim=1)
        features = torch.stack(
            [
                module(torch.cat([part.real, part.imag], dim=1))
                for module, part in zip(self.analysis, parts, strict=True)
            ],
            dim=1,
        )
        features = features * self.condition(embedding)[:, None, :, None]
        for block in self.separation:
            if self.training and self.activation_checkpointing:
                features = checkpoint(block, features, use_reentrant=False)
            else:
                features = block(features)
        estimates = [
            part * self.complex_mask(module(features[:, index]), bins)
            for index, (module, part, bins) in enumerate(
                zip(self.synthesis, parts, self.config.band_widths, strict=True)
            )
        ]
        output = torch.istft(
            torch.cat(estimates, dim=1), 512, 128, window=self.window, length=signal.shape[-1]
        )
        return output[:, None, :samples]

    def forward(self, mixture, reference, reference_lengths=None):
        return self.extract(mixture, self.reference_encoder(reference, reference_lengths))
