"""Independent band-split and scaled residual-reference research candidates.

These use published architectural ideas with explicitly smaller local settings;
they do not import research implementations or external pretrained weights.
"""

import torch
from torch import nn
from torch.nn import functional as F

from tse.model import make_reference_encoder


class Residual2D(nn.Module):
    def __init__(self, inputs, outputs, stride=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(4, outputs),
            nn.ReLU(),
            nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
            nn.GroupNorm(4, outputs),
        )
        self.shortcut = (
            nn.Identity()
            if inputs == outputs and stride == 1
            else nn.Sequential(
                nn.Conv2d(inputs, outputs, 1, stride=stride, bias=False), nn.GroupNorm(4, outputs)
            )
        )

    def forward(self, x):
        return F.relu(self.layers(x) + self.shortcut(x))


class ResNetReferenceEncoder(nn.Module):
    """ResNet34 depth [3,4,6,3], smaller widths, GroupNorm, 64 log-mel bands."""

    def __init__(self, config):
        super().__init__()
        base = config.reference_encoder.resnet_base_channels
        if base % 4:
            raise ValueError("Reference width must be divisible by four GroupNorm groups")
        self.register_buffer("window", torch.hann_window(512), persistent=False)
        mel_points = torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + 8000 / 700)), 66)
        hz = 700 * (10 ** (mel_points / 2595) - 1)
        frequency = torch.linspace(0, 8000, 257)[None]
        lower = (frequency - hz[:-2, None]) / (hz[1:-1, None] - hz[:-2, None])
        upper = (hz[2:, None] - frequency) / (hz[2:, None] - hz[1:-1, None])
        self.register_buffer(
            "mel_filter", torch.minimum(lower, upper).clamp_min(0), persistent=False
        )
        self.stem = nn.Sequential(
            nn.Conv2d(1, base, 3, padding=1, bias=False), nn.GroupNorm(4, base), nn.ReLU()
        )
        blocks = []
        inputs = base
        for stage, count in enumerate([3, 4, 6, 3]):
            outputs = base * 2**stage
            for index in range(count):
                blocks.append(Residual2D(inputs, outputs, 2 if stage and index == 0 else 1))
                inputs = outputs
        self.blocks = nn.Sequential(*blocks)
        self.projection = nn.Linear(inputs * 2, config.reference_encoder.embedding_dim)

    def forward(self, reference, lengths=None):
        if lengths is not None and not bool(torch.all(lengths == reference.shape[-1])):
            # Full-recording normalization must never include right padding.
            return torch.cat(
                [
                    self.forward(x[None, :, : int(length)])
                    for x, length in zip(reference, lengths, strict=True)
                ]
            )
        spectrum = torch.stft(
            reference[:, 0],
            512,
            160,
            window=self.window,
            center=True,
            pad_mode="constant",
            return_complex=True,
        )
        mel = (self.mel_filter @ spectrum.abs().square()).clamp_min(1e-6).log()
        mel = mel - mel.mean(-1, keepdim=True)
        features = self.blocks(self.stem(mel[:, None]))
        variance, mean = torch.var_mean(features, dim=(2, 3), correction=0)
        pooled = torch.cat([mean, (variance + 1e-5).sqrt()], dim=1)
        return F.normalize(self.projection(pooled), dim=1, eps=1e-6)


class BandSequenceBlock(nn.Module):
    def __init__(self, channels, hidden, reference):
        super().__init__()
        self.condition = nn.Linear(reference, channels * 2)
        self.time_norm = nn.LayerNorm(channels)
        self.time = nn.LSTM(channels, hidden, batch_first=True, bidirectional=True)
        self.time_projection = nn.Linear(hidden * 2, channels)
        self.band_norm = nn.LayerNorm(channels)
        self.band = nn.LSTM(channels, hidden, batch_first=True, bidirectional=True)
        self.band_projection = nn.Linear(hidden * 2, channels)

    def forward(self, x, reference):
        batch, bands, frames, channels = x.shape
        scale, shift = self.condition(reference).chunk(2, dim=-1)
        conditioned = self.time_norm(x) * (1 + scale[:, None, None]) + shift[:, None, None]
        temporal, _ = self.time(conditioned.reshape(batch * bands, frames, channels))
        x = x + self.time_projection(temporal).reshape(batch, bands, frames, channels)
        frequency = self.band_norm(x).transpose(1, 2).reshape(batch * frames, bands, channels)
        frequency, _ = self.band(frequency)
        return x + self.band_projection(frequency).reshape(
            batch, frames, bands, channels
        ).transpose(1, 2)


class BandSplitExtractor(nn.Module):
    requires_whole_clip = True

    def __init__(self, config, speaker_classes=0):
        super().__init__()
        self.config = config
        self.reference_encoder = make_reference_encoder(config)
        self.register_buffer("window", torch.hann_window(config.stft_fft_samples), persistent=False)
        d = config.band_channels
        self.band_encoders = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(width * 2), nn.Linear(width * 2, d))
            for width in config.band_widths
        )
        self.blocks = nn.ModuleList(
            BandSequenceBlock(
                d, config.band_hidden_channels, config.reference_encoder.embedding_dim
            )
            for _ in range(config.band_blocks)
        )
        # Both arms retain identical real/imaginary head tensor shapes, so the
        # initial model is exactly matched when testing phase-aware output.
        self.mask_heads = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(d), nn.Linear(d, d * 2), nn.Tanh(), nn.Linear(d * 2, width * 2)
            )
            for width in config.band_widths
        )
        for head in self.mask_heads:
            nn.init.normal_(head[-1].weight, std=0.001)
            nn.init.zeros_(head[-1].bias)
            with torch.no_grad():
                head[-1].weight[head[-1].out_features // 2 :].zero_()
        self.speaker_head = (
            nn.Linear(config.reference_encoder.embedding_dim, speaker_classes)
            if speaker_classes
            else None
        )
        if sum(p.numel() for p in self.parameters()) > config.parameter_budget:
            raise ValueError("Band-split model exceeds the parameter budget")

    def extract(self, mixture, embedding):
        spectrum = torch.stft(
            mixture[:, 0],
            self.config.stft_fft_samples,
            self.config.stft_hop_samples,
            window=self.window,
            center=True,
            pad_mode="constant",
            return_complex=True,
        )
        bands = spectrum.split(self.config.band_widths, dim=1)
        encoded = []
        for band, encoder in zip(bands, self.band_encoders, strict=True):
            # Compressed real/imaginary features preserve phase and limit scale.
            value = torch.cat([band.real, band.imag], dim=1).transpose(1, 2)
            value = torch.sign(value) * torch.log1p(value.abs())
            encoded.append(encoder(value))
        x = torch.stack(encoded, dim=1)
        for block in self.blocks:
            x = block(x, embedding)
        outputs = []
        for index, (band, head) in enumerate(zip(bands, self.mask_heads, strict=True)):
            real, imaginary = head(x[:, index]).transpose(1, 2).chunk(2, dim=1)
            real = real.sigmoid()
            imaginary = (
                imaginary.tanh() if self.config.spectral_mask == "complex" else imaginary * 0
            )
            # Real arithmetic avoids depending on complex-autograd kernels in MPS.
            output = torch.complex(
                band.real * real - band.imag * imaginary, band.imag * real + band.real * imaginary
            )
            outputs.append(output)
        return torch.istft(
            torch.cat(outputs, dim=1),
            self.config.stft_fft_samples,
            self.config.stft_hop_samples,
            window=self.window,
            center=True,
            length=mixture.shape[-1],
        )[:, None]

    def forward(self, mixture, reference, reference_lengths=None):
        return self.extract(mixture, self.reference_encoder(reference, reference_lengths))
