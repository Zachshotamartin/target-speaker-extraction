"""Independent convolutional target extractor with reference-derived affine conditioning."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from tse.config import ModelConfig


class ChannelNorm(nn.Module):
    """Per-frame channel normalization; padding never enters global statistics."""

    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x: Tensor) -> Tensor:
        variance, mean = torch.var_mean(x, dim=1, keepdim=True, correction=0)
        return (x - mean) * torch.rsqrt(variance + 1e-5) * self.weight + self.bias


class GlobalNorm(ChannelNorm):
    """Per-example channel/time statistics for fully valid, offline mixture crops."""

    def forward(self, x: Tensor) -> Tensor:
        variance, mean = torch.var_mean(x, dim=(1, 2), keepdim=True, correction=0)
        return (x - mean) * torch.rsqrt(variance + 1e-5) * self.weight + self.bias


def separator_norm(channels: int, config: ModelConfig) -> nn.Module:
    return (
        GlobalNorm(channels)
        if config.separation_normalization == "global"
        else ChannelNorm(channels)
    )


class ReferenceBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.PReLU(channels),
            ChannelNorm(channels),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        return (x + self.layers(x)) * mask


class ReferenceEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        c = config.reference_encoder
        self.kernel = c.kernel_samples
        self.stride = c.stride_samples
        self.encoder = nn.Conv1d(1, c.channels, self.kernel, stride=self.stride)
        self.blocks = nn.ModuleList(ReferenceBlock(c.channels, d) for d in c.dilations)
        self.projection = nn.Linear(c.channels * 2, c.embedding_dim)

    def forward(self, reference: Tensor, lengths: Tensor | None = None) -> Tensor:
        if lengths is None:
            lengths = torch.full(
                (reference.shape[0],),
                reference.shape[-1],
                device=reference.device,
                dtype=torch.long,
            )
        sample_mask = (
            torch.arange(reference.shape[-1], device=reference.device)[None, None]
            < lengths[:, None, None]
        )
        reference = reference * sample_mask
        left = self.kernel - self.stride
        right = (-reference.shape[-1]) % self.stride
        x = F.relu(self.encoder(F.pad(reference, (left, right))))
        frame_lengths = (lengths + self.stride - 1) // self.stride
        mask = (
            torch.arange(x.shape[-1], device=x.device)[None, None] < frame_lengths[:, None, None]
        ).to(x.dtype)
        x = x * mask
        for block in self.blocks:
            x = block(x, mask)
        count = mask.sum(-1).clamp_min(1)
        mean = (x * mask).sum(-1) / count
        variance = ((x - mean.unsqueeze(-1)).square() * mask).sum(-1) / count
        summary = torch.cat([mean, (variance + 1e-5).sqrt()], dim=1)
        return F.normalize(self.projection(summary), dim=1, eps=1e-6)


def make_reference_encoder(config: ModelConfig) -> nn.Module:
    if config.reference_encoder.family == "scaled_resnet34":
        from tse.advanced_models import ResNetReferenceEncoder

        return ResNetReferenceEncoder(config)
    return ReferenceEncoder(config)


class ConditionedBlock(nn.Module):
    def __init__(self, config: ModelConfig, dilation: int):
        super().__init__()
        hidden = config.hidden_channels
        self.in_projection = nn.Conv1d(config.bottleneck_channels, hidden, 1)
        self.norm1 = separator_norm(hidden, config)
        self.activation1 = nn.PReLU(hidden)
        self.condition = nn.Linear(config.reference_encoder.embedding_dim, hidden * 2)
        self.depthwise = nn.Conv1d(
            hidden,
            hidden,
            config.temporal_kernel,
            padding=dilation * (config.temporal_kernel // 2),
            dilation=dilation,
            groups=hidden,
        )
        self.activation2 = nn.PReLU(hidden)
        self.norm2 = separator_norm(hidden, config)
        self.residual = nn.Conv1d(hidden, config.bottleneck_channels, 1)
        self.skip = nn.Conv1d(hidden, config.skip_channels, 1)

    def forward(self, x: Tensor, reference: Tensor) -> tuple[Tensor, Tensor]:
        h = self.norm1(self.activation1(self.in_projection(x)))
        scale, shift = self.condition(reference).chunk(2, dim=1)
        h = h * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        h = self.norm2(self.activation2(self.depthwise(h)))
        return x + self.residual(h), self.skip(h)


class TargetExtractor(nn.Module):
    def __init__(self, config: ModelConfig, speaker_classes: int = 0):
        super().__init__()
        self.config = config
        self.reference_encoder = make_reference_encoder(config)
        self.encoder = nn.Conv1d(
            1,
            config.encoder_channels,
            config.encoder_kernel_samples,
            stride=config.encoder_stride_samples,
            bias=False,
        )
        self.input_norm = separator_norm(config.encoder_channels, config)
        self.bottleneck = nn.Conv1d(config.encoder_channels, config.bottleneck_channels, 1)
        self.blocks = nn.ModuleList(
            ConditionedBlock(config, dilation)
            for _ in range(config.repeats)
            for dilation in config.dilations
        )
        self.mask = nn.Sequential(
            nn.PReLU(config.skip_channels),
            nn.Conv1d(config.skip_channels, config.encoder_channels, 1),
            nn.ReLU(),
        )
        self.decoder = nn.ConvTranspose1d(
            config.encoder_channels,
            1,
            config.encoder_kernel_samples,
            stride=config.encoder_stride_samples,
            bias=False,
        )
        # A matched random analysis/synthesis initialization starts near a useful
        # waveform representation; both filterbanks remain independently trainable.
        with torch.no_grad():
            self.decoder.weight.copy_(
                self.encoder.weight * (6 * config.encoder_stride_samples / config.encoder_channels)
            )
            nn.init.normal_(self.mask[1].weight, std=0.001)
            nn.init.ones_(self.mask[1].bias)
        self.speaker_head = (
            nn.Linear(config.reference_encoder.embedding_dim, speaker_classes)
            if speaker_classes
            else None
        )
        count = sum(p.numel() for p in self.parameters())
        if count > config.parameter_budget:
            raise ValueError(
                f"Model has {count:,} parameters; budget is {config.parameter_budget:,}"
            )

    def extract(self, mixture: Tensor, embedding: Tensor) -> Tensor:
        length = mixture.shape[-1]
        stride = self.config.encoder_stride_samples
        left = self.config.encoder_kernel_samples - stride
        right = (-length) % stride
        encoded = F.relu(self.encoder(F.pad(mixture, (left, right))))
        x = self.bottleneck(self.input_norm(encoded))
        skip = torch.zeros(
            (x.shape[0], self.config.skip_channels, x.shape[-1]), device=x.device, dtype=x.dtype
        )
        for block in self.blocks:
            x, current = block(x, embedding)
            skip = skip + current
        estimated = self.decoder(encoded * self.mask(skip))
        return estimated[..., left : left + length]

    def forward(
        self, mixture: Tensor, reference: Tensor, reference_lengths: Tensor | None = None
    ) -> Tensor:
        return self.extract(mixture, self.reference_encoder(reference, reference_lengths))


class SpectralTargetExtractor(nn.Module):
    """Reference-conditioned attenuation with a fixed, invertible STFT filterbank."""

    def __init__(self, config: ModelConfig, speaker_classes: int = 0):
        super().__init__()
        self.config = config
        self.reference_encoder = make_reference_encoder(config)
        self.register_buffer("window", torch.hann_window(config.stft_fft_samples), persistent=False)
        self.input_norm = separator_norm(config.encoder_channels, config)
        self.bottleneck = nn.Conv1d(config.encoder_channels, config.bottleneck_channels, 1)
        self.blocks = nn.ModuleList(
            ConditionedBlock(config, dilation)
            for _ in range(config.repeats)
            for dilation in config.dilations
        )
        self.mask = nn.Sequential(
            nn.PReLU(config.skip_channels),
            nn.Conv1d(config.skip_channels, config.encoder_channels, 1),
            nn.Sigmoid(),
        )
        nn.init.normal_(self.mask[1].weight, std=0.001)
        nn.init.zeros_(self.mask[1].bias)
        self.speaker_head = (
            nn.Linear(config.reference_encoder.embedding_dim, speaker_classes)
            if speaker_classes
            else None
        )
        if sum(parameter.numel() for parameter in self.parameters()) > config.parameter_budget:
            raise ValueError("Spectral model exceeds the configured parameter budget")

    @property
    def context_samples(self) -> int:
        receptive_frames = (
            sum(self.config.dilations) * self.config.repeats * (self.config.temporal_kernel // 2)
        )
        return max(
            16000, self.config.stft_fft_samples + receptive_frames * self.config.stft_hop_samples
        )

    def extract(self, mixture: Tensor, embedding: Tensor) -> Tensor:
        spectrum = torch.stft(
            mixture[:, 0],
            n_fft=self.config.stft_fft_samples,
            hop_length=self.config.stft_hop_samples,
            window=self.window,
            center=True,
            pad_mode="constant",
            return_complex=True,
        )
        features = self.bottleneck(self.input_norm(torch.log1p(spectrum.abs())))
        skip = torch.zeros(
            (mixture.shape[0], self.config.skip_channels, spectrum.shape[-1]),
            dtype=mixture.dtype,
            device=mixture.device,
        )
        for block in self.blocks:
            features, current = block(features, embedding)
            skip = skip + current
        output = torch.istft(
            spectrum * self.mask(skip),
            n_fft=self.config.stft_fft_samples,
            hop_length=self.config.stft_hop_samples,
            window=self.window,
            center=True,
            length=mixture.shape[-1],
        )
        return output[:, None]

    def forward(
        self, mixture: Tensor, reference: Tensor, reference_lengths: Tensor | None = None
    ) -> Tensor:
        return self.extract(mixture, self.reference_encoder(reference, reference_lengths))


def make_model(config: ModelConfig, speaker_classes: int = 0) -> nn.Module:
    if config.family == "reference_conditioned_bsrnn":
        from tse.advanced_models import BandSplitExtractor

        return BandSplitExtractor(config, speaker_classes)
    if config.family == "reference_conditioned_stft_tcn":
        return SpectralTargetExtractor(config, speaker_classes)
    return TargetExtractor(config, speaker_classes)


def choose_device(name: str) -> torch.device:
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; choose --device cpu explicitly")
    if name not in {"mps", "cpu"}:
        raise ValueError("Supported devices are cpu and mps")
    if name == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 6))
    return torch.device(name)
