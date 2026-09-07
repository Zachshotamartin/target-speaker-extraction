"""Validated, serializable experiment contracts."""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class AudioConfig(StrictModel):
    sample_rate: Literal[16000] = 16000
    channels: Literal[1] = 1
    crop_seconds: float = Field(default=2, ge=0.25, le=10)
    reference_seconds: float = Field(default=5, ge=0.25, le=10)
    reference_mode: Literal["crop", "full_utterance"] = "crop"


class DataConfig(StrictModel):
    protocol: Literal[
        "custom-librispeech-tse-v1", "libri2mix-16k-min-clean", "known-voice-concept-v1"
    ] = "custom-librispeech-tse-v1"
    training_source: Literal["LibriSpeech/train-clean-100"] = "LibriSpeech/train-clean-100"
    development_source: Literal[
        "LibriSpeech/dev-clean",
        "LibriSpeech/dev-clean+dev-other",
        "LibriSpeech/train-clean-100/reserved-utterances",
    ] = "LibriSpeech/dev-clean"
    test_source: Literal[
        "LibriSpeech/test-clean",
        "LibriSpeech/train-clean-100/reserved-identities",
        "LibriSpeech/test-other",
        "LibriSpeech/train-clean-100/reserved-utterances",
    ] = "LibriSpeech/test-clean"
    pilot_speakers_target: int = Field(default=60, ge=2)
    minimum_utterances_per_speaker: int = Field(default=3, ge=3)
    mixtures_on_demand: Literal[True] = True
    distinct_reference_utterance: Literal[True] = True
    prefer_different_reference_chapter: bool = True
    target_to_interferer_db: tuple[float, float] = (-5, 5)
    overlap_fraction: Literal[1.0, "variable"] = 1.0
    target_present: Literal[True, "mixed"] = True

    @model_validator(mode="after")
    def check_ratio(self):
        if self.target_to_interferer_db[0] > self.target_to_interferer_db[1]:
            raise ValueError("Mixture ratio bounds are reversed")
        if self.protocol == "custom-librispeech-tse-v1" and self.target_to_interferer_db != (-5, 5):
            raise ValueError("The v1 mixture protocol fixes ratio bounds to [-5, 5] dB")
        return self


class ReferenceConfig(StrictModel):
    family: Literal["waveform_tcn", "scaled_resnet34", "resnet34_fbank"] = "waveform_tcn"
    resnet_base_channels: int = Field(default=16, ge=4, le=64)
    channels: int = Field(default=128, ge=8, le=512)
    kernel_samples: int = Field(default=160, ge=4)
    stride_samples: int = Field(default=80, ge=1)
    dilations: list[int] = Field(default_factory=lambda: [1, 2, 4], min_length=1)
    pooling: Literal["masked_mean_std"] = "masked_mean_std"
    embedding_dim: int = Field(default=128, ge=8, le=512)
    shared_with_mixture_encoder: Literal[False] = False


class ModelConfig(StrictModel):
    family: Literal[
        "reference_conditioned_tcn",
        "reference_conditioned_stft_tcn",
        "reference_conditioned_bsrnn",
        "reference_bsrnn",
    ] = "reference_conditioned_tcn"
    weights: Literal["random_initialization", "project_checkpoint", "project_reference"] = (
        "random_initialization"
    )
    encoder_channels: int = Field(default=128, ge=8, le=512)
    encoder_kernel_samples: int = Field(default=32, ge=4)
    encoder_stride_samples: int = Field(default=16, ge=1)
    bottleneck_channels: int = Field(default=64, ge=8, le=512)
    hidden_channels: int = Field(default=128, ge=8, le=1024)
    skip_channels: int = Field(default=64, ge=8, le=512)
    temporal_kernel: int = Field(default=3, ge=3, le=9)
    dilations: list[int] = Field(
        default_factory=lambda: [1, 2, 4, 8, 16, 32, 64, 128], min_length=1
    )
    repeats: int = Field(default=2, ge=1, le=8)
    conditioning: Literal["feature_wise_affine", "multiply_once"] = "feature_wise_affine"
    mask_activation: Literal["relu", "sigmoid", "gated_complex"] = "relu"
    separation_normalization: Literal["per_frame", "global"] = "per_frame"
    stft_fft_samples: int = Field(default=512, ge=64, le=2048)
    stft_hop_samples: int = Field(default=128, ge=16, le=512)
    spectral_mask: Literal["real", "complex", "gated_complex"] = "real"
    band_widths: list[int] = Field(default_factory=lambda: [8] * 8 + [16] * 8 + [32, 33])
    band_channels: int = Field(default=48, ge=8, le=128)
    band_hidden_channels: int = Field(default=64, ge=8, le=256)
    band_blocks: int = Field(default=4, ge=1, le=8)
    causal: Literal[False] = False
    parameter_budget: int = Field(default=3000000, ge=1)
    reference_encoder: ReferenceConfig = Field(default_factory=ReferenceConfig)

    @model_validator(mode="after")
    def check_architecture(self):
        if self.family != "reference_bsrnn" and (
            self.reference_encoder.family == "resnet34_fbank"
            or self.spectral_mask == "gated_complex"
            or self.mask_activation == "gated_complex"
            or self.conditioning == "multiply_once"
        ):
            raise ValueError("Published reference components require the reference_bsrnn family")
        if self.family == "reference_bsrnn":
            if (
                self.mask_activation != "gated_complex"
                or self.spectral_mask != "gated_complex"
                or self.conditioning != "multiply_once"
                or self.separation_normalization != "global"
                or self.reference_encoder.family != "resnet34_fbank"
                or self.reference_encoder.embedding_dim != 256
                or self.stft_fft_samples != 512
                or self.stft_hop_samples != 128
                or self.encoder_channels != 257
                or min(self.band_widths) < 1
                or sum(self.band_widths) != 257
            ):
                raise ValueError(
                    "Reference BSRNN requires its explicit complex, global-normalization, ResNet34 contract"
                )
            return self
        if self.family in {"reference_conditioned_stft_tcn", "reference_conditioned_bsrnn"}:
            if (
                self.mask_activation != "sigmoid"
                or self.encoder_channels != self.stft_fft_samples // 2 + 1
            ):
                raise ValueError(
                    "STFT extractor requires a sigmoid mask and one channel per frequency bin"
                )
            if (
                self.stft_fft_samples & (self.stft_fft_samples - 1)
                or self.stft_fft_samples % self.stft_hop_samples
            ):
                raise ValueError("STFT FFT must be a power of two divisible by the hop")
            if self.stft_hop_samples >= self.stft_fft_samples or 32000 % self.stft_hop_samples:
                raise ValueError(
                    "STFT hop must overlap and align with the two-second inference core"
                )
            if self.family == "reference_conditioned_bsrnn" and (
                min(self.band_widths) < 1 or sum(self.band_widths) != self.encoder_channels
            ):
                raise ValueError("Band widths must partition every STFT frequency exactly once")
        elif self.mask_activation != "relu":
            raise ValueError("Learned waveform filterbank requires its declared ReLU mask")
        if self.spectral_mask == "complex" and self.family != "reference_conditioned_bsrnn":
            raise ValueError("Complex masks are implemented for the band-split family")
        if self.temporal_kernel % 2 != 1:
            raise ValueError("Temporal kernel must be odd")
        if min(self.dilations + self.reference_encoder.dilations) < 1:
            raise ValueError("Dilations must be positive")
        for kernel, stride in [
            (self.encoder_kernel_samples, self.encoder_stride_samples),
            (self.reference_encoder.kernel_samples, self.reference_encoder.stride_samples),
        ]:
            if stride > kernel or kernel % stride:
                raise ValueError("Encoder kernel must be a multiple of its stride")
        return self


class TrainingConfig(StrictModel):
    optimizer: Literal["adamw", "adam"] = "adamw"
    learning_rate: float = Field(default=0.0003, gt=0, le=0.1)
    weight_decay: float = Field(default=0.0001, ge=0)
    microbatch_size: int = Field(default=2, ge=1, le=128)
    gradient_accumulation: int = Field(default=4, ge=1, le=64)
    gradient_clip_norm: float = Field(default=5, gt=0)
    max_optimizer_updates: int = Field(default=1000, ge=1)
    validation_interval_updates: int = Field(default=250, ge=1)
    precision: Literal["float32"] = "float32"
    learning_rate_schedule: Literal["none", "plateau", "cosine", "exponential"] = "none"
    epochs: int = Field(default=100, ge=1)
    checkpoint_interval_updates: int = Field(default=100, ge=1)
    activation_checkpointing: bool = False
    schedule_decay_updates: int = Field(default=5000, ge=1)
    preserve_initialized_classifier: bool = False
    scheduler_patience_validations: int = Field(default=4, ge=1)
    scheduler_factor: float = Field(default=0.5, gt=0, lt=1)
    minimum_learning_rate: float = Field(default=0.00001, gt=0)
    num_workers: Literal[0] = 0
    retain_checkpoints: list[Literal["best", "latest"]] = Field(
        default_factory=lambda: ["best", "latest"]
    )


class LossConfig(StrictModel):
    primary: Literal["negative_si_sdr"] = "negative_si_sdr"
    epsilon: float = Field(default=1e-8, gt=0)
    mask_padding: Literal[True] = True
    zero_mean: Literal[True] = True
    separation_weight: float = Field(default=1, gt=0, le=1)
    speaker_classification_weight: float = Field(default=0, ge=0, le=1)
    speaker_logit_scale: float = Field(default=1, ge=1, le=30)
    waveform_weight: float = Field(default=0.1, ge=0, le=10)
    spectral_weight: float = Field(default=0, ge=0, le=10)
    spectral_fft_sizes: list[int] = Field(default_factory=lambda: [256, 512, 1024], min_length=1)
    absent_target_si_sdr: Literal["unsupported"] = "unsupported"
    absent_target_weight: float = Field(default=10, ge=0)

    @model_validator(mode="after")
    def check_spectral_sizes(self):
        if any(size < 16 or size > 4096 or size & (size - 1) for size in self.spectral_fft_sizes):
            raise ValueError("Spectral FFT sizes must be powers of two from 16 through 4096")
        return self


class AugmentationConfig(StrictModel):
    reference_enabled: bool = False
    mixture_noise_enabled: bool = False
    independent_random_streams: Literal[True] = True
    realistic_enabled: bool = False
    environment_root: str | None = None
    environment_manifest: str | None = None

    @model_validator(mode="after")
    def check_realism(self):
        if self.mixture_noise_enabled and not self.realistic_enabled:
            raise ValueError("Mixture noise requires the declared realistic rendering pipeline")
        if self.realistic_enabled and (
            not self.environment_root or not self.environment_manifest or self.reference_enabled
        ):
            raise ValueError(
                "Realistic training needs explicit resources and replaces the legacy reference augmentation"
            )
        return self


class EvaluationConfig(StrictModel):
    realistic_validation: bool = False
    development_cases_target: int = Field(default=400, ge=2)
    final_test_cases_target: int = Field(default=1000, ge=2)
    mixture_seconds: float = Field(default=4, ge=0.25, le=60)
    reference_seconds: float = Field(default=5, ge=0.25, le=10)
    confusion_margin_db: float = Field(default=3, ge=0)
    bootstrap_target_speaker_clusters: Literal[True] = True
    bootstrap_replicates: int = Field(default=1000, ge=10)
    training_seeds_final_target: int = Field(default=3, ge=1)


class RuntimeConfig(StrictModel):
    preferred_device: Literal["mps", "cpu"] = "mps"
    cpu_path_required: Literal[True] = True
    allow_implicit_cpu_fallback_in_benchmarks: Literal[False] = False
    benchmark_warmup_steps: int = Field(default=20, ge=1)
    benchmark_measured_steps: int = Field(default=100, ge=1)
    mps_memory_fraction: float = Field(default=0.6, gt=0, le=0.8)


class ResourceConfig(StrictModel):
    minimum_free_disk_gib: float = Field(default=15, ge=1)
    persistent_artifact_budget_gib: float = Field(default=20, ge=1)
    process_rss_target_gib: float = Field(default=10, ge=1)


class InferenceConfig(StrictModel):
    mode: Literal["offline"] = "offline"
    maximum_mixture_seconds: float = Field(default=60, ge=1, le=600)
    minimum_reference_seconds: float = Field(default=3, ge=0.25)
    maximum_reference_seconds: float = Field(default=10, ge=1, le=60)
    input_formats: list[Literal["wav", "flac"]] = Field(default_factory=lambda: ["wav", "flac"])
    maximum_concurrent_extractions: Literal[1] = 1
    report_confidence: Literal[False] = False

    @model_validator(mode="after")
    def check_durations(self):
        if self.minimum_reference_seconds > self.maximum_reference_seconds:
            raise ValueError("Reference duration bounds are reversed")
        return self


class ExperimentConfig(StrictModel):
    schema_version: Literal[1] = 1
    status: Literal["proposal", "active"] = "active"
    experiment: str = "clean-reference"
    seed: int = Field(default=42, ge=0)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    resources: ResourceConfig = Field(default_factory=ResourceConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)

    @classmethod
    def load(cls, path: Path) -> ExperimentConfig:
        if path.suffix == ".toml":
            return cls.model_validate(tomllib.loads(path.read_text()))
        return cls.model_validate_json(path.read_text())

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.model_dump(), sort_keys=True).encode()).hexdigest()
