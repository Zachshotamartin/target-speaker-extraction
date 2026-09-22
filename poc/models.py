"""Pretrained inference adapters. Only extract() can transform a waveform.

ASR deliberately accepts no reference, speaker embedding, prompt, or identity.
The same transcribe() call is used for the raw and One Voice comparisons.
"""

import gc
import json
import os
from pathlib import Path

import numpy as np

from poc.common import RATE, digest

ASR_SETTINGS = dict(
    language="en",
    beam_size=5,
    temperature=0.0,
    word_timestamps=True,
    vad_filter=False,
    condition_on_previous_text=False,
    initial_prompt=None,
)


class Runtime:
    def __init__(self, directory, progress=lambda _: None):
        # Disable the native uploader before its library initializes. HF's
        # telemetry flag and DO_NOT_TRACK do not configure ONNX Runtime.
        os.environ["ORT_DISABLE_TELEMETRY"] = "1"
        import onnxruntime
        import torch

        onnxruntime.disable_telemetry_events()
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        progress("Verifying frozen model files")
        if digest(self.directory / "one-voice.pt") != self.manifest["one_voice"]["sha256"]:
            raise ValueError("One Voice checkpoint does not match the provisioned manifest.")
        for spec in self.manifest["models"].values():
            for name, expected in spec.get("files", {}).items():
                path = (self.directory / name).resolve()
                if not path.is_relative_to(self.directory.resolve()) or digest(path) != expected:
                    raise ValueError(
                        "A pretrained file is missing or has changed. Run provisioning again."
                    )
        from faster_whisper.utils import get_assets_path

        vad_files = list(Path(get_assets_path()).glob("*silero*.onnx"))
        if len(vad_files) != 1 or digest(vad_files[0]) != self.manifest["models"]["vad"]["sha256"]:
            raise ValueError("The packaged VAD changed. Re-provision the model manifest.")
        self.separator = self.speaker = self.asr = None

    def speech(self, audio):
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        return [
            [s["start"] / RATE, s["end"] / RATE]
            for s in get_speech_timestamps(
                audio,
                VadOptions(
                    min_speech_duration_ms=250, min_silence_duration_ms=250, speech_pad_ms=60
                ),
            )
        ]

    def embedding(self, audio):
        import torch

        if self.speaker is None:
            from speechbrain.inference.speaker import EncoderClassifier

            local = str(self.directory / "speaker")
            self.speaker = EncoderClassifier.from_hparams(
                source=local,
                savedir=local,
                overrides={"pretrained_path": local},
                run_opts={"device": "cpu"},
            )
        with torch.inference_mode():
            vector = (
                self.speaker.encode_batch(torch.from_numpy(audio.copy())[None]).flatten().numpy()
            )
        return vector / max(float(np.linalg.norm(vector)), 1e-8)

    def extract(self, mixture, reference):
        if self.separator is None:
            from tse.inference import Extractor

            self.separator = Extractor(self.directory / "one-voice.pt", device="cpu")
            # Optimizer state is not needed for inference. The original checkpoint is untouched.
            self.separator.payload = {
                k: v
                for k, v in self.separator.payload.items()
                if k in {"config", "step", "provenance"}
            }
            gc.collect()
        return self.separator.extract_array(mixture, reference, chunked=False)

    def release_separator(self):
        self.separator = None
        gc.collect()

    def transcribe(self, audio):
        """Transcribe the supplied waveform as-is; no speaker selection or cleanup."""
        if self.asr is None:
            from faster_whisper import WhisperModel

            self.asr = WhisperModel(
                str(self.directory / "asr"),
                device="cpu",
                compute_type="int8",
                cpu_threads=1,
                num_workers=1,
                local_files_only=True,
            )
        segments, _ = self.asr.transcribe(audio, **ASR_SETTINGS)
        words = []
        duration = len(audio) / RATE
        for segment in segments:
            for word in segment.words or []:
                start, end = max(0.0, word.start), min(duration, word.end)
                if end > start and word.word.strip():
                    words.append(
                        {
                            "start": round(start, 3),
                            "end": round(end, 3),
                            "text": word.word,
                            "asr_probability": round(word.probability, 4),
                        }
                    )
        return words
