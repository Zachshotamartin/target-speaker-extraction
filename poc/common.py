"""Bounded audio I/O and timeline-preserving result formats."""

import hashlib
import io
import json
from pathlib import Path

import av
import numpy as np
import soundfile as sf

RATE = 16000
MAX_BYTES = 4 * 1024 * 1024
MAX_SECONDS = 30
TTL_SECONDS = 900
TERMINAL = {"ready", "failed", "cancelled", "expired"}
ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, allow_nan=False))
    temporary.replace(path)


def decode(data, maximum=MAX_SECONDS):
    if not data or len(data) > MAX_BYTES:
        raise ValueError("Choose a nonempty recording under 4 MiB.")
    pieces, count = [], 0
    try:
        with av.open(io.BytesIO(data), mode="r") as container:
            streams = list(container.streams.audio)
            if len(streams) != 1:
                raise ValueError("Use a recording with one audio stream.")
            stream = streams[0]
            if (
                stream.duration is not None
                and float(stream.duration * stream.time_base) > maximum + 0.05
            ):
                raise ValueError(f"Recording must be at most {maximum:g} seconds.")
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=RATE)
            for frame in container.decode(stream):
                if (
                    frame.sample_rate < 8000
                    or frame.sample_rate > 192000
                    or len(frame.layout.channels) > 2
                ):
                    raise ValueError("Use mono or stereo audio between 8 and 192 kHz.")
                for converted in resampler.resample(frame):
                    x = converted.to_ndarray().flatten()
                    count += len(x)
                    if count > maximum * RATE + 1:
                        raise ValueError(f"Recording must be at most {maximum:g} seconds.")
                    pieces.append(x)
            for converted in resampler.resample(None):
                pieces.append(converted.to_ndarray().flatten())
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("Audio could not be decoded. Try WAV, FLAC, MP3, M4A or WebM.") from error
    audio = np.concatenate(pieces).astype(np.float32) if pieces else np.array([], np.float32)
    if not len(audio) or len(audio) > maximum * RATE + 1 or not np.isfinite(audio).all():
        raise ValueError(f"Use valid audio of at most {maximum:g} seconds.")
    return audio


def wav(audio, playback=False):
    x = np.asarray(audio, np.float32)
    if playback and len(x):
        x = x * min(1.0, 0.98 / max(float(np.max(np.abs(x))), 1e-8))
    result = io.BytesIO()
    sf.write(result, x, RATE, format="WAV", subtype="PCM_16")
    return result.getvalue()


def merge_spans(spans, gap=0.0):
    result = []
    for start, end in sorted(spans):
        if end <= start:
            continue
        if result and start <= result[-1][1] + gap:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def overlap(start, end, spans):
    return sum(max(0.0, min(end, b) - max(start, a)) for a, b in merge_spans(spans))


def stamp(seconds, srt=False):
    ms = max(0, round(seconds * 1000))
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}{',' if srt else '.'}{ms:03}"


def export_text(result, kind):
    segments = [s for s in result.get("segments", []) if s["attribution"] == "accepted"]
    if kind == "txt":
        return "\n".join(s["text"].strip() for s in segments)
    if kind == "srt":
        return "\n\n".join(
            f"{i}\n{stamp(s['start'], True)} --> {stamp(s['end'], True)}\n{s['text'].strip()}"
            for i, s in enumerate(segments, 1)
        )
    if kind == "json":
        return json.dumps(result, indent=2, allow_nan=False)
    raise ValueError("Choose txt, srt or json.")
