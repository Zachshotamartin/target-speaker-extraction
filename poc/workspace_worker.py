"""Bounded CPU inference and media exports for a locally saved recording project."""

import json
import os
import resource
import subprocess
from pathlib import Path

import numpy as np

from poc.common import RATE, atomic_json, decode, wav
from poc.workspace import FILE_BYTES, MAX_SECONDS


def windowed_extract(extract, mixture, reference, progress=lambda _: None):
    """Overlap-add full context windows, including the final incomplete window."""
    window, hop = 24 * RATE, 20 * RATE
    if len(mixture) <= window:
        return extract(mixture, reference)
    starts = list(range(0, len(mixture) - window + 1, hop))
    if starts[-1] + window < len(mixture):
        starts.append(len(mixture) - window)
    output, weight = np.zeros_like(mixture), np.zeros_like(mixture)
    for index, start in enumerate(starts):
        progress(f"Extracting passage {index + 1} of {len(starts)}")
        prediction = extract(mixture[start : start + window], reference)
        if len(prediction) != window or not np.isfinite(prediction).all():
            raise ValueError("The model returned invalid audio.")
        envelope = np.ones(window, dtype=np.float32)
        if index:
            overlap = starts[index - 1] + window - start
            envelope[:overlap] *= np.linspace(0, 1, overlap, dtype=np.float32)
        if index < len(starts) - 1:
            overlap = start + window - starts[index + 1]
            envelope[-overlap:] *= np.linspace(1, 0, overlap, dtype=np.float32)
        output[start : start + window] += prediction * envelope
        weight[start : start + window] += envelope
    return output / np.maximum(weight, 1e-8)


def discover(runtime, audio, progress=lambda _: None):
    """Suggest audible reference passages; clustering is not verified identification."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    duration = len(audio) / RATE
    speech = runtime.speech(audio)
    candidates = []
    for start in np.arange(0, max(0.001, duration - 3 + 0.001), max(1.5, duration / 240)):
        end = min(duration, float(start) + 4)
        spoken = sum(max(0, min(b, end) - max(a, start)) for a, b in speech)
        if end - start >= 3 and spoken >= 1.5:
            candidates.append((float(start), end))
    if not candidates:
        return []
    vectors = []
    for i, (start, end) in enumerate(candidates):
        progress(f"Comparing voice samples {i + 1} of {len(candidates)}")
        vectors.append(runtime.embedding(audio[int(start * RATE) : int(end * RATE)]))
    vectors = np.asarray(vectors)
    labels = (
        fcluster(
            linkage(np.clip(pdist(vectors, metric="cosine"), 0, 2), method="average"),
            0.35,
            criterion="distance",
        )
        if len(vectors) > 1
        else np.ones(1, dtype=int)
    )
    groups = sorted(set(labels), key=lambda k: -sum(labels == k))[:4]
    suggestions = []
    for group in groups:
        indexes = np.flatnonzero(labels == group)
        centroid = vectors[indexes].mean(axis=0)
        best = int(indexes[np.argmax(vectors[indexes] @ centroid)])
        start, end = candidates[best]
        suggestions.append(
            {
                "id": f"voice-{len(suggestions)}",
                "label": f"Voice {chr(65 + len(suggestions))}",
                "start": start,
                "end": end,
                "sample_count": len(indexes),
                "asset": f"reference-{len(suggestions)}.wav",
            }
        )
    return suggestions


def probe(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-protocol_whitelist",
            "file,pipe",
            "-format_whitelist",
            "mov,matroska,webm,avi",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        timeout=30,
        check=True,
    )
    return json.loads(result.stdout)


def subtitle_text(words):
    import re

    def stamp(seconds):
        milliseconds = round(seconds * 1000)
        return f"{milliseconds // 3600000:02}:{milliseconds // 60000 % 60:02}:{milliseconds // 1000 % 60:02},{milliseconds % 1000:03}"

    cues = []
    for word in sorted(words, key=lambda item: item["start"]):
        text = re.sub(r"[<>\{\}\\\r\n]", "", word["text"]).strip()
        if not text:
            continue
        if (
            cues
            and word["end"] - cues[-1]["start"] < 3.5
            and word["start"] - cues[-1]["end"] < 0.8
            and len(cues[-1]["text"] + text) < 48
        ):
            cues[-1].update(end=word["end"], text=cues[-1]["text"] + " " + text)
        else:
            cues.append({**word, "text": text})
    return (
        "\n\n".join(
            f"{i + 1}\n{stamp(c['start'])} --> {stamp(c['end'])}\n{c['text']}"
            for i, c in enumerate(cues)
        )
        + "\n"
    )


def render_video(directory, options, progress):
    source = directory / "mixture.input"
    metadata = probe(source)
    video = next(
        (
            s
            for s in metadata["streams"]
            if s["codec_type"] == "video" and not s.get("disposition", {}).get("attached_pic")
        ),
        None,
    )
    if not video or video.get("width", 0) > 4096 or video.get("height", 0) > 4096:
        raise ValueError("Choose a video up to 4096 pixels on each side.")
    duration = float(metadata["format"].get("duration", 0))
    if not 0 < duration <= MAX_SECONDS + 0.1 or options["clips"][-1]["end"] > duration + 0.05:
        raise ValueError("Video and edit timings do not match, or exceed 10 minutes.")
    audio = decode(
        (directory / "edited.input").read_bytes(), maximum=MAX_SECONDS, max_bytes=FILE_BYTES
    )
    length = sum(c["end"] - c["start"] for c in options["clips"])
    if abs(len(audio) / RATE - length) > 0.05:
        raise ValueError("The edited audio duration must match the retained video passages.")
    (directory / "render.wav").write_bytes(wav(audio))
    progress("Rendering your cuts and captions")
    # One streaming decoder. Splitting hundreds of trims would buffer frames
    # behind concat inputs and grow memory with the whole source video.
    selection, retiming, offset = [], [], 0.0
    for clip in options["clips"]:
        start, end = clip["start"], clip["end"]
        selection.append(f"gte(t,{start})*lt(t,{end})")
        retiming.append(f"gte(T,{start})*lt(T,{end})*(T-{start}+{offset})")
        offset += end - start
    filters = [
        "[0:v:0]setpts=PTS-STARTPTS,"
        "scale=w='min(1280,iw)':h='min(720,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2,"
        + f"select='{'+'.join(selection)}',setpts='({'+'.join(retiming)})/TB',fps=30,"
        + "tpad=stop_mode=clone:stop_duration=1[joined]"
    ]
    if options["captions"] and options["words"]:
        (directory / "captions.srt").write_text(subtitle_text(options["words"]))
        filters.append(
            "[joined]subtitles=captions.srt:force_style='FontName=DejaVu Sans,FontSize=22,Outline=1,MarginV=24'[out]"
        )
    else:
        filters.append("[joined]null[out]")
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-threads",
        "1",
        "-protocol_whitelist",
        "file,pipe",
        "-format_whitelist",
        "mov,matroska,webm,avi",
        "-i",
        "mixture.input",
        "-i",
        "render.wav",
        "-filter_complex_threads",
        "1",
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[out]",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-threads",
        "1",
        "-preset",
        "fast",
        "-b:v",
        "1200k",
        "-maxrate",
        "1500k",
        "-bufsize",
        "3000k",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        "-movflags",
        "+faststart",
        "-t",
        str(length),
        "captioned.mp4",
    ]
    try:
        subprocess.run(command, cwd=directory, capture_output=True, timeout=2400, check=True)
    except subprocess.CalledProcessError:
        raise ValueError("Video rendering failed. Try an MP4 or MOV recording.") from None
    finally:
        (directory / "render.wav").unlink(missing_ok=True)
        (directory / "captions.srt").unlink(missing_ok=True)
    output = directory / "captioned.mp4"
    if (
        output.stat().st_size > FILE_BYTES
        or abs(float(probe(output)["format"]["duration"]) - length) > 0.15
    ):
        output.unlink(missing_ok=True)
        raise ValueError("The rendered video exceeded the output limit or had invalid timing.")
    return {"kind": "video", "duration": length, "asset": "captioned.mp4"}


def run_workspace(directory, models, options, progress):
    directory = Path(directory)
    if options["kind"] == "video":
        atomic_json(directory / "result.json", render_video(directory, options, progress))
        return
    from poc.models import Runtime
    from poc.pipeline import run

    progress("Reading your recording")
    audio = decode(
        (directory / "mixture.input").read_bytes(), maximum=MAX_SECONDS, max_bytes=FILE_BYTES
    )
    runtime = Runtime(models, progress)
    (directory / "original.wav").write_bytes(wav(audio, playback=True))
    if options["kind"] == "discover":
        candidates = discover(runtime, audio, progress)
        for candidate in candidates:
            sample = audio[int(candidate["start"] * RATE) : int(candidate["end"] * RATE)]
            (directory / candidate["asset"]).write_bytes(wav(sample))
        result = {
            "kind": "discover",
            "duration": len(audio) / RATE,
            "candidates": candidates,
            "notice": "Suggested voice samples, not verified speakers. Overlapping voices can be grouped together; listen and choose a clean solo passage.",
        }
    else:
        original_extract = runtime.extract
        runtime.extract = lambda mixture, reference: windowed_extract(
            original_extract, mixture, reference, progress
        )
        reports = []
        comparison = None
        for i, reference in enumerate(options["references"]):
            progress(f"Processing voice {i + 1} of {len(options['references'])}")
            sample = decode((directory / reference["file"]).read_bytes(), maximum=10)
            report, estimate = run(runtime, audio, sample, options["compare"] and i == 0, progress)
            comparison = report["comparison"] or comparison
            if comparison and not report["comparison"]:
                report["comparison"] = {
                    **comparison,
                    "one_voice": {
                        "text": "".join(s["text"] for s in report["segments"]),
                        "words": [w for s in report["segments"] for w in s["words"]],
                    },
                }
            asset = f"speaker-{i}.wav"
            (directory / asset).write_bytes(wav(estimate, playback=True))
            report["peak_worker_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
                1 if os.uname().sysname == "Darwin" else 1024
            )
            report_asset = f"report-{i}.json"
            atomic_json(directory / report_asset, report)
            reports.append(
                {
                    "id": f"speaker-{i}",
                    "label": reference["label"],
                    "asset": asset,
                    "report_asset": report_asset,
                }
            )
        result = {
            "kind": "extract",
            "duration": len(audio) / RATE,
            "tracks": reports,
            "notice": "Each track is extracted independently. The model was trained on two-speaker mixtures; three or four simultaneous speakers are experimental.",
        }
    atomic_json(directory / "result.json", result)
