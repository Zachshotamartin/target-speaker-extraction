#!/usr/bin/env python3
"""Development-only artifact audit; candidates do not modify the served model."""

from __future__ import annotations

import argparse
import html
import io
import json
from pathlib import Path

import numpy as np
import torch
from scipy.signal import istft, stft

from tse.audio import read_audio, wav_bytes
from tse.data import SpeechCorpus, load_cases
from tse.inference import Extractor
from tse.metrics import measure, summarize
from tse.utils import atomic_json, sha256


def candidates(mixture: np.ndarray, estimate: np.ndarray) -> dict[str, np.ndarray]:
    """Fixed 32 ms Hann windows, 8 ms hop; neither method sees clean sources."""
    options = dict(fs=16000, nperseg=512, noverlap=384)
    mix = stft(mixture, **options)[2]
    prediction = stft(estimate, **options)[2]
    # Closest real attenuation mask to the predicted complex spectrum.
    mask = np.clip(np.real(prediction * np.conj(mix)) / (np.abs(mix) ** 2 + 1e-10), 0, 1)
    limited = prediction * np.minimum(1, np.abs(mix) / (np.abs(prediction) + 1e-10))
    return {
        "raw": estimate,
        "mixture_phase": istft(mask * mix, **options)[1][: len(mixture)].astype(np.float32),
        "spectral_limit": istft(limited, **options)[1][: len(mixture)].astype(np.float32),
    }


def artifact_proxy(estimate: np.ndarray, target: np.ndarray, interferer: np.ndarray) -> float:
    """Residual after scalar least-squares projection onto both true sources.

    This also counts filtering, phase and amplitude-envelope changes. It is not
    a perceptual noise score, BSS Eval SAR, or a measurement of white noise.
    """
    basis = np.stack([target, interferer], axis=1).astype(np.float64)
    centered = basis - basis.mean(axis=0)
    output = estimate.astype(np.float64) - float(estimate.mean())
    fitted = centered @ np.linalg.lstsq(centered, output, rcond=None)[0]
    return float(np.sum((output - fitted) ** 2) / max(np.sum(output**2), 1e-12))


def listening_preview(extractor: Extractor, destination: Path) -> dict:
    gallery = Path("artifacts/gallery")
    source = json.loads((gallery / "index.json").read_text())
    if source["checkpoint_sha256"] != extractor.checkpoint_hash:
        raise ValueError("Gallery and audited checkpoint differ")
    destination.mkdir(parents=True, exist_ok=True)
    entries, cards = [], []
    # Keep the first six requests regardless of outcome, matching the existing gallery.
    for number, entry in enumerate(source["items"][:6], 1):
        mixture = read_audio(gallery / entry["tracks"]["mixture"])
        raw = read_audio(gallery / entry["tracks"]["output"])
        tracks = candidates(mixture, raw)
        tracks["mixture"] = mixture
        tracks["target"] = read_audio(gallery / entry["tracks"]["target"])
        paths = {}
        players = []
        for name, samples in tracks.items():
            filename = f"case-{number:02d}-{name}.wav"
            (destination / filename).write_bytes(wav_bytes(samples))
            paths[name] = filename
            label = {
                "raw": "Current model",
                "mixture_phase": "Candidate A · mixture phase",
                "spectral_limit": "Candidate B · spectral limit",
                "mixture": "Original mixture",
                "target": "Known clean target",
            }[name]
            players.append(
                f'<div class="gallery-track"><span>{label}</span>'
                f'<audio controls preload="none" src="{filename}"></audio></div>'
            )
        entries.append({"original_gallery_request": entry, "tracks": paths})
        cards.append(
            f'<section class="gallery-card"><h2>Request {number:02d}</h2>'
            f'<p class="gallery-caption">{html.escape(entry["case_id"])}</p>'
            f'<div class="gallery-tracks">{"".join(players)}</div></section>'
        )
    manifest = {
        "checkpoint_sha256": extractor.checkpoint_hash,
        "selection": "First six existing development gallery requests; no quality-based selection.",
        "level_policy": "Original amplitudes retained; lower volume alone is not better separation.",
        "attribution": source["attribution"],
        "source": source["source"],
        "items": entries,
    }
    atomic_json(destination / "index.json", manifest)
    (destination / "index.html").write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Quality comparison · One voice</title>"
        '<link rel="stylesheet" href="/assets/style.css"></head><body><div class="page">'
        '<header class="topbar"><a class="brand" href="/">one voice.</a>'
        '<a href="/gallery/">Original gallery</a></header><main>'
        '<section class="introduction"><div><p class="eyebrow">DEVELOPMENT EXPERIMENT</p>'
        '<h1>Compare the artifacts.</h1><p class="intro-copy">Two fixed spectral cleanup '
        "candidates applied to the current model. Neither candidate retrains the separator. "
        "Both can damage speech or retain the wrong voice.</p></div></section>"
        '<p class="gallery-intro">The first six requests from the original gallery, including '
        "paired voice switches. Original amplitudes are retained; lower volume alone is not "
        "better extraction. Model weights are unchanged; spectral candidates are not enabled "
        "in the main app.</p>"
        + "".join(cards)
        + '<p class="gallery-intro">Speech: <a href="https://www.openslr.org/12">LibriSpeech</a>, '
        'Panayotov et al. (2015), <a href="https://creativecommons.org/licenses/by/4.0/">'
        "CC BY 4.0</a>. Cropped, normalized, mixed and processed derivatives. "
        '<a href="index.json">Source identifiers and transformations</a>.</p></main></div>'
        "<script>document.addEventListener('play',e=>{if(e.target.tagName==='AUDIO')"
        "document.querySelectorAll('audio').forEach(a=>{if(a!==e.target)a.pause()})},true);"
        "</script></body></html>"
    )
    return {"selection": manifest["selection"], "path": str(destination)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    parser.add_argument("--cases", type=Path, default=Path("data/manifests/dev-report-cases.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/releases/model.pt"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/quality-audit-development.json")
    )
    args = parser.parse_args()
    if json.loads(args.cases.read_text())["split"] != "dev":
        raise ValueError("This exploratory audit accepts development cases only")
    corpus = SpeechCorpus(Path("data/raw"), Path("data/manifests/inventory.json"), "dev", 4, 5)
    cases = load_cases(args.cases, corpus)
    extractor = Extractor(args.checkpoint, args.device)
    rows = {method: [] for method in ("raw", "mixture_phase", "spectral_limit")}
    checks = {}
    for index, case in enumerate(cases):
        signals = corpus.render(case)
        mixture, reference = signals["mixture"], signals["reference"]
        raw = extractor.extract_array(mixture, reference)
        if index == 0:
            decoded = read_audio(io.BytesIO(wav_bytes(raw)))
            checks["float_wav_roundtrip_max_error"] = float(np.max(np.abs(decoded - raw)))
            if args.device == "mps":
                cpu = Extractor(args.checkpoint, "cpu").extract_array(mixture, reference)
                checks["first_case_cpu_mps_max_error"] = float(np.max(np.abs(cpu - raw)))
            if len(mixture) <= 64000:
                checks["case_path"] = "Whole recording; chunking is bypassed for these cases."
        for name, estimate in candidates(mixture, raw).items():
            measured = measure(
                *[
                    torch.from_numpy(wave)[None, None]
                    for wave in (estimate, mixture, signals["target"], signals["interferer"])
                ]
            )[0]
            rows[name].append(
                {
                    "case_id": case["case_id"],
                    "target_speaker": signals["speaker"],
                    **measured,
                    "artifact_proxy_fraction": artifact_proxy(
                        estimate, signals["target"], signals["interferer"]
                    ),
                    "output_peak": float(np.max(np.abs(estimate))),
                    "fraction_above_playback_range": float(np.mean(np.abs(estimate) > 1)),
                }
            )
        if (index + 1) % 100 == 0:
            print(f"Audited {index + 1}/{len(cases)} development requests", flush=True)
    summaries = {}
    for method, current in rows.items():
        summaries[method] = {
            **summarize(current),
            "mean_artifact_proxy_fraction": float(
                np.mean([r["artifact_proxy_fraction"] for r in current])
            ),
            "cases_above_playback_range": sum(r["output_peak"] > 1 for r in current),
            "maximum_peak": max(r["output_peak"] for r in current),
        }
    output = {
        "status": "Exploratory development comparison; not a release or final test claim.",
        "path": "Raw extract_array output before the playback level guard; candidates are not enabled in the app.",
        "checkpoint_sha256": extractor.checkpoint_hash,
        "case_manifest_sha256": sha256(args.cases),
        "source_manifest_sha256": corpus.manifest_hash,
        "device": args.device,
        "artifact_proxy_definition": artifact_proxy.__doc__,
        "candidate_definition": candidates.__doc__,
        "checks": checks,
        "summaries": summaries,
        "rows": rows,
        "listening_preview": listening_preview(extractor, Path("artifacts/gallery/quality-audit")),
    }
    atomic_json(args.output, output)
    print(json.dumps({"checks": checks, "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
