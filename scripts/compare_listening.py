#!/usr/bin/env python3
"""Create a score-blind development comparison with optional matched playback RMS."""

import argparse
import html
import io
from pathlib import Path

import numpy as np

from tse.audio import read_audio, wav_bytes
from tse.data import SpeechCorpus, load_cases
from tse.inference import Extractor
from tse.utils import atomic_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=Path("artifacts/releases/v0.1.0/model.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/gallery/quality-progress"))
    parser.add_argument("--count", type=int, default=6)
    args = parser.parse_args()
    if args.count < 2 or args.count > 20 or args.count % 2:
        raise ValueError("Use an even count from 2 through 20")
    manifest, cases_path = (
        Path("data/manifests/inventory.json"),
        Path("data/manifests/dev-cases.json"),
    )
    corpus = SpeechCorpus(Path("data/raw"), manifest, "dev", 4, 5)
    cases = load_cases(cases_path, corpus)[: args.count]
    models = {
        name: Extractor(path, "cpu")
        for name, path in (("baseline", args.baseline), ("candidate", args.candidate))
    }
    for model in models.values():
        if set(model.payload["provenance"].get("train_speakers", [])) & set(corpus.speakers):
            raise ValueError("Listening data overlaps checkpoint training speakers")
    args.output.mkdir(parents=True, exist_ok=True)
    labels = {
        "mixture": "Original mixture",
        "target": "Known clean target",
        "baseline": "Original model",
        "candidate": "Training candidate",
        "reference": "Separate voice sample",
    }
    cards, records = [], []
    for number, case in enumerate(cases, 1):
        signals = corpus.render(case)
        tracks = {key: signals[key] for key in ("mixture", "target", "reference")}
        processing = {}
        for name, model in models.items():
            data, processing[name] = model.extract_files(
                io.BytesIO(wav_bytes(signals["mixture"])),
                io.BytesIO(wav_bytes(signals["reference"])),
            )
            tracks[name] = read_audio(io.BytesIO(data))
        normalized = {
            key: value * (0.1 / max(float(np.sqrt(np.mean(value**2))), 1e-8))
            for key, value in tracks.items()
        }
        common = min(1.0, 0.98 / max(float(np.max(np.abs(value))) for value in normalized.values()))
        paths, players = {}, []
        for key, label in labels.items():
            paths[key] = {}
            for mode, audio in (("raw", tracks[key]), ("matched", normalized[key] * common)):
                filename = f"{number:02d}-{key}-{mode}.wav"
                (args.output / filename).write_bytes(wav_bytes(audio))
                paths[key][mode] = filename
                paths[key][f"{mode}_sha256"] = sha256(args.output / filename)
            players.append(
                f'<div class="gallery-track"><span>{label}</span><audio controls preload="none" src="{paths[key]["matched"]}" data-raw="{paths[key]["raw"]}" data-matched="{paths[key]["matched"]}"></audio></div>'
            )
        records.append(
            {
                "case": case,
                "paths": paths,
                "processing": processing,
                "shared_peak_gain_after_rms_matching": common,
            }
        )
        cards.append(
            f'<section class="gallery-card"><h2>Request {number:02d} · voice {html.escape(signals["speaker"])}</h2><p class="gallery-caption">{html.escape(case["case_id"])}</p><div class="gallery-tracks">{"".join(players)}</div></section>'
        )
    index = {
        "models": {name: model.info() for name, model in models.items()},
        "case_manifest_sha256": sha256(cases_path),
        "source_manifest_sha256": sha256(manifest),
        "selection": f"First {len(cases)} development requests, including paired target switches; independent of scores.",
        "level_matching": "Each track is normalized to RMS 0.1, then one shared attenuation across all tracks in that request keeps sample peaks within 0.98. This is for listening only, not model inference or LUFS matching. Raw audio is also retained.",
        "attribution": "LibriSpeech / OpenSLR 12, Panayotov et al. (2015), CC BY 4.0; cropped, normalized, mixed and processed derivatives.",
        "items": records,
    }
    atomic_json(args.output / "index.json", index)
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Training comparison · One voice</title><link rel="stylesheet" href="/assets/style.css"></head><body><div class="page"><header class="topbar"><a class="brand" href="/">one voice.</a><a href="/gallery/">Original gallery</a></header><main><section class="introduction"><div><p class="eyebrow">DEVELOPMENT / TRAINING IN PROGRESS</p><h1>Hear the difference.</h1><p class="intro-copy">The original model and a newer training checkpoint, beside the known clean voice. This is a progress comparison; training and selection continue.</p></div></section><p class="gallery-intro">Consecutive pairs keep different voices from the same mixture. These are the first development examples, selected independently of scores.</p><label class="gallery-intro"><input type="checkbox" id="match" checked> Match listening volume</label><p class="gallery-caption">Matching adjusts average signal power for comparison, with shared headroom. Turn it off to hear the original output levels.</p>'
        + "".join(cards)
        + '<p class="gallery-intro">Speech: <a href="https://www.openslr.org/12">LibriSpeech</a>, Panayotov et al. (2015), <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>. <a href="index.json">Source identities, model hashes and transformations</a>.</p></main></div><script>document.addEventListener("play",e=>{if(e.target.tagName==="AUDIO")document.querySelectorAll("audio").forEach(a=>{if(a!==e.target)a.pause()})},true);document.getElementById("match").addEventListener("change",e=>document.querySelectorAll("audio").forEach(a=>{a.pause();a.src=e.target.checked?a.dataset.matched:a.dataset.raw}));</script></body></html>'
    )
    (args.output / "index.html").write_text(page)
    print(f"Wrote {len(cases)} development comparisons to {args.output}")


if __name__ == "__main__":
    main()
