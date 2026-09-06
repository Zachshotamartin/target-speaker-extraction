"""Attributed development listening gallery with an explicit, score-blind selection."""

from __future__ import annotations

import html
import io
import json
from pathlib import Path

import torch

from tse.audio import read_audio, wav_bytes
from tse.data import SpeechCorpus, load_cases
from tse.inference import Extractor
from tse.metrics import measure
from tse.utils import atomic_json, sha256


def make_gallery(
    checkpoint: Path,
    root: Path,
    manifest: Path,
    cases_path: Path,
    output: Path,
    device: str = "cpu",
    count: int = 20,
) -> dict:
    if count < 2 or count > 100 or count % 2:
        raise ValueError("Choose an even gallery size between 2 and 100")
    protocol = json.loads(cases_path.read_text())
    if protocol["split"] != "dev":
        raise ValueError("The public listening gallery uses development data only")
    corpus = SpeechCorpus(root, manifest, "dev", 4, 5)
    cases = load_cases(cases_path, corpus)[:count]
    extractor = Extractor(checkpoint, device)
    output.mkdir(parents=True, exist_ok=True)
    items = []
    cards = []
    for index, case in enumerate(cases, 1):
        batch = corpus.batch([case], torch.device("cpu"))
        tracks = {}
        for key in ("mixture", "reference", "target"):
            data = wav_bytes(batch[key][0, 0].numpy())
            filename = f"case-{index:02d}-{key}.wav"
            (output / filename).write_bytes(data)
            tracks[key] = filename
        audio, metadata = extractor.extract_files(
            output / tracks["mixture"], output / tracks["reference"]
        )
        prediction = torch.from_numpy(read_audio(io.BytesIO(audio)))[None, None]
        filename = f"case-{index:02d}-output.wav"
        (output / filename).write_bytes(audio)
        tracks["output"] = filename
        metrics = measure(prediction, batch["mixture"], batch["target"], batch["interferer"])[0]
        items.append(
            {
                "case_id": case["case_id"],
                "target_speaker": batch["speakers"][0],
                "sources": case["sources"],
                "references": case["references"],
                "target_index": case["target_index"],
                "metrics": metrics,
                "processing": metadata,
                "tracks": tracks,
            }
        )
        players = "".join(
            f'<div class="gallery-track"><span>{label}</span><audio controls preload="none" src="{tracks[key]}"></audio></div>'
            for key, label in (
                ("mixture", "Original mixture"),
                ("reference", "Separate voice reference"),
                ("target", "Known clean target"),
                ("output", "Model estimate"),
            )
        )
        flag = "Confusion proxy flagged" if metrics["confused"] else "Confusion proxy not flagged"
        source_ids = ", ".join(html.escape(entry["id"]) for entry in case["sources"])
        cards.append(
            f'<section class="gallery-card"><div class="gallery-heading"><h2>Request {index:02d} · voice {html.escape(batch["speakers"][0])}</h2>'
            f"<span>{metrics['si_sdri_db']:+.2f} dB SI-SDR improvement</span></div>"
            f'<p class="gallery-caption">{html.escape(case["case_id"])} · {flag}</p>'
            f'<div class="gallery-tracks">{players}</div><p class="gallery-caption">LibriSpeech sources: {source_ids}.</p></section>'
        )
    index = {
        "checkpoint_sha256": sha256(checkpoint),
        "case_manifest_sha256": sha256(cases_path),
        "source_manifest_sha256": sha256(manifest),
        "selection": f"First {len(cases)} development requests in manifest order, including paired target swaps; no score-based selection.",
        "attribution": "LibriSpeech, Panayotov et al. (2015), CC BY 4.0. Cropped, normalized, mixed and processed derivatives.",
        "source": "https://www.openslr.org/12",
        "items": items,
    }
    atomic_json(output / "index.json", index)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Listening gallery · One voice</title><link rel="stylesheet" href="/assets/style.css"></head>
<body><div class="page"><header class="topbar"><a class="brand" href="/">one voice<span class="brand-dot">.</span></a><nav aria-label="Main navigation"><a href="/">Workspace</a><a class="current" href="/gallery/">Listening gallery</a></nav></header>
<main><section class="introduction"><div><p class="eyebrow">DEVELOPMENT AUDIO / MEASURED EXAMPLES</p><h1>Listen closely.</h1><p class="intro-copy">The mixture, a separate reference, the known target, and the model’s estimate. Consecutive request pairs use the same mixture with a different voice to keep.</p></div></section>
<p class="gallery-intro">{html.escape(index["selection"])} Model {sha256(checkpoint)[:12]}. These examples are development data, not the reserved final test. A signal score cannot replace listening.</p>
{"".join(cards)}
<p class="gallery-intro">Speech: <a href="https://www.openslr.org/12">LibriSpeech</a>, Vassil Panayotov, Guoguo Chen, Daniel Povey and Sanjeev Khudanpur (2015), <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>. Recordings have been cropped, normalized, mixed and processed. <a href="index.json">View source IDs, recipes and per-request metrics</a>.</p>
</main><footer><span>ONE VOICE / LISTENING GALLERY</span><span>Evidence you can hear.</span></footer></div><script>document.addEventListener('play', function(event) {{ if (event.target.tagName === 'AUDIO') document.querySelectorAll('audio').forEach(function(audio) {{ if (audio !== event.target) audio.pause(); }}); }}, true);</script></body></html>"""
    (output / "index.html").write_text(page)
    return index
