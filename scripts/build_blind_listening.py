#!/usr/bin/env python3
"""Generate an anonymous A/B development gallery with separately stored identities."""

import argparse
import html
import io
import json
import secrets
from pathlib import Path

import numpy as np
import soundfile as sf

from tse.audio import read_audio, wav_bytes
from tse.inference import Extractor
from tse.realistic import SCENARIOS, RealisticCorpus
from tse.utils import atomic_json, sha256


def build(candidate, study, device, environment_root):
    if any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in study) or not study:
        raise ValueError("Use a simple lowercase study identifier")
    output = Path("artifacts/gallery") / study
    private = Path("artifacts/listening") / study
    if output.exists() or private.exists():
        raise FileExistsError("Do not change audio or identities in an existing listening study")
    manifest = Path("data/v3/manifests/inventory.json")
    env_manifest = Path("data/v3/manifests/environments.json")
    cases_path = Path("data/v3/manifests/dev-realistic-cases.json")
    corpus = RealisticCorpus(
        Path("data/expanded/raw"), manifest, "dev", 4, 5, environment_root, env_manifest
    )
    cases = json.loads(cases_path.read_text())["cases"]
    selected = [
        case
        for scenario in SCENARIOS
        if scenario != "target_absent"
        for case in [c for c in cases if c["scenario"] == scenario][:2]
    ]
    models = {
        "baseline": Extractor(Path("artifacts/releases/v0.2.0/model.pt"), device),
        "candidate": Extractor(candidate, device),
    }
    output.mkdir(parents=True)
    private.mkdir(parents=True)
    trials, sections = [], []
    for index, case in enumerate(selected, 1):
        signals = corpus.render(case)
        tracks = {"mixture": signals["mixture"], "target": signals["target"]}
        for name, extractor in models.items():
            encoded, _ = extractor.extract_files(
                io.BytesIO(wav_bytes(signals["mixture"])),
                io.BytesIO(wav_bytes(signals["reference"])),
            )
            tracks[name] = read_audio(io.BytesIO(encoded))
        matched = {
            name: wave * (0.1 / max(float(np.sqrt(np.mean(wave**2))), 1e-5))
            for name, wave in tracks.items()
        }
        common_gain = min(1.0, 0.98 / max(float(np.abs(wave).max()) for wave in matched.values()))
        labels = ["baseline", "candidate"]
        if secrets.randbelow(2):
            labels.reverse()
        identity = f"trial-{index:02}"
        mapping = dict(zip(("A", "B"), labels, strict=True))
        urls = {}
        audio_records = {}
        for label, name in {"mixture": "mixture", "target": "target", **mapping}.items():
            path = output / f"{identity}-{label}.wav"
            sf.write(path, matched[name] * common_gain, 16000, subtype="FLOAT")
            urls[label] = path.name
            audio_records[label] = {
                "sha256": sha256(path),
                "model": models[name].checkpoint_hash if name in models else None,
            }
        trials.append(
            {
                "id": identity,
                "case_id": case["case_id"],
                "scenario": case["scenario"],
                "models": {label: models[name].checkpoint_hash for label, name in mapping.items()},
                "audio": audio_records,
            }
        )
        controls = []
        for label in ("A", "B"):
            fields = []
            for criterion, title in [
                ("competing_speech", "Competing speech"),
                ("target_damage", "Damage to the desired voice"),
                ("static", "Static or artificial noise"),
            ]:
                fields.append(
                    f'<label>{title}<select required name="{label}-{criterion}"><option value="">Choose severity</option>{"".join(f"<option value={n}>{n}</option>" for n in range(1, 6))}</select></label>'
                )
            controls.append(
                f'<div><h3>Version {label}</h3><audio controls preload="none" src="{urls[label]}"></audio>{"".join(fields)}</div>'
            )
        sections.append(
            f'<section class="trial"><h2>Trial {index:02} · {html.escape(case["scenario"].replace("_", " "))}</h2><div class="sources"><label>Conversation<audio controls preload="none" src="{urls["mixture"]}"></audio></label><label>Known target<audio controls preload="none" src="{urls["target"]}"></audio></label></div><form data-trial="{identity}"><div class="versions">{"".join(controls)}</div><label>Optional note<textarea name="comment" maxlength="1000"></textarea></label><button type="submit">Save this rating</button><p role="status"></p></form></section>'
        )
    style = ".trial{border-top:1px solid #ccc;padding:28px 0}.sources,.versions{display:grid;grid-template-columns:1fr 1fr;gap:30px}label{display:block;margin:14px 0}select,textarea{display:block;width:100%;padding:9px;margin-top:6px;font:inherit}audio{display:block;max-width:100%;margin:10px 0}button{padding:12px 20px;background:#263c2d;color:white;border:0;font:inherit;cursor:pointer}textarea{min-height:60px}@media(max-width:650px){.sources,.versions{grid-template-columns:1fr}}"
    script = """const study=STUDY;
let participant=localStorage.getItem('tse-listening-participant');
if(!participant){participant=crypto.randomUUID();localStorage.setItem('tse-listening-participant',participant);}
document.querySelectorAll('form').forEach(form=>form.addEventListener('submit',async event=>{
event.preventDefault();const data=new FormData(form);const payload={participant,trial:form.dataset.trial,comment:data.get('comment'),A:{},B:{}};
for(const version of ['A','B'])for(const criterion of ['competing_speech','target_damage','static'])payload[version][criterion]=Number(data.get(version+'-'+criterion));
const status=form.querySelector('[role=status]');status.textContent='Saving…';
try{const response=await fetch('/listening/'+study+'/ratings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const result=await response.json();if(!response.ok)throw new Error(result.detail||'Could not save');status.textContent='Saved locally · '+result.completed_trials+' of '+result.total_trials+' trials rated.';}catch(error){status.textContent='Could not save: '+error.message;}
}));
document.querySelectorAll('audio').forEach(player=>player.addEventListener('play',()=>document.querySelectorAll('audio').forEach(other=>{if(other!==player)other.pause();})));""".replace(
        "STUDY", json.dumps(study)
    )
    page = f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Blind listening · One voice</title><link rel="stylesheet" href="/assets/style.css"><style>{style}</style></head><body><div class="page"><header class="topbar"><a class="brand" href="/">one voice.</a><a href="/gallery/">Listening gallery</a></header><main><p class="eyebrow">DEVELOPMENT / BLIND LISTENING</p><h1>Judge what you hear.</h1><p>Compare A and B with the known target. Rate each problem from 1 (none) to 5 (severe). Model identities are hidden and their order changes between trials. Listen for competing speech, damage to the voice you want, and static separately.</p><p>Playback levels are matched with shared peak headroom. Ratings stay on this computer, associated with a random browser identifier. Submitting another rating for a trial replaces your previous one. These are controlled public-speech examples, not natural recorded conversations.</p>{"".join(sections)}</main><p>LibriSpeech / OpenSLR12, CC BY 4.0. Simulated rooms and public noise recordings from OpenSLR28; cropped, mixed and processed derivatives.</p></div><script>{script}</script></body></html>'
    (output / "index.html").write_text(page)
    atomic_json(
        private / "key.json",
        {
            "study": study,
            "case_manifest_sha256": sha256(cases_path),
            "environment_manifest_sha256": sha256(env_manifest),
            "trials": trials,
            "selection": "First two development requests in each target-present scenario, independent of scores",
            "playback": "Each waveform RMS matched to 0.1, then one shared attenuation keeps all peaks <=0.98",
        },
    )
    print(f"http://127.0.0.1:8000/gallery/{study}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--study", required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    parser.add_argument("--environment-root", type=Path, required=True)
    args = parser.parse_args()
    build(args.candidate, args.study, args.device, args.environment_root)
