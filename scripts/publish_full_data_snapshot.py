"""Freeze best/latest training artifacts and render a reproducible public release.

Reads the run without changing its controls, checkpoints, or training environment.
The latest checkpoint is never assigned an earlier validation checkpoint's score.
"""

import argparse
import importlib.util
import json
import shutil
from pathlib import Path

import matplotlib
import numpy as np
import soundfile as sf
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tse.utils import sha256


def write_json(path, value, compact=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=None if compact else 2, allow_nan=False) + "\n")


def freeze(source, destination):
    with source.open("rb") as handle, destination.open("wb") as target:
        shutil.copyfileobj(handle, target)
    digest = sha256(destination)
    if digest != sha256(source):
        raise RuntimeError(f"{source.name} changed during snapshot; retry the export")
    return digest


def public_snapshot(record, validation, renders, assets, role):
    items, files = [], {}
    for item in record["items"]:
        tracks = {}
        for name, versions in item["tracks"].items():
            source = (
                renders / Path(versions["matched"]).parent.name / Path(versions["matched"]).name
            )
            samples, rate = sf.read(source, dtype="float32")
            assert rate == 16000 and np.isfinite(samples).all() and abs(samples).max() <= 1
            filename = f"{name}-{sha256(source)[:12]}.wav"
            sf.write(assets / filename, samples, rate, subtype="PCM_16")
            files[filename] = sha256(assets / filename)
            track = dict(
                src=f"/assets/one-voice/{filename}",
                duration=len(samples) / rate,
                gain=versions["playback_gain"],
                peaks=[round(float(np.max(np.abs(c))), 4) for c in np.array_split(samples, 192)],
            )
            if name == "estimate":
                raw = renders / Path(versions["raw"]).parent.name / Path(versions["raw"]).name
                raw_name = f"raw-estimate-{sha256(raw)[:12]}.wav"
                shutil.copyfile(raw, assets / raw_name)
                files[raw_name] = sha256(assets / raw_name)
                track["raw"] = f"/assets/one-voice/{raw_name}"
            tracks[name] = track
        items.append(
            dict(
                index=item["index"],
                conversation=item["mixture_number"],
                voice="A" if item["index"] % 2 == 0 else "B",
                caseId=item["case_id"],
                improvement=item["metrics"]["si_sdri_db"],
                tracks=tracks,
            )
        )
    return dict(
        step=record["step"],
        epoch=record["epoch"],
        checkpointSHA256=record["checkpoint_sha256"],
        sourceSHA256=record["source_tree_sha256"],
        dataManifestSHA256=record["manifest_sha256"],
        validation=validation,
        role=role,
        selection="Best mean SI-SDR improvement on all 6,000 development requests; not a held-out test score."
        if role == "best"
        else "Latest saved checkpoint; only the fixed listening examples were evaluated at this exact step.",
        examples="Both target voices in the first six development mixtures, fixed independently of quality.",
        playback="Playback levels matched toward RMS 0.08 with peak ceiling 0.98. No filtering or denoising added. Metrics use raw predictions.",
        attribution="LibriSpeech / OpenSLR 12, Panayotov et al. (2015), CC BY 4.0; Libri2Mix clean mixtures. Cropped, mixed and model-processed derivatives.",
        items=items,
        files=files,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--site", type=Path, default=Path("site"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    assets = args.site / "public/assets/one-voice"
    assets.mkdir(parents=True, exist_ok=True)

    def load(name):
        return json.loads((args.run / name).read_text())

    best = load("best-full-validation.json")
    full = load("latest-full-validation.json")
    monitor = load("best-monitor-validation.json")
    spec = importlib.util.spec_from_file_location(
        "listening", Path(__file__).with_name("watch_full_listening.py")
    )
    listening = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(listening)
    listening.INDICES = list(range(12))
    latest_payload = None
    for role, filename in [("best", "best-full.pt"), ("latest", "latest.pt")]:
        digest = freeze(args.run / filename, args.output / filename)
        payload = torch.load(args.output / filename, map_location="cpu", weights_only=True)
        step = payload["step"]
        if role == "best":
            assert step == best["step"] and digest == best["checkpoint_sha256"]
            validation = dict(
                cases=best["cases"],
                improvement=best["mean_si_sdri_db"],
                positiveFraction=best["positive_improvement_fraction"],
                confusionFraction=best["confusion_fraction"],
            )
        else:
            latest_payload = {k: payload[k] for k in ("step", "epoch", "elapsed_seconds")}
            validation = None
        del payload
        selection = dict(step=step, checkpoint=filename, result=best if role == "best" else {})
        record = listening.render_snapshot(
            args.output,
            dict(root=str(args.root), manifest=str(args.manifest)),
            selection,
            args.output / "renders",
        )
        snapshot = public_snapshot(record, validation, args.output / "renders", assets, role)
        write_json(
            args.site / "src" / ("snapshot.json" if role == "best" else "snapshot-latest.json"),
            snapshot,
            True,
        )
        write_json(
            assets / ("MODEL.json" if role == "best" else "MODEL-latest.json"),
            {k: v for k, v in snapshot.items() if k != "items"},
        )
        write_json(
            args.output / f"{role}-selection.json",
            dict(record=record, validation=best if role == "best" else None),
        )
        if role == "latest":
            data = (args.site / "src/snapshot-latest.json").read_bytes()
            name = f"listening-latest-{sha256(args.site / 'src/snapshot-latest.json')[:12]}.json"
            (assets / name).write_bytes(data)
            write_json(
                args.site / "src/release-assets.json",
                {"latestListening": f"/assets/one-voice/{name}"},
            )
        print(f"Frozen {role}: {step}, {digest}", flush=True)
    rows = [json.loads(line) for line in (args.run / "metrics.jsonl").read_text().splitlines()]
    history = [
        dict(
            epoch=r["epoch"],
            step=r["step"],
            kind=r["kind"],
            cases=r["cases"],
            improvement=r["mean_si_sdri_db"],
        )
        for r in rows
        if r.get("event") == "validation"
    ]

    def compact(value):
        return {k: v for k, v in value.items() if k != "rows"}

    report = dict(
        asOf="2026-09-29",
        status="paused",
        resumable=True,
        reason="Training paused; 100-epoch schedule not completed.",
        completedEpochs=81,
        plannedEpochs=100,
        latestStep=latest_payload["step"],
        elapsedSeconds=latest_payload["elapsed_seconds"],
        trainingMixtures=13900,
        trainingSpeakers=251,
        developmentSpeakers=40,
        bestFull=compact(best),
        latestFull=compact(full),
        bestMonitor=compact(monitor),
        latestCheckpointValidation="The latest checkpoint is 9 updates after epoch-81 validation; its full-suite score is unmeasured.",
        metric="Mean target-specific SI-SDR improvement over the mixture, in dB. Development results, not an unseen test or speech-recognition accuracy.",
        history=history,
    )
    write_json(
        args.site / "src/training-progress.json",
        {k: v for k, v in report.items() if k != "history"},
    )
    write_json(assets / "training-progress.json", report)
    write_json(args.output / "release.json", report)
    for name in (
        "best-full-validation.json",
        "latest-full-validation.json",
        "best-monitor-validation.json",
    ):
        write_json(args.output / name, load(name))
    fig, ax = plt.subplots(figsize=(10, 4.3), layout="constrained")
    fig.patch.set_facecolor("#fafaf8")
    ax.set_facecolor("#fafaf8")
    for kind, label, color in [
        ("monitor", "400-case monitor", "#a2a2a0"),
        ("full", "6,000-case validation", "#24483b"),
    ]:
        points = [r for r in history if r["kind"] == kind]
        ax.plot(
            [r["epoch"] for r in points],
            [r["improvement"] for r in points],
            color=color,
            linewidth=1.6,
            label=label,
        )
    ax.scatter([best["epoch"]], [best["mean_si_sdri_db"]], color="#24483b", s=40, zorder=3)
    ax.annotate(
        f"Best full: {best['mean_si_sdri_db']:.2f} dB",
        (best["epoch"], best["mean_si_sdri_db"]),
        xytext=(-115, -27),
        textcoords="offset points",
        fontsize=11,
    )
    ax.set(xlabel="Completed training epochs", ylabel="SI-SDR improvement (dB)", xlim=(0, 100))
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(assets / "training-progress.svg")
    chart = assets / "training-progress.svg"
    chart.write_text("\n".join(line.rstrip() for line in chart.read_text().splitlines()) + "\n")
    plt.close(fig)
    print(f"Published {len(history)} validation records", flush=True)


if __name__ == "__main__":
    main()
