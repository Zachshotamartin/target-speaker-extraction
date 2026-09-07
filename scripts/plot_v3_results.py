#!/usr/bin/env python3
"""Plot the frozen acoustic comparison and recorded development training curves."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tse.utils import atomic_json, sha256


def read(path):
    return json.loads(Path(path).read_text())


def validations(name):
    path = Path("artifacts/runs") / f"v3-{name}" / "metrics.jsonl"
    if not path.is_file():
        return []
    by_step = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get("event") == "validation":
            by_step[row["step"]] = row
    return [by_step[step] for step in sorted(by_step)]


def main():
    output = Path("reports/figures/v3")
    output.mkdir(parents=True, exist_ok=True)
    freeze = read("reports/v3-selection-freeze.json")
    comparison = read("reports/v3-test-comparison.json")
    if (
        comparison["candidate_checkpoint_sha256"]
        != freeze["checkpoints"]["candidate"]["checkpoint_sha256"]
    ):
        raise ValueError("Figure candidate differs from frozen choice")
    colors = {"baseline": "#aaa99f", "candidate": "#315e4a", "accent": "#ad7540"}
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "#fbfaf6",
            "axes.facecolor": "#fbfaf6",
            "savefig.facecolor": "#fbfaf6",
        }
    )
    conditions = ["clean", "noise", "reverb", "partial_overlap", "pauses", "combined"]
    labels = ["Clean", "Noise", "Reverberation", "Partial overlap", "Pauses", "Combined"]
    metrics = [comparison["conditions"][condition]["si_sdri_db"] for condition in conditions]
    y = np.arange(len(conditions))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), gridspec_kw={"width_ratios": [1.05, 1]})
    for offset, name, label in [
        (-0.18, "baseline", "v0.2.0"),
        (0.18, "candidate", "Frozen candidate"),
    ]:
        axes[0].barh(
            y + offset,
            [metric[f"{name}_mean"] for metric in metrics],
            height=0.34,
            label=label,
            color=colors[name],
        )
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Mean SI-SDR improvement over mixture (dB)")
    axes[0].set_title("Separation under acoustic conditions", loc="left", pad=18)
    axes[0].legend(frameon=False, loc="lower right")
    means = np.array([metric["mean_candidate_minus_baseline"] for metric in metrics])
    bounds = np.array([metric["paired_ci95"] for metric in metrics])
    axes[1].hlines(y, bounds[:, 0], bounds[:, 1], color=colors["candidate"], linewidth=1.5)
    axes[1].scatter(means, y, color=colors["candidate"], s=35, zorder=3)
    axes[1].axvline(0, color="#77776f", linewidth=1, linestyle="--")
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Candidate minus v0.2.0 (dB), approximate 95% interval")
    axes[1].set_title("Paired differences", loc="left", pad=18)
    for ax in axes:
        ax.grid(axis="x", alpha=0.18)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Fresh test · 600 target-present requests / 33 voices",
        x=0.04,
        ha="left",
        fontsize=15,
        weight="bold",
    )
    fig.text(
        0.04,
        0.04,
        "100 requests per condition; conditions share mixtures. Target-speaker cluster bootstrap, 1,000 replicates.\nShared-interferer/room dependence and training-seed uncertainty remain. Static requires human listening.",
        fontsize=9,
        color="#55564f",
    )
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.22, top=0.80, wspace=0.58)
    fig.savefig(output / "acoustic-comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for name, label, color in [
        ("continuation-control", "Plateau continuation", colors["baseline"]),
        ("continuation-cosine", "Cosine continuation", colors["candidate"]),
    ]:
        rows = validations(name)
        axes[0].plot(
            [row["step"] - 20000 for row in rows],
            [row["mean_si_sdri_db"] for row in rows],
            label=label,
            color=color,
            marker=".",
        )
    for name, label, color in [
        ("band-real", "Real mask / project reference", colors["baseline"]),
        ("band-complex", "Complex mask / project reference", colors["candidate"]),
        ("band-resnet", "Complex mask / new ResNet", colors["accent"]),
    ]:
        rows = validations(name)
        axes[1].plot(
            [row["step"] for row in rows],
            [row["mean_si_sdri_db"] for row in rows],
            label=label,
            color=color,
            marker=".",
        )
    axes[0].set_title("Matched learning-rate continuation", loc="left")
    axes[0].set_xlabel("Additional updates after the shared 20,000-update state")
    axes[1].set_title("Band-separator pilots and selected extension", loc="left")
    axes[1].set_xlabel("New separator training updates")
    for ax in axes:
        ax.set_ylabel("Mean development SI-SDRi (dB)")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        ax.grid(alpha=0.18)
    fig.suptitle(
        "Development learning curves · same 80 clean requests",
        x=0.05,
        ha="left",
        fontsize=15,
        weight="bold",
    )
    fig.text(
        0.05,
        0.035,
        "One training seed per comparison. Pilots share separator initialization; reference pretraining differs in the ResNet arm.\nAdaptation uses a different 140-case selection set and is excluded from this chart.",
        fontsize=9,
        color="#55564f",
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.24, top=0.79, wspace=0.32)
    fig.savefig(output / "development-curves.png", dpi=180)
    plt.close(fig)
    atomic_json(
        output / "sources.json",
        {
            "test_comparison_sha256": sha256(Path("reports/v3-test-comparison.json")),
            "selection_freeze_sha256": sha256(Path("reports/v3-selection-freeze.json")),
            "training_logs": {
                str(path): sha256(path)
                for path in sorted(Path("artifacts/runs").glob("v3-*/metrics.jsonl"))
            },
        },
    )
    print(f"Wrote figures to {output}")


if __name__ == "__main__":
    main()
