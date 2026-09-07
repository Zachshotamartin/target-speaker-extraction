#!/usr/bin/env python3
"""Plot a hash-verified paired quality comparison and the development learning curve."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tse.quality import compare_quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--run", type=Path, default=Path("artifacts/runs/quality-stft-expanded-fastlr")
    )
    parser.add_argument("--output", type=Path, default=Path("reports/figures/quality-v2"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    compare_quality(args.baseline, args.candidate, args.output / "paired-comparison.json")
    reports = [json.loads(path.read_text()) for path in (args.baseline, args.candidate)]
    labels = ["Original model", "New model"]
    colors = ["#9A9A92", "#47634C"]
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "font.size": 10,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(11, 4.2), layout="constrained")
    for ax, metric, title, factor in zip(
        axes,
        ("mean_si_sdri_db", "mean_estoi", "confusion_fraction"),
        (
            "Separation improvement · dB ↑",
            "Intelligibility proxy · ESTOI ↑",
            "Speaker confusion proxy · % ↓",
        ),
        (1, 1, 100),
        strict=True,
    ):
        values = [report["summary"][metric] * factor for report in reports]
        ax.bar(labels, values, color=colors, width=0.6)
        ax.set(title=title, ylim=(0, max(values) * 1.25))
        for i, value in enumerate(values):
            ax.text(i, value, f"{value:.2f}", ha="center", va="bottom")
    figure.suptitle(
        f"{reports[1].get('split', 'dev').upper()} · {reports[1]['summary']['cases']:,} matched extraction requests · clean references"
    )
    figure.supxlabel(
        "Same delivered WAV path. Higher ESTOI predicts intelligibility; it is not a human listening score.",
        fontsize=9,
    )
    figure.savefig(args.output / "quality-comparison.png", dpi=170)
    plt.close(figure)
    metrics = [json.loads(line) for line in (args.run / "metrics.jsonl").read_text().splitlines()]
    points = [row for row in metrics if row["event"] == "validation"]
    figure, ax = plt.subplots(figsize=(8, 4.2), layout="constrained")
    ax.plot(
        [row["step"] for row in points], [row["mean_si_sdri_db"] for row in points], color=colors[1]
    )
    ax.set(
        xlabel="Expanded-data optimizer updates",
        ylabel="Mean SI-SDR improvement · dB",
        title="Checkpoint selection · 80 development requests",
    )
    ax.grid(axis="y", alpha=0.2)
    figure.supxlabel(
        "Resumed segments share the same sample schedule and optimizer state. These are development measurements.",
        fontsize=9,
    )
    figure.savefig(args.output / "learning-curve.png", dpi=170)
    plt.close(figure)
    print(f"Saved quality figures to {args.output}")


if __name__ == "__main__":
    main()
