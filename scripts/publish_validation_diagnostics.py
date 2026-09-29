"""Publish descriptive diagnostics from saved validation rows; never run inference."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_rows(best, latest):
    """Align by case identity, never by incidental file order."""
    before = {r["case_id"]: r for r in best["rows"]}
    after = {r["case_id"]: r for r in latest["rows"]}
    if len(before) != len(best["rows"]) or len(after) != len(latest["rows"]):
        raise ValueError("Duplicate validation case")
    if not before or before.keys() != after.keys():
        raise ValueError("Validation suites must contain identical cases")
    if best["manifest_sha256"] != latest["manifest_sha256"]:
        raise ValueError("Validation manifests differ")
    result = []
    for key, a in before.items():
        b = after[key]
        if a["target_speaker"] != b["target_speaker"]:
            raise ValueError("Target speaker changed between evaluations")
        values = [a["si_sdri_db"], b["si_sdri_db"]]
        if not np.isfinite(values).all():
            raise ValueError("Non-finite validation score")
        result.append(
            dict(
                case_id=key,
                target_speaker=a["target_speaker"],
                best_db=values[0],
                latest_db=values[1],
                delta_db=values[1] - values[0],
                best_confused=a["confused"],
                latest_confused=b["confused"],
            )
        )
    return result


def score_summary(rows, field):
    values = np.array([row[field] for row in rows])
    return dict(
        mean=float(values.mean()),
        median=float(np.median(values)),
        p10=float(np.percentile(values, 10)),
        p25=float(np.percentile(values, 25)),
        p75=float(np.percentile(values, 75)),
        p90=float(np.percentile(values, 90)),
        minimum=float(values.min()),
        maximum=float(values.max()),
        improved=int((values > 0).sum()),
        no_improvement=int((values <= 0).sum()),
    )


def analyze(best, latest, metrics):
    rows = paired_rows(best, latest)
    speakers = []
    for speaker in sorted({r["target_speaker"] for r in rows}):
        group = [r for r in rows if r["target_speaker"] == speaker]
        a = score_summary(group, "best_db")
        b = score_summary(group, "latest_db")
        speakers.append(
            dict(
                speaker=speaker,
                cases=len(group),
                best_mean=a["mean"],
                best_p25=a["p25"],
                best_p75=a["p75"],
                latest_mean=b["mean"],
            )
        )
    speakers.sort(key=lambda r: (r["best_mean"], r["speaker"]))
    for rank, row in enumerate(speakers, 1):
        row["rank"] = rank
    failures = []
    seen = set()
    for row in metrics:
        if row.get("event") != "validation" or row.get("kind") != "full":
            continue
        if row["step"] > latest["step"] or row["step"] in seen:
            continue
        if row["cases"] != len(rows) or row["manifest_sha256"] != best["manifest_sha256"]:
            raise ValueError("Historical full-validation suite changed")
        seen.add(row["step"])
        failures.append(
            dict(
                epoch=row["epoch"],
                step=row["step"],
                no_improvement_percent=100 * (1 - row["positive_improvement_fraction"]),
                wrong_speaker_percent=100 * row["confusion_fraction"],
            )
        )
    failures.sort(key=lambda r: r["step"])
    if not failures or failures[-1]["step"] != latest["step"]:
        raise ValueError("Missing latest full evaluation in history")
    changes = np.array([r["delta_db"] for r in rows])
    summary = dict(
        cases=len(rows),
        speakers=len(speakers),
        best=score_summary(rows, "best_db"),
        latest=score_summary(rows, "latest_db"),
        wrong_speaker_cases=sum(r["best_confused"] for r in rows),
        latest_wrong_speaker_cases=sum(r["latest_confused"] for r in rows),
        paired=dict(
            mean_change=float(changes.mean()),
            improved_over_1db=int((changes > 1).sum()),
            within_1db=int((abs(changes) <= 1).sum()),
            worsened_over_1db=int((changes < -1).sum()),
        ),
        speaker_mean_range=[speakers[0]["best_mean"], speakers[-1]["best_mean"]],
        speaker_case_range=[min(r["cases"] for r in speakers), max(r["cases"] for r in speakers)],
    )
    for data, key in ((best, "best"), (latest, "latest")):
        if not np.isclose(summary[key]["mean"], data["mean_si_sdri_db"], atol=1e-9, rtol=0):
            raise ValueError("Saved aggregate does not match case rows")
        if not np.isclose(
            summary[key]["improved"] / len(rows), data["positive_improvement_fraction"]
        ):
            raise ValueError("Saved improvement fraction does not match rows")
    return dict(summary=summary, paired=rows, speakers=speakers, failures=failures)


def publish(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    paths = [args.best, args.latest, args.metrics]
    best, latest = [json.loads(path.read_text()) for path in paths[:2]]
    metrics = [json.loads(line) for line in args.metrics.read_text().splitlines()]
    data = analyze(best, latest, metrics)
    summary = data["summary"]
    assets = args.site / "public/assets/one-voice"
    assets.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"font.size": 13, "svg.fonttype": "none", "svg.hashsalt": "onevoice-diagnostics-v1"}
    )
    green, blue, rust, gray = "#24483b", "#496b8e", "#a3482d", "#929691"

    def plot():
        fig, ax = plt.subplots(figsize=(10, 5.4), layout="constrained")
        fig.patch.set_facecolor("#fafaf8")
        ax.set_facecolor("#fafaf8")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
        return fig, ax

    def asset(name, value, extension):
        token = hashlib.sha256(value).hexdigest()[:12]
        path = assets / f"{name}-{token}.{extension}"
        path.write_bytes(value)
        return f"/assets/one-voice/{path.name}"

    def save(fig, name):
        stream = io.StringIO()
        fig.savefig(stream, format="svg", metadata={"Date": None})
        plt.close(fig)
        clean = "\n".join(line.rstrip() for line in stream.getvalue().splitlines()) + "\n"
        return asset(name, clean.encode(), "svg")

    values = np.array([r["best_db"] for r in data["paired"]])
    recent = np.array([r["latest_db"] for r in data["paired"]])
    bins = np.arange(
        np.floor(min(values.min(), recent.min()) / 2) * 2,
        np.ceil(max(values.max(), recent.max()) / 2) * 2 + 2,
        2,
    )
    fig, ax = plot()
    ax.hist(
        values,
        bins=bins,
        weights=np.full(len(values), 100 / len(values)),
        color=green,
        alpha=0.8,
        label=f"Best · epoch {best['epoch']:g}",
    )
    ax.hist(
        recent,
        bins=bins,
        weights=np.full(len(recent), 100 / len(recent)),
        color=blue,
        histtype="step",
        linewidth=2,
        label=f"Latest validation · epoch {latest['epoch']:g}",
    )
    ax.axvline(0, color=rust, linestyle="--", linewidth=1.3, label="No improvement")
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set(xlabel="SI-SDR improvement over the mixture (dB)", ylabel="Share of validation requests")
    ax.legend(frameon=False, loc="upper left")
    distribution = save(fig, "quality-distribution")

    fig, ax = plot()
    epochs = [r["epoch"] for r in data["failures"]]
    ax.plot(
        epochs,
        [r["no_improvement_percent"] for r in data["failures"]],
        color=rust,
        linewidth=2,
        label="No improvement over mixture",
    )
    ax.plot(
        epochs,
        [r["wrong_speaker_percent"] for r in data["failures"]],
        color=blue,
        linewidth=2,
        linestyle="--",
        label="Wrong-speaker match",
    )
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set(
        xlabel="Completed training epochs",
        ylabel="Share of validation requests",
        ylim=(0, 50),
        xlim=(0, latest["epoch"]),
    )
    ax.legend(frameon=False, loc="upper right")
    failures = save(fig, "failure-rates")

    fig, ax = plot()
    rank = [r["rank"] for r in data["speakers"]]
    ax.vlines(
        rank,
        [r["best_p25"] for r in data["speakers"]],
        [r["best_p75"] for r in data["speakers"]],
        color=gray,
        linewidth=2,
        label="Best: middle 50% of that speaker’s requests",
    )
    ax.scatter(
        rank,
        [r["best_mean"] for r in data["speakers"]],
        color=green,
        s=27,
        label="Best: mean per speaker",
        zorder=3,
    )
    ax.scatter(
        rank,
        [r["latest_mean"] for r in data["speakers"]],
        color=blue,
        marker="x",
        s=25,
        label="Latest validation: mean per speaker",
        zorder=3,
    )
    ax.axhline(summary["best"]["mean"], color=green, linestyle=":", linewidth=1)
    ax.set(
        xlabel="Development speakers, ordered by best-model mean (low → high)",
        ylabel="SI-SDR improvement (dB)",
        xlim=(0, len(rank) + 1),
    )
    ax.legend(frameon=False, loc="lower right", fontsize=11)
    speakers = save(fig, "speaker-performance")

    fig, ax = plot()
    limits = [
        np.floor(min(values.min(), recent.min()) / 5) * 5,
        np.ceil(max(values.max(), recent.max()) / 5) * 5,
    ]
    changes = recent - values
    colors = np.where(changes > 1, green, np.where(changes < -1, rust, gray))
    ax.scatter(values, recent, color=colors, s=8, alpha=0.35, edgecolors="none")
    ax.plot(limits, limits, color="#343b37", linewidth=1, label="Same score")
    ax.fill_between(limits, np.array(limits) - 1, np.array(limits) + 1, color=gray, alpha=0.12)
    ax.set(
        xlabel=f"Best · epoch {best['epoch']:g}: improvement (dB)",
        ylabel=f"Latest validation · epoch {latest['epoch']:g}: improvement (dB)",
        xlim=limits,
        ylim=limits,
    )
    ax.text(
        0.04,
        0.94,
        "Above diagonal: latest validation is better",
        transform=ax.transAxes,
        va="top",
        color=green,
        fontsize=12,
    )
    ax.text(
        0.96,
        0.06,
        "Below diagonal: best model is better",
        transform=ax.transAxes,
        ha="right",
        color=rust,
        fontsize=12,
    )
    paired = save(fig, "paired-checkpoints")

    report = dict(
        schema=1,
        best_epoch=best["epoch"],
        latest_epoch=latest["epoch"],
        best_step=best["step"],
        latest_step=latest["step"],
        best_checkpoint_sha256=best["checkpoint_sha256"],
        manifest_sha256=best["manifest_sha256"],
        source_sha256={path.name: digest(path) for path in paths},
        definitions={
            "scope": "Same 6,000 development requests from 40 unseen-in-training speakers; not an untouched test set.",
            "wrong_speaker": "Prediction SI-SDR against the interferer exceeds SI-SDR against the target by more than 3 dB.",
            "no_improvement": "Target SI-SDR improvement is at most zero. This can overlap wrong-speaker cases; rates must not be added.",
            "speaker_bars": "25th to 75th request-score percentiles, not confidence intervals. Dots are means; speaker groups have unequal sizes.",
            "paired_buckets": "Differences greater than +1 dB, within inclusive ±1 dB, and less than -1 dB. Descriptive thresholds, not a significance test.",
            "latest": "Epoch-81 validation at 281,475 updates. The saved checkpoint at 281,484 is newer and not represented by these scores.",
        },
        **data,
    )
    downloads = {}
    for name in ("paired", "speakers", "failures"):
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=list(data[name][0]))
        writer.writeheader()
        writer.writerows(data[name])
        downloads[name] = asset(f"diagnostics-{name}", stream.getvalue().encode(), "csv")
    downloads["report"] = asset(
        "validation-diagnostics", (json.dumps(report, indent=2) + "\n").encode(), "json"
    )
    frontend = dict(
        summary=summary,
        best_epoch=best["epoch"],
        latest_epoch=latest["epoch"],
        charts=dict(distribution=distribution, failures=failures, speakers=speakers, paired=paired),
        downloads=downloads,
    )
    (args.site / "src/validation-diagnostics.json").write_text(
        json.dumps(frontend, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--best", type=Path, required=True)
    parser.add_argument("--latest", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--site", type=Path, default=Path("site"))
    publish(parser.parse_args())
