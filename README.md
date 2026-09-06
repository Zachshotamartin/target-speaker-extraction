# Target Speaker Extraction

Train a compact audio model that isolates a chosen speaker from an overlapping recording, using a separate example of that person's voice.

The practical focus is reference quality: a voice sample recorded on a phone should still be useful when the conversation comes from another microphone or room. Whether the model achieves that is an experimental question, not a current capability claim.

**Status: project plan and repository foundation. No audio model has been implemented or trained, and no performance results are available.**

## What we are building

Inputs:

- A recording containing two overlapping speakers.
- A separate 3–10 second reference recording of the intended speaker; the default experiment uses 5 seconds.

Output:

- A mono audio file containing the estimated target voice, aligned to the input recording.
- Processing metadata: model version, sample rate, duration, runtime, and input validation messages.

The first version processes recorded audio locally. Live microphone use is a later milestone requiring a causal architecture and measured latency.

## Independent implementation

We will write the data pipeline, network, training loop, evaluation, and inference code ourselves using PyTorch and general numerical/audio libraries. The core experiment trains its weights from random initialization.

SpeakerBeam is a research reference, not a codebase or checkpoint dependency. Public datasets and established mathematical ideas will be attributed. This project claims an independent implementation and measured engineering contributions; it does not claim to have invented target speaker extraction.

## Read the plan

| Document | Contents |
| --- | --- |
| [Project plan](docs/PROJECT_PLAN.md) | Product scope, research question, success criteria, hardware budget, delivery strategy |
| [Data plan](docs/DATA_PLAN.md) | Sources, speaker splits, manifests, mixture generation, reference corruption, storage |
| [Model design](docs/MODEL_DESIGN.md) | Tensor contracts, proposed architecture, losses, training and inference |
| [Evaluation plan](docs/EVALUATION.md) | Baselines, experiments, leakage controls, metrics, uncertainty, release criteria |
| [Engineering plan](docs/ENGINEERING.md) | Environment, package layout, tracking, tests, API, deployment and reproducibility |
| [Roadmap and backlog](docs/ROADMAP.md) | Milestones, dependencies, implementation tickets and acceptance checks |
| [Decisions](docs/DECISIONS.md) | Agreed boundaries and provisional design choices |
| [Sources and attribution](docs/SOURCES.md) | Primary references, data licenses and origin tracking |

## Local machine

The initial plan targets an Apple M3 Pro with 18 GiB unified memory and approximately 44 GiB free disk at setup. Training speed and usable batch size are unmeasured. A short forward/backward benchmark is the first implementation gate.

The planning configuration is [configs/pilot.toml](configs/pilot.toml). It records proposed defaults; there is no training command consuming it yet.

## Available command

With Python 3.12 installed:

```sh
python3.12 scripts/check_environment.py
```

Or with `uv`:

```sh
uv run --no-project --python 3.12 scripts/check_environment.py
```

The command reports the active interpreter, platform, disk space, tool availability, and installed PyTorch version if present. It does not install packages, download data, or run training. Model-specific MPS compatibility remains an implementation task.

## Repository conventions

- `main` holds reviewed, reproducible work; short feature branches hold implementation changes.
- Audio, datasets, checkpoints, run logs, credentials, and local environments stay out of Git.
- Small manifests, checksums, configurations, aggregate results, and documentation can be versioned when they contain no private paths or recordings.
- Each reported experiment must identify its code commit, configuration, data manifest, checkpoint, and evaluation protocol.
- Document capabilities as planned, implemented, measured, or released. Do not substitute proposed targets for results.

The repository is intended for public portfolio review. A license for original project code has not yet been selected; third-party data and dependencies retain their own licenses.
