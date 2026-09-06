# Reproducing One voice

Run commands from the repository root. Python 3.12 is pinned by the project, and `uv.lock` pins dependencies. Apple Silicon uses native PyTorch/MPS; Linux uses the explicit PyTorch CPU wheel index. CUDA training is not configured by this lock.

## Setup and verification

```sh
git clone https://github.com/Zachshotamartin/target-speaker-extraction.git
cd target-speaker-extraction
uv sync --frozen
uv run python scripts/check_environment.py
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run pytest -q
uv build
```

Tests use generated signals, not downloaded speech. They cover metric correctness, leakage rejection, deterministic mixtures, length/padding, reference gradients, exact CPU resume, paired comparison contracts, upload errors, readiness, and chunk alignment. MPS speed and real-speech quality are measured separately.

## One-command study

```sh
uv run python scripts/run_study.py --download --steps 5000 --device mps --evaluate-test
```

For development, omit `--evaluate-test`. Existing verified downloads are reused. Interrupted training resumes from `latest.pt`; completed runs are skipped when their recorded budget matches. Do not launch two study/training processes writing the same run directory. Change `--device` to `cpu` when MPS is unavailable; CPU training is slower.

The final test freeze records both checkpoint hashes, the case-manifest hash, and development selection before evaluation. A second invocation with `--evaluate-test` fails if that freeze already exists. If evaluation was interrupted after the freeze, verify its hashes still match and use the individual evaluation commands below to finish the same frozen artifacts. Do not remove the freeze to choose a new model using test feedback.

## Data preparation, individually

```sh
uv run tse data acquire train-clean-100 --speakers 60 --utterances 50
uv run tse data acquire dev-clean --speakers 40 --utterances 30
uv run tse data acquire test-clean --speakers 40 --utterances 30
uv run tse data inventory
uv run tse data audit
uv run tse data build-cases --split train --count 16 --seed 44000 --seconds 2 --output data/manifests/train-cases.json
uv run tse data build-cases --split dev --count 80 --seed 88000 --output data/manifests/dev-cases.json
uv run tse data build-cases --split dev --count 400 --seed 188000 --output data/manifests/dev-report-cases.json
uv run tse data build-cases --split test --count 1000 --seed 99000 --output data/manifests/test-cases.json
```

Acquisition metadata records source URLs, license, selected speakers, utterance hashes, and partial-archive limitations. A download selection changes if the upstream archive ordering changes; compare the published metadata hashes. An existing selection with different arguments is rejected. Preserve the source inventory and frozen case manifests with any checkpoint.

## Train and evaluate

```sh
uv run tse benchmark --device mps --output reports/mps-benchmark.json
uv run tse train --config configs/overfit.json --run artifacts/runs/overfit --fixed-cases data/manifests/train-cases.json --device mps
uv run tse train --config configs/control-study.json --run artifacts/runs/control --device mps
uv run tse train --config configs/augmented-study.json --run artifacts/runs/augmented --device mps
```

The study runner writes the two `*-study.json` configs from the control config and requested update budget. Both train from scratch with the same seed; treatment changes reference augmentation. To continue an interrupted run, repeat its command with `--resume`. Changing any configuration except the maximum updates is rejected on resume.

```sh
uv run tse evaluate --checkpoint artifacts/runs/control/best.pt --cases data/manifests/dev-report-cases.json --output reports/control-development.json --device mps
uv run tse evaluate --checkpoint artifacts/runs/augmented/best.pt --cases data/manifests/dev-report-cases.json --output reports/augmented-development.json --device mps
uv run tse compare --control reports/control-development.json --treatment reports/augmented-development.json --output reports/development-comparison.json
```

Evaluation defaults to all five conditions. To finish an already frozen final test, replace `dev-report-cases.json` with `test-cases.json` and write `control-test.json`/`augmented-test.json`, then compare those files. The study runner performs the initial freeze and selection.

## Export and use

The study exports the model selected on development data. For an explicitly chosen checkpoint:

```sh
uv run tse export --checkpoint artifacts/runs/control/best.pt
uv run tse examples
uv run tse gallery
uv run tse serve --device cpu
```

Export strips optimizer/RNG state and writes `artifacts/releases/model.pt` plus hash/config/provenance JSON. It remains a trusted local PyTorch artifact; do not load arbitrary checkpoints. Exporting control manually overrides the study's selected artifact, so keep selection metadata consistent.

```sh
uv run tse extract --mixture conversation.wav --reference voice.wav --output isolated.wav --device cpu
curl -f http://127.0.0.1:8000/ready
curl -f -F mixture=@conversation.wav -F reference=@voice.wav http://127.0.0.1:8000/extract -o isolated.wav
```

The local web app offers two references for the first fixed development mixture. Examples are selected by manifest order, not by quality score, and are attributed to LibriSpeech/CC BY 4.0.

The [local listening gallery](http://127.0.0.1:8000/gallery/) contains the first 20 development requests, with original/reference/known-target/model audio and per-case metrics. Build it before starting the service; the API mounts the gallery when its directory exists. The gallery remains local and uses no uploaded private recordings.

## Delivery checks

```sh
uv run tse evaluate-delivery --cases data/manifests/test-cases.json --output reports/delivered-test.json --device cpu
uv run tse diagnose --checkpoint artifacts/releases/model.pt --output reports/reference-diagnostics.json --device cpu
```

Use development cases for delivery checks before the final freeze. The first command scores the same WAV decoding, normalization, extraction, and WAV encoding functions used by the API. HTTP behavior has independent integration tests.

For runtime profiling, supply valid development mixture/reference WAVs (the example index lists their filenames):

```sh
uv run tse profile-delivery --mixture artifacts/examples/voice-1-mixture.wav --reference artifacts/examples/voice-1-reference.wav --device cpu --output reports/delivery-cpu.json
uv run tse profile-delivery --mixture artifacts/examples/voice-1-mixture.wav --reference artifacts/examples/voice-1-reference.wav --device mps --output reports/delivery-mps.json
```

Run one device at a time without concurrent training. Profiles distinguish model load, first request, repeated warm end-to-end processing, and model-only time. They exclude browser/network/HTTP parsing. Repeated audio tests length-dependent runtime and numeric chunk agreement, not real long-conversation quality. Process RSS includes whole-clip diagnostic memory.

After the frozen test and both device profiles are complete, regenerate the published records and figures:

```sh
uv run python scripts/snapshot_metadata.py
uv run python scripts/build_report.py
```

The report builder verifies that test, export, and runtime reports identify the same artifact before writing `docs/MODEL_CARD.md`, `docs/CASE_STUDY.md`, and `reports/figures/`. This release report describes the measured Mac and requires its CPU and MPS profiles; CPU-only training/evaluation remains available independently.

## CPU container

```sh
docker build -t one-voice .
docker run --rm -p 127.0.0.1:8000:8000 -v "$PWD/artifacts/releases:/model:ro" one-voice
```

The container runs as a non-root user and requires the model mount. It uses CPU and does not include public audio or examples. A Docker build requires a running Docker engine; refer to the verification report for whether this environment exercised it.

## Artifacts and disk

- `data/raw`: selected public FLACs and acquisition manifests.
- `data/manifests`: audited source inventory and deterministic case recipes.
- `artifacts/runs/<name>`: best/latest checkpoint, config, provenance, metrics, validation, summary.
- `artifacts/releases`: exported selected model and identity metadata.
- `artifacts/examples`: generated attributed audio for local UI.
- `reports`: aggregate/per-case study results, selection, freeze, and runtime measurements.

Raw audio, checkpoints, and environments are Git-ignored. Do not delete a manifest needed to reproduce a checkpoint. The source acquisition and trainer check a minimum free-disk reserve; training uses a bounded decoded-audio cache and creates mixtures on demand.
