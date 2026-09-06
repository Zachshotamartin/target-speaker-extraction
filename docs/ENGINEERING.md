# Engineering and delivery

The repository currently contains planning documents, proposed configuration and an environment-report utility. The modules and interfaces below are implementation targets, not existing functionality.

## 1. Environment

Use native ARM Python 3.12 in a project `.venv`, managed with `uv`. Add the package manifest and lockfile during environment setup after selecting and testing compatible dependencies.

Proposed direct dependencies:

- PyTorch for tensor operations, models, differentiation and checkpoints.
- NumPy, SciPy and SoundFile for numerical checks, resampling and WAV/FLAC I/O.
- Pytest and Ruff for meaningful checks and code quality.
- FastAPI, Uvicorn and multipart support when the API milestone begins.
- Plotting and optional metric packages only when required by the evaluation implementation.

Do not install an entire research toolkit to obtain the extraction network. Use SpeakerBeam as literature only. Pin exact package versions after CPU/MPS and audio round-trip checks, then commit the lockfile. Record Python patch version, macOS version and architecture in run metadata.

The available `scripts/check_environment.py` reports environment facts without downloading anything. The current default `python3` launcher returned no version during setup, while native `/opt/homebrew/bin/python3.12` returned Python 3.12.5. Use a verified interpreter when creating the environment; do not silently rely on that launcher.

## 2. Planned package layout

```text
src/tse/
  audio.py                 # decode, mono policy, resampling, scaling, valid lengths
  config.py                # strict typed configuration and validation
  data/
    inventory.py           # source registry and audio inventory
    manifests.py           # schemas, hashes, split audits
    mixtures.py            # seeded mixing and timing
    augmentation.py        # independently seeded reference transforms
    dataset.py             # training crops and deterministic evaluation cases
  models/
    encoders.py            # mixture and reference encoders
    blocks.py              # temporal and conditioning blocks
    extractor.py           # end-to-end network
  training/
    losses.py              # masked SI-SDR and optional auxiliary losses
    engine.py              # optimizer, accumulation, validation and resume
    checkpoints.py         # save/load contracts and metadata
  evaluation/
    metrics.py             # per-case metrics and degeneracy policy
    suites.py              # condition manifests and reference diagnostics
    reporting.py           # aggregates, paired intervals and figures
    profiling.py           # synchronized CPU/MPS measurements
  inference.py             # shared file/chunk inference implementation
  cli.py                   # commands with explicit inputs and outputs
  api.py                   # bounded local serving interface
tests/
  unit/
  integration/
  fixtures/                # tiny generated fixtures; raw recordings remain ignored
web/                       # local audio comparison interface, added in its milestone
configs/
docs/
scripts/
reports/                   # reviewed aggregate reports; local raw outputs ignored
```

Add modules when they contain working behavior. Do not create placeholder functions that pretend training or inference already works.

## 3. Configuration

Use TOML for experiment configuration and JSON/JSONL for machine-readable metadata. `configs/pilot.toml` records initial choices; the first implementation adds strict validation for unknown keys, types and incompatible settings.

Separate architecture, data protocol, training controls, evaluation, resource budgets and inference settings. Resolve paths relative to a declared project/data root. Reject inconsistent sample rates, impossible reference durations, nonpositive batch sizes and configurations that leak test partitions into training.

Store the fully resolved configuration with each run. Command-line overrides are captured in it. Do not mutate the saved configuration halfway through a run.

## 4. Training and experiment tracking

Start with local structured files rather than a remote tracking service:

```text
artifacts/runs/<run_id>/
  config.toml
  environment.json
  provenance.json
  metrics.jsonl
  checkpoints/best.pt
  checkpoints/latest.pt
  evaluation/per_case.jsonl
  evaluation/summary.json
```

A run ID combines a timestamp and a short config hash. Save the Git commit and dirty-state flag, seed, manifest hashes, model count and device information. Checkpoints use atomic replacement and include state dictionaries, optimizer state, step, configuration and random generator state.

Resume from `latest`; select a release candidate using the declared development metric. Handle interrupted saves without overwriting the last valid checkpoint. Load only project-produced or explicitly trusted model artifacts. The inference API never accepts an uploaded checkpoint.

Add MLflow only if the number of runs creates a real comparison or registry need. It is not a prerequisite for sound experiment records.

## 5. Planned command contract

These are intended command names, not commands available today:

| Command | Required behavior |
| --- | --- |
| `tse data inventory` | Produce a source manifest, duration statistics and provenance |
| `tse data build-cases` | Generate reproducible case definitions from a named protocol |
| `tse data audit` | Validate splits, references, offsets and source identity |
| `tse benchmark` | Measure model forward/backward or inference under a saved config |
| `tse train` | Start or resume a run with explicit config and output directory |
| `tse evaluate` | Produce per-case and aggregate results for a fixed checkpoint/manifest |
| `tse extract` | Process mixture and reference files into an output WAV |
| `tse serve` | Start the local API with a selected release artifact |

Commands return nonzero exit codes for invalid configuration, data failure or missing artifacts. Training must never start merely because a config file is imported.

## 6. Meaningful verification

Add checks alongside the behavior they protect:

| Area | Checks that matter |
| --- | --- |
| Audio | Resampling duration, exact crop offsets, silent input, unsupported format, clipping and common gain |
| Data | Cross-split speaker/utterance overlap, correct enrollment identity, deterministic case reconstruction |
| Mixtures | Source level ratio, sum relationship, delay alignment and valid-length handling |
| Metrics | Identical signal, scaled signal, added interference, wrong speaker and all-zero target policy |
| Model | Output lengths around stride boundaries, gradients into both encoders, masking and reference sensitivity |
| Training | One real optimizer step, accumulation scaling, finite values and checkpoint resume |
| Inference | Full/chunk boundaries, exact duration, fixed reference caching and device selection |
| API | File limits, invalid input, missing model, concurrency, successful output and cleanup |

Use analytically known signals for numerical checks and a tiny licensed or generated fixture for integration. Keep real training and large downloads outside CI. Tiny-set overfitting is a manual ML gate with retained evidence, not a long test on every commit.

## 7. CI and Git workflow

The foundation commit contains no model test suite; it must not show a misleading training badge. Add a CPU GitHub Actions job once executable package code exists. The job installs locked development dependencies, runs linting and bounded tests without network dataset access.

Branch examples: `feat/data-manifests`, `feat/conditioned-model`, `feat/training-loop`, `feat/local-inference`. Each change explains resulting behavior, relevant checks and limitations. Avoid mixing an architecture change with a data-protocol change in the same experimental comparison.

Never commit raw audio, environments, checkpoints, tokens or machine-specific paths. The root ignore rules cover common cases. Before each publication, inspect the staged file list and diff. Reviewed aggregate reports and diagrams belong in Git; large release artifacts require explicit provenance and license review.

## 8. Local API

Start with a single-process, single-model service bound to loopback. Load the checkpoint once at startup and advertise readiness only after a small inference check.

Proposed routes:

| Route | Contract |
| --- | --- |
| `GET /health` | Process alive; no claim that a model is ready |
| `GET /ready` | Model loaded with usable device, or an explicit not-ready response |
| `GET /model` | Model ID, sample rate, supported input bounds and artifact hash |
| `POST /extract` | Multipart `mixture` and `reference`; return aligned WAV plus minimal processing headers |

Initial limits: WAV/FLAC only, mixture at most 60 seconds, reference 3–10 seconds, one in-flight extraction, and a declared upload-byte ceiling. Validate actual decoded duration and channel layout instead of trusting extensions. Standardize channel downmixing and resampling. Reject silent/too-short references using an explicit rule whose limitations are documented.

Use request-scoped temporary files and guaranteed cleanup on success or failure. Raw recordings are not logged or retained by default. Log request ID, status, durations, model version and timing. Queue capacity is bounded; a busy service responds clearly instead of exhausting memory.

The first API returns no uncalibrated confidence percentage. Target presence and extraction quality cannot be inferred reliably from mask magnitude alone.

## 9. Interface

Build a small local page with two file selectors, duration validation, an extraction button, progress state, synchronized original/output playback, and WAV download. Show the exact supported operating envelope and useful input guidance.

Display model version and runtime in an expandable details area. Put research metrics in the evaluation report, not in a form a user must understand to process a recording. An output that fails validation should produce a clear error, not an empty audio player.

## 10. Packaging and runtime

Run natively on macOS for MPS acceleration. Add a CPU Docker image later for portable serving and CI parity; do not assume a Docker container on this Mac has access to MPS.

Bundle model configuration, trained weights, preprocessing version and a checksum as one versioned artifact. Verify the artifact before serving. Keep previous candidates available for rollback if a later model regresses.

Any public hosted demonstration, cloud training or paid service is a separate deployment choice. The repository can be public while training data, checkpoints and user recordings remain local.

## 11. Monitoring and release evidence

The local service measures latency, errors, input duration and resource pressure. Audio quality usually has no live ground truth, so input statistics cannot establish accuracy drift. Log coarse validation statistics, then evaluate representative labeled cases before retraining or promotion.

Retraining is a deliberate experiment triggered by evidence of a failure class. Compare the candidate against the retained baseline on the frozen protocol. Promote only with a documented tradeoff decision.

The final model card includes intended use, data sources, independent-implementation boundary, trained configuration, update count, hardware, quality by condition, resource use and known failures. A portfolio write-up connects each engineering decision to measured evidence.
