# Reproducing the quality experiments

This extends the original [v0.1.0 study](REPRODUCING.md). Run commands from the repository root with Python 3.12 and the locked environment. No external pretrained model is required. The lineage begins with this project's original trained reference encoder, so a fresh clone must first reproduce that study or obtain its exact locally retained artifact.

## Preserve the original and prepare expanded data

Keep the original exported model as `artifacts/releases/v0.1.0/model.pt` and its identity JSON alongside it. Do not overwrite the baseline while running pilots. The recorded baseline hash is `f3271f1decf7ec0e8e9b0f1fab578a693f51c4a1d1778adeeb9109a5e64a5c20`; reproducing training on different software/hardware need not reproduce these exact bytes.

```sh
uv sync --frozen
uv run tse data acquire train-clean-100 --root data/expanded/raw --speakers 1000 --utterances 10000
uv run python scripts/prepare_expanded_data.py
```

The acquisition caps exceed this collection's counts; the recorded download contains all 251 train-clean-100 speakers. Preparation links the existing dev-clean recordings into the expanded root, audits source hashes and reserves 20 identities not used in the original training. Hard links require the same filesystem during preparation. Do not regenerate or alter a frozen source inventory/test recipe during a run. The source pool has 231 eligible training speakers, 24,377 utterances and 90.582 clean-source hours. The original test-clean split is not in this expanded inventory.

## Pilot and expanded training

```sh
uv run tse train --config configs/quality-stft-pilot.json --run artifacts/runs/quality-stft-pilot --initialize-reference-from artifacts/releases/v0.1.0/model.pt --device mps
uv run tse train --config configs/quality-stft-expanded-fastlr.json --root data/expanded/raw --manifest data/expanded/manifests/inventory.json --dev-cases data/expanded/manifests/dev-cases.json --run artifacts/runs/quality-stft-expanded-fastlr --initialize-from artifacts/runs/quality-stft-pilot/best.pt --prefetch --device mps
```

The spectral pilot retains only the original model's reference-encoder weights. The expanded run retains that pilot's extractor and creates a fresh 231-label speaker classifier. Neither initialization imports external weights. The expanded config has 20,000 updates; the original execution first ran 2,000 and then resumed with that increased budget. The preserved metrics describe its actual process segments.

To resume, repeat the command with `--resume` and omit `--initialize-from`. Only the maximum update budget may change. `--prefetch` prepares one CPU batch ahead with the same per-example seeds. Optional `--compile-blocks` uses the tested MPS temporal-layer compiler configuration, retaining eager STFT/ISTFT. Compilation can change floating-point arithmetic slightly; its provenance is recorded. The historical run adopted compilation after update 5,247, including a documented failed/repeated segment. A fully eager rerun is a method reproduction, not a byte-identical historical replay.

Start the fixed-step collector in another terminal **before update 8,000**:

```sh
uv run python scripts/checkpoint_average.py capture --run artifacts/runs/quality-stft-expanded-fastlr --steps 8000 10000 12000 14000 16000 18000 20000 --output artifacts/quality-checkpoints/trajectory
```

Only one collector may write this directory. It verifies steps/hashes and refuses to replace captured checkpoints or substitute a missed step. The predeclared averages are final three (16k/18k/20k) and final five (12k/14k/16k/18k/20k):

```sh
uv run python scripts/checkpoint_average.py average --checkpoints artifacts/quality-checkpoints/trajectory/step-16000.pt artifacts/quality-checkpoints/trajectory/step-18000.pt artifacts/quality-checkpoints/trajectory/step-20000.pt --output artifacts/quality-checkpoints/average-last3.pt
uv run python scripts/checkpoint_average.py average --checkpoints artifacts/quality-checkpoints/trajectory/step-12000.pt artifacts/quality-checkpoints/trajectory/step-14000.pt artifacts/quality-checkpoints/trajectory/step-16000.pt artifacts/quality-checkpoints/trajectory/step-18000.pt artifacts/quality-checkpoints/trajectory/step-20000.pt --output artifacts/quality-checkpoints/average-last5.pt
```

Averages are inference candidates without resumable optimizer state. They require development scoring; averaging is not automatically an improvement.

## Controlled normalization pilot

After the main run, test the predeclared change independently:

```sh
uv run tse train --config configs/quality-stft-globalnorm-pilot.json --root data/expanded/raw --manifest data/expanded/manifests/inventory.json --dev-cases data/expanded/manifests/dev-cases.json --run artifacts/runs/quality-stft-globalnorm-pilot --initialize-from artifacts/runs/quality-stft-pilot/best.pt --initialize-normalization-transfer --prefetch --device mps
```

Compare this 2,000-update pilot to the existing expanded high-rate 2,000-update checkpoint. Seed, batches, fresh classifier, optimizer, loss and initialization tensors are matched; separator normalization changes. The original weights were trained with per-frame normalization, so this is an adaptation experiment. Reference normalization remains unchanged. The global variant uses whole-clip inference because its statistics depend on the whole sequence. Do not claim finite-context chunk equivalence or promote it without long-clip checks.

## Development evaluation and diagnostics

The default evaluator uses the original 400 dev-clean recipes/source manifest, compatible with all disjoint-training candidates:

```sh
uv run python scripts/evaluate_quality.py --checkpoint artifacts/runs/quality-stft-expanded-fastlr/best.pt --output reports/quality-expanded-final-development.json --device cpu
uv run python scripts/compare_listening.py --candidate artifacts/runs/quality-stft-expanded-fastlr/best.pt
uv run python scripts/diagnose_mask_capacity.py
```

Use immutable candidate copies when training is still writing best/latest files. The quality evaluator scores decoded delivered WAVs, with SI-SDRi, speaker confusion, ESTOI and a source-projection distortion proxy. Its output identifies source/case/checkpoint hashes. The mask-capacity diagnostic explicitly uses clean targets and rejects test recipes; its numbers are not model performance. Listening comparisons preserve raw tracks and provide separately labeled RMS-matched playback.

## Freeze, fresh test and release

Select a candidate from development evidence and export it into a versioned directory:

```sh
uv run tse export --checkpoint artifacts/quality-checkpoints/selected.pt --output artifacts/releases/v0.2.0/model.pt
```

`selected.pt` must be an explicit copy of the development-selected artifact. Before fresh-test scoring, record `reports/quality-v2-selection.json` with status `frozen_before_fresh_test`, exact `case_manifest_sha256` and `source_manifest_sha256` for the expanded fresh-test inputs, and a `checkpoints` mapping containing `baseline` and `candidate`, each with its exported `checkpoint_sha256`. Include the development decision and report identities. The release's committed record documents the actual selection. For a new run, use a new record; do not overwrite a previously opened test freeze.

```sh
uv run python scripts/evaluate_quality.py --checkpoint artifacts/releases/v0.1.0/model.pt --root data/expanded/raw --manifest data/expanded/manifests/inventory.json --cases data/expanded/manifests/fresh-test-cases.json --freeze reports/quality-v2-selection.json --output reports/quality-v2-baseline-test.json --device mps
uv run python scripts/evaluate_quality.py --checkpoint artifacts/releases/v0.2.0/model.pt --root data/expanded/raw --manifest data/expanded/manifests/inventory.json --cases data/expanded/manifests/fresh-test-cases.json --freeze reports/quality-v2-selection.json --output reports/quality-v2-selected-test.json --device mps
```

The evaluator rejects a changed checkpoint, inventory or test recipe. Test results describe the frozen selection and must not become tuning feedback for this release. Evaluate runtime one device at a time with training stopped:

```sh
uv run tse profile-delivery --checkpoint artifacts/releases/v0.2.0/model.pt --mixture artifacts/examples/voice-1-mixture.wav --reference artifacts/examples/voice-1-reference.wav --device cpu --output reports/quality-v2-delivery-cpu.json
uv run tse profile-delivery --checkpoint artifacts/releases/v0.2.0/model.pt --mixture artifacts/examples/voice-1-mixture.wav --reference artifacts/examples/voice-1-reference.wav --device mps --output reports/quality-v2-delivery-mps.json
uv run python scripts/build_quality_report.py
```

The report builder verifies matching frozen test/export/runtime identities before replacing the current model card. Original cards are preserved as `MODEL_CARD_V0_1.md` and `CASE_STUDY_V0_1.md`. `build_report.py` rebuilds those historical records using `metadata/model.json`.

After promotion, copy the selected export and JSON to `artifacts/releases/model.pt`/`model.json`, restart the service, regenerate the main gallery and release comparison, then verify actual browser extraction/download and `/model` identity. Snapshot new metadata with `snapshot_quality_metadata.py --release artifacts/releases/v0.2.0/model.json`; do not replace the original metadata directory with new experiment records.

The external SSD is available if storage grows; current preparation remains on the internal disk and enforces a free-space reserve. All audio/checkpoints stay outside Git. Public metadata records identities and recipes without redistributing recordings or model weights.
