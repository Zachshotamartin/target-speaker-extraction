# Full-data training with pause and resume

The active experiment starts a **fresh random model** on all prepared Libri2Mix clean train-100 data: **13,900 mixtures, 251 speakers and 27,800 target requests per epoch**. Both sources are requested once per epoch, in a deterministic shuffled order. New three-second mixture crops and distinct-utterance voice references are drawn each epoch.

The efficient architecture keeps three BSRNN blocks, the ResNet34 enrollment encoder, and three-second enrollment crops. This is a declared compute compromise relative to the six-block, full-reference baseline; using its dataset does not make this an exact paper reproduction. No pretrained or eight-voice checkpoint initializes this run.

## Schedule and session controls

- 100 complete epochs; effective batch 8 (separator microbatch 2, accumulation 4).
- 3,475 optimizer updates per epoch; **347,500 updates total**.
- Adam, initial learning rate 0.001, exponential decay toward 0.000025 over the complete schedule. No learning-rate reset or compressed schedule at a session boundary.
- An explicitly started session lasts at most eight hours, including evaluation. It can pause earlier. Reaching the time limit does not schedule another session.
- Checkpoints every 50 updates, before evaluation, at each epoch end, and on a requested pause.
- The MPS allocator limit is 60% of recommended working-set memory; activation checkpointing is disabled and fixed-size enrollment crops avoid variable-shape training overhead.

Open [full-data progress](http://127.0.0.1:8000/experiments/full/). **Pause and save** requests a stop after the current complete optimizer update or evaluation request. Wait for **Paused and saved** before disconnecting the SSD. **Resume · up to 8 hours** begins a new bounded session. Neither button changes the dataset, configuration or schedule.

The registered run is local-only in `artifacts/full-training-active.json`. The API accepts only a pause/resume action for this fixed registration, rejects extra parameters and cross-origin requests, and prevents duplicate launches. An exclusive trainer lock also prevents concurrent command-line workers from writing the same run. On macOS, idle-sleep prevention is tied to the trainer PID and capped at eight hours; it does not override lid sleep or change permanent power settings.

Command-line equivalent (substitute the dataset root and a new run directory):

```bash
.venv/bin/python scripts/train_full_dataset.py \
  --root "$DATASET_ROOT" \
  --manifest data/reference/manifest.json \
  --config configs/full-data-efficient.json \
  --run "$TRAINING_RUN" --device mps --minutes 480
```

Add `--resume` for an existing checkpoint. A command-line pause is SIGINT/SIGTERM or a `pause.request` file in the run directory. Remove that request file before a command-line resume; the interface handles this automatically.

## Checkpoints and recoverable evaluation

`latest.pt` contains model and BatchNorm buffers, Adam state, CPU/MPS random state, completed update count, epoch cursor, configuration, manifest hash, implementation identity, evaluation completion markers, best scores and elapsed compute time. An atomic write replaces it only after a complete update. A hard interruption can lose at most the work since the last saved checkpoint; an incomplete update is replayed from that checkpoint.

Resuming requires the same configuration, data manifest, Python source fingerprint, PyTorch version and device. Changing training code deliberately requires an explicit migration or a new experiment; silently loading weights alone would not be equivalent to resuming. Retain this Git revision and environment for the run. Dataset audio hashes are verified when files are loaded.

Two independent development records are maintained:

| Evaluation | Frequency | Latest result | Best checkpoint / result |
|---|---|---|---|
| Fixed speaker-balanced 400 requests, 10 per unseen development speaker | Initialization and every 500 updates | `latest-monitor-validation.json` | `best.pt` / `best-monitor-validation.json` |
| All 6,000 requests from 3,000 official development mixtures | Every completed epoch | `latest-full-validation.json` | `best-full.pt` / `best-full-validation.json` |

Best selection uses mean SI-SDR improvement within the same evaluation suite. Full mixture lengths and deterministic three-second enrollment crops are used for both development suites. **Development evaluation runs explicitly on CPU** using a copy of the current model; training remains on MPS. The first live startup reached the MPS memory ceiling during variable-length recurrent evaluation at update zero. CPU evaluation retains every full recording and avoids changing normalization or recurrent context through temporal chunking. The failed startup remains archived; no trained weights were lost. Each result records its update and case count; the newest training state can be newer than the most recently measured score. The two suites' scores are not interchanged. Individual-case results and validation history are retained.

Evaluation writes complete-case progress every 25 requests and on a requested pause. A resumed evaluation continues at its saved cursor using the same model step and frozen request order. A partial evaluation cannot become a best result. Model-only snapshots of the final five epochs are retained for a later, separately evaluated averaging candidate. No test evaluation or automatic default-model promotion occurs during training.

All development speakers are excluded from training. The official test-clean corpus has historical project exposure, so a future score on it must be labeled as a benchmark result, not a pristine project holdout. Test-other remains outside this run. SpeakerBeam's enrollment tables remain local; public source metadata and licensing records remain unchanged.

## Preserved earlier experiment

The eight-voice continuation was safely paused at update **6,000** on September 7, 2026. Its best development improvement was **6.66185 dB at update 5,350** across its separate 32-request familiar-voice suite. Those checkpoints and galleries remain intact. Its score must not be compared directly with this run's 40 unseen development speakers. The default application still serves the earlier v0.2.0 release.

## Verification

Automated checks compare uninterrupted training with a pause mid-epoch and a pause mid-development evaluation. They require exact CPU equality of final weights, optimizer state, random state, learning-rate progression and best scores. Additional checks cover balanced development selection, split leakage rejection, incomplete evaluation recovery, earlier-best retention, session expiry, duplicate trainer exclusion and fixed-run API controls. A real MPS pause/resume check is recorded separately after launch; CPU bitwise equivalence does not promise bitwise equality of every MPS kernel across processes.
