# Listening during full-data training

The full-training dashboard includes four fixed development requests: both target
voices in the first two Libri2Mix development mixtures. Selection is independent
of model scores. Each request has a voice reference, original conversation, clean
target, best model estimate and latest model estimate.

“Best” uses the checkpoint selected by all 6,000 development requests. “Latest”
uses the most recent full validation, including an explicitly labeled in-progress
preview. These are separate from the 400-request monitor. If best and latest are
the same update, the page explains why their audio is identical. These few examples
are listening aids, not a representative estimate of overall quality.

## Running the publisher

From the repository root, with the dataset SSD mounted and the full run registered:

```sh
.venv/bin/python scripts/watch_full_listening.py
```

Use `--once` for one refresh. The normal mode checks every 30 seconds and a file
lock prevents duplicate publishers. A detached launch can redirect output to
`artifacts/full-listening/worker.log`. Its PID is recorded in
`artifacts/full-listening/active.json`. Restart this publisher after a reboot; the
page warns if its heartbeat becomes stale. Previously rendered audio remains
available even when the publisher is stopped.

The publisher runs on one CPU thread at reduced scheduling priority. It performs
eight model inferences on initial setup and four for each new validation update;
cached samples need no additional inference. It neither pauses training nor uses
MPS. Temporary CPU and disk activity can still modestly affect training speed.

The implementation lives in `scripts/` and web assets. It does not change any
Python file under `src/tse`, preserving the source fingerprint required by the
existing full-training checkpoint's strict resume check. No server restart is
needed because the existing static gallery mount serves newly written artifacts.

## Exact predictions and provenance

The publisher opens the trainer's atomically written checkpoint, hashes that open
file and loads from the same descriptor. It checks the optimizer update, dataset
identity and, where recorded, checkpoint hash before generating audio. It uses the
same full recording, deterministic reference crop, CPU model evaluation and raw
SI-SDR metric as full validation. Existing completed case scores are checked within
0.03 dB. There is no temporal chunking, filtering, denoising or target-informed
postprocessing. Clean targets are used only for comparison and metrics.

The worker captures each full validation while its checkpoint is still current.
Older non-best weights overwritten before the worker was installed cannot be
recovered; it reports their absence rather than labeling newer training weights as
those validation results. Samples and metadata publish atomically after all four
requests succeed. Checkpoint weights are never copied into the gallery or retained
by the publisher on disk.

Float WAV files preserve raw model values. Optional matched playback applies only
a constant gain per track toward RMS 0.08, capped at peak 0.98 and gain 100. This
is not LUFS matching; unusually peaky audio may remain quieter. Scores always use
unmodified predictions. Switching the option pauses playback and preserves its
position. Only one audio track plays at a time, and an automatic refresh waits for
active playback to finish or for the user to load new samples.

Generated audio, checkpoint/source hashes, case identities and metrics live under
`artifacts/gallery/full-validation/`. The four most recent render directories,
currently selected best/latest and any directories less than 24 hours old are
retained; older unused renders are removed. Training checkpoints and source data
are never part of that cleanup. Generated files are Git-ignored; the code, tests
and documentation are tracked.
