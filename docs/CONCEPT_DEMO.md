# A practical known-voice separation demo

The goal is an audible demonstration of reference-conditioned extraction within a short Mac session. The full 100-epoch research baseline is on hold. This experiment narrows the domain to eight familiar LibriSpeech voices in clean two-person mixtures. Different reference recordings tell the model which of the two voices to keep.

## Budget and efficiency

- At most **600 optimizer updates**, effective batch eight. The sampler changes source utterances, references, pairings and offsets on every update.
- **30 minutes per session by default**, measured from startup. The trainer finishes its current update, saves and stops. Evaluation also checks the deadline between requests. Saving can add a few seconds; this is not a hard real-time deadline.
- Checkpoints every 25 updates. Validation every 100 updates, at initialization and at completion. A shorter session may stop before 600 updates. There is no automatic next session.
- Three of the full model's six separator blocks; full 32-band resolution, 128 band features, 256 recurrent hidden features and the ResNet34 reference encoder. The model has **22,995,427 parameters**.
- Three-second training mixtures and three-second reference crops keep execution shapes fixed. Separator microbatch two, gradient accumulation four, no activation checkpoint recomputation, and MPS graph-cache reuse during training. Validation clears the MPS cache. The allocator retains a bounded approximately 8 GiB allowance.
- Adam, learning rate 0.0003 decaying to 0.00003 over the declared 600-update cosine schedule. Loss remains 0.9 × negative SI-SDR + 0.1 × voice-classification cross entropy.

The [discarded speed probe](../reports/concept-reuse-profile.json) measured five warm updates averaging **2.57 seconds**, versus the original full-model estimate of 8.29 seconds. That suggests about 26 training-only minutes for 600 updates; evaluation and checkpoint writes are extra. This short measurement is not sustained-run or quality evidence. A full-model microbatch-four probe ran out of MPS memory and was rejected. [Decision and identities](../reports/concept-compute-decision.json).

The concept trainer uses update and time limits, not the generic configuration schema's unused `epochs` field. Its development targets are at least 6 dB mean SI-SDR improvement, 3 dB at the 25th percentile, 90% of requests improving over the mixture and 90% preferring the requested source. Two successive evaluations passing these targets stop training early. They are engineering targets, not paper results or a substitute for listening. Completing 600 updates without passing them is reported honestly.

## Reused learning and reserved recordings

Initialization is our own full-size model's best 224-update learning-check checkpoint. Transfer retains the first three separator blocks and every other matching weight, including the speaker encoder and classifier. No externally trained weights or extraction implementation are imported. Removing later blocks can initially damage performance; adaptation and evaluation determine whether it recovers.

The sixteen fixed requests used by the initializer are reconstructed from their exact seed and manifest. All their source and reference utterances are excluded from every concept pool. The first eight familiar speaker IDs in sorted order are chosen before scoring. A deterministic hash ordering reserves ten development and ten test utterances per voice, leaving **686 training, 80 development and 80 test utterances**. The [public reservation](../metadata/concept-demo/manifest.json) records identities and hashes.

Training uses all eight voices in each effective batch. Each mixture generates two requests with the same waveform and opposite references/targets. Sources and references are different utterances. Relative levels vary from −3 to +3 dB. Development contains 16 fixed pairs / 32 requests, using four-second mixtures and three-second references. Test has 32 pairs / 64 requests and is not opened by the trainer.

This evaluates new **utterances**, not new speakers. There is no established claim for unfamiliar voices, real noisy conversations, live audio or removal of all static. Neither the old release's test scores nor the initializer's memorization score can stand in for the concept's own evaluation.

## Run, pause and resume

The existing project Mac has the source audio and initializer on its SSD. From the repository root:

```sh
CONCEPT_DATA="/Volumes/Zach's SSD/target-speaker-extraction/reference-baseline"
CONCEPT_INIT="$CONCEPT_DATA/runs/tiny-learning-memory/best.pt"
CONCEPT_RUN="$CONCEPT_DATA/runs/concept-eight-voices"

# Run once when creating a new reservation; preserve an existing one.
uv run python scripts/prepare_concept_demo.py --root "$CONCEPT_DATA" --checkpoint "$CONCEPT_INIT"

uv run python scripts/train_concept_demo.py --root "$CONCEPT_DATA" --checkpoint "$CONCEPT_INIT" --run "$CONCEPT_RUN" --minutes 30

# Only when choosing another bounded session:
uv run python scripts/train_concept_demo.py --root "$CONCEPT_DATA" --checkpoint "$CONCEPT_INIT" --run "$CONCEPT_RUN" --minutes 30 --resume
```

Ctrl+C / SIGINT or SIGTERM requests a stop after a complete update. An abrupt power loss or forced termination resumes from the last atomic checkpoint. Resume requires matching source code, device, configuration, initialization and data identities; optimizer and RNG state are restored. CPU tests compare resumed and uninterrupted model states exactly and verify that an expired session preserves a checkpoint. No sleep-prevention process or paid instance is required.

A fresh clone contains the reservation and code, not audio or weights. Recreate the data and initializer using the [reference guide](REFERENCE_BASELINE.md), then generate a new reservation tied to that new checkpoint and record its identities. The recorded checkpoint hash must not be silently substituted.

## Review the result

The [progress page](http://127.0.0.1:8000/experiments/concept/) shows actual update counts, the session limit, development results and a link to the current best candidate. The generated gallery shows the first eight development requests, selected before scoring. It provides mixture, independent voice reference, model estimate and known target. Listening levels are matched, so simply turning down the mixture cannot masquerade as extraction. Audio filenames include the checkpoint identity, and the page switches only after every new file is written.

Training saves `latest.pt`, `best.pt`, metrics, status and detailed development rows on the SSD. The reserved test remains unopened until a candidate is frozen for a separate final evaluation. The default app checkpoint is preserved; this concept gallery is a separately identified candidate. Listen for competing speech, missing or damaged target words and static separately before describing it as a successful audible demonstration.
