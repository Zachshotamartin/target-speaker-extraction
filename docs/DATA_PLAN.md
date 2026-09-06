# Data plan

All quantities below are proposed starting points. Dataset download and preparation have not started.

## 1. Sources and protocol identity

Use [LibriSpeech](https://www.openslr.org/12) as the initial source of clean, 16 kHz English speech. Its official page lists the corpus under CC BY 4.0. Record original URLs, archive checksums, extracted member identifiers and attribution in the dataset registry.

[LibriMix](https://github.com/JorisCos/LibriMix) supplies an established mixture benchmark and useful published conventions. Our first pipeline will make its own deterministic mixtures from LibriSpeech and use our own reference-selection code. Name this protocol `custom-librispeech-tse-v1`; do not call its scores official Libri2Mix results.

A later benchmark adapter may use unmodified official Libri2Mix test definitions and a separately documented enrollment protocol. Preserve the official test examples, sampling rate and mixture condition, and disclose training-data differences. Any reused metadata retains its attribution. No SpeakerBeam implementation or checkpoints are required.

## 2. Data acquisition stages

| Stage | Source selection | Purpose |
| --- | --- | --- |
| Fixture | Procedurally generated signals and a tiny training-only speech selection | Audio contracts, metrics, shape and pipeline debugging |
| Pilot | Approximately 30 speakers and at least 20 usable utterances per speaker from `train-clean-100` | Learn the task with limited decoding and storage |
| Development | `dev-clean`, with deterministic case definitions | Hyperparameters, checkpoints and validation thresholds |
| Main training | Expand toward `train-clean-100` if the pilot passes | Improve speaker and phonetic coverage |
| Final test | `test-clean`, frozen before final comparisons | Report final quality and robustness |
| Optional challenge | Additional licensed noise, room responses or consented microphone recordings | Test beyond simulated clean audiobook mixtures |

The official archive listing gives approximately 6.3 GB for `train-clean-100`, 337 MB for `dev-clean` and 346 MB for `test-clean`. These are archive sizes, not a prediction of peak disk use. Full LibriMix generation produces many configurations and is outside the starting storage budget.

For the pilot, use a streaming archive reader or a small explicitly selected extraction. Never silently download and unpack the full benchmark family. Inventory source durations, usable utterances and speaker coverage before assigning a pilot size.

## 3. Storage and integrity

Planned local layout, ignored by Git:

```text
data/
  raw/librispeech/
  manifests/
  derived/cache/
  evaluation/
artifacts/
  checkpoints/
  runs/
reports/local/
```

Keep original FLAC where possible. Decode only needed utterances or crops. Generate training mixtures in memory. Cache only a bounded set of frequently used audio, with a documented byte limit.

Before acquisition, estimate `archive + extracted files + environment/cache + run artifacts + safety margin`. Preserve at least 15 GiB free throughout. The initial persistent-artifact budget is 20 GiB, including dataset and checkpoints. Pause automatic acquisition when the measured budget would be exceeded; recalculate or reduce the selection.

Download to a partial file, verify the publisher checksum when supplied, then record SHA-256 for local provenance. A streaming partial extraction cannot claim that the entire archive was checksum-verified; record member hashes and later verify the full archive if retained. Reject corrupt audio and unsafe archive member paths. Never execute code from an archive to prepare data.

## 4. Split rules

1. Respect the official train/development/test partition as the first boundary.
2. Explicitly assert that speaker ID sets are disjoint; do not assume directory names guarantee this.
3. Every utterance used as a mixture source or reference belongs to its assigned partition.
4. The reference speaker matches the target speaker; its utterance ID differs from every utterance used in that mixture.
5. Prefer another chapter for the reference when possible. Record same-chapter and different-chapter cases for analysis.
6. A reused reference or utterance must never bridge training and evaluation.
7. Freeze development and test mixture definitions. Training recipes may change by epoch using saved seeds.
8. If external recordings are added, partition by person and recording session rather than random audio chunks.

Do not use evaluation speakers for speaker-encoder pretraining, normalization fitting or auxiliary classification. Label vocabularies are learned from training speakers only.

## 5. Manifest contracts

The utterance manifest has one record per source utterance:

| Field | Meaning |
| --- | --- |
| `utterance_id` | Stable source identifier |
| `speaker_id`, `chapter_id` | Public corpus identifiers |
| `split` | `train`, `dev`, or `test` |
| `relative_path` | Path relative to configured dataset root; no personal absolute paths |
| `sample_rate`, `num_samples`, `channels` | Validated audio properties |
| `sha256` | File identity |
| `source_name`, `source_version`, `license_id` | Provenance |
| `usable_speech_seconds`, `quality_flags` | Versioned checks and rejection reasons |

The case manifest specifies one target extraction task:

| Field | Meaning |
| --- | --- |
| `case_id`, `protocol_version`, `split` | Stable evaluation identity |
| `target_id`, `interferer_id`, `reference_id` | Utterance identifiers |
| `target_speaker_id`, `interferer_speaker_id`, `reference_speaker_id` | Auditable identity relationships |
| `source_offsets`, `reference_offset`, `num_samples` | Crop definitions in the declared sample domain |
| `target_to_interferer_db`, `common_gain` | Mixture scale and level |
| `overlap_fraction`, `source_delays` | Timing pattern |
| `reference_condition`, `reference_transform_parameters` | Exact corruption identity |
| `mixture_condition`, `mixture_transform_parameters` | Any mixture-side augmentation |
| `target_present`, `seed` | Task variant and deterministic generation seed |
| `source_manifest_sha256`, `case_manifest_sha256` | Links to immutable data definitions |

Store generation code version and random generator version alongside manifests. Identical manifest identifiers with different contents are errors.

## 6. Mixture construction

For target `s`, interferer `i`, and optional noise `n`, form `x = g * (s + a*i + b*n)`. The supervised target is `y = g*s`. The reference `r` is another utterance by the target speaker.

The pilot uses no external noise: `b = 0`. Choose target-to-interferer ratio from −5 to +5 dB over active source regions. Compute `a` from the measured source RMS values. Do not make the target consistently louder than the interferer. Randomize which source is selected as the target and include paired cases where the same mixture is requested with each valid reference.

Choose `g` once for the complete mixture and apply it to every supervised component. If peak protection is necessary, rescale the mixture and target together; never clip the mixture and leave the target unmodified. Reference level normalization is separate and uses only reference audio.

Use 2-second training crops for the hardware pilot, expanding to 4 seconds after profiling. The starting reference is 5 seconds. Reject or resample inadequate crops; do not repeat a short speech fragment to fake a longer reference. Padding requires explicit valid-length masks.

Begin with fully overlapping two-speaker crops to isolate the separation task. Add partial overlaps in a distinct experiment using 25%, 50%, 75% and 100% overlap bins. Verify actual active speech overlap rather than relying only on file duration.

For future reverberant mixtures, define the target as the target's signal at the mixture microphone, including the simulated acoustic response. Removing room reverberation is a different objective. Preserve delay alignment and crop all mixture components consistently.

## 7. Reference robustness

Train a clean-reference control first. Then independently corrupt references while keeping all other experimental settings fixed.

| Family | Proposed training variation | Evaluation approach |
| --- | --- | --- |
| Duration | Fixed 5 seconds for control | 1, 3, 5 and 10 seconds where genuine speech is available |
| Level | Random gain within a validated range | Fixed levels; distinguish level effects from silence |
| Background noise | Begin with generated noise at 5–20 dB reference SNR | Fixed noise realizations and unseen seeds; later add separate real noise sources |
| Channel response | Smooth random filters and bandwidth limits | Held-out filter parameters, including a declared narrow-band case |
| Reverberation | Simple documented simulated responses | Held-out response seeds/parameters; label as synthetic |
| Combined mismatch | A controlled subset of moderate transformations | Prespecified combinations, not an uncontrolled Cartesian product |

These transforms are approximations. Performance under simulated filters does not establish performance on all real microphones. A later small consented recording set can measure actual cross-device behavior.

Apply reference transformations independently from mixture transformations. Do not assign a stable filter to a particular speaker, split or target position. That would create an artificial identity shortcut. Record all random choices.

Heavy pitch shifting and speed transformations are deferred because they can alter perceived identity and complicate the target definition. Codec simulation is deferred until a reproducible, licensed implementation is chosen.

## 8. Negative and edge cases

Maintain separately labeled cases for silence, insufficient reference speech, clipped input, one speaker, target pauses, unrelated references and completely absent targets.

An absent reference speaker is distinct from the other speaker in the mixture. Using the interferer's reference is a valid target switch, not an absent-target example. The first model reports absent-target behavior as a limitation; suppressing audio requires a later trained and calibrated mechanism.

## 9. Quality checks and acceptance

- Every file decodes, has finite values and an expected sample rate/channel count.
- Every case has valid offsets, correct reference identity and the requested output length.
- Speaker and utterance leakage assertions pass.
- Measured mixture ratio matches the requested ratio within a declared tolerance.
- A common scale preserves the mixture/target relationship.
- Repeating a fixed case reconstructs the same arrays within the documented resampling tolerance.
- Padding and silence cannot dominate the loss or improve a score accidentally.
- A listening audit covers at least 20 cases across conditions before training.
- Dataset attribution and manifest hashes accompany every experimental report.

## 10. Data release policy

Version source URLs, scripts, seeds, schema and suitable small manifests. Keep raw audio and personal recordings out of Git. A public example gallery should use audio that can be redistributed, with source attribution and its transformation history. Review data terms before distributing trained artifacts or derived samples; a repository code license does not override dataset terms.
