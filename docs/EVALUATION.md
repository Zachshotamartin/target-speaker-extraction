> Original planning record. For the implemented system, see [IMPLEMENTATION.md](IMPLEMENTATION.md), [REPRODUCING.md](REPRODUCING.md), and the measured reports. Proposed targets below are not current capability claims.

# Evaluation and experiments

Status: protocol proposal. No experiment has been run. Final test thresholds and case manifests must be frozen before the final comparison.

## 1. Questions evaluation must answer

1. Does the model improve the target signal over returning the unprocessed mixture?
2. Does it extract the requested speaker rather than whichever source is easier or louder?
3. Does reference augmentation improve performance when the reference is degraded?
4. What does that improvement cost on clean speech, inference time and memory?
5. Which conditions remain unreliable?

## 2. Baselines and controls

| ID | System | Purpose |
| --- | --- | --- |
| B0 | Return the original mixture | Required no-learning baseline; SI-SDR improvement is zero by definition |
| B1 | Our network trained with clean references | Main learned control |
| B2 | Same network and training budget, with reference augmentation | Main proposed improvement |
| D0 | B1/B2 with a valid reference for the other mixture speaker | Test whether selection follows the reference |
| D1 | B1/B2 with an unrelated reference speaker | Measure absent-target behavior, outside the first supported envelope |
| D2 | Reference conditioning replaced with a constant vector | Diagnose reference use; report as an ablation, not a strong separation baseline |
| O0 | Ground-truth target returned directly | Pipeline sanity check only; not an achievable system |

A published SpeakerBeam score is context, not a directly comparable baseline under a different dataset, reference protocol or sample rate. Running an external implementation is optional future work and must be separately identified; it is not needed to implement or train our model.

## 3. Evaluation populations

- Training diagnostics use training-only examples and never become headline results.
- Development speakers drive checkpoint choice, hyperparameters and threshold selection.
- Final test speakers are disjoint from all training and development speakers.
- The custom protocol is `custom-librispeech-tse-v1`; official Libri2Mix results require a separate faithful adapter.

Proposed main sizes: 400 development extraction cases and 1,000 final test extraction cases. Construct balanced two-target pairs: one mixture can be evaluated with a separate valid reference for each speaker. Balance target assignment, speaker coverage and target-to-interferer ratios. Determine feasible counts from the source inventory and freeze the exact manifests.

Start with 4-second target-present mixture cases for controlled comparisons. Evaluate 10, 30 and 60 second inputs separately for product behavior and throughput. Do not merge long-file product timing with short-case quality numbers.

## 4. Prespecified condition suites

Use the same base mixtures for each reference condition so differences are paired.

| Suite | What changes | Main interpretation |
| --- | --- | --- |
| Clean | Clean 5-second reference | Baseline extraction quality |
| Duration | 1, 3, 5 or 10 seconds of real reference speech | Sensitivity to available enrollment speech |
| Noise | Reference SNR fixed at declared levels, initially 5, 10 and 20 dB | Noise robustness |
| Channel | Held-out deterministic frequency responses | Simulated microphone mismatch |
| Reverb | Held-out documented impulse-response generation | Simulated room mismatch |
| Combined | A small frozen set of moderate transformations | Interaction of realistic imperfections |
| Partial overlap | Source overlap bins, with target pauses | Behavior outside continuous overlap |
| Wrong/absent reference | Unrelated third speaker | Failure behavior and future presence work |

Freeze which condition families constitute the primary mismatch average. Weight those families equally rather than letting whichever family has more files dominate. Report each family separately. Keep extreme corruptions as a challenge set rather than silently changing the primary suite.

For reference-duration comparisons, preserve a common eligible case set where possible. If some speakers lack long clips, report the exclusion count and use paired eligible subsets. Do not claim improved performance by dropping hard cases.

## 5. Metrics

### Separation quality

Primary metric: `SI-SDRi = SI-SDR(estimate, target) - SI-SDR(mixture, target)`, using the exact target at mixture scale and aligned valid samples. See [the SI-SDR paper](https://arxiv.org/abs/1811.02508) and the proposed implementation in [MODEL_DESIGN.md](MODEL_DESIGN.md).

Report mean, median, 10th percentile, proportion of cases with negative improvement, and the paired change from B1 to B2. Keep raw per-case scores so aggregates can be audited.

There is one specified target. Do not use permutation-invariant scoring to switch the output to whichever ground-truth speaker scores better; that would hide target-selection errors.

Secondary measures include gain-sensitive waveform error or SDR, output/input level ratio, clipping incidence, and an intelligibility measure after validating its implementation and license. SI-SDR alone does not certify naturalness, correct volume or word preservation.

### Speaker selection

For each nondegenerate output, compare SI-SDR against the requested target and the interferer. A proposed confusion flag is `SI-SDR(output, interferer) > SI-SDR(output, target) + 3 dB`. Finalize the margin on development data and report sensitivity to it. This is a signal-based proxy, not biometric identity verification.

Report silent or near-zero outputs separately. Such outputs must not be silently dropped from quality aggregates or counted as successful selection. Keep declared energy thresholds and affected-case counts.

For valid reference swaps, measure quality against each newly requested target. A system that produces the same source for both requests fails this check even if one output sounds clean.

### Absent targets

SI-SDR against an all-zero target is not a valid absence metric. Report output energy relative to mixture energy, how often another voice is emitted, and listening examples. A later presence classifier must report false alarms, missed target speech and calibration before controlling output suppression.

### Efficiency

- Real-time factor: elapsed processing seconds divided by input audio seconds.
- Cold and warm timing, separately.
- Decode/resample, reference encoding, model processing and output encoding time.
- Peak process RSS and MPS memory counters, separately labeled; do not sum overlapping unified-memory counters.
- Parameter count, checkpoint size and input duration.
- CPU/MPS device, precision, batch size, chunk/context configuration and power conditions.

Warm up and synchronize MPS around timed regions. Use repeated measurements, report median and p95 over a stated sample count, and distinguish model-only from end-to-end timing. Warm real-time factor below 1 does not establish low-latency live streaming.

## 6. Experimental sequence

| Experiment | Intervention | Gate and output |
| --- | --- | --- |
| E00 | Signal, shape and loss correctness | Numerical checks pass; silence policy documented |
| E01 | Overfit 16 fixed training cases | Clear quality improvement and functioning target switches |
| E02 | 1,000-update pilot | Positive held-out learning; measured runtime and memory |
| E03 | Clean-reference control B1 | Saved checkpoint, stable development results, reference diagnostics |
| E04 | Reference augmentation B2 | Paired clean/mismatch report against B1 |
| E05 | Noise, channel and reverb ablations | Identify which transformations help and which hurt |
| E06 | Reference duration and auxiliary identity loss | Quantify additional tradeoffs only if core model works |
| E07 | Smaller model and bounded-memory inference | Quality/time/memory comparison |
| E08 | Final frozen comparison and product checks | Reproducible report, model card and release decision |

Keep architecture, optimizer updates, training speakers, base mixture sequence and evaluation cases identical between B1 and B2. Use separate random streams for mixture sampling and augmentation so adding transformations does not unintentionally change the training examples.

Start with one seed for feasibility. For the final primary comparison, target three paired training seeds if the measured budget permits. If only one is possible, report that limitation and do not claim the result is stable across training randomness.

## 7. Uncertainty and reporting

Compute paired bootstrap confidence intervals over target-speaker clusters, using the same resampled cases for B1 and B2. Record the bootstrap seed and use an initial 1,000 resamples. This estimates variation across the sampled target speakers; shared interferers create additional dependence, so label these intervals as approximate. Report variation across training seeds separately.

The primary hypothesis is that B2 improves mean SI-SDRi on the prespecified mismatch suite without an unacceptable clean-quality or target-confusion regression. An initial practical target is +1 dB mismatch improvement with at most 0.5 dB clean degradation. Assess both magnitude and uncertainty. Failure to meet that target is reported directly.

Do not tune on final test scores. If a test failure motivates changes, treat that test as inspected, record the iteration and reserve a fresh evaluation set or declare the resulting comparison exploratory.

## 8. Listening evaluation

Maintain a fixed gallery containing typical cases, difficult cases, low scores, loudness imbalances, short references and target absence. Include both improvements and regressions.

For a small human comparison, hide model labels and randomize A/B order. Ask about intended-speaker preservation, competing-voice leakage and artifacts. Record listener count and protocol. A small informal listening study supports failure analysis; it does not justify broad population claims.

Keep a calibrated evaluation playback convention. If demonstration audio is loudness-matched for comfortable listening, label that and retain the unmodified metric audio separately.

## 9. Results contract

Every per-case record contains `run_id`, `case_id`, `condition`, target/interferer identifiers, checkpoint SHA-256, metric implementation version, valid lengths, quality scores, confusion/degeneracy flags and runtime when measured.

Each summary identifies code commit, environment lock hash, config hash, training-data hash, evaluation-manifest hash, seeds, exclusions, confidence interval method and aggregation weights. JSON is the source of truth; tables and plots are generated from it.

## 10. Release checks

- All claimed results can be regenerated from retained metadata and model artifacts.
- Speaker leakage and invalid reference relationships are absent.
- The reference-switch test passes within the declared operating envelope.
- Quality claims are supported by complete test aggregates, not selected audio clips.
- Failed and unsupported conditions are documented in the model card and interface.
- Exported or chunked inference is evaluated as the exact version served to users.
- The data protocol, training resources and differences from external benchmarks are explicit.
