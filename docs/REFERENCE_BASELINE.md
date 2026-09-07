# Independent full-size reference baseline

This experiment addresses the [research gaps](V3_RESEARCH_GAPS.md) directly. The model is independently implemented from published architectural specifications; no SpeakerBeam/WeSep model implementation or externally trained weights are imported. Public mixture metadata and enrollment mappings are data inputs. Completed compact experiments and the v0.2.0 serving artifact remain preserved.

**Current direction:** the user chose a [short known-voice concept demo](CONCEPT_DEMO.md) instead of the long main run described here. The concept adapts our own 224-update learning-check checkpoint, removes three separator blocks and reserves different recordings of eight familiar voices. It has a 600-update cap and a 30-minute default session limit. This is a separate experiment, not progress toward the 100-epoch baseline below.

## Architecture and objective

The new model has **28,134,115 parameters**, including a 6,634,336-parameter speaker encoder. It uses 32 spectral bands, 128 band features, recurrent hidden width 256 and six time/frequency blocks. A 256-dimensional ResNet34 enrollment vector conditions the separator through one multiplicative fusion. Sequence-spanning normalization and unbounded gated complex coefficients replace the compact model's per-frame normalization and restricted coefficients. STFT/ISTFT uses a 512-sample Hann window and 128-sample hop at 16 kHz. Inference uses the whole clip.

The enrollment network uses residual depths 3/4/6/3, channel widths 32/64/128/256, BatchNorm, temporal mean/sample-standard-deviation pooling with frequency positions retained, and one 256-dimensional projection. There is no embedding L2 normalization. The independent frontend computes 80 log filterbanks from 25 ms Hamming frames with a 10 ms hop, per-frame DC removal, pre-emphasis 0.97 and utterance mean normalization. References retain their full isolated-source duration. Training dither has a standard deviation of one PCM unit; inference disables dither for deterministic delivery.

The objective is 0.9 × negative SI-SDR + 0.1 × speaker-classification cross entropy. Classification logits have no extra temperature multiplier. There are no waveform/spectral reconstruction penalties or acoustic augmentations in this clean baseline. The optimizer is Adam with weight decay 0.0001, gradient clipping at norm 5 and exponential learning-rate decay.

Specifications: [pinned WeSep configuration](https://github.com/wenet-e2e/wesep/blob/99eca54b60300d39b9353d93cf285a14bba37854/examples/librimix/tse/v2/confs/bsrnn.yaml), [separator definition](https://github.com/wenet-e2e/wesep/blob/99eca54b60300d39b9353d93cf285a14bba37854/wesep/models/bsrnn.py), [ResNet definition](https://github.com/wenet-e2e/wespeaker/blob/dfa741957e5c11f477623b6e583d67d0af25ee88/wespeaker/models/resnet.py), [temporal pooling](https://github.com/wenet-e2e/wespeaker/blob/dfa741957e5c11f477623b6e583d67d0af25ee88/wespeaker/models/pooling_layers.py). Our implementation is [reference_model.py](../src/tse/reference_model.py).

## Resolved choices and remaining reproduction limits

The [paper](https://arxiv.org/html/2409.09589v1#S4) specifies 100 epochs and a final-five-checkpoint average. Both the [first public configuration](https://github.com/wenet-e2e/wesep/blob/34997628e54d37f96a870b68e1ce0a51be889482/examples/librimix/tse/v2/confs/bsrnn.yaml) and current pinned configuration instead specify 150 epochs, two averaged checkpoints and an initial learning rate of 0.001. The paper prints an initial rate of 10e-3. We use **100 epochs, a final-five average, and 0.001 → 0.000025**. These choices are explicit; this is a paper-aligned independent baseline, not a claim of an exact historical reproduction.

Other differences are deterministic evaluation dither, independent filterbank arithmetic, global epoch shuffling rather than shard-buffer shuffling, enforced distinct-utterance enrollment, and the local effective batch. Numerical agreement with a complete original training implementation has not been established. Do not infer parity from matching architectural dimensions alone.

The independent filterbank frontend was separately compared with `torchaudio.compliance.kaldi.fbank` on real training enrollments, both with and without dither. [Measured differences](../reports/reference-fbank-parity.json) are reported in log-feature units; the results are close but not bit-identical. Torchaudio was installed only in an ignored verification directory and is not required by the model at runtime.

## Official mixture recipe and evaluation identities

Preparation reads the [pinned LibriMix metadata](https://github.com/JorisCos/LibriMix/tree/b898ca375c03e4039cdb3dbe3dd485aad58ec7d1/metadata/Libri2Mix). It independently applies the supplied source gains, trims both sources to the shorter duration and writes sources and their pre-quantization sum as PCM16. It preserves 13,900 train mixtures and 3,000 each for development and benchmark test. Both target choices are trained, giving **27,800 target requests per epoch** and 2,780,000 across 100 epochs. At effective batch eight this is **347,500 optimizer updates**.

All 251 train-clean-100 speakers now participate. The old 20-speaker custom test drawn from that partition is therefore not held out for this model. The official dev/test enrollment identities come from [the authors' published map](https://github.com/BUTSpeechFIT/speakerbeam/tree/91af02cc617afa35fedfbdbf32533012cd0a8672/egs/libri2mix/data/wav8k/min). Test-clean has historical evaluation exposure in this project; its forthcoming score is a standard benchmark result, not a fresh unseen test. The unopened test-other cases remain available for a separately frozen transfer evaluation.

Generated duration is measured from the actual `min` waveforms: 43.270 training mixture hours, 4.513 development hours and 4.187 test hours. We report those measurements rather than substituting the paper's corpus-summary durations. [Preparation evidence](../reports/reference-data-preparation.json) records source commits, hashes, counts and verified dev/test archive MD5 values. The existing full train acquisition has per-file identities but no verified publisher archive checksum. Audio and checkpoints live on the SSD and are excluded from Git.

The [public metadata snapshot](../metadata/reference-baseline/index.json) contains mixture/source identities and provenance, with the LibriMix MIT notice. It omits the enrollment tables obtained from SpeakerBeam: that repository has a custom evaluation license, so this project records pinned URLs and checksums rather than redistributing their contents. The complete local manifest includes the evaluation mapping; its hash is recorded separately from the public snapshot.

## Memory, learning checks and training duration

The full batch of eight enrollment feature sequences is padded after individual mean normalization. Separator microbatches are processed individually. A reference forward without retained activations produces detached embeddings; separator/classifier backpropagation accumulates their gradients; a recomputed reference forward receives those gradients. The recomputation uses training batch statistics but does not update BatchNorm running buffers a second time. Activation checkpointing also recomputes recurrent and reference residual blocks. CPU float64 tests compare gradients and running buffers with ordinary joint-batch training. Float32 kernel partitioning introduces rounding differences.

MPS allocator caches are released between phases and updates. This keeps short-lived recurrent/CNN workspaces from competing with the next full enrollment batch. It does not reduce model width, truncate enrollment or change the batch objective. Sustained variable-length learning must pass; a successful single batch is insufficient evidence of memory feasibility.

The initial sustained diagnostic exceeded the conservative 8 GiB allocation allowance after 74 complete updates. Its checkpoint, optimizer and RNG were preserved in an [explicit runtime transition](../reports/reference-memory-transition.json), then continued with a bounded allowance of approximately 10 GiB (`mps_memory_fraction: 0.75`). This does not disable the MPS memory limit. Earlier failed diagnostic runs remain on the SSD. The longest training enrollment is 16.6 seconds; a separate discarded profile can exercise these full inputs using `--longest-enrollment`.

The initial three-update profile measured about 8.29 seconds per warm update and extrapolated roughly **800 training hours** for 100 epochs. This estimate excludes validation, checkpoint writes and thermal changes. It is a preliminary hardware estimate, not a promised completion date. [Initial profile](../reports/reference-training-profile.json).

Before the full run, a separate 256-update diagnostic repeats sixteen training requests covering both targets of eight mixtures. It uses a constant learning rate of 0.001. Its gate requires at least 8 dB mean improvement and the correct source preference in all sixteen requests. These weights do not initialize the main model and are not deployed. A learning failure blocks the full run instead of being hidden by a long training queue.

The diagnostic completed all **256 updates**. Its [learning gate passed](../reports/reference-learning-gate.json): the best checkpoint (update 224) achieved **15.42 dB SI-SDR improvement** and preferred the requested source in all sixteen cases. The final update scored 13.96 dB; the best-checkpoint result is identified explicitly. A [CPU check of update 192](../reports/reference-cpu-validation.json) matched the corresponding MPS per-case SI-SDRi values within 0.000004 dB. These are fixed training examples, not evidence of generalization or a new release.

The user cannot dedicate this Mac to the projected month-long run. [Full training is on hold](../reports/reference-compute-decision.json), with no automatic main run queued and no paid GPU provisioned. Subsequent [concept efficiency measurements](../reports/concept-compute-decision.json) rejected a full-model microbatch of four after an out-of-memory failure, then measured a smaller fixed-shape configuration. [Compute options](COMPUTE_OPTIONS.md) record the new bounded local direction. Repeated passes over sixteen diagnostic examples are distinct from the 27,800 requests in each full-data epoch.

## Run and resume

From the repository root:

The commands below reuse the complete training acquisition at `data/expanded/raw` on this Mac. For a fresh clone, first run `uv run tse data acquire train-clean-100 --root /path/on/your/ssd/training-speech --speakers 1000 --utterances 10000`, then pass that directory as `--training-root` to the preparation command. The caps deliberately exceed the collection's size; preparation requires all 251 speakers and 28,539 original utterances. The generated audio is independent of the machine-specific acquisition-record hash, so regenerated provenance may have a different manifest hash; do not substitute a gate from a different manifest.

```sh
uv sync --frozen
REFERENCE_DATA="/Volumes/Zach's SSD/target-speaker-extraction/reference-baseline"
uv run python scripts/prepare_reference_data.py --root "$REFERENCE_DATA"
uv run python scripts/profile_reference_training.py --root "$REFERENCE_DATA"
uv run python scripts/train_reference_baseline.py --config configs/reference-overfit.json --root "$REFERENCE_DATA" --run "$REFERENCE_DATA/runs/tiny-learning" --fixed-indices 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
uv run python scripts/validate_reference_learning.py --root "$REFERENCE_DATA" --run "$REFERENCE_DATA/runs/tiny-learning"
uv run python scripts/train_reference_baseline.py --root "$REFERENCE_DATA" --run "$REFERENCE_DATA/runs/main"
```

Use a new directory for a new diagnostic. Keep prior failed runs and their logs. Resume an interrupted run using the identical command plus `--resume`; it restores the optimizer, RNG and exact sample position. Existing data, configuration and implementation digests must match. SIGINT/SIGTERM stops after a complete update; unexpected failures retain the last atomic checkpoint, discarding partial update state. CPU tests compare interrupted/resumed training with uninterrupted training exactly.

For a repeat of the gate, use `validate_reference_learning.py --output /path/to/a/new/gate.json` and pass that file with the trainer's `--learning-gate` option. Recorded results are preserved rather than overwritten. Full training checks the gate's implementation, model and dataset identities before initializing random weights.

Checkpoints are saved every 100 updates and at epoch boundaries. Full development evaluation runs at each main-training epoch. The final five epoch states are averaged; floating BatchNorm buffers are averaged and integer observation counters come from the final epoch. A final training summary is written only after the configured budget finishes.

The [live progress page](http://127.0.0.1:8000/experiments/reference/) distinguishes the learning diagnostic from main training and shows the currently served model. It reads saved counters; elapsed work is not a quality claim.

## Benchmark and release boundary

```sh
uv run tse export --checkpoint "$REFERENCE_DATA/runs/main/average-last5.pt" --output artifacts/releases/reference-candidate/model.pt
uv run python scripts/evaluate_reference_baseline.py --checkpoint artifacts/releases/reference-candidate/model.pt --root "$REFERENCE_DATA" --split dev --output reports/reference-official-development.json
```

Benchmark test scoring additionally requires a JSON freeze with `status: frozen_before_benchmark_test`, exact `checkpoint_sha256`, `manifest_sha256` and `source_tree_sha256`; commit it before scoring. Use the evaluator's `--freeze` and `--split test` arguments. Report absolute SI-SDR, SI-SDR improvement and the fraction exceeding 1 dB improvement separately. Raw benchmark inference is distinct from delivered-WAV validation.

Do not promote merely because training completes. Export identity, numerical/runtime delivery checks, paired custom acoustic development evaluation and actual listening still determine app suitability. Preserve the prior model and all negative results. The full baseline is the experimental anchor for subsequent controlled changes, not a claim that clean benchmark performance guarantees realistic conversational separation.
