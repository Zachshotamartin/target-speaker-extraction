# Implemented system

This describes the delivered code. Original design documents retain the proposal; measured reports and the model card describe results.

## Model and loss

The default network has 1,223,296 parameters. Its mixture encoder has 128 filters, length 32 samples, stride 16 at 16 kHz. A 64-channel bottleneck feeds two repeats of eight temporal blocks with dilations 1–128. Each block expands to 128 channels, applies reference-derived affine conditioning and a depthwise convolution, then produces residual and 64-channel skip outputs. A nonnegative mask acts on the analysis representation before transposed-convolution synthesis.

The separate reference encoder has 128 filters, length 160, stride 80, three residual convolutions, masked mean/std pooling, and an L2-normalized 128-dimensional representation. No pretrained speaker model is used. Per-frame channel normalization keeps temporal padding out of global statistics; each reference block reapplies its valid-frame mask. The model is noncausal.

Both filterbanks train independently. Initially, synthesis receives a scaled copy of the random analysis filters, and the mask starts near one. This improves initial waveform reconstruction without imported weights; it is not a perfect-reconstruction guarantee.

The loss is negative zero-mean target-specific SI-SDR plus **0.1 × normalized waveform L1**. L1 anchors amplitude because SI-SDR is gain-invariant. This resolves the initial plan's gain-sensitive-loss option. The optional speaker classifier is implemented but unused in the release comparison. AdamW uses learning rate 0.0003, weight decay 0.0001, clipping at 5, microbatch 2, accumulation 4, float32, and no scheduler.

The separate 16-case learning diagnostic uses 600 updates, learning rate 0.001, and accumulation 1. It cannot establish generalization.

## Actual data protocol

The downloader takes the first requested speakers and utterances in each official archive. This archive-order selection is a convenience sample, not a representative population. It retains selected FLAC files and member hashes without saving full archives. Because the stream stops early, the full archive checksum is **not verified**; acquisition metadata explicitly records this.

The acquired inventory contains 60 training speakers (2,974 usable utterances, 10.784 hours), 40 development speakers (1,186, 2.5742 hours), and 40 test speakers (1,187, 2.7218 hours). Inventory rejects utterances shorter than two seconds. The experiment further requires clips long enough for its crops and at least three eligible utterances per speaker. Inventory hours describe the source pool, not exactly the hours sampled during training. Manifests are authoritative; historical planning fields such as `pilot_speakers_target` do not select downloaded files.

Mixtures use distinct speakers, fully overlapping clean crops, a uniform −5 to +5 dB level ratio, and a common anti-clipping gain. Targets remain at mixture scale. References come from distinct utterances, preferably another chapter. IDs and sample offsets are explicit. Each evaluation pair shares exactly one mixture and reverses the target/reference assignment.

Training uses 2-second mixtures and 5-second references on demand. Evaluation uses 4-second mixtures and 5-second references. The development checkpoint-selection set has 80 cases; its reporting set has 400. The reserved test has 1,000 extraction requests, corresponding to 500 underlying mixture pairs. Repeated speakers create dependence. Approximate target-speaker cluster confidence intervals do not fully capture shared-interferer dependence.

## Reference robustness experiment

The control receives clean references. Treatment rotates evenly through clean, noise, channel, reverb, and combined conditions while retaining the control's initialization and mixture schedule. Corruption RNG is independent of mixture RNG.

Noise uses Gaussian noise at 10 dB SNR; combined uses 15 dB. Channel corruption uses a first-order low-pass coefficient from 0.6–0.9. Reverb uses a generated 180 ms sparse impulse response with twelve decaying taps. Combined applies channel, reverb, then noise. All references finish at RMS 0.1. These synthetic stress tests are not measured microphones or real rooms. Evaluation holds out random corruption seeds, using the same families and severity distribution as training.

Treatment is selected on development data only when equally weighted mismatch improvement is positive, clean degradation is at most 0.5 dB, and clean confusion does not regress. This is an artifact-selection rule, not proof of statistical significance. One paired seed supports a pilot; it cannot establish robustness across training seeds.

## Delivered audio path

WAV/WAVEX and FLAC inputs with one or two channels and encoded rates from 8–192 kHz are decoded, downmixed, and polyphase-resampled to 16 kHz. Invalid formats, non-finite/silent/very inactive audio, and duration violations receive errors. References must be 3–10 seconds; mixtures can be up to 60 seconds.

The shared inference path removes DC, scales the mixture to RMS 0.14 and reference to RMS 0.1, predicts, then undoes mixture gain. A separate delivered-path evaluation verifies this preprocessing. Float WAV avoids silently clipping estimates outside ±1; browser playback may itself clip these samples. Output peak is available in metadata.

Inputs up to four seconds use one pass. Longer inputs use disjoint two-second output cores with one second of context on each side, caching the reference representation. Only each core is kept. Context exceeds the default separator's receptive-field radius, enabling comparison to whole-file output without crossfade. This assumption should be rechecked after architecture changes. The profiler measures numeric boundary error on 10/30/60-second repeated development audio, not long natural conversations.

The API binds to loopback, accepts one extraction at a time, limits combined input to 24 MiB, validates hosts, and rejects cross-origin browser uploads. Framework-managed multipart spool files are closed after processing. Application logs record duration, runtime, and model ID, not raw recordings. Browser audio object URLs are revoked when replaced/reset.

There are no accounts, persistent jobs, target-presence classifier, or confidence score. The container is a CPU packaging recipe, not a hosted deployment or Apple MPS container.

## Reproducibility and limits

Atomic checkpoints contain model, optimizer, CPU/MPS RNG, config, source/case hashes, commit, source digest, and device details. The next mixture is determined by its absolute sample index. A CPU test verifies exact uninterrupted-versus-resumed weights. Bitwise equality across MPS hardware/library versions is not claimed. Resume permits only a changed update budget; other config/data changes are rejected.

Only best/latest checkpoints remain. An interruption discards incomplete accumulated gradients. A resumed summary's elapsed time covers its latest process segment; prior logs preserve earlier segments. Account for this when reporting full-run time.

Source digests cover package Python files, so unrelated serving/reporting additions can change them. The first control was developed/resumed while delivery code was being completed; matched treatment retained the same model, data generator, loss, and optimizer behavior. Git history and the dependency lock preserve context. Future multi-seed studies should begin from one frozen checkout.

Absent-reference diagnostics measure emitted energy, not SI-SDR against silence. Constant-reference diagnostics are out-of-distribution interventions, not trained baselines. Reference-duration reports retain eligible IDs and per-case scores; changing eligible subsets cannot establish a pure duration effect. SI-SDR and confusion proxies do not certify intelligibility or perceived quality.
