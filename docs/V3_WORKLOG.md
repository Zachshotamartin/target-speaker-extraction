# All four improvement tracks

The user authorized the complete follow-up: learning-rate continuation, a stronger time/frequency separator, phase-aware reconstruction, and realistic acoustic training with listening evaluation. Version 0.2.0 stays preserved as the baseline. No new test result determines model selection.

## Experimental contracts

- [Predeclared plan](../reports/v3-plan.json): two 5,000-update continuation arms; three 4,000-update architecture pilots; continue the selected architecture to 10,000 updates; paired 3,000-update clean/realistic adaptation.
- [Evaluation policy](../reports/v3-evaluation-policy.json): automatic promotion requires at least +0.3 dB mean improvement on acoustic development conditions, with clean SI-SDRi regression no greater than 0.5 dB, ESTOI regression no greater than 0.01, and confusion increase no greater than one percentage point. If no candidate qualifies, the baseline remains the default.
- [Continuation initialization](../reports/v3-continuation-initialization.json): exact model, classifier, optimizer moments and CPU/MPS RNG match at step 20,000. The control retains the plateau scheduler state; the alternative begins a 5,000-update cosine decay from 0.001 to 0.00003. Each follows the same new source samples.

## Independent architectural experiments

The new separator groups all 257 complex STFT bins into 18 bands. Each block models time within bands and frequency across bands with bidirectional LSTMs, conditioned on the reference embedding. It uses whole-recording inference because recurrent context spans the sequence. This is a compact adaptation of band-split modeling, not a reproduction of the published BSRNN recipe.

The real- and complex-mask variants have identical tensor shapes, initial weights, reference encoders, classifiers and initial predictions. The imaginary mask starts at zero; only the complex arm can learn phase corrections. The real mask is sigmoid-bounded and the imaginary mask is tanh-bounded. Unit tests verify exact initial output matching, reconstruction alignment and the different imaginary gradients.

A third arm changes the reference encoder to a scaled ResNet34 with block counts [3,4,6,3], 16 initial channels, GroupNorm and 64 log-mel bands. Its separator and classifier initialization match the complex arm. Its reference encoder starts randomly, while the other two use our own v0.2.0 reference weights. That difference is explicit: it is not a comparison of equally pretrained encoders. [Initialization record](../reports/v3-architecture-initialization.json).

The smaller widths, GroupNorm, masks, conditioning and training schedule are project choices. Primary references: [enrollment augmentation / BSRNN study](https://arxiv.org/html/2409.09589v1), [TF-GridNet complex spectral prediction](https://arxiv.org/abs/2211.12433). No research source code or external pretrained weights enter these models.

## New speech and acoustic data

The existing 231 training identities remain unchanged. Development includes 40 dev-clean and 33 dev-other identities. The final test uses 33 previously unscored test-other identities; the original and v0.2.0 test identities remain historical and are excluded from the new test. Both downloaded speech archives match the publisher's MD5. Source manifests record SHA-256 identities. [Preparation report](../reports/v3-data-preparation.json).

New archives and acoustic files are on the external SSD under `target-speaker-extraction/v3/`. Additional speech is extracted beside the existing local corpus. None of these audio files enters Git.

Noise comes from the Freesound portion of MUSAN distributed in OpenSLR28, whose included license states the selected recordings were marked public domain. Simulated room responses are also from OpenSLR28, listed under Apache 2.0. Noise recordings are partitioned by content hash; room responses are partitioned by room group. The complete ZIP and extracted files have SHA-256 identities; ZIP member CRCs were checked, but no publisher archive checksum was verified for that ZIP. [OpenSLR28](https://www.openslr.org/28/), [LibriSpeech](https://www.openslr.org/12/).

There are 661/87/78 eligible noise files and 1,880/276/244 selected room responses in train/dev/test. Training repeats clean, noise, reverb, partial overlap, pauses, target absence and combined scenarios with fixed per-example seeds. Clean receives two of eight slots; each other scenario receives one. Noise SNR is 0–20 dB in mixtures and 5–20 dB in references. Different source positions share a room; reference-room selection is independent. Room targets retain reverberation, aligned by the strongest arrival and truncated to 0.8 seconds of decay. This evaluates extraction, not dry-source dereverberation.

Target-absent examples remove the requested source while retaining the other voice and environmental noise. They use normalized output-energy loss. SI-SDR and ESTOI are not computed against silent targets. The presence objective is not a calibrated absence detector.

## Development and listening

The acoustic development suite has 350 requests: 50 source requests repeated under seven conditions. The final suite has 700 requests, 100 per condition. Dependence between conditions and paired targets must be retained in interpretation. Clean regression is also checked on the original 400 development requests.

The baseline acoustic development mean across target-present conditions is 5.117 dB. Its clean dev-other result is 6.498 dB; combined conditions fall to 1.697 dB. These measurements motivate realistic adaptation but are not fresh-test claims. [Baseline report](../reports/v3-baseline-realistic-development.json).

The blind listening tool hides model identity and randomly assigns A/B per trial. It presents equal-RMS playback with shared peak headroom and separate 1–5 severity ratings for competing speech, target damage and static. Anonymous browser identifiers and ratings stay in the local service. Model mappings live outside the served gallery. Test-generated ratings are confined to temporary test directories. No human rating is inferred from successful playback or objective metrics.

## Commands

Prepare data from the completed archives:

```sh
uv run python scripts/prepare_v3_data.py --archives "/Volumes/Zach's SSD/target-speaker-extraction/v3/archives" --environment-root "/Volumes/Zach's SSD/target-speaker-extraction/v3/environments"
uv run python scripts/prepare_v3_models.py
```

Preparation preserves existing manifests and refuses to replace recorded initializers. Source paths can be changed explicitly for another machine. The exact continuation source must be a full training checkpoint, including optimizer/RNG state; the averaged v0.2.0 export is not resumable.

```sh
uv run python scripts/fork_training.py --source artifacts/quality-checkpoints/trajectory/step-20000.pt --config configs/v3-continuation-cosine.json --run artifacts/runs/v3-continuation-cosine
uv run tse train --config configs/v3-continuation-cosine.json --root data/expanded/raw --manifest data/expanded/manifests/inventory.json --dev-cases data/expanded/manifests/dev-cases.json --run artifacts/runs/v3-continuation-cosine --resume --prefetch --compile-blocks --device mps
uv run python scripts/evaluate_realistic.py --checkpoint artifacts/releases/v0.2.0/model.pt --environment-root "/Volumes/Zach's SSD/target-speaker-extraction/v3/environments" --output reports/v3-baseline-realistic-development.json --device cpu
```

Existing reports are retained; choose a new output for a new evaluation. Final scoring additionally requires a freeze that binds the exact model, speech manifest, case recipes and acoustic manifest.

Run the remaining development sequence serially:

```sh
uv run python scripts/run_v3_experiments.py --environment-root "/Volumes/Zach's SSD/target-speaker-extraction/v3/environments"
```

The runner resumes incomplete checkpoints, preserves pilot candidates before extending their training, compares all three pilots on acoustic development data, and extends the highest-scoring architecture to 10,000 updates. It selects the adaptation source by acoustic SI-SDRi among models satisfying the clean-retention gates. Both adaptation arms receive the same 140-case composite checkpoint-selection set. The runner stops after the final development decision; it does not open fresh test data or replace the served model. Its current stage is recorded in `artifacts/verification/v3-suite-status.json`.

Numerical checks cover complex reconstruction and gradients, reference padding, CPU resume with acoustic augmentation, resource-change rejection, classifier-label identity, and frozen-test identities. The first band experiment uses microbatches of four with two-step accumulation; that setting was chosen before training after an MPS feasibility check. Only the band input and mask projections are compiled; recurrent layers remain eager. Measurements taken while other jobs run establish feasibility, not isolated inference performance.
