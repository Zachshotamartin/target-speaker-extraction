# Voice identity and target-speaker activity: implementation proposal

Current milestone: the user selected a complete proof of concept with existing
models and no new training. Follow [PRETRAINED_TRANSCRIPTION_POC_PLAN.md](PRETRAINED_TRANSCRIPTION_POC_PLAN.md).
The custom-training routes below remain background research, not authorized work
for this milestone.

Prepared September 21, 2026 against repository main `aca75b3`.
Status: planning only. No models downloaded, dependencies installed, training
started, or application behavior changed for this proposal.

User constraint: use public datasets for training, threshold calibration and
evaluation. Do not request additional personal recordings or a user-supplied
training/evaluation corpus. Personal enrollment audio, if supplied for selecting
a voice, is inference input only: never use it for model training, fine-tuning or
threshold fitting. An existing clean recording can serve as that reference;
recording a new one is not required. If no reference is available, the system can
offer an explicitly selected speaker from an existing recording, but cannot
automatically know which speaker is the user.

## Intended behavior

A user supplies clean enrollment speech, selects that profile, and receives a
transcript of their speech in a recording that may include another person. The
system must represent uncertainty and must be evaluated when the enrolled person
is silent or completely absent. New users enroll through examples rather than
requiring a new model-training run.

There are two related tasks:

- Speaker verification: do this reference and this speech segment belong to the
  same speaker? This is useful for enrollment and segment-level checks.
- Personalized voice activity detection (PVAD): at what times is the enrolled
  speaker talking, including overlap with someone else? This more directly
  matches the transcription requirement.

A conventional verification score on a mixed recording is not automatically a
target-presence probability. Nor does detecting the target prove that every word
in the separated signal belongs to the target.

## Existing code and constraints

`src/tse/inference.py` already computes a reference embedding and conditions
extraction with it. The active `ReferenceBSRNN` uses a 256-dimensional enrollment
representation, bidirectional recurrent layers, global normalization and
whole-clip inference. Its current training configuration assumes target-present
two-speaker mixtures. The inference interface exposes no calibrated confidence.

`site/` and `src/tse/public_api.py` already provide uploads and audio extraction.
The public API currently accepts a mixture up to 30 seconds, a 3–10-second
reference, and a combined upload below 4 MiB. A persistent voice profile, voice
presence detection and transcription are proposed additions.

The running separator remains isolated in its existing process and frozen source
copy. Future work must use a separate environment and output directory, with no
package upgrades in the trainer's environment or edits to its source snapshot.
Use an immutable evaluated checkpoint for experiments; never read a live
checkpoint while it is being replaced. This proposal does not change the current
training configuration or its checkpoint compatibility rules.

## Options

| Route | What we train | Benefit | Main limitation |
| --- | --- | --- | --- |
| Pretrained baseline | No neural network; calibrate scores and temporal rules | Fastest way to measure feasibility | Segment verification is fragile during overlap and on very short speech |
| Recommended hybrid | Small custom PVAD network; keep a pretrained speaker encoder frozen | Original ML work focused on target absence and overlap | Requires frame labels, calibration and evaluation on public recordings |
| Full custom | Speaker embedding network, then a PVAD network | Maximum ownership of training and model design | Needs greater speaker diversity and substantially more experimentation |

The earlier 2–4-day estimate concerned an integration prototype. It is not an
estimate for training a competitive speaker encoder from scratch or establishing
robustness across recording conditions.

## Data

| Resource | Proposed use | Availability and limitations |
| --- | --- | --- |
| [LibriSpeech](https://www.openslr.org/12/) | Speaker-labeled clean speech, distinct enrollment utterances, verification pairs and synthetic conversations | Start with the prepared train-clean-100 source. Optional train-clean-360 and train-other-500 expansion; CC BY 4.0. Audiobook speech does not establish microphone or conversational robustness. |
| [MUSAN](https://www.openslr.org/17/) | Noise, music and distracting speech augmentation | Public download; listed as CC BY 4.0. Prefer the noise subset initially and retain source attribution. |
| [RIRS_NOISES](https://www.openslr.org/28/) | Reverberation and room variation | Public download; listed as Apache 2.0. Split room responses between fitting and evaluation. |
| [VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) | A larger speaker-verification training option with interview/channel variation | Official site describes an audio-access request process. Successful audio access is not verified here; do not make the first milestone depend on it. OpenSLR 49 contains supporting metadata/trials, not the audio corpus. |

Construct four conditions: target only, another speaker only, target plus another
speaker, and no speech. Include changes between these conditions within a clip.
Vary overlap, speaker level, microphone/channel filtering and background noise.
Include cases in which the target is never present, not just short pauses.

Enrollment must use a different utterance from the target content. Split speakers
before generating training, development and evaluation mixtures. Also separate
recordings/sessions, noise excerpts and room responses. A clip's file duration
does not imply continuous speech: derive activity labels from source-level speech
boundaries, audit automatically generated boundaries, and manually check an
evaluation subset. Speech recognizers' transcripts are not speaker-identity labels.

Preserve the existing One Voice evaluation protocol. Its development speakers
must not become training speakers for the new component if results are reported
as generalization of the combined system. Existing test-clean exposure remains
documented; do not present that corpus as a new pristine project holdout.
TitaNet's training includes LibriSpeech, so a LibriSpeech-only comparison using it
cannot establish wholly unseen-speaker performance. Record pretraining exposure
for each baseline and reserve public speakers and recording sessions unused by
our training or threshold selection. Where pretrained-model exposure is unknown,
report that limitation rather than claiming a pristine holdout. No personal
recording collection is a prerequisite or planned work item.

## Route 1: use pretrained components

First candidate: [SpeechBrain ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb).
It provides speaker embeddings and cosine-based verification, is trained on
VoxCeleb, and lists an Apache 2.0 license. Use a pinned release/model revision and
check its CPU compatibility in an isolated environment.

Alternative: [NVIDIA TitaNet-Large](https://huggingface.co/nvidia/speakerverification_en_titanet_large).
Its card lists about 23 million parameters and CC BY 4.0. Use it as a comparison
if the first model struggles; benchmark the NeMo runtime on this Mac before
committing to it. Neither model's published benchmark EER predicts error rates
on our mixtures or extracted audio.

[Silero VAD](https://github.com/snakers4/silero-vad) supplies a pretrained generic
speech detector and an ONNX option under MIT. It identifies speech activity, not
the enrolled person's identity.

Baseline procedure:

1. Accept an existing clean enrollment recording and allow the user to listen
   before saving. Check for silence/clipping. Use one valid reference initially;
   additional personal samples are not required. Any future aggregation rule must
   be developed and evaluated on public data.
2. Store a profile ID, encoder revision, preprocessing version and enrollment
   representation. Keep the reference audio needed by One Voice under the user's
   storage preference. Never compare vectors from different encoders or revisions.
3. Detect speech on the original recording and compute identity evidence on
   overlapping short windows. Tune window/context length against short-utterance
   errors rather than declaring a default threshold to be universally valid.
4. Extract the requested speaker with the frozen One Voice checkpoint. Compare
   verification evidence from original and extracted audio; extracted-audio
   similarity alone is not independent proof because the separator is already
   conditioned on the reference. Treat conflicting evidence as uncertain.
5. Apply development-calibrated acceptance and rejection thresholds with smoothing
   and a reject region. Preserve context around accepted speech. Pass accepted
   regions to the chosen pretrained ASR and retain original timeline offsets.

This route requires calibration and temporal decision logic but no new neural
network training. It is a baseline, not a guaranteed solution for heavy overlap.
ASR confidence must not be substituted for speaker-identity confidence.

## Route 2: build a small personalized activity model

Recommended first custom model: a speaker-conditioned CNN plus a small causal
GRU, targeting roughly 0.2–1 million trainable parameters as a design budget.
This size is a proposal, not a benchmarked implementation. Input is 16 kHz audio
represented as log-mel frames plus the frozen encoder's enrollment embedding.
The pretrained speaker encoder's parameters and runtime are additional costs.

Use two sigmoid outputs per frame: target-active and other-speaker-active. This
represents simultaneous speakers naturally: both can be active. Silence maps to
neither. Train with weighted binary cross-entropy, sample all four conditions,
and select thresholds using false-accept/miss tradeoffs on development speakers.
Do not make the model learn only a fixed "Zach versus everyone" classifier.

Train on the original mixture first, so target-presence evidence does not rely
entirely on what the separator produces. A later ablation can add extracted-audio
features if measurements justify the added computation and dependencies.

Keep the speaker encoder frozen initially. Cache enrollment embeddings and
reusable source features on the SSD; synthesize noisy mixtures as needed. A
frozen copy of One Voice's existing reference encoder is a low-download ablation,
but its representations must be independently evaluated for this task.

Google's [Personal VAD](https://google.github.io/speaker-id/publications/PersonalVAD/)
demonstrates the speaker-conditioned activity formulation and reports a small
detector. Our proposed overlap outputs, data recipe and evaluation would be our
own implementation choices, not a claim to reproduce that paper's results.

## Route 3: train the speaker encoder ourselves

Use 2–4-second speech crops and a compact TDNN or residual CNN, statistics
pooling, and a 192- or 256-dimensional embedding. Start with a few-million-
parameter budget and profile it before scaling. Train on many labeled speakers
with an additive angular-margin classification objective; evaluate same-speaker
and different-speaker trials using embedding similarity. The training classifier
is discarded for enrollment of new speakers.

Use balanced speaker sampling and different utterances per speaker, add noise,
reverberation and channel augmentation, and reserve all evaluation identities.
The current 251-speaker source is suitable for a controlled learning baseline,
but insufficient evidence for broad real-world robustness. More diverse
speaker-labeled data is the principal scaling need.

Then condition the Route 2 activity model on this encoder. A custom verifier by
itself does not solve frame-level overlap. Compare the complete system with the
pretrained baseline before choosing it for users. Training from scratch is
feasible as an experiment; outperforming a widely trained encoder is not promised.

## Evaluation and decision gates

Use a frozen, versioned suite covering unfamiliar speakers, different reference
utterances, short responses, overlap, complete target absence, room noise and
different microphones represented in the selected public data. Use public
recordings from reserved speakers/sessions for all calibration and evaluation.
Performance on the user's own microphone and voice remains unmeasured unless
the user independently chooses to try inference; such input is not training data.

- Verification: ROC/DET curves, EER as a diagnostic, and false rejection at a
  chosen false-accept operating point. Cosine similarity is not a probability.
- Activity: target miss duration, incorrectly accepted non-target duration, false
  activation events per hour, and boundary delay. Report overlap separately.
- Transcription: target-word error rate, non-target word insertions, and words
  produced per hour of target-absent audio. Include direct ASR, generic VAD plus
  verification, extraction without identity gating, and the proposed full system.
- Runtime: end-to-end real-time factor, peak resident/unified memory and latency
  distribution on this M3 Pro with 18 GiB RAM. Training beside another workload
  must be measured explicitly.

Provisional development target: at most 1% non-target speech duration accepted
while retaining at least 90% of target speech, reported by condition, plus fewer
non-target transcript insertions than the ungated pipeline. These are proposed
operating goals, not achieved results or universal product requirements. Tighten
or revise them based on the user's tolerated errors and measured tradeoff curves.
Report uncertainty across speakers/sessions and test enough negative audio to
support any claimed low false-activation rate; a few successful clips are insufficient.

Prefer the custom detector only when it improves rejection at a comparable
target-miss rate and stays within the runtime budget. A low average loss or a
higher separation dB score alone does not satisfy this decision gate.

## Execution order, once implementation is requested

1. Enrollment/profile handling and immutable model interfaces.
2. Versioned evaluation suite plus pretrained verification/VAD baseline.
3. Recorded-clip pipeline using One Voice and a pretrained speech recognizer.
4. Small custom PVAD experiment and controlled comparison.
5. Optional custom encoder only if learning goals or measured limitations justify it.
6. Buffered microphone use after recorded-clip behavior is acceptable.

Begin with short recordings matching the existing API. A causal activity detector
does not make the bidirectional One Voice separator causal: continuous live use
still needs a buffering/context strategy and fresh evaluation of boundary errors.
Do not simply concatenate aggressively trimmed clips and lose timestamp mapping.

Before any long training commitment, run a bounded throughput/memory benchmark,
then estimate total updates from measured seconds per update and evaluation cost.
Trainable parameter count alone cannot predict elapsed time. The current active
One Voice job keeps priority; no competing GPU training is launched as part of
this plan. Cache and CPU budgets should be explicit for future experiments.
Retain latest/best checkpoints, optimizer/RNG states, data order, calibration
settings and source identity for resumable custom training.

Deliverables would be a saved voice profile UI, versioned inference components,
evaluation report, reproducible training recipe for the custom route, and a model
card separating pretrained dependencies from our trained contribution.
