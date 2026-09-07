# Research alignment audit

Checked on 2026-09-06 after listening feedback and the user's request to verify the referenced guide. The project uses a research-supported target-speaker-extraction formulation, but it is **not a faithful reproduction of a published system**. Citation of a paper does not validate our substituted components or training budget.

The local [model-design guide](MODEL_DESIGN.md) is our original proposal, not an external recipe. The external references include the [SpeakerBeam authors' tutorial repository](https://github.com/BUTSpeechFIT/speakerbeam) and the [enrollment-augmentation study](https://arxiv.org/html/2409.09589v1). No SpeakerBeam implementation or checkpoints are imported. Independent implementation can follow established mathematical methods; it does not make those methods new.

## What matches, and what differs

| Component | Research reference | Actual project choice and implication |
| --- | --- | --- |
| Task | Enrollment audio identifies the desired speaker in a mixture. | Matches the formulation. A different utterance supplies the reference; clean targets supervise training only. |
| Speaker supervision | SpEx jointly optimizes extraction and speaker classification. | Current quality runs use both. Our small raw-waveform reference encoder and loss weights differ; this is not SpEx. [SpEx](https://arxiv.org/pdf/2004.08326). |
| SpeakerBeam recipe | 512 learned filters, 128 bottleneck/skip channels, 512 hidden channels, 24 temporal blocks, ReLU masks; 8 kHz, three-second mixture/reference segments, Adam at 0.001 and a 200-epoch budget. | Original model was much smaller; the current quality candidate instead uses a 16 kHz STFT, 96/192/96 channel widths, 18 blocks, four-second mixtures and five-second references. Its 20,000-update budget is not equivalent to the published schedule. [Authors' configuration](https://github.com/BUTSpeechFIT/speakerbeam/blob/main/egs/libri2mix/local/conf.yml). |
| Enrollment-augmentation guide | ResNet34 speaker encoder, band-split recurrent separator, STFT/ISTFT, speaker CE plus SI-SDR; 100 epochs and averaging the last five checkpoints. It studies MUSAN noise, simulated room responses, SpecAugment and self-estimated references. | Our encoder and separator are different. Original Gaussian noise/filter/echo corruptions are simpler substitutes. The active quality run uses clean references. Planned checkpoint averaging is related, but our update schedule is different. Therefore our results do not reproduce that paper's augmentation findings. [Study, methods and experimental settings](https://arxiv.org/html/2409.09589v1). |
| Normalization | Noncausal Conv-TasNet uses statistics across channels and time. Its causal normalization accumulates past statistics. | Our existing `ChannelNorm` uses only channels at each frame. It is neither of those. This makes finite-context chunking straightforward, but its quality cost was not isolated before release. A separator-global-normalization experiment is now explicit. [Conv-TasNet, section II-D](https://arxiv.org/pdf/1809.07454). |
| Output representation | Learned waveform representations can avoid retaining mixture phase. | The new spectral candidate predicts real masks in [0,1] and retains mixture phase. This restricts reconstruction under destructive interference; it is a baseline hypothesis, not the highest-fidelity design. The original learned-waveform model had a different failure profile. |
| Evaluation | Research systems use specified mixture datasets and metric conventions. | Our on-demand LibriSpeech mixtures and reserved identities are custom. Published Libri2Mix/WSJ0 numbers and paper-specific accuracy definitions cannot be compared directly with our SI-SDRi and three-dB confusion proxy. |

## Evidence about the current bottleneck

On 400 development requests, the original delivered model scores 2.062 dB mean SI-SDR improvement and 0.5387 ESTOI. The expanded spectral model at 10,000 updates scores **4.890 dB and 0.6223**, with 9.75% confusion and 15.75% negative improvements. These are development measurements during tuning, not an independent final-test claim. [Measured report](../reports/quality-stft-expanded-fastlr-10000.json).

A separate source-informed diagnostic calculates masks from the known clean target. With our same STFT and a mask constrained to [0,1], the clipped phase-sensitive result reaches **14.559 dB**; an ideal magnitude ratio reaches 12.598 dB. An unconstrained real mask reaches 16.319 dB. [Diagnostic implementation](../scripts/diagnose_mask_capacity.py), [all development results](../reports/mask-capacity-development.json).

These are reconstruction examples using unavailable ground truth, not learned predictions, promised attainable performance, or strict waveform SI-SDR upper bounds. The large gap nevertheless shows that bounded masks can represent much cleaner outputs on these cases than our model currently predicts. Phase restrictions alone do not explain the current low score. Source-informed masks never enter the application or model evaluation.

The distortion projection metric includes phase, filtering and envelope errors; it is not a measurement of perceived static. ESTOI predicts intelligibility; it is not transcription accuracy or a human judgment. Matched-volume listening remains necessary.

## Corrective experimental sequence

1. Retain the current 20,000-update run as a measured control, preserve fixed checkpoints, and compare the two predeclared averages on development.
2. Test global separator normalization with the same pilot weights, new classifier initialization, seed, source pool, loss, optimizer, 2,000-update budget and validation schedule as the existing expanded-data pilot. Only separator normalization changes; the reference encoder stays unchanged. An explicit transfer flag records this architectural change. This is a short adaptation experiment, not a full from-scratch architecture comparison.
3. Global statistics invalidate finite-context chunk equivalence. This variant uses whole-clip offline inference and must pass long-clip quality/memory checks before promotion. Do not swap normalization in a served checkpoint without training and evaluation.
4. Select using development separation, intelligibility, voice confusion and listening. Freeze the selected checkpoint before evaluating the newly reserved speaker identities. Preserve the opened v0.1.0 test as historical evidence.

A future faithful baseline would also need a pinned published architecture/data recipe, matching training exposure and convergence checks, followed by controlled modifications. The current custom system must not be described as that reproduction. A laptop-first prototype was a reasonable implementation milestone; presenting its operational completeness as adequate separation quality was not supported by its results.
