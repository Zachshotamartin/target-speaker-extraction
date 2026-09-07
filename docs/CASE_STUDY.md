# Improving target-speaker extraction on one Mac

The first One voice release had a working training/evaluation/API pipeline, but weak separation and audible artifacts. The user heard the competing voice become quieter without being adequately removed. The quality follow-up's frozen fresh test measures **5.887 dB mean SI-SDR improvement**, versus 1.453 dB for the original model on the same requests. Remaining failures are part of the result: 13.90% of requests worsen and 7.60% trigger the confusion proxy.

## Diagnose the signal path before redesigning the model

A development audit found 58 of 400 original predictions above sample magnitude one. A uniform attenuation guard fixes that playback risk without changing relative speaker levels. However, the first 20 original gallery outputs were already below full scale. Four-second gallery requests bypass chunking, float WAV round-trips preserved samples, and a CPU/MPS comparison was numerically close. Those checks did not support treating clipping or serialization as the sole explanation for the reported static. [Audio diagnosis](QUALITY_IMPROVEMENT.md).

Fixed cleanup filters and a matched additional-training comparison with spectral loss produced modest gains. Increasing mask strength suppressed more interference while damaging intelligibility. These experiments discouraged treating quieter background audio as sufficient evidence of better extraction.

## Rebuild the learning experiment and preserve an independent test

The new candidate uses independently implemented reference-conditioned temporal convolutions over an STFT, with a bounded real mask and inverse transform. Its reference encoder is learned within this project. Joint speaker classification, waveform and multi-resolution spectral objectives supplement target-specific SI-SDR. Expanded training draws from 231 speakers and 90.582 eligible source hours, using four-second mixtures and separate five-second references.

The original test had already been opened. Twenty previously unused train-clean-100 identities were therefore reserved before expanded training, while the same 40 dev-clean identities continued to guide development. Final selection binds exact model, inventory and recipe hashes before the new 1,000-request test. These are custom LibriSpeech mixtures, not official Libri2Mix results.

The principal development changes combine architecture, speaker supervision, crop length, data diversity and training exposure. Their aggregate improvement cannot be attributed to one component. A model small enough for local experiments remains a hypothesis, not evidence that the published research recipe has been reproduced.

## Check the research assumptions explicitly

The [research audit](RESEARCH_AUDIT.md) compares the implementation with SpeakerBeam, SpEx, Conv-TasNet and the enrollment-augmentation study. It identifies substantial substitutions, including the small reference encoder, per-frame separator normalization, real mixture-phase masks, custom augmentation and much shorter training schedule. No external model weights or SpeakerBeam implementation enter the core.

A source-informed diagnostic reaches 14.559 dB with the same bounded-mask format. This uses unavailable clean targets, so it is not deployable performance or a guaranteed learning ceiling. It demonstrates representational headroom and helps distinguish output-format restrictions from imperfect mask prediction.

A separate 2,000-update normalization adaptation matches initialization tensors, fresh classifier, seed, data sequence, losses and optimizer. Changing only separator normalization gives 3.673 dB / 0.5916 ESTOI, versus 3.484 dB / 0.5852 for the matched per-frame control on 400 development requests. The starting weights were already trained with per-frame normalization, so this short comparison cannot settle which architecture is better when trained from scratch. The global variant also requires whole-clip inference because its statistics span time.

The main expanded run completed 20,000 updates. The predeclared final-three and final-five checkpoint averages were scored alongside its best single checkpoint. Final selection and all compared report identities are retained in the [freeze](../reports/quality-v2-selection.json). Uniformly averaged project checkpoints at expanded-data steps 16000, 18000, 20000. Source hashes are retained in the exported provenance.

## Measure the delivered result

| Metric | Original model | Updated model |
| --- | ---: | ---: |
| Mean SI-SDR improvement | 1.453 dB | 5.887 dB |
| Median SI-SDR improvement | 1.656 dB | 6.270 dB |
| ESTOI intelligibility proxy | 0.5457 | 0.6534 |
| Requests worse than mixture | 30.80% | 13.90% |
| Speaker confusion proxy | 17.30% | 7.60% |
| Scalar distortion proxy | 11.58% | 11.20% |

The paired separation gain is +4.434 dB, with approximate 95% interval [3.455, 5.529]. The [model card](MODEL_CARD.md) specifies metrics, uncertainty, failure slices, runtime and artifact identities. The original [case study](CASE_STUDY_V0_1.md) remains available separately; its opened test was not relabeled as unseen.

The service and evaluator share decoding, normalization, prediction, peak handling and float-WAV output. Long-file checks verify duration and the applicable inference strategy. Training uses deterministic sample seeds, atomic resumable checkpoints and provenance records; compiled temporal layers were numerically checked before use on MPS. Exact CPU resume has regression coverage. Code and metadata are public, while audio and weights remain local.

The [comparison gallery](http://127.0.0.1:8000/gallery/quality-progress/) contains examples chosen by manifest order and separately labeled matched-RMS playback. ESTOI predicts intelligibility and the distortion projection is a proxy; neither constitutes a human judgment that static is gone. The model remains experimental for two target-present voices in recorded audio. Real microphones, reverberation, background noise, target absence and broader language/domain transfer require their own evidence.

[Reproduce the quality experiments](REPRODUCING_QUALITY.md).
