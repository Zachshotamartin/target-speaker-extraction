# Why the published results do not yet transfer to this project

Audit made on 2026-09-06 (local time) following feedback that the mixture and both continuation estimates sounded almost identical. The project follows the target-speaker-extraction formulation, but neither the released TCN nor the compact v3 band separator is a reproduction of the referenced system. The missing experimental anchor is a faithfully specified baseline trained to convergence. A functioning application and successful numerical tests do not establish that baseline.

## Verified differences

The [enrollment-augmentation paper](https://arxiv.org/html/2409.09589v1#S4) reports 100 epochs, three-second mixture segments, full enrollment utterances and last-five checkpoint averaging. Its unaugmented, jointly trained train-100 system achieves 14.31 dB SI-SDR on clean Libri2Mix. Thus pretrained weights, extra augmentation and train-460 are not prerequisites for strong clean separation. Our SI-SDR **improvement** on custom mixtures is not directly comparable with that absolute SI-SDR result.

For implementation details, this audit pins the authors' WeSep repository at `99eca54b60300d39b9353d93cf285a14bba37854`. Its current configuration differs from the paper: 150 epochs and two averaged checkpoints. It is useful implementation evidence, but cannot silently substitute for the historical experiment. The exact historical configuration remains unresolved.

| Component | Pinned authors' implementation | Our active band pilots |
| --- | --- | --- |
| Band resolution | 32 bands, including narrow low-frequency bands | 18 bands; eight bins in the narrowest band |
| Separator width / depth | 128 features, recurrent hidden width 256, six blocks | 48 features, hidden width 64, four blocks |
| Reference | ResNet34, 80 features, 256-dimensional embedding | Existing waveform encoder for real/complex arms; scaled ResNet34 with 64 mel bands and 128-dimensional embedding in the third arm |
| Conditioning | Multiplicative fusion once in the configured recipe | Affine scale and shift inside every block |
| Normalization | Group normalization spanning sequence positions | Per-position layer normalization in the separator |
| Output | Gated complex coefficients; coefficient amplitudes are not bounded by the sigmoid gate | Real coefficient in [0,1]; imaginary coefficient in [-1,1] for complex variants |

Sources: [pinned configuration](https://github.com/wenet-e2e/wesep/blob/99eca54b60300d39b9353d93cf285a14bba37854/examples/librimix/tse/v2/confs/bsrnn.yaml), [pinned model definition](https://github.com/wenet-e2e/wesep/blob/99eca54b60300d39b9353d93cf285a14bba37854/wesep/models/bsrnn.py). Reading these specifications for an audit does not import their implementation or weights. The local implementation remains in [advanced_models.py](../src/tse/advanced_models.py).

These are architectural changes, not just runtime optimizations. They may affect optimization and capacity; the audit does not establish a causal ranking or a guaranteed gain from reverting any one of them.

Each new pilot receives 4,000 updates at effective batch eight: **32,000 four-second target requests**. The selected architecture's planned 10,000 updates amount to **80,000 requests**, or 88.9 hours of mixture-crop exposure with reuse. This is neither 10,000 epochs nor evidence of convergence. The older TCN's training cannot be counted as separator training for these new randomly initialized recurrent models. The two waveform-reference arms inherit only our previously trained reference encoder. The ResNet reference starts randomly.

Our speech source remains 231 training identities with on-demand mixtures, fixed-length references and custom development cases. The [authors' tutorial](https://github.com/wenet-e2e/wesep/blob/99eca54b60300d39b9353d93cf285a14bba37854/examples/librimix/tse/v2/README.md) specifies Libri2Mix data preparation and enrollment mappings. Matching a source corpus name does not match its mixture recipe or evaluation population.

Our objective combines negative SI-SDR with waveform, multi-resolution spectral and speaker-classification terms. Those additional reconstruction terms and our classification scale have not been independently ablated. Realistic adaptation adds our own noise/room/overlap/absence recipe. Self-estimated enrollment augmentation is absent. More acoustic variation can improve robustness, but its presence alone does not validate clean separation or reproduce an augmentation study.

## What our own measurements establish

- The served release achieves 6.046 dB SI-SDRi on 400 clean development requests. Cosine continuation reaches 6.590 dB, but its acoustic gain is only 0.189 dB and its paired interval includes no gain. The listening feedback does not support an audible improvement. [Continuation results](../reports/v3-continuation-results.json).
- The early listening form repeated one source pair across six conditions. Its twelve requests average only 0.483 dB improvement for the release and 1.019 dB for cosine, with six and four negative-improvement requests respectively. The files are distinct; weak extraction is real. The broad development average concealed this failure. [Feedback](../reports/v3-continuation-listening-feedback.json).
- With the known target spectrum, a real mask in our allowed range reaches 14.390 dB improvement on the acoustic suite's clean subset; the released model reaches 6.498 dB on that subset. This diagnostic uses unavailable ground truth, not a learned predictor or a strict SI-SDR bound. It demonstrates substantial representational room even before adding phase correction. [Source-informed diagnostic](../reports/v3-acoustic-mask-diagnostic.json), [matched subset baseline](../reports/v3-baseline-realistic-development.json).
- The real-mask recurrent pilot completed at 2.756 dB clean development improvement, well below the release. A short random-separator pilot cannot establish its eventual quality. [Pilot report](../reports/v3-band-real-pilot-clean-development.json).

The evidence points to inadequate learned extraction, with training exposure and our architectural substitutions as plausible causes. It does not identify a single missing trick. Phase reconstruction, speaker recognition, and clean versus realistic generalization must be distinguished experimentally.

## Consequence for the research direction

Finish the predeclared v3 comparisons under their existing contracts and report failures alongside gains. Do not change a running arm's architecture, increase only a favored pilot's selection budget, or claim these trials reproduce the paper.

The next reference experiment should independently implement one fixed, published clean baseline; reconcile the historical settings; match data/enrollment conventions and metric definitions; prove tiny-set learning and reference switching for that model; and train with explicit exposure and convergence tracking. Measure actual Mac memory and throughput before setting its budget. Follow with one controlled modification at a time. This would preserve independent code and locally trained weights while making the quality comparison interpretable. It is a research direction, not an additional training job already completed or scheduled.
