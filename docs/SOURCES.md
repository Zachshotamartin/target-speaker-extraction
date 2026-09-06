# Sources, attribution and data origin

Primary sources checked on 2026-09-06. These references support the task, public data and established methods; they do not establish performance of this untrained project.

## Public data

### LibriSpeech

- [Official dataset page](https://www.openslr.org/12)
- Public read-speech corpus with 16 kHz audio; official page lists CC BY 4.0.
- Intended source of clean utterances for our independently generated mixtures.
- Preserve dataset attribution, original identifiers, archive/member hashes and split names.

### LibriMix

- [Dataset repository and generation documentation](https://github.com/JorisCos/LibriMix)
- [Dataset paper](https://arxiv.org/abs/2005.11262)
- Benchmark resource for speech mixtures, including noisy variants.
- Its default generator creates several large configurations. Our initial custom protocol is separate from official benchmark results.
- The repository's code license does not replace the licenses or attribution of constituent audio sources. Check the exact assets used, including any WHAM noise, before acquisition or redistribution.

## Model and evaluation literature

### SpeakerBeam

- [Authors' implementation and paper references](https://github.com/BUTSpeechFIT/speakerbeam)
- Prior work on extracting a target voice using an enrollment utterance.
- Research reference only for our core model: no implementation or weights are being imported.

### Conv-TasNet

- [Luo and Mesgarani: Conv-TasNet](https://arxiv.org/abs/1809.07454)
- Prior work informing learned time-domain representations and temporal convolutional separation.
- Attribute the architectural influence; independently implementing related components does not invent the underlying method.

### FiLM

- [Perez et al.: FiLM, a general conditioning layer](https://arxiv.org/abs/1709.07871)
- Prior work informing feature-wise affine modulation from a conditioning vector.

### SI-SDR

- [Le Roux et al.: SDR — half-baked or well done?](https://arxiv.org/abs/1811.02508)
- Source for the scale-invariant signal-to-distortion evaluation concept.
- Document our implementation's centering, masking, epsilon, silence and aggregation conventions.

### Enrollment augmentation

- [On the effectiveness of enrollment speech augmentation for Target Speaker Extraction](https://arxiv.org/abs/2409.09589)
- Prior research on changing reference audio during training.
- This establishes that the topic is not new. Our proposed contribution is an independent local implementation and a controlled robustness/efficiency study, with results still to be measured.

## Runtime

- [PyTorch MPS documentation](https://docs.pytorch.org/docs/stable/notes/mps.html): Apple GPU device support. Individual operations, installed versions and training throughput still require local verification.

## Origin registry to implement

For every external dataset, metadata file, optional checkpoint or code excerpt, record its exact source URL, retrieved version/commit, retrieval date, license, checksum and intended use. Default to writing our own implementation from documented ideas and mathematical specifications.

The source tree currently includes no third-party model code, copied research recipe or downloaded weights. General libraries will be normal declared dependencies. If externally licensed material is later added, preserve its notices and describe its role explicitly.

## Public repository and release

No original project code license has been selected yet. Choose and add one deliberately before a reusable code/model release. Third-party material remains under its own terms regardless of the repository's eventual license.

For public derived audio examples, document the source and transformation, confirm redistribution terms and include attribution. Private recordings remain local unless their owners explicitly permit inclusion.
