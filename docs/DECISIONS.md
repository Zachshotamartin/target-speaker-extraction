# Decision log

Recorded on 2026-09-06. Update decisions with evidence and preserve the reason for a change.

| ID | Decision | Status | Reason / reconsider when |
| --- | --- | --- | --- |
| D01 | Implement our own model, training, data and evaluation code; train core weights from scratch | Agreed scope | The user wants substantive ML engineering and an independent implementation |
| D02 | Use SpeakerBeam as attributed research only; no code or checkpoints in the core | Agreed scope | Avoid repackaging its implementation |
| D03 | Public repository named `target-speaker-extraction` | User requested | User explicitly changed the initial private preference to public |
| D04 | Initialize the existing `ML` workspace directly on `main` | Established | The workspace was empty and not inside another Git repository |
| D05 | Focus the primary study on mismatched reference quality | Proposed research focus | Useful failure class with a controlled training intervention; already has prior literature |
| D06 | Start with two speakers, 16 kHz mono and offline recordings | Proposed first version | Keeps the core learnable and measurable on the local machine |
| D07 | Use custom LibriSpeech mixtures before an official benchmark adapter | Proposed data protocol | Saves storage and supports independent data code; results must be named accurately |
| D08 | Use a compact reference-conditioned temporal convolutional network | Proposed architecture | Clear reference mechanism and bounded local compute; revisit after tiny-set/compute gates |
| D09 | Use native ARM Python 3.12 and test CPU/MPS | Proposed environment | Python 3.12.5 is available natively; package/device compatibility remains unverified |
| D10 | Generate mixtures on demand and keep audio/checkpoints outside Git | Established repository convention | Limited disk and reproducibility through manifests rather than copied artifacts |
| D11 | Keep the first model target-present; measure absence failures separately | Proposed operating envelope | Zero-target losses and reliable presence calibration require separate design |
| D12 | Use local JSON/TOML records before introducing tracking infrastructure | Proposed engineering choice | Sufficient for the first controlled experiments |
| D13 | Defer streaming, deployment optimization and broader speaker counts | Proposed sequencing | They require a reliable baseline and separate quality/latency evidence |
| D14 | Leave the original code license unselected for the foundation | Open release choice | Public visibility does not itself establish a reuse license; record a license before a code/model release |

## Open technical decisions and their evidence

- **Exact dependency versions:** decide after environment and audio/MPS smoke checks, then lock.
- **Usable model width, crop length and batch size:** decide from a measured forward/backward benchmark.
- **Pilot speaker selection:** decide from inventory and reference eligibility; preserve source split boundaries.
- **Main training update budget:** decide from learning curves and throughput, before comparing B1/B2.
- **Auxiliary speaker supervision:** add only if the reference representation fails or a controlled experiment supports it.
- **Reference corruption strengths:** finalize using development listening/metrics before final evaluation.
- **Quality and release thresholds:** prespecify on development data, then freeze for the final test.
- **Noise/room source expansion:** choose after checking actual download size and license terms.
- **Streaming:** requires a causal model/state design; current chunking does not satisfy it.

These decisions do not block the foundation work. They have explicit milestones for resolution.
