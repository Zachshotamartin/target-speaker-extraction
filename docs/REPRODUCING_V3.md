# Reproducing the v3 improvement sequence

Run from the repository root in the locked Python 3.12 environment. This sequence starts from the project's own v0.2.0 export and full 20,000-update training checkpoint. A fresh clone first needs the [original study](REPRODUCING.md) and [v0.2.0 quality sequence](REPRODUCING_QUALITY.md), or the exact local artifacts. There are no externally pretrained weights. Reproducing a method on other hardware need not reproduce checkpoint bytes.

## Acquire and prepare the additional public data

The speech archives are LibriSpeech dev-other and test-other from OpenSLR12. The acoustic archive is RIRS_NOISES from OpenSLR28. Set paths for the machine running the experiment; these variables are examples for the project Mac and can point to another suitable volume.

```sh
uv sync --frozen
V3_ARCHIVES="/Volumes/Zach's SSD/target-speaker-extraction/v3/archives"
V3_ACOUSTICS="/Volumes/Zach's SSD/target-speaker-extraction/v3/environments"
mkdir -p "$V3_ARCHIVES" "$V3_ACOUSTICS"
curl --fail --location --continue-at - --output "$V3_ARCHIVES/dev-other.tar.gz" https://openslr.trmal.net/resources/12/dev-other.tar.gz
curl --fail --location --continue-at - --output "$V3_ARCHIVES/test-other.tar.gz" https://openslr.trmal.net/resources/12/test-other.tar.gz
curl --fail --location --continue-at - --output "$V3_ARCHIVES/rirs_noises.zip" https://www.openslr.org/resources/28/rirs_noises.zip
uv run python scripts/prepare_v3_data.py --archives "$V3_ARCHIVES" --environment-root "$V3_ACOUSTICS"
```

Do not redownload complete local archives unnecessarily. Preparation verifies publisher MD5 values for both speech archives, checks extracted acoustic ZIP member CRCs, and records SHA-256 identities. The acoustic archive lacks a verified publisher checksum. Noise is split by content identity and room responses by room group. This creates the 350-case acoustic development suite, 700-case fresh test, and shared 140-case adaptation selection set. Existing prepared manifests cannot be silently replaced. The expanded speech root retains the original 231 training identities and excludes previous opened test identities from the new test.

## Complete matched learning-rate continuation

The source must include optimizer and RNG state. An averaged inference export cannot substitute for it.

```sh
uv run python scripts/fork_training.py --source artifacts/quality-checkpoints/trajectory/step-20000.pt --config configs/v3-continuation-control.json --run artifacts/runs/v3-continuation-control
uv run python scripts/fork_training.py --source artifacts/quality-checkpoints/trajectory/step-20000.pt --config configs/v3-continuation-cosine.json --run artifacts/runs/v3-continuation-cosine
uv run tse train --config configs/v3-continuation-control.json --root data/expanded/raw --manifest data/expanded/manifests/inventory.json --dev-cases data/expanded/manifests/dev-cases.json --run artifacts/runs/v3-continuation-control --resume --prefetch --compile-blocks --device mps
uv run tse train --config configs/v3-continuation-cosine.json --root data/expanded/raw --manifest data/expanded/manifests/inventory.json --dev-cases data/expanded/manifests/dev-cases.json --run artifacts/runs/v3-continuation-cosine --resume --prefetch --compile-blocks --device mps
```

Both arms run 5,000 further updates from the exact same state. Do not launch concurrent GPU training. The control retains the saved plateau-scheduler history; cosine starts a fixed 5,000-update decay. Each branch has its own resumable directory.

## Architecture and acoustic adaptation

```sh
uv run python scripts/prepare_v3_models.py
uv run python scripts/run_v3_experiments.py --environment-root "$V3_ACOUSTICS"
```

The initializer script creates exact matching separator/classifier tensors for all three pilots and matching real/complex initial predictions. It refuses to replace initializers. The runner trains each pilot for 4,000 updates, snapshots its best checkpoint, scores clean/acoustic development sets, and continues the selected architecture to 10,000 total updates. It then selects an adaptation source using the declared clean-retention criteria and runs matched clean/realistic 3,000-update arms. Both use the shared 140-case checkpoint-selection set. A full clean/acoustic comparison applies the promotion gates to all candidates.

Re-running the suite resumes incomplete runs and verifies completed checkpoint/report identities. Only maximum training budget may change on resume. The selected architecture's config will show 10,000 after extension; its immutable pilot snapshot still records the 4,000-update experiment. Preserve its full metric stream and initialization lineage. `artifacts/verification/v3-suite-status.json` identifies the active subprocess/log. The live local [progress page](http://127.0.0.1:8000/experiments/v3/) reads those records.

The suite ends at development selection. It does not score fresh test or replace the app's model. Learning-rate, architecture, acoustic adaptation and listening comparisons are documented separately; combined gains should not be attributed to a single component.

## Export and verify the fixed candidate

After development completes, this executable selection follows the predeclared research fallback if no model passes the deployment gates:

```sh
uv run python - <<'PY'
import json
import subprocess
from pathlib import Path

decision = json.loads(Path('reports/v3-development-selection.json').read_text())
candidate = decision['selected']
if candidate['name'] == 'baseline':
    candidate = max((row for row in decision['candidates'] if row['name'] != 'baseline'),
                    key=lambda row: row['acoustic_si_sdri_db'])
destination = Path('artifacts/releases/v3-candidate/model.pt')
if destination.exists():
    raise FileExistsError('Preserve the existing candidate export')
subprocess.run(['.venv/bin/tse', 'export', '--checkpoint', candidate['checkpoint'],
                '--output', str(destination)], check=True)
PY
uv run python scripts/check_v3_delivery.py --checkpoint artifacts/releases/v3-candidate/model.pt --environment-root "$V3_ACOUSTICS"
uv run tse profile-delivery --checkpoint artifacts/releases/v3-candidate/model.pt --mixture artifacts/examples/voice-1-mixture.wav --reference artifacts/examples/voice-1-reference.wav --device cpu --output reports/v3-delivery-cpu.json
uv run tse profile-delivery --checkpoint artifacts/releases/v3-candidate/model.pt --mixture artifacts/examples/voice-1-mixture.wav --reference artifacts/examples/voice-1-reference.wav --device mps --output reports/v3-delivery-mps.json
```

Run profiles without other training. They test 10/30/60-second delivered file processing. Whole-clip recurrent/global-normalization models have no chunk-equivalence claim. CPU/MPS agreement is checked on a fixed development request in each acoustic condition. Finish any inference implementation changes before these checks; the freeze verifies the exact evaluated source digest.

## Freeze before fresh-test scoring

```sh
uv run python scripts/freeze_v3_selection.py --candidate-export artifacts/releases/v3-candidate/model.pt
git add reports/v3-selection-freeze.json
git commit -m "Freeze v3 development selection before acoustic test"
uv run python scripts/evaluate_realistic.py --checkpoint artifacts/releases/v0.2.0/model.pt --cases data/v3/manifests/test-realistic-cases.json --environment-root "$V3_ACOUSTICS" --freeze reports/v3-selection-freeze.json --output reports/v3-baseline-test.json --device mps
uv run python scripts/evaluate_realistic.py --checkpoint artifacts/releases/v3-candidate/model.pt --cases data/v3/manifests/test-realistic-cases.json --environment-root "$V3_ACOUSTICS" --freeze reports/v3-selection-freeze.json --output reports/v3-candidate-test.json --device mps
```

The freeze binds source and case manifests, acoustic inventory, candidate and baseline weights, implementation digest, runtime evidence, and development decisions. Do not overwrite a historical freeze or rename an already opened test as fresh. A rerun of the published protocol is reproduction, not a newly unseen evaluation. Fresh-test outcomes cannot change this release's model choice or thresholds.

## Human listening, results and public metadata

```sh
uv run python scripts/build_blind_listening.py --candidate artifacts/releases/v3-candidate/model.pt --study v3-final-listening --device cpu --environment-root "$V3_ACOUSTICS" --selection-plan reports/v3-final-listening-plan.json
uv run tse serve --device mps --port 8000
```

The form is at `http://127.0.0.1:8000/gallery/v3-final-listening/`. Only real submissions count as subjective ratings. Do not generate test ratings in a real study or change A/B identities/audio after rating begins. The early continuation study retains the difficult one-pair examples and the reviewer's negative feedback. The final study follows fixed positions across six different development mixtures. Neither is a population-level listening test.

After scoring and listening preparation:

```sh
uv run python scripts/build_v3_report.py
uv run python scripts/snapshot_v3_metadata.py --release artifacts/releases/v3-candidate/model.json
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run pytest -q
uv build
```

The report builder refuses mismatched freeze, test, comparison or runtime evidence. It explicitly reports pending human ratings. Metadata includes public corpus identities, case recipes and training records, excluding audio, checkpoints, individual ratings, browser IDs and hidden listening keys. Checkpoint provenance links the source experiment and older project initialization.

Promote only the development-eligible artifact after delivery verification: preserve v0.2.0, copy the exact exported bytes to the chosen release/default paths, restart the service, regenerate the development gallery, and verify `/model` plus actual browser extraction and download. If the development gates failed, keep v0.2.0 as default and present the candidate as a research result. Do not use test scores to reverse that choice.
