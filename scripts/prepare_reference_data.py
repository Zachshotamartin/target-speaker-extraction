#!/usr/bin/env python3
"""Prepare official 16 kHz/min/clean Libri2Mix on an external volume.

Downloads public metadata and missing complete dev/test speech archives. Reuses
the existing complete training speech acquisition; does not import research code.
"""

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

import soundfile as sf

from tse.librimix import PROTOCOL, render_clean
from tse.utils import atomic_json, sha256

LIBRIMIX = "b898ca375c03e4039cdb3dbe3dd485aad58ec7d1"
ENROLLMENT = "91af02cc617afa35fedfbdbf32533012cd0a8672"
PARTITIONS = {"train": "train-clean-100", "dev": "dev-clean", "test": "test-clean"}
PUBLISHER_MD5 = {
    "dev-clean": "42e2234ba48799c1f50f24a7926300a1",
    "test-clean": "32fa31d27d2e1cad72775fee3f4849a9",
}


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    partial = path.with_suffix(path.suffix + ".download")
    subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--retry",
            "3",
            "--continue-at",
            "-",
            "--output",
            str(partial),
            url,
        ],
        check=True,
    )
    partial.replace(path)


def prepare(root, training_root, manifest):
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < 35 * 1024**3:
        raise OSError("Preparation requires 35 GiB free on the selected dataset volume")
    if manifest.exists():
        existing = json.loads(manifest.read_text())
        for rows in existing["splits"].values():
            for row in rows:
                for record in [row["mixture"], *row["sources"]]:
                    if sha256(root / record["path"]) != record["sha256"]:
                        raise ValueError("Prepared audio differs from its immutable manifest")
        print("Verified existing complete dataset", flush=True)
        return
    metadata = root / "metadata"
    origins, archives = [], []
    for split, partition in PARTITIONS.items():
        url = f"https://raw.githubusercontent.com/JorisCos/LibriMix/{LIBRIMIX}/metadata/Libri2Mix/libri2mix_{partition}.csv"
        path = metadata / f"libri2mix_{partition}.csv"
        download(url, path)
        origins.append({"url": url, "sha256": sha256(path)})
        if split == "train":
            continue
        url = f"https://raw.githubusercontent.com/BUTSpeechFIT/speakerbeam/{ENROLLMENT}/egs/libri2mix/data/wav8k/min/{split}/map_mixture2enrollment"
        path = metadata / f"{split}-enrollment.txt"
        download(url, path)
        origins.append({"url": url, "sha256": sha256(path)})
        archive = root / "archives" / f"{partition}.tar.gz"
        download(f"https://openslr.trmal.net/resources/12/{partition}.tar.gz", archive)
        with archive.open("rb") as handle:
            digest = hashlib.file_digest(handle, "md5").hexdigest()
        if digest != PUBLISHER_MD5[partition]:
            raise ValueError("Speech archive failed publisher MD5 verification")
        archives.append({"partition": partition, "md5": digest, "sha256": sha256(archive)})
        with tarfile.open(archive) as source:
            for member in source:
                if not member.isfile() or not member.name.endswith(".flac"):
                    continue
                parts = PurePosixPath(member.name).parts
                if len(parts) != 5 or parts[:2] != ("LibriSpeech", partition) or ".." in parts:
                    raise ValueError("Unexpected speech archive path")
                path = root / "speech" / member.name
                data = source.extractfile(member).read()
                if path.exists() and path.read_bytes() != data:
                    raise ValueError("Existing speech source differs")
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    path.write_bytes(data)
    acquisition = json.loads((training_root / "train-clean-100.acquisition.json").read_text())
    if len(acquisition["speakers"]) != 251 or len(acquisition["files"]) != 28539:
        raise ValueError("This recipe requires the complete train-clean-100 speech collection")
    for record in acquisition["files"]:
        if sha256(training_root / record["path"]) != record["sha256"]:
            raise ValueError("Existing training acquisition changed")
    splits, enrollments = {}, {}
    for split, partition in PARTITIONS.items():
        rows = list(csv.DictReader((metadata / f"libri2mix_{partition}.csv").open()))
        if len(rows) != (13900 if split == "train" else 3000):
            raise ValueError("Unexpected official mixture count")
        prepared = []
        speech_root = (
            training_root / "LibriSpeech" if split == "train" else root / "speech/LibriSpeech"
        )
        for index, row in enumerate(rows):
            signals = [
                sf.read(speech_root / row[f"source_{side}_path"], dtype="float32")[0]
                for side in (1, 2)
            ]
            sources, mix = render_clean(
                *signals, [float(row[f"source_{side}_gain"]) for side in (1, 2)]
            )
            records = []
            for folder, signal in zip(("s1", "s2", "mix_clean"), [*sources, mix], strict=True):
                relative = f"{split}/{folder}/{row['mixture_ID']}.wav"
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    existing, rate = sf.read(path, dtype="float32")
                    if (
                        rate != 16000
                        or existing.shape != signal.shape
                        or (existing != signal).any()
                    ):
                        raise ValueError(f"Incomplete dataset contains changed audio: {relative}")
                else:
                    sf.write(path, signal, 16000, subtype="PCM_16")
                record = {"path": relative, "sha256": sha256(path)}
                if folder != "mix_clean":
                    utterance = Path(row[f"source_{folder[-1]}_path"]).stem
                    record.update(utterance=utterance, speaker=utterance.split("-")[0])
                records.append(record)
            prepared.append(
                {
                    "id": row["mixture_ID"],
                    "samples": len(mix),
                    "sources": records[:2],
                    "mixture": records[2],
                }
            )
            if index % 500 == 0:
                print(
                    json.dumps({"event": "render", "split": split, "mixtures": index}), flush=True
                )
        splits[split] = prepared
        if split != "train":
            mapping = {}
            for line in (metadata / f"{split}-enrollment.txt").read_text().splitlines():
                mixture_id, utterance, enrollment = line.split()
                side = mixture_id.split("_").index(utterance)
                mapping[f"{mixture_id}:{side}"] = f"{split}/{enrollment}.wav"
            if len(mapping) != 6000:
                raise ValueError("Expected both target enrollments for every evaluation mixture")
            enrollments[split] = mapping
    speaker_sets = {
        split: {s["speaker"] for row in rows for s in row["sources"]}
        for split, rows in splits.items()
    }
    if any(
        speaker_sets[a] & speaker_sets[b]
        for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))
    ):
        raise ValueError("Official partitions unexpectedly share a speaker")
    payload = {
        "protocol": PROTOCOL,
        "sample_rate": 16000,
        "splits": splits,
        "enrollments": enrollments,
        "sources": origins,
        "verified_archives": archives,
        "training_acquisition_sha256": sha256(training_root / "train-clean-100.acquisition.json"),
        "note": "Official mixture gains/identities and enrollment mapping. Independent PCM16 rendering. Existing train acquisition has file identities but no publisher-verified archive. Test-clean contains historical evaluated identities, so benchmark scoring will not be called a fresh test.",
    }
    atomic_json(manifest, payload)
    atomic_json(
        Path("reports/reference-data-preparation.json"),
        {
            "status": "complete",
            "manifest_sha256": sha256(manifest),
            "protocol": PROTOCOL,
            "counts": {
                split: {
                    "mixtures": len(rows),
                    "target_requests": 2 * len(rows),
                    "speakers": len(speaker_sets[split]),
                    "hours": sum(r["samples"] for r in rows) / 16000 / 3600,
                }
                for split, rows in splits.items()
            },
            "sources": origins,
            "verified_archives": archives,
            "note": payload["note"],
        },
    )
    print("Completed official clean mixture preparation", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, default=Path("data/expanded/raw"))
    parser.add_argument("--manifest", type=Path, default=Path("data/reference/manifest.json"))
    arguments = parser.parse_args()
    prepare(arguments.root, arguments.training_root, arguments.manifest)
