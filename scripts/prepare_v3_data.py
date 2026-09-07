#!/usr/bin/env python3
"""Prepare unused speech evaluation and recording-disjoint acoustic resources."""

import argparse
import hashlib
import json
import shutil
import tarfile
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

import soundfile as sf

from tse.data import SpeechCorpus, audit_records, build_cases
from tse.utils import atomic_json, sha256

SCENARIOS = ("clean", "noise", "reverb", "partial_overlap", "pauses", "target_absent", "combined")
MD5 = {
    "dev-other": "c8d0bcc9cca99d4f8b62fcc847357931",
    "test-other": "fb5a50374b501bb3bac4815ee91d3135",
}


def rank(value):
    return hashlib.sha256(("tse-v3-20260907:" + value).encode()).hexdigest()


def write_immutable(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Refusing to alter prepared metadata {path}")
    else:
        atomic_json(path, value)


def prepare(archives, speech_root, output, environment_root):
    old = json.loads(Path("data/expanded/manifests/inventory.json").read_text())
    records = [row for row in old["records"] if row["split"] in {"train", "dev"}]
    acquisition = []
    for partition in MD5:
        archive = archives / f"{partition}.tar.gz"
        with archive.open("rb") as handle:
            digest = hashlib.file_digest(handle, "md5").hexdigest()
        if digest != MD5[partition]:
            raise ValueError("LibriSpeech archive differs from publisher MD5")
        acquisition.append(
            {
                "archive": archive.name,
                "url": f"https://openslr.trmal.net/resources/12/{archive.name}",
                "md5": digest,
                "publisher_md5_verified": True,
                "sha256": sha256(archive),
            }
        )
        with tarfile.open(archive) as tar:
            for member in tar:
                parts = PurePosixPath(member.name).parts
                if not member.isfile() or not member.name.endswith(".flac"):
                    continue
                if len(parts) != 5 or parts[:2] != ("LibriSpeech", partition) or ".." in parts:
                    raise ValueError("Unexpected speech archive path")
                if (
                    member.size > 32 * 1024**2
                    or shutil.disk_usage(speech_root).free < 15 * 1024**3 + member.size
                ):
                    raise ValueError("Speech extraction exceeds declared resource reserve")
                target = speech_root.joinpath(*parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                content = tar.extractfile(member).read()
                if target.exists() and sha256(target) != hashlib.sha256(content).hexdigest():
                    raise ValueError("Existing speech file differs")
                if not target.exists():
                    target.write_bytes(content)
                info = sf.info(target)
                if info.samplerate != 16000 or info.channels != 1 or info.duration < 2:
                    continue
                records.append(
                    {
                        "id": target.stem,
                        "speaker": parts[2],
                        "chapter": parts[3],
                        "split": "dev" if partition == "dev-other" else "test",
                        "path": str(PurePosixPath(*parts)),
                        "samples": info.frames,
                        "sample_rate": info.samplerate,
                        "sha256": sha256(target),
                    }
                )
    manifest = output / "inventory.json"
    payload = {
        "records": records,
        "audit": audit_records(records),
        "license": "CC-BY-4.0",
        "source": "LibriSpeech / OpenSLR12",
        "note": "231 previous training identities; dev-clean and dev-other development; previously unscored test-other final evaluation. The opened v2 test identities are excluded.",
    }
    prior_tests = {row["speaker"] for row in old["records"] if row["split"] == "test"}
    original = json.loads(Path("data/manifests/inventory.json").read_text())
    prior_tests |= {row["speaker"] for row in original["records"] if row["split"] == "test"}
    assert not prior_tests & {r["speaker"] for r in records if r["split"] == "test"}
    write_immutable(manifest, payload)
    for name in ("dev-cases.json", "dev-report-cases.json"):
        cases = json.loads((Path("data/expanded/manifests") / name).read_text())
        cases["source_manifest_sha256"] = sha256(manifest)
        write_immutable(output / name, cases)
    for split, base_seed, count in [("dev", 2026090700, 50), ("test", 2026091700, 100)]:
        corpus = SpeechCorpus(speech_root, manifest, split, 4, 5)
        if split == "dev":
            allowed = {r["speaker"] for r in records if "dev-other" in r["path"]}
            corpus.by_speaker = {s: rows for s, rows in corpus.by_speaker.items() if s in allowed}
            corpus.speakers = sorted(corpus.by_speaker)
        temporary = output / f"{split}-base.json"
        if temporary.exists():
            bases = json.loads(temporary.read_text())
        else:
            bases = build_cases(corpus, count, base_seed, temporary)
        expanded = []
        for scenario in SCENARIOS:
            for case in bases["cases"]:
                expanded.append(
                    {
                        **case,
                        "case_id": f"{case['case_id']}-{scenario}",
                        "scenario": scenario,
                        "environment_seed": case["seed"] + 3000000000,
                    }
                )
        write_immutable(
            output / f"{split}-realistic-cases.json",
            {
                **bases,
                "rendering_protocol": "realistic-tse-v1",
                "cases": expanded,
                "evaluation_note": "Repeated source pairs across controlled conditions; target-absence requests exclude SI-SDR and ESTOI.",
            },
        )

    env_archive = archives / "rirs_noises.zip"
    with zipfile.ZipFile(env_archive) as archive:
        rooms = defaultdict(list)
        noises = []
        for info in archive.infolist():
            parts = PurePosixPath(info.filename).parts
            if ".." in parts or PurePosixPath(info.filename).is_absolute():
                raise ValueError("Unsafe acoustic archive path")
            if not info.filename.endswith(".wav"):
                continue
            if len(parts) == 5 and parts[1] == "simulated_rirs":
                rooms["/".join(parts[2:4])].append(info.filename)
            elif len(parts) == 3 and parts[1] == "pointsource_noises":
                noises.append(info.filename)
        selected = [
            (name, "rir", group)
            for group in sorted(rooms, key=rank)[:600]
            for name in sorted(rooms[group], key=rank)[:4]
        ]
        selected += [(name, "noise", name) for name in sorted(noises)]
        env_records = []
        for name, kind, group in selected:
            content = archive.read(name)  # ZipFile verifies each extracted member CRC.
            if len(content) > 128 * 1024**2:
                raise ValueError("Unexpectedly large acoustic file")
            digest = hashlib.sha256(content).hexdigest()
            target = environment_root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and sha256(target) != digest:
                raise ValueError("Existing acoustic resource changed")
            if not target.exists():
                target.write_bytes(content)
            info = sf.info(target)
            if info.samplerate != 16000 or (kind == "noise" and info.duration < 0.5):
                continue
            # Content-identical noise recordings cannot leak across partitions.
            split_group = group if kind == "rir" else digest
            bucket = int(rank(kind + ":" + split_group)[:8], 16) % 10
            partition = "train" if bucket < 8 else "dev" if bucket == 8 else "test"
            env_records.append(
                {
                    "id": name,
                    "kind": kind,
                    "group": split_group,
                    "split": partition,
                    "path": name,
                    "sha256": digest,
                    "samples": info.frames,
                    "channels": info.channels,
                    "sample_rate": info.samplerate,
                }
            )
        license_text = archive.read("RIRS_NOISES/pointsource_noises/LICENSE").decode()
    env_payload = {
        "source": "OpenSLR28 RIRS_NOISES",
        "url": "https://www.openslr.org/28/",
        "archive_sha256": sha256(env_archive),
        "publisher_archive_checksum_verified": False,
        "integrity": "Complete ZIP downloaded; extracted member CRC checked; SHA256 of archive and selected files retained",
        "resource_license": "Apache-2.0 as listed by OpenSLR28",
        "noise_license_text": license_text,
        "selection": "First 600 room groups by fixed SHA256 rank, first four RIRs per room by rank; all noise recordings at least 0.5 seconds",
        "split_rule": "Fixed SHA256 rank modulo 10: 0..7 train, 8 dev, 9 test; grouped by room or noise content hash",
        "records": env_records,
    }
    write_immutable(output / "environments.json", env_payload)
    write_immutable(
        Path("reports/v3-data-preparation.json"),
        {
            "speech_acquisition": acquisition,
            "speech_audit": payload["audit"],
            "speech_manifest_sha256": sha256(manifest),
            "environment_manifest_sha256": sha256(output / "environments.json"),
            "acoustic_counts": {
                split: {
                    kind: sum(r["split"] == split and r["kind"] == kind for r in env_records)
                    for kind in ("noise", "rir")
                }
                for split in ("train", "dev", "test")
            },
            "fresh_test_cases_sha256": sha256(output / "test-realistic-cases.json"),
            "fresh_test_status": "Prepared before model selection; not scored",
            "noise_license_text": license_text,
        },
    )
    print(json.dumps({"speech": payload["audit"], "acoustic_files": len(env_records)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--environment-root", type=Path, required=True)
    parser.add_argument("--speech-root", type=Path, default=Path("data/expanded/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/v3/manifests"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prepare(args.archives, args.speech_root, args.output, args.environment_root)
