"""Bounded acquisition of public LibriSpeech audio; no research code is downloaded."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import time
import urllib.request
from pathlib import Path, PurePosixPath


def acquire(
    root: Path,
    split: str,
    speakers: int = 60,
    utterances: int = 50,
    minimum_free_gib: float = 15,
) -> dict:
    if split not in {"train-clean-100", "dev-clean", "test-clean"}:
        raise ValueError("Only the three declared clean corpus partitions are supported")
    if speakers < 2 or utterances < 3:
        raise ValueError("Acquisition needs at least two speakers and three utterances each")
    root.mkdir(parents=True, exist_ok=True)
    provenance = root / f"{split}.acquisition.json"
    if provenance.exists():
        previous = json.loads(provenance.read_text())
        if previous["speaker_limit"] == speakers and previous["utterance_limit"] == utterances:
            for record in previous["files"]:
                path = root / record["path"]
                if (
                    not path.is_file()
                    or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]
                ):
                    raise ValueError(f"Acquired source has changed or is missing: {record['path']}")
            return previous
        raise ValueError("A different selection already exists; use a separate data root")
    url = f"https://openslr.trmal.net/resources/12/{split}.tar.gz"
    counts: dict[str, int] = {}
    files = []
    started = time.monotonic()
    print(json.dumps({"event": "acquisition_start", "split": split, "url": url}), flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "target-speaker-extraction/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        with tarfile.open(fileobj=response, mode="r|gz") as archive:
            for member in archive:
                parts = PurePosixPath(member.name).parts
                if not member.isfile() or not member.name.endswith(".flac"):
                    continue
                if len(parts) != 5 or parts[0] != "LibriSpeech" or parts[1] != split:
                    raise ValueError("Unexpected archive path")
                if any(part in {"..", ".", ""} for part in parts):
                    raise ValueError("Unsafe archive path")
                speaker = parts[2]
                if speaker not in counts and len(counts) >= speakers:
                    break
                if speaker not in counts:
                    counts[speaker] = 0
                    print(
                        json.dumps({"event": "speaker", "split": split, "speakers": len(counts)}),
                        flush=True,
                    )
                if counts[speaker] >= utterances:
                    continue
                if shutil.disk_usage(root).free < minimum_free_gib * 1024**3 + member.size:
                    raise OSError(
                        "Acquisition stopped to preserve the configured free-disk reserve"
                    )
                if member.size > 32 * 1024**2:
                    raise ValueError("Unexpectedly large utterance")
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError("Archive member cannot be read")
                payload = source.read()
                if len(payload) != member.size:
                    raise ValueError("Truncated archive member")
                destination = root.joinpath(*parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".partial")
                temporary.write_bytes(payload)
                temporary.replace(destination)
                files.append(
                    {
                        "path": str(PurePosixPath(*parts)),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "bytes": len(payload),
                    }
                )
                counts[speaker] += 1
    result = {
        "source": "LibriSpeech / OpenSLR SLR12",
        "url": url,
        "license": "CC-BY-4.0",
        "attribution": "Vassil Panayotov, Guoguo Chen, Daniel Povey and Sanjeev Khudanpur; LibriSpeech (2015).",
        "split": split,
        "speaker_limit": speakers,
        "utterance_limit": utterances,
        "selection": "First archive speakers and first utterances per speaker; deterministic bounded pilot selection",
        "archive_checksum_verified": False,
        "integrity_note": "Stream was stopped after the selected members; SHA-256 identities are recorded per extracted file, not publisher-verified archive integrity.",
        "speakers": counts,
        "files": files,
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    temporary = provenance.with_suffix(".partial")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(provenance)
    print(
        json.dumps(
            {
                "event": "acquisition_complete",
                "split": split,
                "files": len(files),
                "seconds": result["elapsed_seconds"],
            }
        ),
        flush=True,
    )
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=["train-clean-100", "dev-clean", "test-clean"])
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--speakers", type=int, default=60)
    parser.add_argument("--utterances", type=int, default=50)
    arguments = parser.parse_args()
    acquire(arguments.root, arguments.split, arguments.speakers, arguments.utterances)
