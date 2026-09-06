"""Artifact identity and atomic metadata writes."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w") as target:
        json.dump(value, target, indent=2, allow_nan=False)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def git_state() -> dict:
    def run(arguments: list[str]) -> str | None:
        result = subprocess.run(["git", *arguments], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "commit": run(["rev-parse", "HEAD"]),
        "dirty": bool(run(["status", "--porcelain"])),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def source_digest() -> str:
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
