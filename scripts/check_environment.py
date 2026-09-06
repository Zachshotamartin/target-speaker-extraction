#!/usr/bin/env python3
"""Report local prerequisites without installing packages or loading a model."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def read_sysctl(key: str) -> str | None:
    """Read one non-sensitive macOS hardware field when available."""
    if platform.system() != "Darwin":
        return None
    try:
        result = subprocess.run(
            ["/usr/sbin/sysctl", "-n", key],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    free_gib = shutil.disk_usage(project_root).free / 1024**3
    memory_bytes = read_sysctl("hw.memsize")
    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        torch_version = None

    warnings: list[str] = []
    if sys.version_info[:2] != (3, 12):
        warnings.append("The planned project interpreter is Python 3.12.")
    if platform.system() == "Darwin" and platform.machine() != "arm64":
        warnings.append("Use a native ARM interpreter for the Apple Silicon plan.")
    if free_gib < 15:
        warnings.append("Free disk is below the planned 15 GiB reserve.")

    report = {
        "python_version": platform.python_version(),
        "architecture": platform.machine(),
        "operating_system": platform.system(),
        "os_version": platform.mac_ver()[0] or platform.release(),
        "processor": read_sysctl("machdep.cpu.brand_string") or platform.processor(),
        "memory_gib": (
            round(int(memory_bytes) / 1024**3, 2)
            if memory_bytes and memory_bytes.isdigit()
            else None
        ),
        "free_disk_gib": round(free_gib, 2),
        "tools_available": {name: shutil.which(name) is not None for name in ("git", "uv", "gh")},
        "torch_version_in_this_interpreter": torch_version,
        "mps_model_compatibility": "not tested",
        "warnings": warnings,
        "note": "Read-only environment report; no packages, data or models were downloaded.",
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
