"""Local controls for one explicitly registered full-data training run."""

import fcntl
import json
import subprocess
import sys
from pathlib import Path

from tse.utils import atomic_json

_WORKERS = {}


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def process_alive(pid):
    if not pid:
        return False
    worker = _WORKERS.get(pid)
    if worker is not None:
        return worker.poll() is None
    result = subprocess.run(["ps", "-p", str(pid), "-o", "args="], capture_output=True, text=True)
    return result.returncode == 0 and "scripts/train_full_dataset.py" in result.stdout


def full_progress(workspace=Path(".")):
    pointer = read(workspace / "artifacts/full-training-active.json")
    run = Path(pointer["run"]) if pointer.get("run") else None
    state = read(run / "status.json") if run else {}
    alive = process_alive(pointer.get("pid"))
    if not alive and state.get("status") in {"starting", "training", "validating"}:
        state["status"] = "interrupted"
    results = {}
    for kind in ("monitor", "full"):
        for version in ("latest", "best"):
            value = read(run / f"{version}-{kind}-validation.json") if run else {}
            results[f"{version}_{kind}"] = {k: v for k, v in value.items() if k != "rows"}
    return {
        "status": "not_started",
        **state,
        "process_alive": alive,
        "registered": bool(pointer),
        "pause_requested": bool(run and (run / "pause.request").exists()),
        "can_resume": bool(run and (run / "latest.pt").exists())
        and not alive
        and state.get("status") != "complete",
        **results,
    }


def control(action, workspace=Path(".")):
    workspace = workspace.resolve()
    pointer_path = workspace / "artifacts/full-training-active.json"
    pointer = read(pointer_path)
    if not pointer:
        raise ValueError("No full-data run has been registered")
    run = Path(pointer["run"])
    # Do not create missing mount points when the external SSD is disconnected.
    if not Path(pointer["root"]).is_dir() or not run.is_dir():
        raise ValueError("Reconnect the dataset SSD before controlling this run")
    with (run / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        pointer = read(pointer_path)
        alive = process_alive(pointer.get("pid"))
        if action == "pause":
            if alive:
                (run / "pause.request").touch()
            return {"status": "pause_requested" if alive else "already_stopped"}
        if action != "resume":
            raise ValueError("Choose pause or resume")
        if alive:
            return {"status": "already_running", "pid": pointer["pid"]}
        if read(run / "status.json").get("status") == "complete":
            return {"status": "complete"}
        (run / "pause.request").unlink(missing_ok=True)
        command = [
            sys.executable,
            "scripts/train_full_dataset.py",
            "--root",
            pointer["root"],
            "--manifest",
            pointer["manifest"],
            "--config",
            pointer["config"],
            "--run",
            str(run),
            "--device",
            pointer["device"],
        ]
        minutes = pointer.get("session_minutes")
        if minutes is not None:
            command.extend(["--minutes", str(minutes)])
        if (run / "latest.pt").exists():
            command.append("--resume")
        with (run / "process.log").open("ab") as log:
            worker = subprocess.Popen(
                command,
                cwd=workspace,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        _WORKERS[worker.pid] = worker
        atomic_json(pointer_path, {**pointer, "pid": worker.pid})
        if sys.platform == "darwin":
            # Idle sleep only, scoped to the trainer; closing the lid can still suspend the Mac.
            with (run / "caffeinate.log").open("ab") as log:
                helper = subprocess.Popen(
                    ["caffeinate", "-i", "-w", str(worker.pid)],
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
            _WORKERS[helper.pid] = helper
        return {"status": "started", "pid": worker.pid}
