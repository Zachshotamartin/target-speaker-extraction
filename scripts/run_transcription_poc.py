"""Start only the local product preview and transcription API; never touch training."""

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def occupied(port):
    with socket.socket() as connection:
        return connection.connect_ex(("127.0.0.1", port)) == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ui-port", type=int, default=5295, choices=[5295, 5297])
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    runtime = ROOT / ".venv-poc/bin/python"
    manifest = ROOT / "artifacts/poc/models/manifest.json"
    if not runtime.exists() or not manifest.exists():
        raise SystemExit("Provision .venv-poc and the models first; see poc/README.md.")
    if occupied(args.ui_port):
        raise SystemExit(
            f"Port {args.ui_port} is already serving. Open http://127.0.0.1:{args.ui_port}/#transcribe or use the other preview port. No processes were stopped."
        )
    node = shutil.which("node")
    bundled = Path("/opt/homebrew/opt/node@22/bin/node")
    if bundled.exists():
        node = str(bundled)
    if not node or not (ROOT / "site/dist/index.html").exists():
        raise SystemExit("Install Node 22, then run npm ci and npm run build in site/.")
    reuse_api = occupied(5296)
    if reuse_api:
        try:
            with urllib.request.urlopen("http://127.0.0.1:5296/health", timeout=3) as response:
                info = json.load(response)
            if info.get("workspace", {}).get("version") != 1:
                raise ValueError("The existing service needs the recording workspace update")
            if info.get("models") != json.loads(manifest.read_text()):
                raise ValueError("Different model manifest")
        except Exception:
            raise SystemExit(
                "Port 5296 is occupied by another or unready service. No processes were stopped."
            ) from None
    log_dir = ROOT / "artifacts/poc"
    if args.detach:
        with (log_dir / "launcher.log").open("ab") as log:
            process = subprocess.Popen(
                [str(runtime), str(Path(__file__)), "--ui-port", str(args.ui_port)],
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
        print(
            f"Local preview starting at http://127.0.0.1:{args.ui_port}/#transcribe (supervisor PID {process.pid})."
        )
        return
    children = []
    stopping = False

    def stop(_signal=None, _frame=None):
        nonlocal stopping
        stopping = True
        for child in children:
            if child.poll() is None:
                child.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT)}
    try:
        if not reuse_api:
            children.append(
                subprocess.Popen(
                    [
                        str(runtime),
                        "-m",
                        "uvicorn",
                        "poc.server:create_app",
                        "--factory",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        "5296",
                        "--no-access-log",
                    ],
                    cwd=ROOT,
                    env=env,
                )
            )
        children.append(
            subprocess.Popen(
                [
                    node,
                    str(ROOT / "site/node_modules/vite/bin/vite.js"),
                    "preview",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(args.ui_port),
                    "--strictPort",
                ],
                cwd=ROOT / "site",
            )
        )
        (log_dir / "services.json").write_text(
            json.dumps(
                {
                    "supervisor_pid": os.getpid(),
                    "owned_child_pids": [c.pid for c in children],
                    "ui_port": args.ui_port,
                    "reused_api": reuse_api,
                }
            )
        )
        print(
            f"Open http://127.0.0.1:{args.ui_port}/#transcribe. Ctrl+C stops only these POC services.",
            flush=True,
        )
        while not stopping and all(child.poll() is None for child in children):
            time.sleep(0.5)
    finally:
        stop()
        for child in children:
            try:
                child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=2)


if __name__ == "__main__":
    main()
