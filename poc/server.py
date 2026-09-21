"""Bounded transcription jobs, local by default and isolated from training."""

import asyncio
import hmac
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from email.parser import BytesParser
from email.policy import default
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response

from poc.common import MAX_BYTES, ROOT, TERMINAL, TTL_SECONDS, atomic_json, decode, export_text, wav


class Jobs:
    def __init__(self, directory, models, timeout=600, memory_mib=4096):
        self.directory, self.models = Path(directory), Path(models)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.directory, 0o700)
        # Ephemeral jobs cannot resume. Remove only this service's UUID directories.
        for entry in self.directory.iterdir():
            if entry.is_dir() and not entry.is_symlink():
                try:
                    uuid.UUID(entry.name)
                except ValueError:
                    continue
                shutil.rmtree(entry)
        self.timeout, self.memory_mib = timeout, memory_mib
        self.jobs, self.lock = {}, threading.RLock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def submit(self, mixture, reference, compare, owner=None):
        with self.lock:
            if sum(j["status"] not in TERMINAL for j in self.jobs.values()) >= 2:
                raise HTTPException(
                    429, "One job is running and one is waiting. Try again shortly."
                )
            key = str(uuid.uuid4())
            directory = self.directory / key
            directory.mkdir(mode=0o700)
            (directory / "mixture.input").write_bytes(mixture)
            (directory / "reference.input").write_bytes(reference)
            atomic_json(directory / "options.json", {"compare": compare})
            self.jobs[key] = {
                "id": key,
                "status": "queued",
                "stage": "Waiting for the CPU worker",
                "created_at": time.time(),
                "process": None,
                "owner": owner,
            }
            return self.public(key)

    def public(self, key, owner=None):
        with self.lock:
            if key not in self.jobs or (owner is not None and self.jobs[key]["owner"] != owner):
                raise HTTPException(
                    404, "This result expired or does not exist. Submit the recording again."
                )
            job = {k: v for k, v in self.jobs[key].items() if k not in {"process", "owner"}}
            directory = self.directory / key
            if job["status"] == "running" and (directory / "progress.json").exists():
                job.update(json.loads((directory / "progress.json").read_text()))
            if job["status"] == "ready":
                job["result"] = json.loads((directory / "result.json").read_text())
            return job

    def cancel(self, key, owner=None):
        with self.lock:
            if key not in self.jobs or (owner is not None and self.jobs[key]["owner"] != owner):
                raise HTTPException(404, "Job does not exist.")
            job = self.jobs[key]
            job.update(status="cancelled", stage="Deleted", expires_at=time.time() + TTL_SECONDS)
            process = job["process"]
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            shutil.rmtree(self.directory / key, ignore_errors=True)
            return self.public(key)

    def loop(self):
        while not self.stop.wait(0.3):
            with self.lock:
                now = time.time()
                for key, job in list(self.jobs.items()):
                    if job.get("expires_at", now + 1) <= now:
                        shutil.rmtree(self.directory / key, ignore_errors=True)
                        del self.jobs[key]
                active = next((j for j in self.jobs.values() if j["status"] == "running"), None)
                if active:
                    process = active["process"]
                    code = process.poll()
                    directory = self.directory / active["id"]
                    if code is not None:
                        success = code == 0 and (directory / "result.json").exists()
                        error = "The inference worker failed. Retry or check local model setup."
                        if (directory / "error.json").exists():
                            error = json.loads((directory / "error.json").read_text())["message"]
                        active.update(
                            status="ready" if success else "failed",
                            stage="Ready" if success else error,
                            expires_at=now + TTL_SECONDS,
                        )
                        if not success:
                            shutil.rmtree(directory, ignore_errors=True)
                        continue
                    try:
                        rss = int(
                            subprocess.check_output(
                                ["/bin/ps", "-o", "rss=", "-p", str(process.pid)], timeout=2
                            ).strip()
                            or 0
                        )
                    except (subprocess.SubprocessError, ValueError):
                        rss = 0
                    if now - active["started_at"] > self.timeout or rss > self.memory_mib * 1024:
                        reason = (
                            "The job reached its 10-minute limit."
                            if now - active["started_at"] > self.timeout
                            else "The job exceeded its 4 GiB CPU memory budget. Try a shorter clip."
                        )
                        self.cancel(active["id"])
                        active.update(status="failed", stage=reason)
                    continue
                queued = next((j for j in self.jobs.values() if j["status"] == "queued"), None)
                if queued:
                    env = dict(os.environ)
                    env.update(
                        PYTHONPATH=str(ROOT / "src") + os.pathsep + str(ROOT),
                        HF_HUB_OFFLINE="1",
                        HF_HUB_DISABLE_TELEMETRY="1",
                        OMP_NUM_THREADS="1",
                        OPENBLAS_NUM_THREADS="1",
                        VECLIB_MAXIMUM_THREADS="1",
                    )
                    try:
                        process = subprocess.Popen(
                            [
                                sys.executable,
                                "-m",
                                "poc.worker",
                                "--job",
                                str(self.directory / queued["id"]),
                                "--models",
                                str(self.models),
                            ],
                            cwd=ROOT,
                            env=env,
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        queued.update(
                            status="running",
                            stage="Loading local models",
                            started_at=now,
                            process=process,
                        )
                    except OSError:
                        queued.update(
                            status="failed",
                            stage="Could not start the local worker.",
                            expires_at=now + TTL_SECONDS,
                        )
                        shutil.rmtree(self.directory / queued["id"], ignore_errors=True)

    def close(self):
        self.stop.set()
        with self.lock:
            for key in list(self.jobs):
                self.cancel(key)
        self.thread.join(timeout=3)


def parse_upload(body, content_type):
    if len(body) > MAX_BYTES:
        raise HTTPException(413, "The combined upload must be under 4 MiB.")
    if not content_type.startswith("multipart/form-data"):
        raise HTTPException(415, "Send multipart audio fields named mixture and reference.")
    message = BytesParser(policy=default).parsebytes(
        b"Content-Type: "
        + content_type.encode("ascii", "replace")
        + b"\r\nMIME-Version: 1.0\r\n\r\n"
        + body
    )
    if not message.is_multipart() or message.defects:
        raise HTTPException(400, "Invalid multipart upload.")
    fields = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if name not in {"mixture", "reference", "compare"} or name in fields or part.is_multipart():
            raise HTTPException(
                400, "Use one mixture, one reference, and an optional compare field."
            )
        fields[name] = part.get_payload(decode=True)
    if not fields.get("mixture") or not fields.get("reference"):
        raise HTTPException(400, "Choose a recording and a reference voice.")
    if fields.get("compare", b"true") not in {b"true", b"false"}:
        raise HTTPException(400, "compare must be true or false.")
    return fields["mixture"], fields["reference"], fields.get("compare", b"true") == b"true"


def demo_items():
    return json.loads((ROOT / "site/src/snapshot.json").read_text())["items"]


def demo_audio(index, track):
    items = demo_items()
    if (
        index < 0
        or index >= len(items)
        or track not in {"mixture", "reference", "target", "absent", "silence"}
    ):
        raise HTTPException(404, "Example does not exist.")
    item = items[index]
    if track == "absent":
        item = next(
            i
            for i in items
            if i["conversation"] == item["conversation"] and i["voice"] != item["voice"]
        )
        track = "target"
    path = (
        ROOT
        / "site/public"
        / item["tracks"]["mixture" if track == "silence" else track]["src"].lstrip("/")
    )
    if not path.resolve().is_relative_to((ROOT / "site/public/assets/one-voice").resolve()):
        raise HTTPException(500, "Invalid example configuration.")
    data = path.read_bytes()
    return wav(np.zeros_like(decode(data))) if track == "silence" else data


def create_app(directory=None, models=None, *, access_token=None):
    directory = Path(directory or ROOT / "artifacts/poc/jobs")
    models = Path(models or ROOT / "artifacts/poc/models")
    if access_token is not None and len(access_token) < 32:
        raise ValueError("Hosted transcription requires a secret of at least 32 characters.")

    def owner(request):
        return request.state.owner if access_token else None

    @asynccontextmanager
    async def lifespan(app):
        # A second API must not delete or share the first API's job directory.
        import fcntl

        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (directory / ".service.lock").open("w") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            app.state.jobs = Jobs(directory, models)
            try:
                yield
            finally:
                app.state.jobs.close()

    app = FastAPI(
        title="One Voice transcription",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.middleware("http")
    async def access_guard(request, call_next):
        from urllib.parse import urlsplit

        host = request.headers.get("host", "").split(":")[0]
        origin = request.headers.get("origin")
        if access_token:
            if not hmac.compare_digest(
                request.headers.get("authorization", "").encode(),
                ("Bearer " + access_token).encode(),
            ):
                return Response(status_code=401)
            path = request.url.path.removeprefix(request.scope.get("root_path", ""))
            if path.startswith("/transcriptions"):
                try:
                    value = request.headers.get("x-onevoice-owner", "")
                    parsed = uuid.UUID(value)
                    if parsed.version != 4 or str(parsed) != value:
                        raise ValueError("Invalid browser session")
                    request.state.owner = value
                except ValueError:
                    return Response(status_code=401)
        elif host not in {"127.0.0.1", "localhost", "testserver"} or (
            origin and urlsplit(origin).hostname not in {"localhost", "127.0.0.1"}
        ):
            return Response("This service is local only.", status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/health")
    async def health():
        manifest = models / "manifest.json"
        return {
            "ready": manifest.exists(),
            "processing_location": "hosted" if access_token else "local",
            "models": json.loads(manifest.read_text()) if manifest.exists() else None,
            "limits": {
                "seconds": 30,
                "reference_seconds": [3, 10],
                "combined_bytes": MAX_BYTES,
                "active_jobs": 1,
                "waiting_jobs": 1,
                "result_minutes": 15,
                "worker_memory_mib": 4096,
            },
            "setup": "Run python -m poc.provision; see poc/README.md",
        }

    @app.get("/demos")
    async def demos():
        return [
            {
                "id": i,
                "conversation": item["conversation"],
                "voice": item["voice"],
                "case_id": item["caseId"],
            }
            for i, item in enumerate(demo_items())
        ]

    @app.get("/demos/{index}/{track}")
    async def example(index: int, track: str):
        return Response(await asyncio.to_thread(demo_audio, index, track), media_type="audio/wav")

    @app.post("/transcriptions", status_code=202)
    async def transcribe(request: Request):
        if not (models / "manifest.json").exists():
            raise HTTPException(503, "Local models are not provisioned. See poc/README.md.")
        body = bytearray()
        async for part in request.stream():
            body.extend(part)
            if len(body) > MAX_BYTES:
                raise HTTPException(413, "The combined upload must be under 4 MiB.")
        mixture, reference, compare = parse_upload(
            bytes(body), request.headers.get("content-type", "")
        )
        return app.state.jobs.submit(mixture, reference, compare, owner(request))

    @app.get("/transcriptions/{key}")
    async def status(key: str, request: Request):
        return app.state.jobs.public(key, owner(request))

    @app.delete("/transcriptions/{key}")
    async def cancel(key: str, request: Request):
        return await asyncio.to_thread(app.state.jobs.cancel, key, owner(request))

    @app.get("/transcriptions/{key}/audio/{track}")
    async def audio(key: str, track: str, request: Request):
        if app.state.jobs.public(key, owner(request))["status"] != "ready":
            raise HTTPException(409, "Audio is not ready.")
        if track not in {"original", "extracted"}:
            raise HTTPException(404, "Audio does not exist.")
        return FileResponse(
            directory / key / f"{track}.wav",
            media_type="audio/wav",
            filename=f"one-voice-{track}.wav",
        )

    @app.get("/transcriptions/{key}/export/{kind}")
    async def export(key: str, kind: str, request: Request):
        job = app.state.jobs.public(key, owner(request))
        if job["status"] != "ready":
            raise HTTPException(409, "Transcript is not ready.")
        if kind not in {"txt", "srt", "json"}:
            raise HTTPException(404, "Choose txt, srt or json.")
        return Response(
            export_text(job["result"], kind),
            media_type="application/json" if kind == "json" else "text/plain",
            headers={"Content-Disposition": f'attachment; filename="one-voice.{kind}"'},
        )

    return app
