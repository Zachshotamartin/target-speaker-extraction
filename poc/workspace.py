"""Resumable bounded uploads and owned assets for the recording workspace."""

import asyncio
import hashlib
import json
import math
import re
import threading
import time
import uuid
from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.responses import Response

CHUNK_BYTES = 1024 * 1024
FILE_BYTES = 128 * CHUNK_BYTES
DISK_BYTES = 768 * CHUNK_BYTES
MAX_SECONDS = 600
ASSET = re.compile(
    r"^(?:(?:original|speaker-[0-3]|reference-[0-3])\.wav|captioned\.mp4|report-[0-3]\.json)$"
)


def identifier(value):
    try:
        parsed = uuid.UUID(value)
        if parsed.version != 4 or str(parsed) != value:
            raise ValueError()
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(404, "This upload or result does not exist.") from None
    return value


def finite(value, minimum, maximum):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise HTTPException(400, "A time or size is outside the supported range.")
    return value


async def json_body(request, limit=256 * 1024):
    if not request.headers.get("content-type", "").startswith("application/json"):
        raise HTTPException(415, "Send JSON options.")
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > limit:
            raise HTTPException(413, "Too many options in this request.")
    try:
        value = json.loads(body)
        if not isinstance(value, dict):
            raise ValueError()
        return value
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(400, "Invalid options.") from None


class Uploads:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.lock, self.items = threading.RLock(), {}
        # Only this service's locked, ephemeral upload area is touched.
        for path in self.directory.iterdir():
            if path.is_file() and re.fullmatch(r"[a-f0-9-]{36}\.input", path.name):
                path.unlink()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.reap, daemon=True)
        self.thread.start()

    def reap(self):
        while not self.stop.wait(30):
            self.expire()

    def expire(self):
        with self.lock:
            for key, item in list(self.items.items()):
                if item["expires_at"] <= time.time():
                    (self.directory / f"{key}.input").unlink(missing_ok=True)
                    del self.items[key]

    def get(self, key, owner):
        identifier(key)
        self.expire()
        item = self.items.get(key)
        if item is None or item["owner"] != owner:
            raise HTTPException(404, "Upload expired. Upload your recording again.")
        return item

    def public(self, key, owner):
        with self.lock:
            item = self.get(key, owner)
            return {k: v for k, v in item.items() if k not in {"owner", "hashes"}}

    def create(self, size, owner):
        size = finite(size, 1, FILE_BYTES)
        if int(size) != size:
            raise HTTPException(400, "Invalid upload size.")
        with self.lock:
            self.expire()
            if sum(i["owner"] == owner for i in self.items.values()) >= 8:
                raise HTTPException(429, "Finish or remove an earlier upload first.")
            used = sum(p.stat().st_size for p in self.directory.parent.rglob("*") if p.is_file())
            reserved = sum(i["size"] - i["received"] for i in self.items.values())
            pending_outputs = (
                sum(1 for p in self.directory.parent.glob("*/options.json")) * 160 * CHUNK_BYTES
            )
            if used + reserved + pending_outputs + size > DISK_BYTES:
                raise HTTPException(
                    429, "The demo is at capacity. Try again after older results expire."
                )
            key = str(uuid.uuid4())
            (self.directory / f"{key}.input").touch(mode=0o600)
            self.items[key] = {
                "id": key,
                "size": int(size),
                "received": 0,
                "chunks": 0,
                "hashes": [],
                "owner": owner,
                "expires_at": time.time() + 3600,
            }
            return self.public(key, owner)

    def write(self, key, index, data, owner):
        with self.lock:
            item = self.get(key, owner)
            expected = min(CHUNK_BYTES, item["size"] - index * CHUNK_BYTES)
            if index < 0 or expected <= 0 or len(data) != expected:
                raise HTTPException(400, "Upload chunk has the wrong size.")
            digest = hashlib.sha256(data).hexdigest()
            if index < item["chunks"]:
                if item["hashes"][index] != digest:
                    raise HTTPException(409, "This chunk differs from the saved upload.")
                return self.public(key, owner)
            if index != item["chunks"]:
                raise HTTPException(409, "Resume at the next missing chunk.")
            with (self.directory / f"{key}.input").open("ab") as handle:
                handle.write(data)
            item["hashes"].append(digest)
            item.update(
                received=item["received"] + len(data),
                chunks=index + 1,
                expires_at=time.time() + 3600,
            )
            return self.public(key, owner)

    def complete(self, key, owner):
        with self.lock:
            item = self.get(key, owner)
            if item["received"] != item["size"]:
                raise HTTPException(409, "The upload is not finished.")
            return self.directory / f"{key}.input"

    def delete(self, key, owner):
        with self.lock:
            self.get(key, owner)
            (self.directory / f"{key}.input").unlink(missing_ok=True)
            del self.items[key]

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
        with self.lock:
            for key in list(self.items):
                self.delete(key, self.items[key]["owner"])


def render_options(body):
    clips, words = body.get("clips"), body.get("words", [])
    if (
        not isinstance(clips, list)
        or not 1 <= len(clips) <= 500
        or not isinstance(words, list)
        or len(words) > 10000
    ):
        raise HTTPException(400, "Choose 1–500 retained passages and at most 10,000 caption words.")
    last, length, normalized = 0, 0, []
    for clip in clips:
        if not isinstance(clip, dict):
            raise HTTPException(400, "Invalid cut.")
        start, end = (
            finite(clip.get("start"), last, MAX_SECONDS),
            finite(clip.get("end"), 0, MAX_SECONDS),
        )
        if end - start < 0.01:
            raise HTTPException(400, "Retained passages must be at least 10 milliseconds.")
        last, length = end, length + end - start
        normalized.append({"start": start, "end": end})
    clean_words = []
    for word in words:
        if (
            not isinstance(word, dict)
            or not isinstance(word.get("text"), str)
            or len(word["text"]) > 120
        ):
            raise HTTPException(400, "Invalid caption word.")
        start, end = finite(word.get("start"), 0, length), finite(word.get("end"), 0, length)
        if end <= start:
            raise HTTPException(400, "Invalid caption timing.")
        clean_words.append({"start": start, "end": end, "text": word["text"]})
    return {
        "kind": "video",
        "clips": normalized,
        "words": clean_words,
        "captions": body.get("captions", True) is True,
    }


def attach_routes(app, directory, models, owner):
    @app.post("/workspace/uploads", status_code=201)
    async def begin_upload(request: Request):
        body = await json_body(request, 1024)
        return app.state.uploads.create(body.get("size"), owner(request))

    @app.get("/workspace/uploads/{key}")
    async def upload_status(key: str, request: Request):
        return app.state.uploads.public(key, owner(request))

    @app.delete("/workspace/uploads/{key}", status_code=204)
    async def delete_upload(key: str, request: Request):
        app.state.uploads.delete(key, owner(request))
        return Response(status_code=204)

    @app.post("/workspace/uploads/{key}/chunks/{index}")
    async def upload_chunk(key: str, index: int, request: Request):
        app.state.uploads.get(key, owner(request))
        data = bytearray()
        async for part in request.stream():
            data.extend(part)
            if len(data) > CHUNK_BYTES:
                raise HTTPException(413, "Each chunk must be at most 1 MiB.")
        return app.state.uploads.write(key, index, data, owner(request))

    @app.post("/workspace/jobs", status_code=202)
    async def submit(request: Request):
        body = await json_body(request)
        identity = owner(request)
        request_id = identifier(body["request_id"]) if "request_id" in body else None
        if request_id:
            with app.state.jobs.lock:
                for job in app.state.jobs.jobs.values():
                    if job.get("request_id") == request_id and job["owner"] == identity:
                        return app.state.jobs.public(job["id"], identity)
        kind = body.get("kind")
        if kind not in {"discover", "extract", "video"}:
            raise HTTPException(400, "Choose voice discovery, extraction, or video export.")
        if kind != "video" and not (models / "manifest.json").exists():
            raise HTTPException(503, "Models are not provisioned.")
        files = {"mixture.input": app.state.uploads.complete(body.get("recording"), identity)}
        uploads = [body.get("recording")]
        options = {"kind": kind}
        if kind == "extract":
            references = body.get("references")
            if not isinstance(references, list) or not 1 <= len(references) <= 4:
                raise HTTPException(400, "Choose one to four voice references.")
            options.update(compare=body.get("compare", True) is True, references=[])
            for i, reference in enumerate(references):
                if not isinstance(reference, dict):
                    raise HTTPException(400, "Invalid voice reference.")
                label = reference.get("label", f"Voice {i + 1}")
                if not isinstance(label, str) or not 1 <= len(label.strip()) <= 60:
                    raise HTTPException(400, "Use a voice name under 60 characters.")
                name = f"reference-{i}.input"
                files[name] = app.state.uploads.complete(reference.get("upload"), identity)
                if files[name].stat().st_size > 4 * CHUNK_BYTES:
                    raise HTTPException(413, "A reference must be under 4 MiB.")
                options["references"].append({"file": name, "label": label.strip()})
                uploads.append(reference.get("upload"))
        if kind == "video":
            options = render_options(body)
            files["edited.input"] = app.state.uploads.complete(body.get("audio"), identity)
            uploads.append(body.get("audio"))
        with app.state.jobs.lock, app.state.uploads.lock:
            if request_id:
                for previous in app.state.jobs.jobs.values():
                    if previous.get("request_id") == request_id and previous["owner"] == identity:
                        return app.state.jobs.public(previous["id"], identity)
            used = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
            # Reserve room for generated tracks/video and the decoded render WAV.
            active = sum(
                j["status"] not in {"ready", "failed", "cancelled", "expired"}
                for j in app.state.jobs.jobs.values()
            )
            if (
                used
                + sum(p.stat().st_size for p in files.values())
                + (active + 1) * 160 * CHUNK_BYTES
                > DISK_BYTES
            ):
                raise HTTPException(429, "The demo is at capacity. Try again shortly.")
            options["request_id"] = request_id
            job = app.state.jobs.submit_workspace(options, files, identity)
            for key in set(uploads):
                app.state.uploads.delete(key, identity)
            return job

    @app.get("/workspace/requests/{key}")
    async def request_status(key: str, request: Request):
        identifier(key)
        identity = owner(request)
        with app.state.jobs.lock:
            for job in app.state.jobs.jobs.values():
                if job.get("request_id") == key and job["owner"] == identity:
                    return app.state.jobs.public(job["id"], identity)
        raise HTTPException(404, "No job was submitted with this request.")

    @app.get("/workspace/jobs/{key}")
    async def status(key: str, request: Request):
        return app.state.jobs.public(identifier(key), owner(request))

    @app.delete("/workspace/jobs/{key}")
    async def cancel(key: str, request: Request):
        return await asyncio.to_thread(app.state.jobs.cancel, identifier(key), owner(request))

    def asset_file(key, asset, request):
        job = app.state.jobs.public(identifier(key), owner(request))
        if job["status"] != "ready":
            raise HTTPException(409, "The result is not ready.")
        if not ASSET.fullmatch(asset):
            raise HTTPException(404, "Asset does not exist.")
        path = directory / key / asset
        if not path.is_file():
            raise HTTPException(404, "Asset does not exist.")
        return path

    @app.get("/workspace/jobs/{key}/assets/{asset}/info")
    async def asset_info(key: str, asset: str, request: Request):
        path = asset_file(key, asset, request)
        size = path.stat().st_size
        return {
            "bytes": size,
            "chunks": math.ceil(size / CHUNK_BYTES),
            "type": "application/json"
            if asset.endswith(".json")
            else "video/mp4"
            if asset.endswith(".mp4")
            else "audio/wav",
        }

    @app.get("/workspace/jobs/{key}/assets/{asset}/chunks/{index}")
    async def asset_chunk(key: str, asset: str, index: int, request: Request):
        path = asset_file(key, asset, request)
        if index < 0 or index * CHUNK_BYTES >= path.stat().st_size:
            raise HTTPException(404, "Chunk does not exist.")
        with path.open("rb") as handle:
            handle.seek(index * CHUNK_BYTES)
            data = handle.read(CHUNK_BYTES)
        return Response(data, media_type="application/octet-stream")
