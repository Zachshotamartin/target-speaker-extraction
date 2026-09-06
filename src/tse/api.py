"""Bounded loopback API and local audio interface."""

from __future__ import annotations

import io
import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from tse.inference import Extractor

LOGGER = logging.getLogger("tse.api")
MAX_UPLOAD = 24 * 1024**2


class UploadLimitMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        origin = headers.get(b"origin")
        expected = scope.get("scheme", "http").encode() + b"://" + headers.get(b"host", b"")
        if origin and origin != expected:
            await JSONResponse(
                {"detail": "Cross-origin uploads are not accepted"}, status_code=403
            )(scope, receive, send)
            return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > MAX_UPLOAD:
                await JSONResponse({"detail": "Combined upload exceeds 24 MiB"}, status_code=413)(
                    scope, receive, send
                )
                return
            if not message.get("more_body", False):
                break
        delivered = False

        async def bounded_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, bounded_receive, send)


def create_app(
    checkpoint: Path | None = None, device: str = "cpu", examples_root: Path | None = None
) -> FastAPI:
    checkpoint = checkpoint or Path(os.environ.get("TSE_CHECKPOINT", "artifacts/releases/model.pt"))
    gate = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.extractor = None
        app.state.load_error = None
        if checkpoint.is_file():
            try:
                extractor = Extractor(checkpoint, device)
                extractor.warmup()
                app.state.extractor = extractor
            except Exception as error:
                app.state.load_error = type(error).__name__
                LOGGER.exception("Model startup failed")
        else:
            app.state.load_error = "No trained checkpoint is configured"
        yield
        app.state.extractor = None

    app = FastAPI(title="Target Speaker Extraction", version="0.1.1", lifespan=lifespan)
    app.state.gate = gate
    app.add_middleware(UploadLimitMiddleware)
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver", "[::1]"]
    )

    @app.get("/health")
    def health():
        return {"status": "alive"}

    @app.get("/ready")
    def ready():
        if app.state.extractor is None:
            return JSONResponse({"ready": False, "detail": app.state.load_error}, status_code=503)
        return {"ready": True}

    @app.get("/model")
    def model_info():
        if app.state.extractor is None:
            return JSONResponse({"ready": False, "detail": app.state.load_error}, status_code=503)
        return app.state.extractor.info()

    @app.post("/extract", response_class=Response)
    def extract(mixture: Annotated[UploadFile, File()], reference: Annotated[UploadFile, File()]):
        if app.state.extractor is None:
            raise HTTPException(503, "A trained model is not ready")
        if not gate.acquire(blocking=False):
            raise HTTPException(429, "Another recording is processing. Try again shortly.")
        try:
            sources = []
            for upload in (mixture, reference):
                content = upload.file.read(MAX_UPLOAD + 1)
                if len(content) > MAX_UPLOAD:
                    raise HTTPException(413, "Upload exceeds 24 MiB")
                sources.append(io.BytesIO(content))
            audio, metadata = app.state.extractor.extract_files(*sources)
            LOGGER.info(
                "extraction_complete duration=%.2f elapsed=%.3f model=%s",
                metadata["duration_seconds"],
                metadata["processing_seconds"],
                metadata["model_id"],
            )
            return Response(
                audio,
                media_type="audio/wav",
                headers={
                    "Content-Disposition": 'attachment; filename="isolated-voice.wav"',
                    "X-TSE-Metadata": json.dumps(metadata),
                    "Cache-Control": "no-store",
                },
            )
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        except RuntimeError as error:
            LOGGER.exception("Extraction failed")
            raise HTTPException(
                500, "Audio processing failed; inspect the local service log"
            ) from error
        finally:
            mixture.file.close()
            reference.file.close()
            gate.release()

    examples_root = examples_root or Path("artifacts/examples")
    gallery = Path("artifacts/gallery")

    @app.get("/examples")
    def examples():
        index = examples_root / "index.json"
        result = json.loads(index.read_text()) if index.is_file() else {"items": []}
        return {**result, "gallery_available": (gallery / "index.html").is_file()}

    if examples_root.is_dir():
        app.mount("/example-audio", StaticFiles(directory=examples_root), name="example-audio")

    if gallery.is_dir():
        app.mount("/gallery", StaticFiles(directory=gallery, html=True), name="gallery")

    static = Path(__file__).parent / "web"
    if static.is_dir():
        app.mount("/assets", StaticFiles(directory=static), name="assets")

        @app.get("/", include_in_schema=False)
        def index():
            return FileResponse(static / "index.html", headers={"Cache-Control": "no-cache"})

    return app
