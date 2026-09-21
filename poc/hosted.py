"""Authenticated inference container; never imports the training/dashboard API."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from poc.server import create_app as create_transcription_app
from tse.public_api import create_public_app


def create_app():
    token = os.environ.get("TSE_API_TOKEN", "")
    if len(token) < 32:
        raise ValueError("Set TSE_API_TOKEN to a random secret of at least 32 characters.")
    models = Path(os.environ.get("ONE_VOICE_MODELS", "/app/artifacts/poc/models"))
    extraction = create_public_app(models / "one-voice.pt")
    transcription = create_transcription_app(
        directory=os.environ.get("ONE_VOICE_JOBS", "/tmp/one-voice-jobs"),
        models=models,
        access_token=token,
    )

    @asynccontextmanager
    async def lifespan(app):
        async with extraction.router.lifespan_context(extraction):
            async with transcription.router.lifespan_context(transcription):
                yield

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/")
    async def status():
        return {"service": "OneVoice inference", "website": "https://one-voice.vercel.app"}

    app.mount("/voice", extraction)
    app.mount("/speech", transcription)
    return app
