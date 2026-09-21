"""Inference-only serving surface. Never exposes the local training controls."""
from contextlib import asynccontextmanager
import hmac
import io
import json
import os
from pathlib import Path
import threading

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
import numpy as np
import torch

from tse.audio import read_audio, wav_bytes
from tse.inference import Extractor

MAX_BYTES = 4 * 1024 * 1024


def create_public_app(checkpoint=None, extractor_factory=Extractor):
    checkpoint = Path(checkpoint or os.environ['TSE_CHECKPOINT'])
    token = os.environ.get('TSE_API_TOKEN', '')
    gate = threading.Lock()

    @asynccontextmanager
    async def lifespan(app):
        torch.set_num_threads(1)
        app.state.extractor = extractor_factory(checkpoint, 'cpu')
        yield
        app.state.extractor = None

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware('http')
    async def guard(request: Request, call_next):
        if token and not hmac.compare_digest(request.headers.get('authorization', ''), 'Bearer ' + token):
            return Response(status_code=401)
        # Limit the complete multipart body before Starlette can spool it to disk.
        if request.method == 'POST':
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > MAX_BYTES:
                    return Response(json.dumps({'detail': 'Combined upload must be under 4 MiB.'}), status_code=413, media_type='application/json')
            request._body = bytes(body)
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        return response

    @app.get('/model')
    def model(request: Request):
        info = request.app.state.extractor.info()
        return {**info, 'limits': {'mixture_seconds': 30, 'reference_seconds': [3, 10], 'combined_upload_mib': 4}}

    @app.post('/extract')
    async def extract(request: Request):
        if not gate.acquire(blocking=False):
            raise HTTPException(429, 'Another recording is processing. Try again shortly.')
        try:
            # In-memory parsing avoids UploadFile temporary files for user audio.
            from email.parser import BytesParser
            from email.policy import default
            content_type = request.headers.get('content-type', '')
            if not content_type.startswith('multipart/form-data;') or '\r' in content_type or '\n' in content_type:
                raise HTTPException(415, 'Upload a mixture and a separate voice sample.')
            message = BytesParser(policy=default).parsebytes(('Content-Type: ' + content_type + '\r\nMIME-Version: 1.0\r\n\r\n').encode() + await request.body())
            parts = list(message.iter_parts())
            if len(parts) != 2:
                raise HTTPException(422, 'Provide exactly two recordings.')
            files = {part.get_param('name', header='content-disposition'): part.get_payload(decode=True) for part in parts}
            if set(files) != {'mixture', 'reference'}:
                raise HTTPException(422, 'Provide the mixture and the voice sample.')
            mixture = read_audio(io.BytesIO(files['mixture']), max_seconds=30)
            reference = read_audio(io.BytesIO(files['reference']), max_seconds=10)
            if len(reference) < 3 * 16000:
                raise ValueError('The voice sample must be between 3 and 10 seconds.')
            from starlette.concurrency import run_in_threadpool
            result = await run_in_threadpool(request.app.state.extractor.extract_array, mixture, reference)
            gain = min(1.0, .98 / max(float(np.abs(result).max()), 1e-8))
            return Response(wav_bytes(result * gain), media_type='audio/wav', headers={'X-Checkpoint-SHA256': request.app.state.extractor.checkpoint_hash, 'Content-Disposition': 'attachment; filename="extracted-voice.wav"'})
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        finally:
            gate.release()

    return app
