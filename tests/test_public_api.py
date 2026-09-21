import io
import numpy as np
from fastapi.testclient import TestClient
from tse.audio import wav_bytes
from tse.public_api import create_public_app, MAX_BYTES


class FakeExtractor:
    checkpoint_hash = 'test-checkpoint'
    def __init__(self, *_): pass
    def info(self): return {'ready': True, 'training_updates': 135525}
    def extract_array(self, mixture, reference): return mixture * .5


def recordings():
    wave = wav_bytes((.1 * np.sin(np.arange(64000) / 11)).astype(np.float32))
    return {name: ('test.wav', io.BytesIO(wave), 'audio/wav') for name in ['mixture', 'reference']}


def test_public_surface_and_audio(monkeypatch):
    monkeypatch.delenv('TSE_API_TOKEN', raising=False)
    with TestClient(create_public_app('unused', FakeExtractor)) as client:
        assert client.get('/model').json()['limits']['mixture_seconds'] == 30
        assert client.post('/experiments/full/control', json={'action': 'pause'}).status_code == 404
        response = client.post('/extract', files=recordings())
        assert response.status_code == 200
        assert response.content.startswith(b'RIFF')
        assert response.headers['x-checkpoint-sha256'] == 'test-checkpoint'
        assert response.headers['cache-control'] == 'no-store'
        assert client.post('/extract', content=b'x' * (MAX_BYTES + 1)).status_code == 413
        assert client.post('/extract', files={'mixture': ('bad.wav', b'bad')}).status_code == 422


def test_service_requires_configured_token(monkeypatch):
    monkeypatch.setenv('TSE_API_TOKEN', 'private-test-key')
    with TestClient(create_public_app('unused', FakeExtractor)) as client:
        assert client.get('/model').status_code == 401
        assert client.get('/model', headers={'Authorization': 'Bearer private-test-key'}).status_code == 200
