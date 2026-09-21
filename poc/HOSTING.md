# Public inference on Hugging Face

The standalone website uses a protected Docker Space on CPU Basic. The Space's
app is reachable publicly; its source and frozen weights remain private. Pro is
the subscription; CPU Basic has no hourly hardware charge. Do not select a paid
CPU/GPU tier or persistent storage without a separate budget decision. Idle
workers may sleep, and the interface offers a retry while they wake.

## Service boundary

`poc.hosted:create_app` mounts only the extraction and transcription APIs. It
does not import the training/dashboard server or expose training controls.
Both APIs require a random `TSE_API_TOKEN` of at least 32 characters. All
transcription result routes also require a browser owner ID from the trusted
website proxy. The proxy signs an HttpOnly, Secure, SameSite=Strict session
cookie and never sends the service token to the browser.

The same frozen One Voice checkpoint runs extraction and the transcription
pipeline. ECAPA supplies identity evidence, Silero detects speech, and Whisper
transcribes. One Voice remains the only waveform separator. Comparison uses
the same ASR settings for original and extracted audio.

## Bundle and deploy

1. Provision `artifacts/poc/models` as described in `poc/README.md`; verify the
   manifest and checkpoint SHA before deployment. Never bundle the live run,
   dataset, jobs, saved user audio, environments, or credentials.
2. Create a **protected**, Docker SDK Space with **CPU Basic** hardware. Keep
   the model bundle out of the public GitHub repository.
3. Use `Dockerfile.public` as the Space's `Dockerfile` and
   `Dockerfile.public.dockerignore` as `.dockerignore`. Include `pyproject.toml`,
   the project README, `src`, `poc`, the frozen model directory,
   `site/src/snapshot.json`, and `site/public/assets/one-voice`. Exclude Python
   caches. Add this YAML header to the Space README:

   ```yaml
   ---
   title: OneVoice Inference
   emoji: 🎙️
   colorFrom: gray
   colorTo: gray
   sdk: docker
   app_port: 8080
   pinned: false
   ---
   ```

4. Set `TSE_API_TOKEN` as a Space **secret**, not a variable or committed file.
   The image runs as UID 1000 and reads the packaged models without downloading
   additional models during inference. Its result directory is ephemeral.
5. In the Vercel `one-voice` project, configure these server-side variables:

   | Variable | Value |
   | --- | --- |
   | `ONE_VOICE_SERVICE_URL` | Space app URL followed by `/voice` |
   | `ONE_VOICE_TRANSCRIPTION_URL` | Space app URL followed by `/speech` |
   | `ONE_VOICE_SERVICE_TOKEN` | Same value as the Space secret |
   | `ONE_VOICE_TRANSCRIPTION_TOKEN` | Same value as the Space secret |
   | `ONE_VOICE_PUBLIC_ORIGIN` | `https://one-voice.vercel.app` |

   Use HTTPS URLs. Do not use `VITE_` variables for credentials. Deploy the site
   after setting the variables; existing deployments retain their old values.

## Limits and verification

Uploads are bounded to 4 MiB total, a 30-second recording, and a 3–10 second
reference. Extraction permits one concurrent request. Transcription permits one
active job and one waiting job, with a ten-minute worker timeout and 4 GiB
resident-memory limit. Results expire after 15 minutes or earlier on a restart.
These bounds limit resource use; they do not guarantee capacity under abuse.
The public demo uses no paid API or autoscaling hardware.

Before routing production traffic, verify unauthenticated requests are rejected,
an example produces extracted WAV, and a complete transcription produces both
comparison paths, audio, and TXT/SRT/JSON exports. Check that another browser
cannot read or delete the result. Then verify the same flow through Vercel.

```sh
PYTHONPATH=src:. .venv-poc/bin/python -m pytest tests/test_public_api.py tests/test_transcription_poc.py -q
node site/scripts/check-transcription-proxy.mjs
npm run build --prefix site
```

Local development keeps its existing loopback services and Vite proxy. Deploying
or restarting the hosted worker has no effect on the local training process.
